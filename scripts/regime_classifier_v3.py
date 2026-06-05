from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REGIMES = ["BULL", "CHOP", "BEAR"]


def load_daily_close(path: Path, symbol_name: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns or "close" not in d.columns:
        raise ValueError(f"{path} must contain timestamp and close")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return (
        d.set_index("timestamp")["close"]
        .resample("1D")
        .last()
        .dropna()
        .to_frame(f"{symbol_name}_close")
    )


def load_perp_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns:
        raise ValueError(f"{path} must contain timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    cols = {}
    if "funding_rate" in d.columns:
        d["funding_rate"] = pd.to_numeric(d["funding_rate"], errors="coerce")
        cols["funding_rate"] = "mean"
    if "basis" in d.columns:
        d["basis"] = pd.to_numeric(d["basis"], errors="coerce")
        cols["basis"] = "mean"
    if not cols:
        return pd.DataFrame()
    out = (
        d.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("1D")
        .agg(cols)
        .sort_index()
    )
    return out


def smooth_short_islands(labels: pd.Series, min_persistence: int) -> pd.Series:
    x = labels.astype(str).copy().reset_index(drop=True)
    n = len(x)
    if n == 0 or min_persistence <= 1:
        return x

    changed = True
    while changed:
        changed = False
        run_id = (x != x.shift(1)).cumsum()
        runs = (
            pd.DataFrame({"idx": np.arange(n), "label": x, "run": run_id})
            .groupby("run")
            .agg(start=("idx", "min"), end=("idx", "max"), length=("idx", "size"), label=("label", "first"))
            .reset_index(drop=True)
        )
        for i in range(len(runs)):
            r = runs.iloc[i]
            if int(r["length"]) >= int(min_persistence):
                continue
            if i == 0 or i == len(runs) - 1:
                continue
            prev_label = runs.iloc[i - 1]["label"]
            next_label = runs.iloc[i + 1]["label"]
            if prev_label == next_label:
                s, e = int(r["start"]), int(r["end"])
                x.iloc[s : e + 1] = prev_label
                changed = True
    return x


def classify_v3(
    daily: pd.DataFrame,
    dd_window: int,
    dd_bear_th: float,
    min_persistence: int,
    recovery_window: int,
    recovery_th: float,
    fwd_days: int,
    fwd_bear_th: float,
    fwd_bull_th: float,
) -> pd.DataFrame:
    d = daily.copy()
    d["ret_1d"] = d["eth_close"].pct_change()
    d["ret_5d"] = d["eth_close"].pct_change(5)
    d["ret_10d"] = d["eth_close"].pct_change(10)
    d["ema10"] = d["eth_close"].ewm(span=10, adjust=False).mean()
    d["ema20"] = d["eth_close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["eth_close"].ewm(span=50, adjust=False).mean()
    d["ema20_slope_3d"] = d["ema20"] / d["ema20"].shift(3) - 1.0

    d["vol20"] = d["ret_1d"].rolling(20, min_periods=8).std()
    d["vol_q40_120"] = d["vol20"].rolling(120, min_periods=40).quantile(0.40)
    d["vol_q75_120"] = d["vol20"].rolling(120, min_periods=40).quantile(0.75)
    d["had_expansion_30d"] = d["vol20"].rolling(30, min_periods=10).max() > d["vol_q75_120"]
    d["vol_compress_after_expansion"] = (d["vol20"] < d["vol_q40_120"]) & d["had_expansion_30d"]

    d["rolling_dd"] = d["eth_close"] / d["eth_close"].rolling(int(dd_window), min_periods=max(5, dd_window // 2)).max() - 1.0
    d["recovery_from_local_low"] = d["eth_close"] / d["eth_close"].rolling(int(recovery_window), min_periods=max(5, recovery_window // 2)).min() - 1.0
    d["mom_pos_3d"] = (d["ret_1d"] > 0).rolling(3, min_periods=3).sum() == 3

    if "funding_rate" in d.columns:
        d["funding_3d_chg"] = d["funding_rate"].diff(3)
        d["funding_inflect"] = (d["funding_3d_chg"] > 0.0) & (d["funding_rate"] > d["funding_rate"].rolling(30, min_periods=8).quantile(0.40))
    else:
        d["funding_inflect"] = False

    if "basis" in d.columns:
        d["basis_3d_chg"] = d["basis"].diff(3)
        d["basis_turn_pos"] = (d["basis"] > 0.0) & ((d["basis"].shift(3) < 0.0) | (d["basis_3d_chg"] > 0.0))
    else:
        d["basis_turn_pos"] = False

    lead_any = d["funding_inflect"].fillna(False) | d["basis_turn_pos"].fillna(False) | d["vol_compress_after_expansion"].fillna(False)

    bull_core = (
        (d["ret_5d"] > 0.0)
        & (d["ret_10d"] > 0.0)
        & (d["eth_close"] > d["ema20"])
        & (d["ema20_slope_3d"] > 0.0)
    )
    bull_raw = bull_core & (lead_any | (d["eth_close"] > d["ema50"] * 0.995))

    bear_core = (
        (d["eth_close"] < d["ema20"])
        & (d["ret_5d"] < 0.0)
        & (d["ret_10d"] < 0.0)
        & (d["ema20_slope_3d"] < 0.0)
    )

    regime_raw = np.where(bull_raw, "BULL", np.where(bear_core, "BEAR", "CHOP"))
    d["regime_raw"] = regime_raw

    # Rule 1: drawdown hard override.
    d.loc[d["rolling_dd"] < float(dd_bear_th), "regime_raw"] = "BEAR"

    # Faster BEAR exit on recovery + positive short momentum.
    bear_escape = (
        (d["regime_raw"] == "BEAR")
        & (d["recovery_from_local_low"] > float(recovery_th))
        & d["mom_pos_3d"].fillna(False)
    )
    d.loc[bear_escape, "regime_raw"] = "CHOP"

    # Rule 2: minimum persistence filter (v3 defaults lower than v2).
    smoothed = smooth_short_islands(d["regime_raw"], int(min_persistence))
    d["regime_smoothed"] = pd.Series(smoothed.to_numpy(), index=d.index)

    # Live regime: 1-day lag.
    d["regime_v3"] = d["regime_smoothed"].shift(1).fillna("CHOP")

    # Rule 3: forward-return labels for attribution only.
    d["fwd_20d_return"] = d["eth_close"].pct_change(int(fwd_days)).shift(-int(fwd_days))
    d["regime_fwd_label"] = pd.cut(
        d["fwd_20d_return"],
        bins=[-np.inf, float(fwd_bear_th), float(fwd_bull_th), np.inf],
        labels=["BEAR", "CHOP", "BULL"],
    ).astype(str)

    return d


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    cur = labels.astype(str)
    nxt = cur.shift(-1)
    tab = pd.crosstab(cur, nxt, normalize="index")
    tab = tab.reindex(index=REGIMES, columns=REGIMES, fill_value=0.0)
    tab.index.name = "regime_from"
    tab.columns.name = "regime_to"
    return tab


def lag_diagnostics(regime_daily: pd.DataFrame, momentum_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(momentum_csv)
    if "timestamp" not in m.columns or "weight" not in m.columns:
        raise ValueError("momentum csv must contain timestamp and weight")
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True, errors="coerce")
    m["weight"] = pd.to_numeric(m["weight"], errors="coerce").fillna(0.0)
    m = m.dropna(subset=["timestamp"]).sort_values("timestamp")
    m["day"] = m["timestamp"].dt.floor("D")
    md = m.groupby("day", as_index=False).agg(momentum_on=("weight", lambda s: int((s > 1e-9).any())))
    md["prev_on"] = md["momentum_on"].shift(1).fillna(0).astype(int)
    entry_days = md.loc[(md["momentum_on"] == 1) & (md["prev_on"] == 0), "day"].reset_index(drop=True)

    rd = regime_daily[["day", "regime_v3"]].copy().sort_values("day")
    rd["prev"] = rd["regime_v3"].shift(1)
    bull_starts = rd.loc[(rd["regime_v3"] == "BULL") & (rd["prev"] != "BULL"), "day"].reset_index(drop=True)

    # A) Entry -> next bull start (days to next BULL switch)
    rows_a = []
    for ed in entry_days:
        future = bull_starts[bull_starts >= ed]
        if len(future) == 0:
            continue
        rows_a.append({"entry_day": ed, "next_bull_start": future.iloc[0], "days_to_next_bull_start": int((future.iloc[0] - ed).days)})
    a = pd.DataFrame(rows_a)

    # B) Bull start -> latest prior entry (lead in days before switch)
    rows_b = []
    for bd in bull_starts:
        past = entry_days[entry_days <= bd]
        if len(past) == 0:
            continue
        rows_b.append({"bull_start": bd, "prior_entry_day": past.iloc[-1], "days_since_prior_entry": int((bd - past.iloc[-1]).days)})
    b = pd.DataFrame(rows_b)

    checks = []
    checks.append({"check": "entry_days", "value": float(len(entry_days))})
    checks.append({"check": "bull_starts", "value": float(len(bull_starts))})
    checks.append({"check": "entry_to_next_pairs", "value": float(len(a))})
    checks.append({"check": "bull_to_prior_pairs", "value": float(len(b))})
    if len(a):
        sa = a["days_to_next_bull_start"]
        checks.extend(
            [
                {"check": "entry_to_next_median_days", "value": float(sa.median())},
                {"check": "entry_to_next_mean_days", "value": float(sa.mean())},
                {"check": "entry_to_next_pct_lead_or_same", "value": float((sa <= 0).mean() * 100.0)},
                {"check": "entry_to_next_pct_within_7d", "value": float(((sa >= 0) & (sa <= 7)).mean() * 100.0)},
            ]
        )
    if len(b):
        sb = b["days_since_prior_entry"]
        checks.extend(
            [
                {"check": "bull_to_prior_median_days", "value": float(sb.median())},
                {"check": "bull_to_prior_mean_days", "value": float(sb.mean())},
                {"check": "bull_to_prior_pct_within_7d", "value": float((sb <= 7).mean() * 100.0)},
            ]
        )
    return pd.DataFrame(checks), a


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime classifier v3 (faster bull detection + DD override + persistence)")
    ap.add_argument("--eth-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="", help="Optional perp features CSV (basis/funding)")
    ap.add_argument("--momentum-csv", default="artifacts/backtest/offense_momentum_only_v1_5y.csv")
    ap.add_argument("--dd-window", type=int, default=20)
    ap.add_argument("--dd-bear-th", type=float, default=-0.15)
    ap.add_argument("--min-persistence", type=int, default=3)
    ap.add_argument("--recovery-window", type=int, default=20)
    ap.add_argument("--recovery-th", type=float, default=0.08)
    ap.add_argument("--fwd-days", type=int, default=20)
    ap.add_argument("--fwd-bear-th", type=float, default=-0.05)
    ap.add_argument("--fwd-bull-th", type=float, default=0.05)
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/regime_classifier_v3_daily.csv")
    ap.add_argument("--out-checks-csv", default="artifacts/backtest/regime_classifier_v3_checks.csv")
    ap.add_argument("--out-transition-csv", default="artifacts/backtest/regime_classifier_v3_transition.csv")
    ap.add_argument("--out-lag-csv", default="artifacts/backtest/regime_classifier_v3_lag_pairs.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_classifier_v3_report.html")
    args = ap.parse_args()

    eth = load_daily_close(Path(args.eth_csv), "eth")
    daily = eth.copy()
    if Path(args.btc_csv).exists():
        btc = load_daily_close(Path(args.btc_csv), "btc")
        daily = daily.join(btc, how="left")
        daily["btc_close"] = daily["btc_close"].ffill()
    else:
        daily["btc_close"] = np.nan

    if args.perp_csv and Path(args.perp_csv).exists():
        perp = load_perp_daily(Path(args.perp_csv))
        if len(perp):
            daily = daily.join(perp, how="left")
    daily = daily.dropna(subset=["eth_close"]).copy()

    cls = classify_v3(
        daily=daily,
        dd_window=int(args.dd_window),
        dd_bear_th=float(args.dd_bear_th),
        min_persistence=int(args.min_persistence),
        recovery_window=int(args.recovery_window),
        recovery_th=float(args.recovery_th),
        fwd_days=int(args.fwd_days),
        fwd_bear_th=float(args.fwd_bear_th),
        fwd_bull_th=float(args.fwd_bull_th),
    ).reset_index().rename(columns={"index": "day", "timestamp": "day"})
    cls["day"] = pd.to_datetime(cls["day"], utc=True, errors="coerce")

    dist = cls["regime_v3"].value_counts(normalize=True).mul(100).reindex(REGIMES, fill_value=0.0)
    dist_df = pd.DataFrame({"check": "distribution_pct", "regime": dist.index, "value": dist.values})

    fwd_mean = cls.groupby("regime_v3")["fwd_20d_return"].mean().reindex(REGIMES)
    fwd_df = pd.DataFrame({"check": "fwd20d_mean_return", "regime": fwd_mean.index, "value": fwd_mean.values})

    trans = transition_matrix(cls["regime_v3"])
    trans_long = trans.stack().rename("prob").reset_index()

    lag_checks = pd.DataFrame(columns=["check", "value"])
    lag_pairs = pd.DataFrame(columns=["entry_day", "next_bull_start", "days_to_next_bull_start"])
    if args.momentum_csv and Path(args.momentum_csv).exists():
        lag_checks, lag_pairs = lag_diagnostics(cls[["day", "regime_v3"]], Path(args.momentum_csv))

    checks = pd.concat([dist_df, fwd_df, lag_checks], ignore_index=True)

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    cls.to_csv(out_daily, index=False)

    out_checks = Path(args.out_checks_csv)
    out_checks.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_checks, index=False)

    out_trans = Path(args.out_transition_csv)
    out_trans.parent.mkdir(parents=True, exist_ok=True)
    trans_long.to_csv(out_trans, index=False)

    out_lag = Path(args.out_lag_csv)
    out_lag.parent.mkdir(parents=True, exist_ok=True)
    lag_pairs.to_csv(out_lag, index=False)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["ETH/BTC Price + Regime", "Forward 20d Return", "Rolling DD"])
    fig.add_trace(go.Scatter(x=cls["day"], y=cls["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=1, col=1)
    if cls["btc_close"].notna().any():
        b = cls["btc_close"] / cls["btc_close"].iloc[0] * cls["eth_close"].iloc[0]
        fig.add_trace(go.Scatter(x=cls["day"], y=b, name="BTC Close (scaled)", line=dict(color="#1f77b4")), row=1, col=1)

    for rg, color in [("BULL", "rgba(46,204,113,0.10)"), ("BEAR", "rgba(231,76,60,0.10)"), ("CHOP", "rgba(149,165,166,0.08)")]:
        m = cls["regime_v3"] == rg
        starts = cls.loc[m & ~m.shift(1, fill_value=False), "day"]
        ends = cls.loc[m & ~m.shift(-1, fill_value=False), "day"]
        for sdt, edt in zip(starts, ends):
            fig.add_vrect(x0=sdt, x1=edt, fillcolor=color, line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=cls["day"], y=cls["fwd_20d_return"], name="fwd_20d_return", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_hline(y=float(args.fwd_bull_th), line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.add_hline(y=float(args.fwd_bear_th), line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_trace(go.Scatter(x=cls["day"], y=cls["rolling_dd"], name="rolling_dd", line=dict(color="#8c564b")), row=3, col=1)
    fig.add_hline(y=float(args.dd_bear_th), line_dash="dash", line_color="#d62728", row=3, col=1)
    fig.update_layout(height=1100, title="Regime Classifier v3")

    html = (
        "<html><head><meta charset='utf-8'><title>Regime Classifier v3</title></head><body>"
        "<h3>Regime Classifier v3</h3>"
        f"<p>dd_window={args.dd_window} dd_bear_th={args.dd_bear_th} min_persistence={args.min_persistence} recovery_th={args.recovery_th}</p>"
        "<h4>Distribution (%)</h4>"
        + dist_df.round(6).to_html(index=False, border=0)
        + "<h4>Forward 20d Mean Return by regime_v3</h4>"
        + fwd_df.round(6).to_html(index=False, border=0)
        + "<h4>Lag checks (momentum entries vs BULL starts)</h4>"
        + lag_checks.round(6).to_html(index=False, border=0)
        + "<h4>Transition Matrix (regime_v3 -> next day)</h4>"
        + trans.round(6).to_html(border=0)
        + fig.to_html(full_html=False, include_plotlyjs="cdn")
        + "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_daily}")
    print(f"wrote {out_checks}")
    print(f"wrote {out_trans}")
    print(f"wrote {out_lag}")
    print(f"wrote {out_html}")
    print("\nDistribution (%):")
    print(dist.to_string())
    print("\nMean fwd_20d_return by regime_v3:")
    print(fwd_mean.to_string())
    if len(lag_checks):
        print("\nLag checks:")
        print(lag_checks.to_string(index=False))


if __name__ == "__main__":
    main()
