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


def _confirm_consecutive(x: pd.Series, n: int) -> pd.Series:
    n = int(max(1, n))
    return x.astype(int).rolling(n, min_periods=n).min().fillna(0).astype(bool)


def build_bear_state(
    bear_enter_sig: pd.Series,
    bear_exit_sig: pd.Series,
    bear_enter_n: int,
    bear_exit_n: int,
) -> pd.Series:
    enter_conf = _confirm_consecutive(bear_enter_sig, bear_enter_n)
    exit_conf = _confirm_consecutive(bear_exit_sig, bear_exit_n)

    state = False
    out = np.zeros(len(bear_enter_sig), dtype=bool)
    for i in range(len(out)):
        if not state and bool(enter_conf.iloc[i]):
            state = True
        elif state and bool(exit_conf.iloc[i]):
            state = False
        out[i] = state
    return pd.Series(out, index=bear_enter_sig.index)


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

    rd = regime_daily[["day", "regime_v4"]].copy().sort_values("day")
    rd["prev"] = rd["regime_v4"].shift(1)
    bull_starts = rd.loc[(rd["regime_v4"] == "BULL") & (rd["prev"] != "BULL"), "day"].reset_index(drop=True)

    rows = []
    for ed in entry_days:
        future = bull_starts[bull_starts >= ed]
        if len(future) == 0:
            continue
        rows.append({"entry_day": ed, "next_bull_start": future.iloc[0], "days_to_next_bull_start": int((future.iloc[0] - ed).days)})
    pairs = pd.DataFrame(rows)

    checks = [
        {"check": "entry_days", "value": float(len(entry_days))},
        {"check": "bull_starts", "value": float(len(bull_starts))},
        {"check": "entry_to_next_pairs", "value": float(len(pairs))},
    ]
    if len(pairs):
        s = pairs["days_to_next_bull_start"]
        checks.extend(
            [
                {"check": "entry_to_next_median_days", "value": float(s.median())},
                {"check": "entry_to_next_mean_days", "value": float(s.mean())},
                {"check": "entry_to_next_pct_lead_or_same", "value": float((s <= 0).mean() * 100.0)},
                {"check": "entry_to_next_pct_within_7d", "value": float(((s >= 0) & (s <= 7)).mean() * 100.0)},
            ]
        )
    return pd.DataFrame(checks), pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime classifier v4 (asymmetric bear/bull detector)")
    ap.add_argument("--eth-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--momentum-csv", default="artifacts/backtest/offense_momentum_only_v1_5y.csv")
    ap.add_argument("--dd-window", type=int, default=20)
    ap.add_argument("--dd-bear-th", type=float, default=-0.15)
    ap.add_argument("--dd-exit-th", type=float, default=-0.08)
    ap.add_argument("--bear-ret10d-th", type=float, default=-0.08)
    ap.add_argument("--bull-ret5d-th", type=float, default=0.00)
    ap.add_argument("--bear-enter-persist", type=int, default=5)
    ap.add_argument("--bear-exit-persist", type=int, default=5)
    ap.add_argument("--bull-persist", type=int, default=3)
    ap.add_argument("--fwd-days", type=int, default=20)
    ap.add_argument("--fwd-bear-th", type=float, default=-0.05)
    ap.add_argument("--fwd-bull-th", type=float, default=0.05)
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/regime_classifier_v4_daily.csv")
    ap.add_argument("--out-checks-csv", default="artifacts/backtest/regime_classifier_v4_checks.csv")
    ap.add_argument("--out-transition-csv", default="artifacts/backtest/regime_classifier_v4_transition.csv")
    ap.add_argument("--out-lag-csv", default="artifacts/backtest/regime_classifier_v4_lag_pairs.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_classifier_v4_report.html")
    args = ap.parse_args()

    eth = load_daily_close(Path(args.eth_csv), "eth")
    daily = eth.copy()
    if Path(args.btc_csv).exists():
        btc = load_daily_close(Path(args.btc_csv), "btc")
        daily = daily.join(btc, how="left")
        daily["btc_close"] = daily["btc_close"].ffill()
    else:
        daily["btc_close"] = np.nan
    daily = daily.dropna(subset=["eth_close"]).copy()

    d = daily.copy()
    d["ret_1d"] = d["eth_close"].pct_change()
    d["ret_5d"] = d["eth_close"].pct_change(5)
    d["ret_10d"] = d["eth_close"].pct_change(10)
    d["ema20"] = d["eth_close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["eth_close"].ewm(span=50, adjust=False).mean()
    d["ema20_slope_3d"] = d["ema20"] / d["ema20"].shift(3) - 1.0
    d["rolling_dd"] = d["eth_close"] / d["eth_close"].rolling(int(args.dd_window), min_periods=max(5, int(args.dd_window // 2))).max() - 1.0

    # Conservative BEAR detector.
    bear_enter_sig = (d["rolling_dd"] < float(args.dd_bear_th)) | (
        (d["ema20"] < d["ema50"]) & (d["ret_10d"] < float(args.bear_ret10d_th))
    )
    # Slow BEAR exit condition.
    bear_exit_sig = (
        (d["rolling_dd"] > float(args.dd_exit_th))
        & (d["ret_5d"] > 0.0)
        & (d["ema20_slope_3d"] > 0.0)
    )
    d["bear_state"] = build_bear_state(
        bear_enter_sig=bear_enter_sig.fillna(False),
        bear_exit_sig=bear_exit_sig.fillna(False),
        bear_enter_n=int(args.bear_enter_persist),
        bear_exit_n=int(args.bear_exit_persist),
    )

    # Fast BULL detector (but blocked while BEAR is confirmed).
    bull_sig = (
        (d["ret_5d"] > float(args.bull_ret5d_th))
        & (d["ret_10d"] > 0.0)
        & (d["eth_close"] > d["ema20"])
        & (d["ema20_slope_3d"] > 0.0)
        & (~d["bear_state"])
    )
    d["bull_state"] = _confirm_consecutive(bull_sig.fillna(False), int(args.bull_persist))

    d["regime_raw"] = np.where(d["bull_state"], "BULL", "CHOP")
    d.loc[d["bear_state"], "regime_raw"] = "BEAR"  # BEAR always overrides BULL.

    d["regime_v4"] = pd.Series(d["regime_raw"], index=d.index).shift(1).fillna("CHOP")

    d["fwd_20d_return"] = d["eth_close"].pct_change(int(args.fwd_days)).shift(-int(args.fwd_days))
    d["regime_fwd_label"] = pd.cut(
        d["fwd_20d_return"],
        bins=[-np.inf, float(args.fwd_bear_th), float(args.fwd_bull_th), np.inf],
        labels=["BEAR", "CHOP", "BULL"],
    ).astype(str)

    cls = d.reset_index().rename(columns={"index": "day", "timestamp": "day"})
    cls["day"] = pd.to_datetime(cls["day"], utc=True, errors="coerce")

    dist = cls["regime_v4"].value_counts(normalize=True).mul(100).reindex(REGIMES, fill_value=0.0)
    fwd_mean = cls.groupby("regime_v4")["fwd_20d_return"].mean().reindex(REGIMES)
    trans = transition_matrix(cls["regime_v4"])
    trans_long = trans.stack().rename("prob").reset_index()

    lag_checks, lag_pairs = lag_diagnostics(cls[["day", "regime_v4"]], Path(args.momentum_csv))

    checks = pd.concat(
        [
            pd.DataFrame({"check": "distribution_pct", "regime": dist.index, "value": dist.values}),
            pd.DataFrame({"check": "fwd20d_mean_return", "regime": fwd_mean.index, "value": fwd_mean.values}),
            lag_checks,
        ],
        ignore_index=True,
    )

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
        m = cls["regime_v4"] == rg
        starts = cls.loc[m & ~m.shift(1, fill_value=False), "day"]
        ends = cls.loc[m & ~m.shift(-1, fill_value=False), "day"]
        for sdt, edt in zip(starts, ends):
            fig.add_vrect(x0=sdt, x1=edt, fillcolor=color, line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=cls["day"], y=cls["fwd_20d_return"], name="fwd_20d_return", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_hline(y=float(args.fwd_bull_th), line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.add_hline(y=float(args.fwd_bear_th), line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_trace(go.Scatter(x=cls["day"], y=cls["rolling_dd"], name="rolling_dd", line=dict(color="#8c564b")), row=3, col=1)
    fig.add_hline(y=float(args.dd_bear_th), line_dash="dash", line_color="#d62728", row=3, col=1)
    fig.add_hline(y=float(args.dd_exit_th), line_dash="dash", line_color="#1f77b4", row=3, col=1)
    fig.update_layout(height=1100, title="Regime Classifier v4")

    html = (
        "<html><head><meta charset='utf-8'><title>Regime Classifier v4</title></head><body>"
        "<h3>Regime Classifier v4</h3>"
        f"<p>bear_enter_persist={args.bear_enter_persist}, bear_exit_persist={args.bear_exit_persist}, bull_persist={args.bull_persist}</p>"
        "<h4>Distribution (%)</h4>"
        + pd.DataFrame({"regime": dist.index, "pct": dist.values}).round(6).to_html(index=False, border=0)
        + "<h4>Forward 20d Mean Return by regime_v4</h4>"
        + pd.DataFrame({"regime": fwd_mean.index, "fwd_20d_mean": fwd_mean.values}).round(6).to_html(index=False, border=0)
        + "<h4>Lag checks (momentum entries vs BULL starts)</h4>"
        + lag_checks.round(6).to_html(index=False, border=0)
        + "<h4>Transition Matrix (regime_v4 -> next day)</h4>"
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
    print("\nMean fwd_20d_return by regime_v4:")
    print(fwd_mean.to_string())
    print("\nLag checks:")
    print(lag_checks.to_string(index=False))


if __name__ == "__main__":
    main()
