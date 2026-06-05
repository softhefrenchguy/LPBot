from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = ["BULL", "CHOP", "BEAR"]


def load_daily_close(path: Path, symbol_name: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns or "close" not in d.columns:
        raise ValueError(f"{path} must contain timestamp and close")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out = (
        d.set_index("timestamp")["close"]
        .resample("1D")
        .last()
        .dropna()
        .to_frame(f"{symbol_name}_close")
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
        # Recompute runs if changed.
    return x


def classify_v2(
    daily: pd.DataFrame,
    dd_window: int,
    dd_bear_th: float,
    min_persistence: int,
    fwd_days: int,
    fwd_bear_th: float,
    fwd_bull_th: float,
) -> pd.DataFrame:
    d = daily.copy()
    d["ret_1d"] = np.log(d["eth_close"] / d["eth_close"].shift(1)).fillna(0.0)
    d["ema20"] = d["eth_close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["eth_close"].ewm(span=50, adjust=False).mean()
    d["ret_5d"] = d["eth_close"].pct_change(5)
    d["ret_10d"] = d["eth_close"].pct_change(10)
    d["rolling_dd"] = d["eth_close"] / d["eth_close"].rolling(int(dd_window), min_periods=max(5, dd_window // 2)).max() - 1.0

    bull_raw = (
        (d["eth_close"] > d["ema20"])
        & (d["ema20"] > d["ema50"])
        & (d["ret_5d"] > 0.0)
        & (d["ret_10d"] > 0.0)
    )
    bear_raw = (
        (d["eth_close"] < d["ema20"])
        & (d["ema20"] < d["ema50"])
        & (d["ret_5d"] < 0.0)
    )
    regime_raw = np.where(bull_raw, "BULL", np.where(bear_raw, "BEAR", "CHOP"))
    d["regime_raw"] = regime_raw

    # Rule 1: drawdown hard override.
    d.loc[d["rolling_dd"] < float(dd_bear_th), "regime_raw"] = "BEAR"

    # Rule 2: minimum persistence filter.
    smoothed = smooth_short_islands(d["regime_raw"], int(min_persistence))
    d["regime_smoothed"] = pd.Series(smoothed.to_numpy(), index=d.index)

    # Live regime: 1-day lag to avoid lookahead.
    d["regime_v2"] = d["regime_smoothed"].shift(1).fillna("CHOP")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime classifier v2 with DD override + persistence + forward labels.")
    ap.add_argument("--eth-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--dd-window", type=int, default=20)
    ap.add_argument("--dd-bear-th", type=float, default=-0.15)
    ap.add_argument("--min-persistence", type=int, default=5)
    ap.add_argument("--fwd-days", type=int, default=20)
    ap.add_argument("--fwd-bear-th", type=float, default=-0.05)
    ap.add_argument("--fwd-bull-th", type=float, default=0.05)
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/regime_classifier_v2_checks.csv")
    ap.add_argument("--out-transition-csv", default="artifacts/backtest/regime_classifier_v2_transition.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_classifier_v2_report.html")
    ap.add_argument("--skip-html", action="store_true", help="Skip HTML report generation (no plotly required).")
    args = ap.parse_args()

    eth = load_daily_close(Path(args.eth_csv), "eth")
    if Path(args.btc_csv).exists():
        btc = load_daily_close(Path(args.btc_csv), "btc")
        daily = eth.join(btc, how="left")
        daily["btc_close"] = daily["btc_close"].ffill()
    else:
        daily = eth.copy()
        daily["btc_close"] = np.nan
    daily = daily.dropna(subset=["eth_close"]).copy()

    cls = classify_v2(
        daily=daily,
        dd_window=int(args.dd_window),
        dd_bear_th=float(args.dd_bear_th),
        min_persistence=int(args.min_persistence),
        fwd_days=int(args.fwd_days),
        fwd_bear_th=float(args.fwd_bear_th),
        fwd_bull_th=float(args.fwd_bull_th),
    ).reset_index().rename(columns={"index": "day", "timestamp": "day"})
    cls["day"] = pd.to_datetime(cls["day"], utc=True, errors="coerce")

    # Check 2: label distribution.
    dist = cls["regime_v2"].value_counts(normalize=True).mul(100).reindex(REGIMES, fill_value=0.0)
    dist_df = pd.DataFrame({"check": "distribution_pct", "regime": dist.index, "value": dist.values})

    # Check 3: forward return validation.
    fwd_mean = cls.groupby("regime_v2")["fwd_20d_return"].mean().reindex(REGIMES)
    fwd_df = pd.DataFrame({"check": "fwd20d_mean_return", "regime": fwd_mean.index, "value": fwd_mean.values})

    # Check 4: transition matrix.
    trans = transition_matrix(cls["regime_v2"])
    trans_long = trans.stack().rename("prob").reset_index()

    checks = pd.concat([dist_df, fwd_df], ignore_index=True)
    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    cls.to_csv(out_daily, index=False)

    out_checks = Path(args.out_summary_csv)
    out_checks.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_checks, index=False)

    out_trans = Path(args.out_transition_csv)
    out_trans.parent.mkdir(parents=True, exist_ok=True)
    trans_long.to_csv(out_trans, index=False)

    if not args.skip_html:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["ETH/BTC Price + Regime", "Forward 20d Return", "Rolling DD"])
        fig.add_trace(go.Scatter(x=cls["day"], y=cls["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=1, col=1)
        if cls["btc_close"].notna().any():
            b = cls["btc_close"] / cls["btc_close"].iloc[0] * cls["eth_close"].iloc[0]
            fig.add_trace(go.Scatter(x=cls["day"], y=b, name="BTC Close (scaled)", line=dict(color="#1f77b4")), row=1, col=1)

        for rg, color in [("BULL", "rgba(46,204,113,0.10)"), ("BEAR", "rgba(231,76,60,0.10)"), ("CHOP", "rgba(149,165,166,0.08)")]:
            m = cls["regime_v2"] == rg
            starts = cls.loc[m & ~m.shift(1, fill_value=False), "day"]
            ends = cls.loc[m & ~m.shift(-1, fill_value=False), "day"]
            for sdt, edt in zip(starts, ends):
                fig.add_vrect(x0=sdt, x1=edt, fillcolor=color, line_width=0, row=1, col=1)

        fig.add_trace(go.Scatter(x=cls["day"], y=cls["fwd_20d_return"], name="fwd_20d_return", line=dict(color="#9467bd")), row=2, col=1)
        fig.add_hline(y=float(args.fwd_bull_th), line_dash="dash", line_color="#2ca02c", row=2, col=1)
        fig.add_hline(y=float(args.fwd_bear_th), line_dash="dash", line_color="#d62728", row=2, col=1)
        fig.add_trace(go.Scatter(x=cls["day"], y=cls["rolling_dd"], name="rolling_dd", line=dict(color="#8c564b")), row=3, col=1)
        fig.add_hline(y=float(args.dd_bear_th), line_dash="dash", line_color="#d62728", row=3, col=1)
        fig.update_layout(height=1100, title="Regime Classifier v2")

        html = (
            "<html><head><meta charset='utf-8'><title>Regime Classifier v2</title></head><body>"
            "<h3>Regime Classifier v2</h3>"
            f"<p>dd_window={args.dd_window} dd_bear_th={args.dd_bear_th} min_persistence={args.min_persistence} fwd_days={args.fwd_days}</p>"
            "<h4>Check 2: Distribution (%)</h4>"
            + dist_df.round(6).to_html(index=False, border=0)
            + "<h4>Check 3: Forward 20d Mean Return by regime_v2</h4>"
            + fwd_df.round(6).to_html(index=False, border=0)
            + "<h4>Check 4: Transition Matrix (regime_v2 -> next day)</h4>"
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
    if not args.skip_html:
        print(f"wrote {args.out_html}")
    print("\nCheck 2 - distribution (%):")
    print(dist.to_string())
    print("\nCheck 3 - mean fwd_20d_return by regime_v2:")
    print(fwd_mean.to_string())
    print("\nCheck 4 - transition matrix:")
    print(trans.to_string())


if __name__ == "__main__":
    main()
