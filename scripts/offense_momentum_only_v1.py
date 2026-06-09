from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bars_per_year)),
        "sharpe": float((np.mean(x) * bars_per_year) / (np.std(x) * np.sqrt(bars_per_year) + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offense sleeve: momentum-only (daily features, 5m execution)")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--ema-fast-days", type=int, default=20)
    ap.add_argument("--ema-slow-days", type=int, default=50)
    ap.add_argument("--mom-short-days", type=int, default=5)
    ap.add_argument("--mom-long-days", type=int, default=10)
    ap.add_argument("--up-days-window", type=int, default=10)
    ap.add_argument("--min-up-days-frac", type=float, default=0.60)
    ap.add_argument("--min-mom-sharpe", type=float, default=0.20)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/backtest/offense_momentum_only_v1.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/offense_momentum_only_v1_summary.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/offense_momentum_only_v1.html")
    args = ap.parse_args()

    p = pd.read_csv(args.price_csv)
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    p["r"] = np.log(p["close"] / p["close"].shift(1)).fillna(0.0)

    d = p.set_index("timestamp").resample("1D").agg(close=("close", "last"), r=("r", "sum"))
    d = d.dropna(subset=["close"]).copy()
    d["ema_fast"] = d["close"].ewm(span=int(args.ema_fast_days), adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=int(args.ema_slow_days), adjust=False).mean()
    d["ret_short"] = d["close"].pct_change(int(args.mom_short_days))
    d["ret_long"] = d["close"].pct_change(int(args.mom_long_days))
    d["daily_ret"] = d["close"].pct_change()
    d["vol_long"] = d["daily_ret"].rolling(int(args.mom_long_days), min_periods=max(3, int(args.mom_long_days // 2))).std()
    d["mom_sharpe"] = d["ret_long"] / (d["vol_long"] + 1e-12)
    d["pct_up_days"] = (d["daily_ret"] > 0).rolling(int(args.up_days_window), min_periods=max(3, int(args.up_days_window // 2))).mean()

    trend_ok = (d["close"] > d["ema_fast"]) & (d["ema_fast"] > d["ema_slow"])
    mom_ok = (d["ret_short"] > 0.0) & (d["ret_long"] > 0.0)
    consistency_ok = d["pct_up_days"] >= float(args.min_up_days_frac)
    voladj_ok = d["mom_sharpe"] >= float(args.min_mom_sharpe)

    d["offense_on_raw"] = (trend_ok & mom_ok & consistency_ok & voladj_ok).astype(int)
    d["weight_daily"] = d["offense_on_raw"].astype(float)
    d["weight_exec"] = d["weight_daily"].shift(1).fillna(0.0)

    intraday = p[["timestamp", "close", "r"]].copy()
    intraday["day"] = intraday["timestamp"].dt.floor("D")
    dm = d.reset_index().rename(columns={"timestamp": "day"})
    intraday = intraday.merge(
        dm[["day", "ema_fast", "ema_slow", "ret_short", "ret_long", "mom_sharpe", "pct_up_days", "offense_on_raw", "weight_exec"]],
        on="day",
        how="left",
    )

    intraday["weight"] = intraday["weight_exec"].ffill().fillna(0.0)
    intraday["turnover"] = intraday["weight"].diff().abs().fillna(0.0)
    cost = intraday["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    intraday["cost_r"] = -cost
    intraday["strat_r"] = intraday["weight"].shift(1).fillna(0.0) * intraday["r"] - cost
    intraday["spot_r"] = intraday["r"]
    intraday["eq"] = np.exp(np.cumsum(intraday["strat_r"].to_numpy()))
    intraday["spot_eq"] = np.exp(np.cumsum(intraday["spot_r"].to_numpy()))

    m = perf(intraday["strat_r"], 5)
    s = perf(intraday["spot_r"], 5)
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(intraday)),
                "start": str(intraday["timestamp"].iloc[0]),
                "end": str(intraday["timestamp"].iloc[-1]),
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": s["ret"],
                "excess_vs_spot": m["ret"] - s["ret"],
                "time_in_market_pct": float((intraday["weight"] > 0).mean() * 100.0),
                "avg_weight": float(intraday["weight"].mean()),
                "turnover": float(intraday["turnover"].sum()),
                "offense_days_pct": float(d["offense_on_raw"].mean() * 100.0),
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    intraday.to_csv(out_csv, index=False)
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Equity", "Weight", "Price", "Daily Momentum Features"])
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["eq"], name="Offense Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["close"], name="ETH Close", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ret_short"], name="ret_short", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ret_long"], name="ret_long", line=dict(color="#17becf")), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["mom_sharpe"], name="mom_sharpe", line=dict(color="#8c564b")), row=4, col=1)
    fig.update_layout(height=1200, title="Offense Momentum-only v1")

    html = (
        "<html><head><meta charset='utf-8'><title>Offense Momentum-only v1</title></head><body>"
        "<h3>Offense Momentum-only v1</h3>"
        + summary.round(6).to_html(index=False, border=0)
        + fig.to_html(full_html=False, include_plotlyjs='cdn')
        + "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_html}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
