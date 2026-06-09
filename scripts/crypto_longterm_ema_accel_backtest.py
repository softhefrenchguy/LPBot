from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    bpy = 365.0
    ann_vol = float(np.std(x) * np.sqrt(bpy))
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bpy / len(eq)) - 1.0),
        "ann_vol": ann_vol,
        "sharpe": float((np.mean(x) * bpy) / (ann_vol + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-term EMA + acceleration crypto sleeve backtest.")
    ap.add_argument("--in-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--window-days", type=int, default=1826)
    ap.add_argument("--ema-fast", type=int, default=50, help="Daily EMA fast")
    ap.add_argument("--ema-slow", type=int, default=200, help="Daily EMA slow")
    ap.add_argument("--vel-lookback", type=int, default=10, help="Days for EMA-fast velocity")
    ap.add_argument("--accel-lookback", type=int, default=5, help="Days for acceleration")
    ap.add_argument("--accel-z-window", type=int, default=90)
    ap.add_argument("--entry-accel-z", type=float, default=-0.10)
    ap.add_argument("--exit-accel-z", type=float, default=-0.50)
    ap.add_argument("--min-hold-days", type=int, default=5)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/backtest/crypto_longterm_ema_accel_5y.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/crypto_longterm_ema_accel_5y.html")
    ap.add_argument("--summary-csv", default="artifacts/backtest/crypto_longterm_ema_accel_5y_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if not {"timestamp", "close", "r"}.issubset(df.columns):
        raise ValueError("Input must contain timestamp, close, r columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")
    df["day"] = df["timestamp"].dt.floor("D")

    daily = (
        df.groupby("day", as_index=False)
        .agg(close=("close", "last"), r_daily=("r", "sum"))
        .sort_values("day")
    )
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()
    if daily.empty:
        raise ValueError("No rows in selected window")

    # Long-term trend + acceleration features.
    daily["ema_fast"] = daily["close"].ewm(span=int(args.ema_fast), adjust=False).mean()
    daily["ema_slow"] = daily["close"].ewm(span=int(args.ema_slow), adjust=False).mean()
    daily["vel"] = daily["ema_fast"].pct_change(int(args.vel_lookback))
    daily["accel"] = daily["vel"].diff(int(args.accel_lookback))
    accel_std = daily["accel"].rolling(int(args.accel_z_window), min_periods=max(20, int(args.accel_z_window // 3))).std()
    daily["accel_z"] = daily["accel"] / (accel_std + 1e-12)
    daily["bull_long"] = daily["close"] > daily["ema_slow"]
    daily["peak_risk"] = (daily["close"] < daily["ema_fast"]) & (daily["accel_z"] < float(args.exit_accel_z))

    # Stateful long/flat logic.
    state = np.zeros(len(daily), dtype=float)
    on = False
    hold = 0
    for i in range(len(daily)):
        bull = bool(daily["bull_long"].iat[i])
        az = float(daily["accel_z"].iat[i]) if np.isfinite(daily["accel_z"].iat[i]) else -999.0
        peak = bool(daily["peak_risk"].iat[i])

        if not on:
            if bull and az >= float(args.entry_accel_z):
                on = True
                hold = 0
        else:
            hold += 1
            if (not bull or peak) and hold >= int(args.min_hold_days):
                on = False
                hold = 0
        state[i] = 1.0 if on else 0.0

    daily["w_target"] = state
    daily["w_exec"] = daily["w_target"].shift(1).fillna(0.0)
    daily["turnover"] = daily["w_target"].diff().abs().fillna(0.0)
    cost_k = float(args.trade_cost_bps) / 10000.0
    daily["cost"] = daily["turnover"] * cost_k
    daily["strat_r"] = daily["w_exec"] * daily["r_daily"] - daily["cost"]

    daily["eq_strat"] = np.exp(daily["strat_r"].cumsum())
    daily["eq_spot"] = np.exp(daily["r_daily"].cumsum())
    daily["eq_strat"] = daily["eq_strat"] / float(daily["eq_strat"].iloc[0])
    daily["eq_spot"] = daily["eq_spot"] / float(daily["eq_spot"].iloc[0])

    m_strat = _perf(daily["strat_r"])
    m_spot = _perf(daily["r_daily"])
    summary = pd.DataFrame(
        [
            {
                "model": "ema_accel_longterm",
                **m_strat,
                "avg_weight": float(daily["w_exec"].mean()),
                "time_in_market": float((daily["w_exec"] > 1e-12).mean()),
                "total_cost": float(daily["cost"].sum()),
                "entry_count": int((daily["turnover"] > 0.5).sum()),
            },
            {
                "model": "spot",
                **m_spot,
                "avg_weight": 1.0,
                "time_in_market": 1.0,
                "total_cost": 0.0,
                "entry_count": 0,
            },
        ]
    )
    summary["window_start"] = str(daily["day"].iloc[0])
    summary["window_end"] = str(daily["day"].iloc[-1])
    summary["rows"] = int(len(daily))
    summary["ema_fast"] = int(args.ema_fast)
    summary["ema_slow"] = int(args.ema_slow)
    summary["entry_accel_z"] = float(args.entry_accel_z)
    summary["exit_accel_z"] = float(args.exit_accel_z)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    Path(args.summary_csv).write_text(summary.to_csv(index=False), encoding="utf-8")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Equity", "Price + EMAs", "Acceleration Z", "Weight"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_strat"], name="Strategy Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_spot"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["close"], name="Close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema_fast"], name=f"EMA{args.ema_fast}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema_slow"], name=f"EMA{args.ema_slow}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["accel_z"], name="accel_z"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_exec"], name="w_exec"), row=4, col=1)
    fig.update_layout(height=1300, title="Crypto Long-Term EMA + Acceleration Backtest")

    html = (
        "<html><head><meta charset='utf-8'><title>Crypto EMA Accel</title></head><body>"
        "<h3>Crypto Long-Term EMA + Acceleration</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    Path(args.out_html).write_text(html, encoding="utf-8")

    print("wrote", args.out_csv)
    print("wrote", args.summary_csv)
    print("wrote", args.out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
