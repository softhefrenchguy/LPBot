from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_etf_close(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = next((cols[c] for c in ("timestamp", "date", "datetime") if c in cols), None)
    pcol = next((cols[c] for c in ("close", "adj close", "adj_close") if c in cols), None)
    if dcol is None or pcol is None:
        raise ValueError(f"{path} missing date/close columns")
    out = pd.DataFrame(
        {
            "day": pd.to_datetime(df[dcol], utc=True, errors="coerce"),
            "close": pd.to_numeric(df[pcol], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("day").drop_duplicates("day", keep="last")
    return out.set_index("day")["close"]


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


def _build_longterm_state(
    close: pd.Series,
    ema_fast: int,
    ema_slow: int,
    vel_lookback: int,
    accel_lookback: int,
    accel_z_window: int,
    entry_accel_z: float,
    exit_accel_z: float,
    min_hold_days: int,
) -> pd.Series:
    x = pd.DataFrame({"close": close})
    x["ema_fast"] = x["close"].ewm(span=int(ema_fast), adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=int(ema_slow), adjust=False).mean()
    x["vel"] = x["ema_fast"].pct_change(int(vel_lookback))
    x["accel"] = x["vel"].diff(int(accel_lookback))
    std = x["accel"].rolling(int(accel_z_window), min_periods=max(20, accel_z_window // 3)).std()
    x["accel_z"] = x["accel"] / (std + 1e-12)
    bull = x["close"] > x["ema_slow"]
    peak = (x["close"] < x["ema_fast"]) & (x["accel_z"] < float(exit_accel_z))

    state = np.zeros(len(x), dtype=float)
    on = False
    hold = 0
    for i in range(len(x)):
        az = float(x["accel_z"].iat[i]) if np.isfinite(x["accel_z"].iat[i]) else -999.0
        if not on:
            if bool(bull.iat[i]) and az >= float(entry_accel_z):
                on = True
                hold = 0
        else:
            hold += 1
            if (not bool(bull.iat[i]) or bool(peak.iat[i])) and hold >= int(min_hold_days):
                on = False
                hold = 0
        state[i] = 1.0 if on else 0.0
    return pd.Series(state, index=close.index, name="bull_longterm")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed crypto/ETF sleeves + long-term bull floor inside crypto sleeve.")
    ap.add_argument("--crypto-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--window-days", type=int, default=1826)
    ap.add_argument("--alloc-crypto-sleeve", type=float, default=0.30)
    ap.add_argument("--crypto-floor-bull", type=float, default=0.25)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--etf-dir", default="data/etf")
    ap.add_argument("--etf-symbols", default="GLD,DBC,DBA,DBB")
    ap.add_argument("--etf-weights", default="0.50,0.1666666667,0.1666666667,0.1666666666")
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--vel-lookback", type=int, default=10)
    ap.add_argument("--accel-lookback", type=int, default=5)
    ap.add_argument("--accel-z-window", type=int, default=90)
    ap.add_argument("--entry-accel-z", type=float, default=-0.2)
    ap.add_argument("--exit-accel-z", type=float, default=-0.6)
    ap.add_argument("--min-hold-days", type=int, default=5)
    ap.add_argument("--out-csv", default="artifacts/backtest/fixed_sleeve_combo_longterm_floor.csv")
    ap.add_argument("--summary-csv", default="artifacts/backtest/fixed_sleeve_combo_longterm_floor_summary.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/fixed_sleeve_combo_longterm_floor.html")
    args = ap.parse_args()

    a = float(args.alloc_crypto_sleeve)
    if not (0.0 <= a <= 1.0):
        raise ValueError("alloc-crypto-sleeve must be in [0,1]")

    # Crypto daily features.
    cdf = pd.read_csv(args.crypto_csv)
    need = {"timestamp", "close", "r", "weight"}
    if not need.issubset(cdf.columns):
        raise ValueError(f"{args.crypto_csv} missing required columns {sorted(need)}")
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"], utc=True, errors="coerce")
    cdf["close"] = pd.to_numeric(cdf["close"], errors="coerce")
    cdf["r"] = pd.to_numeric(cdf["r"], errors="coerce")
    cdf["weight"] = pd.to_numeric(cdf["weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    cdf = cdf.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")
    cdf["day"] = cdf["timestamp"].dt.floor("D")
    daily = (
        cdf.groupby("day", as_index=False)
        .agg(crypto_r_daily=("r", "sum"), w_signal=("weight", "mean"), close=("close", "last"))
        .sort_values("day")
    )

    # ETF basket daily.
    symbols = [x.strip().upper() for x in args.etf_symbols.split(",") if x.strip()]
    w = np.array([float(x) for x in args.etf_weights.split(",")], dtype=float)
    if len(symbols) != len(w):
        raise ValueError("etf-symbols and etf-weights lengths must match")
    if not np.isclose(w.sum(), 1.0, atol=1e-9):
        raise ValueError("ETF weights must sum to 1.0")

    parts: list[pd.Series] = []
    for sym, ww in zip(symbols, w):
        s = _load_etf_close(Path(args.etf_dir) / f"{sym}.csv")
        parts.append(np.log(s / s.shift(1)) * float(ww))
    etf_r = pd.concat(parts, axis=1).sum(axis=1, min_count=1).rename("etf_r").reset_index()
    daily = daily.merge(etf_r, on="day", how="left").sort_values("day")
    daily["etf_r"] = pd.to_numeric(daily["etf_r"], errors="coerce").fillna(0.0)

    # Window.
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()

    # Long-term bull state, lagged for execution.
    bull = _build_longterm_state(
        close=daily.set_index("day")["close"],
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        vel_lookback=int(args.vel_lookback),
        accel_lookback=int(args.accel_lookback),
        accel_z_window=int(args.accel_z_window),
        entry_accel_z=float(args.entry_accel_z),
        exit_accel_z=float(args.exit_accel_z),
        min_hold_days=int(args.min_hold_days),
    )
    daily = daily.merge(bull.reset_index(), on="day", how="left")
    daily["bull_longterm_exec"] = daily["bull_longterm"].shift(1).fillna(0.0)

    # Base vs floor-adjusted internal crypto weight.
    daily["w_base"] = daily["w_signal"].shift(1).fillna(0.0).clip(0.0, 1.0)
    daily["w_floor"] = np.where(
        daily["bull_longterm_exec"] > 0.5,
        np.maximum(daily["w_base"], float(args.crypto_floor_bull)),
        daily["w_base"],
    )

    cost_k = float(args.trade_cost_bps) / 10000.0
    daily["turn_base"] = daily["w_base"].diff().abs().fillna(0.0)
    daily["turn_floor"] = daily["w_floor"].diff().abs().fillna(0.0)
    daily["cost_base"] = daily["turn_base"] * cost_k
    daily["cost_floor"] = daily["turn_floor"] * cost_k

    daily["crypto_sleeve_base_r"] = daily["w_base"] * daily["crypto_r_daily"] - daily["cost_base"]
    daily["crypto_sleeve_floor_r"] = daily["w_floor"] * daily["crypto_r_daily"] - daily["cost_floor"]

    daily["combo_base_r"] = a * daily["crypto_sleeve_base_r"] + (1.0 - a) * daily["etf_r"]
    daily["combo_floor_r"] = a * daily["crypto_sleeve_floor_r"] + (1.0 - a) * daily["etf_r"]

    for col in ("combo_base_r", "combo_floor_r", "etf_r", "crypto_r_daily"):
        daily[f"eq_{col}"] = np.exp(daily[col].cumsum())
        daily[f"eq_{col}"] = daily[f"eq_{col}"] / float(daily[f"eq_{col}"].iloc[0])

    s_combo_base = _perf(daily["combo_base_r"])
    s_combo_floor = _perf(daily["combo_floor_r"])
    s_etf = _perf(daily["etf_r"])
    s_spot = _perf(daily["crypto_r_daily"])

    summary = pd.DataFrame(
        [
            {
                "model": "combo_base",
                **s_combo_base,
                "alloc_crypto_sleeve": a,
                "avg_w_inside_crypto": float(daily["w_base"].mean()),
                "effective_crypto_notional": float(a * daily["w_base"].mean()),
            },
            {
                "model": "combo_with_longterm_floor",
                **s_combo_floor,
                "alloc_crypto_sleeve": a,
                "avg_w_inside_crypto": float(daily["w_floor"].mean()),
                "effective_crypto_notional": float(a * daily["w_floor"].mean()),
            },
            {"model": "etf_only", **s_etf, "alloc_crypto_sleeve": 0.0, "avg_w_inside_crypto": 0.0, "effective_crypto_notional": 0.0},
            {"model": "eth_spot", **s_spot, "alloc_crypto_sleeve": 1.0, "avg_w_inside_crypto": 1.0, "effective_crypto_notional": 1.0},
        ]
    )
    summary["window_start"] = str(daily["day"].iloc[0])
    summary["window_end"] = str(daily["day"].iloc[-1])
    summary["rows"] = int(len(daily))
    summary["bull_floor"] = float(args.crypto_floor_bull)
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
        subplot_titles=["Equity", "Internal Crypto Weight", "Long-Term Bull State", "ETH Price"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_combo_base_r"], name="Combo Base Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_combo_floor_r"], name="Combo Floor Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_etf_r"], name="ETF Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_crypto_r_daily"], name="ETH Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_base"], name="w_base"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_floor"], name="w_floor"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["bull_longterm_exec"], name="bull_longterm_exec"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["close"], name="ETH close"), row=4, col=1)
    fig.update_layout(height=1400, title="Fixed Sleeve Combo with Long-Term Crypto Floor")

    html = (
        "<html><head><meta charset='utf-8'><title>Fixed Sleeve + Long-Term Floor</title></head><body>"
        "<h3>Fixed Sleeve Combo with Long-Term Crypto Floor</h3>"
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
