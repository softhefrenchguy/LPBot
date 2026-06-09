from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_price(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must include timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def _resampled_no_lookahead(ts: pd.Series, close: pd.Series, bar_minutes: int, timeframe: str) -> pd.Series:
    tf_map = {"same": f"{bar_minutes}min", "1h": "1h", "4h": "4h", "1d": "1D"}
    tf = tf_map[timeframe]
    tmp = pd.DataFrame({"close": close.to_numpy()}, index=pd.DatetimeIndex(ts.to_numpy()))
    if tf == f"{bar_minutes}min":
        return tmp["close"]
    return tmp["close"].resample(tf).last().shift(1).ffill().reindex(tmp.index, method="ffill").ffill()


def _step_cap(target: np.ndarray, max_dw: float, decision_step_bars: int = 1, min_rebalance_delta: float = 0.0) -> np.ndarray:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    step = int(max(1, decision_step_bars))
    min_delta = float(max(0.0, min_rebalance_delta))
    for i, x in enumerate(np.nan_to_num(target, nan=0.0)):
        if i % step != 0:
            out[i] = prev
            continue
        if abs(float(x) - prev) < min_delta:
            out[i] = prev
            continue
        lo, hi = prev - max_dw, prev + max_dw
        v = min(max(float(x), lo), hi)
        out[i] = v
        prev = v
    return out


def _perf(log_r: np.ndarray, bar_minutes: int = 5) -> dict:
    x = np.asarray(log_r, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bpy = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    ret = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (bpy / len(x)) - 1.0)
    ann_vol = float(np.std(x) * np.sqrt(bpy))
    sharpe = float((np.mean(x) * bpy) / (ann_vol + 1e-12))
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"ret": ret, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd}


def _write_html(df: pd.DataFrame, summary: pd.DataFrame, out: Path, title: str) -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity", "Weight", "Trend Signal", "Price"],
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["trend_on"].astype(int), name="Trend On", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ratio"], name="Trend Ratio", line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="#17becf")), row=4, col=1)
    fig.update_layout(height=1200, title=title)
    html = (
        "<html><head><meta charset='utf-8'><title>Momentum Risk-On</title></head><body>"
        f"<h3>{title}</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple momentum risk-on sleeve.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--bar-minutes", type=int, default=5)
    ap.add_argument("--trend-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--hyst-on", type=float, default=0.002)
    ap.add_argument("--hyst-off", type=float, default=0.002)
    ap.add_argument("--persist-on-bars", type=int, default=3)
    ap.add_argument("--persist-off-bars", type=int, default=3)
    ap.add_argument("--target-vol", type=float, default=0.35)
    ap.add_argument("--vol-window-bars", type=int, default=72)
    ap.add_argument("--w-max", type=float, default=1.0)
    ap.add_argument("--w-min-on", type=float, default=0.10)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.08)
    ap.add_argument("--decision-step-bars", type=int, default=1)
    ap.add_argument("--min-rebalance-delta", type=float, default=0.0)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/paper/momentum_riskon.csv")
    ap.add_argument("--out-html", default="artifacts/paper/momentum_riskon.html")
    args = ap.parse_args()

    df = _load_price(args.price_csv)
    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()

    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    bars_per_year = 365 * 24 * (60 / args.bar_minutes)
    sigma_ann = df["r"].rolling(int(args.vol_window_bars), min_periods=max(20, int(args.vol_window_bars // 3))).std() * np.sqrt(bars_per_year)
    w_raw = (float(args.target_vol) / (sigma_ann + 1e-12)).clip(lower=0.0, upper=float(args.w_max))

    tclose = _resampled_no_lookahead(df["timestamp"], df["close"], int(args.bar_minutes), args.trend_timeframe)
    ef = tclose.ewm(span=int(args.ema_fast), adjust=False).mean()
    es = tclose.ewm(span=int(args.ema_slow), adjust=False).mean()
    ratio = (ef / (es + 1e-12)) - 1.0
    df["ratio"] = ratio.to_numpy()

    on_cond = ratio > float(args.hyst_on)
    off_cond = ratio < -float(args.hyst_off)
    state = np.zeros(len(df), dtype=bool)
    on = False
    on_st, off_st = 0, 0
    on_need = max(1, int(args.persist_on_bars))
    off_need = max(1, int(args.persist_off_bars))
    for i in range(len(df)):
        on_st = on_st + 1 if bool(on_cond.iat[i]) else 0
        off_st = off_st + 1 if bool(off_cond.iat[i]) else 0
        if not on and on_st >= on_need:
            on = True
            off_st = 0
        elif on and off_st >= off_need:
            on = False
            on_st = 0
        state[i] = on
    df["trend_on"] = state

    target_w = np.where(state, np.maximum(w_raw.to_numpy(), float(args.w_min_on)), 0.0)
    df["weight_target"] = np.clip(target_w, 0.0, float(args.w_max))
    df["weight"] = _step_cap(
        df["weight_target"].to_numpy(),
        float(args.max_dw_per_bar),
        decision_step_bars=int(args.decision_step_bars),
        min_rebalance_delta=float(args.min_rebalance_delta),
    )
    df["turnover"] = pd.Series(df["weight"]).diff().abs().fillna(0.0).to_numpy()
    k = float(args.trade_cost_bps) / 10000.0
    df["strat_r"] = pd.Series(df["weight"]).shift(1).fillna(0.0).to_numpy() * df["r"].to_numpy() - df["turnover"].to_numpy() * k
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["r"].to_numpy()))

    s = _perf(df["strat_r"].to_numpy(), int(args.bar_minutes))
    ss = _perf(df["r"].to_numpy(), int(args.bar_minutes))
    summary = pd.DataFrame(
        [
            {
                "rows": len(df),
                "start": str(df["timestamp"].iloc[0]),
                "end": str(df["timestamp"].iloc[-1]),
                "ret": s["ret"],
                "cagr": s["cagr"],
                "ann_vol": s["ann_vol"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_dd"],
                "spot_ret": ss["ret"],
                "spot_cagr": ss["cagr"],
                "excess_ret": s["ret"] - ss["ret"],
                "avg_weight": float(df["weight"].mean()),
                "time_in_market": float((df["weight"] > 1e-12).mean()),
                "turnover_sum": float(df["turnover"].sum()),
                "trend_on_pct": float(df["trend_on"].mean()),
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    _write_html(df.tail(3000).copy(), summary, out_html, "Momentum Risk-On Sleeve")
    print("wrote", out_csv)
    print("wrote", out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
