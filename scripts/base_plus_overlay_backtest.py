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
        raise ValueError("price csv must contain timestamp, close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def _load_overlay(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("overlay csv must contain timestamp")
    # allow either weight or strat_r output from overlay script
    if "weight" not in df.columns and "strat_r" not in df.columns:
        raise ValueError("overlay csv must contain weight or strat_r")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "weight" in df.columns:
        df["ov_w"] = pd.to_numeric(df["weight"], errors="coerce")
    else:
        df["ov_w"] = np.nan
    if "strat_r" in df.columns:
        df["ov_r"] = pd.to_numeric(df["strat_r"], errors="coerce")
    else:
        df["ov_r"] = np.nan
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df[["timestamp", "ov_w", "ov_r"]]


def _apply_step_cap(target: np.ndarray, max_dw: float) -> np.ndarray:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, x in enumerate(np.nan_to_num(target, nan=0.0)):
        lo = prev - max_dw
        hi = prev + max_dw
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


def _write_html(df: pd.DataFrame, summary: pd.DataFrame, out_html: Path, title: str) -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity", "Weights", "Signals", "Price"],
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq_combo"], name="Combo Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq_base"], name="Base Eq", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq_spot"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["w_base"], name="w_base", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["w_overlay"], name="w_overlay", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["w_combo"], name="w_combo", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["trend_on"].astype(int), name="trend_on", line=dict(color="#17becf")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ov_gate"].astype(int), name="ov_gate", line=dict(color="#8c564b")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="close", line=dict(color="#1f77b4")), row=4, col=1)
    fig.update_layout(height=1200, title=title)

    html = (
        "<html><head><meta charset='utf-8'><title>Base+Overlay</title></head><body>"
        f"<h3>{title}</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Base beta sleeve + overlay sleeve backtest.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--overlay-csv", default="artifacts/paper/funding_basis_mr_365d_db.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--target-vol", type=float, default=0.30)
    ap.add_argument("--vol-window-bars", type=int, default=72)
    ap.add_argument("--w-base-max", type=float, default=1.0)
    ap.add_argument("--w-base-floor", type=float, default=0.05)
    ap.add_argument("--trend-ema", type=int, default=200)
    ap.add_argument("--trend-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    ap.add_argument("--trend-hyst-on", type=float, default=0.002)
    ap.add_argument("--trend-hyst-off", type=float, default=0.002)
    ap.add_argument("--trend-persist-on-bars", type=int, default=3)
    ap.add_argument("--trend-persist-off-bars", type=int, default=3)
    ap.add_argument("--base-max-dw", type=float, default=0.08)
    ap.add_argument("--overlay-weight", type=float, default=0.40, help="How much overlay reduces base exposure.")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/paper/base_plus_overlay.csv")
    ap.add_argument("--out-html", default="artifacts/paper/base_plus_overlay.html")
    args = ap.parse_args()

    px = _load_price(args.price_csv)
    ov = _load_overlay(args.overlay_csv)
    df = px.merge(ov, on="timestamp", how="left").sort_values("timestamp").copy()

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()

    df["ov_w"] = pd.to_numeric(df["ov_w"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

    # Base: vol-targeted long beta
    bars_per_year = 365 * 24 * 12
    sigma_ann = df["r"].rolling(int(args.vol_window_bars), min_periods=max(20, int(args.vol_window_bars // 3))).std() * np.sqrt(bars_per_year)
    w_base_raw = (float(args.target_vol) / (sigma_ann + 1e-12)).clip(lower=float(args.w_base_floor), upper=float(args.w_base_max))

    # Trend guard with hysteresis
    ts = df["timestamp"]
    close = df["close"]
    tf_map = {"same": "5min", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = tf_map[args.trend_timeframe]
    tmp = pd.DataFrame({"close": close.to_numpy()}, index=ts)
    if tf == "5min":
        tclose = tmp["close"]
    else:
        tclose = tmp["close"].resample(tf).last().shift(1).ffill().reindex(ts, method="ffill").ffill()
    ema = tclose.ewm(span=int(args.trend_ema), adjust=False).mean()
    ratio = (tclose / (ema + 1e-12)) - 1.0
    on_cond = ratio > float(args.trend_hyst_on)
    off_cond = ratio < -float(args.trend_hyst_off)

    state = np.zeros(len(df), dtype=bool)
    is_on = False
    on_st = 0
    off_st = 0
    on_need = max(1, int(args.trend_persist_on_bars))
    off_need = max(1, int(args.trend_persist_off_bars))
    for i in range(len(df)):
        on_st = on_st + 1 if bool(on_cond.iat[i]) else 0
        off_st = off_st + 1 if bool(off_cond.iat[i]) else 0
        if not is_on and on_st >= on_need:
            is_on = True
            off_st = 0
        elif is_on and off_st >= off_need:
            is_on = False
            on_st = 0
        state[i] = is_on
    df["trend_on"] = state

    w_base = np.where(df["trend_on"].to_numpy(), w_base_raw.to_numpy(), float(args.w_base_floor))
    w_base = _apply_step_cap(w_base, max_dw=float(args.base_max_dw))
    df["w_base"] = pd.Series(w_base, index=df.index).clip(0.0, float(args.w_base_max))

    # Overlay gate from overlay weight presence
    df["ov_gate"] = df["ov_w"] > 0
    df["w_overlay"] = df["ov_w"]

    # Combine: reduce base in risk-off overlay periods
    ov_effect = float(args.overlay_weight) * df["w_overlay"].to_numpy()
    w_combo = df["w_base"].to_numpy() * (1.0 - ov_effect)
    w_combo = np.clip(w_combo, 0.0, float(args.w_base_max))
    df["w_combo"] = w_combo

    cost_k = float(args.trade_cost_bps) / 10000.0
    to_base = pd.Series(df["w_base"]).diff().abs().fillna(0.0).to_numpy()
    to_combo = pd.Series(df["w_combo"]).diff().abs().fillna(0.0).to_numpy()

    df["ret_base"] = pd.Series(df["w_base"]).shift(1).fillna(0.0).to_numpy() * df["r"].to_numpy() - to_base * cost_k
    df["ret_combo"] = pd.Series(df["w_combo"]).shift(1).fillna(0.0).to_numpy() * df["r"].to_numpy() - to_combo * cost_k
    df["ret_spot"] = df["r"]

    df["eq_base"] = np.exp(np.cumsum(df["ret_base"].to_numpy()))
    df["eq_combo"] = np.exp(np.cumsum(df["ret_combo"].to_numpy()))
    df["eq_spot"] = np.exp(np.cumsum(df["ret_spot"].to_numpy()))

    s_base = _perf(df["ret_base"].to_numpy())
    s_combo = _perf(df["ret_combo"].to_numpy())
    s_spot = _perf(df["ret_spot"].to_numpy())
    summary = pd.DataFrame(
        [
            {"model": "base", **s_base, "avg_w": float(df["w_base"].mean()), "time_in_mkt": float((df["w_base"] > 1e-12).mean())},
            {"model": "combo", **s_combo, "avg_w": float(df["w_combo"].mean()), "time_in_mkt": float((df["w_combo"] > 1e-12).mean())},
            {"model": "spot", **s_spot, "avg_w": 1.0, "time_in_mkt": 1.0},
        ]
    )
    summary["start"] = str(df["timestamp"].iloc[0])
    summary["end"] = str(df["timestamp"].iloc[-1])
    summary["rows"] = len(df)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    _write_html(df.tail(3000).copy(), summary, out_html, "Base Beta + Overlay (Funding/Basis)")

    print("wrote", out_csv)
    print("wrote", out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
