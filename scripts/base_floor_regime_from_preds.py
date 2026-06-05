from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def apply_step_cap(target: pd.Series, max_dw: float, w_max: float) -> pd.Series:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    arr = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for i, x in enumerate(arr):
        x = float(np.clip(x, 0.0, w_max))
        lo, hi = prev - max_dw, prev + max_dw
        v = min(max(x, lo), hi)
        out[i] = v
        prev = v
    return pd.Series(out, index=target.index)


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bpy = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bpy / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bpy)),
        "sharpe": float((np.mean(x) * bpy) / (np.std(x) * np.sqrt(bpy) + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Always-on floor regime model from existing prediction CSV")
    ap.add_argument("--pred-csv", required=True, help="CSV with at least timestamp, r, p_up. trend_gap optional.")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--bull-th", type=float, default=0.58)
    ap.add_argument("--bear-th", type=float, default=0.42)
    ap.add_argument("--bull-floor", type=float, default=0.55)
    ap.add_argument("--chop-floor", type=float, default=0.25)
    ap.add_argument("--bear-floor", type=float, default=0.05)
    ap.add_argument("--dd-cut", type=float, default=-0.05, help="spot drawdown threshold for de-risking")
    ap.add_argument("--vol-window-bars", type=int, default=288)
    ap.add_argument("--vol-z-window-bars", type=int, default=288)
    ap.add_argument("--vol-spike-z", type=float, default=1.5)
    ap.add_argument("--drop-scale", type=float, default=0.35)
    ap.add_argument("--out-csv", default="artifacts/backtest/base_floor_regime.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/base_floor_regime.html")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if "timestamp" not in df.columns or "r" not in df.columns or "p_up" not in df.columns:
        raise ValueError("pred csv must contain timestamp, r, p_up")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(0.0)
    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce").fillna(0.5)
    if "trend_gap" in df.columns:
        df["trend_gap"] = pd.to_numeric(df["trend_gap"], errors="coerce").fillna(0.0)
    else:
        df["trend_gap"] = 0.0
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Regime label from probability + trend.
    bull = (df["p_up"] >= float(args.bull_th)) & (df["trend_gap"] > 0.0)
    bear = (df["p_up"] <= float(args.bear_th)) & (df["trend_gap"] < 0.0)
    floor = pd.Series(float(args.chop_floor), index=df.index)
    floor = floor.where(~bull, float(args.bull_floor))
    floor = floor.where(~bear, float(args.bear_floor))
    df["regime_bull"] = bull.astype(int)
    df["regime_bear"] = bear.astype(int)
    df["floor"] = floor

    # Confidence from p_up, long-only.
    conf = ((df["p_up"] - 0.5) / 0.5).clip(lower=0.0, upper=1.0)
    df["conf"] = conf
    w_tgt = floor + (1.0 - floor) * conf

    # Drop detector from spot drawdown + vol spike.
    vol = df["r"].rolling(int(args.vol_window_bars), min_periods=max(20, int(args.vol_window_bars) // 4)).std()
    vol_mu = vol.rolling(int(args.vol_z_window_bars), min_periods=max(20, int(args.vol_z_window_bars) // 4)).mean()
    vol_sd = vol.rolling(int(args.vol_z_window_bars), min_periods=max(20, int(args.vol_z_window_bars) // 4)).std()
    vol_z = (vol - vol_mu) / (vol_sd + 1e-12)
    spot_eq = np.exp(np.cumsum(df["r"].to_numpy()))
    spot_peak = np.maximum.accumulate(spot_eq)
    spot_dd = pd.Series(spot_eq / spot_peak - 1.0, index=df.index)
    drop_flag = (spot_dd <= float(args.dd_cut)) | (vol_z >= float(args.vol_spike_z))
    df["drop_flag"] = drop_flag.astype(int)
    df["spot_dd"] = spot_dd
    df["vol_z"] = vol_z.fillna(0.0)

    w_tgt = w_tgt.where(~drop_flag, w_tgt * float(args.drop_scale))
    w_tgt = w_tgt.clip(lower=0.0, upper=float(args.max_weight))
    df["weight_target"] = w_tgt
    df["weight"] = apply_step_cap(df["weight_target"], max_dw=float(args.max_dw_per_bar), w_max=float(args.max_weight))

    turnover = df["weight"].diff().abs().fillna(0.0)
    cost = turnover * (float(args.trade_cost_bps) / 10000.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["r"] - cost
    df["spot_r"] = df["r"]
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["spot_r"].to_numpy()))
    df["turnover"] = turnover
    df["active"] = (df["weight"] > 1e-12).astype(int)

    m = perf(df["strat_r"], 5)
    ms = perf(df["spot_r"], 5)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity", "Weight", "P(up) + Floor", "Drop Detector", "Returns"],
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["p_up"], name="P(up)", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["floor"], name="Floor", line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spot_dd"], name="Spot DD", line=dict(color="#8c564b")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_z"], name="Vol Z", line=dict(color="#17becf")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["drop_flag"], name="Drop Flag", line=dict(color="#d62728")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["r"], name="Spot LogRet", line=dict(color="#7f7f7f")), row=5, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["strat_r"], name="Strat LogRet", line=dict(color="#1f77b4")), row=5, col=1)
    fig.update_layout(height=1500, title="Base-On + Drop-Reduce Regime Model")

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "excess_vs_spot": m["ret"] - ms["ret"],
                "time_in_market_pct": float(df["active"].mean() * 100.0),
                "avg_weight": float(df["weight"].mean()),
                "turnover": float(df["turnover"].sum()),
                "bull_floor": float(args.bull_floor),
                "chop_floor": float(args.chop_floor),
                "bear_floor": float(args.bear_floor),
                "drop_scale": float(args.drop_scale),
                "dd_cut": float(args.dd_cut),
                "vol_spike_z": float(args.vol_spike_z),
            }
        ]
    )
    html = (
        "<html><head><meta charset='utf-8'><title>Base-On Drop-Reduce</title></head><body>"
        "<h3>Base-On + Drop-Reduce Regime Model</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(
        "ret={:.4f} cagr={:.4f} sharpe={:.4f} max_dd={:.4f} spot_ret={:.4f} excess={:.4f} tim={:.2f}%".format(
            m["ret"],
            m["cagr"],
            m["sharpe"],
            m["max_dd"],
            ms["ret"],
            m["ret"] - ms["ret"],
            float(df["active"].mean() * 100.0),
        )
    )


if __name__ == "__main__":
    main()
