import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _safe_ret(log_r: pd.Series) -> float:
    return float(np.exp(np.nansum(log_r.to_numpy())) - 1.0)


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose HMM+ENet bull-regime behavior.")
    ap.add_argument("--input-csv", default="artifacts/paper/hmm_enet_scaled.csv")
    ap.add_argument("--out-html", default="artifacts/paper/hmm_enet_bull_diagnosis.html")
    ap.add_argument("--bull-timeframe", default="1D", help="Resample timeframe for bull regime.")
    ap.add_argument("--bull-ema", type=int, default=200)
    ap.add_argument("--require-ema-slope-up", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    for c in ["close", "r", "strat_r", "weight", "regime_scale", "bull_on"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["close"] = df["close"].ffill()
    df["r"] = df["r"].fillna(0.0)
    df["strat_r"] = df["strat_r"].fillna(0.0)
    df["weight"] = df["weight"].fillna(0.0)
    if "regime_scale" not in df.columns:
        df["regime_scale"] = np.nan
    if "bull_on" not in df.columns:
        df["bull_on"] = 0.0

    dfx = df.set_index("timestamp")
    close_tf = dfx["close"].resample(args.bull_timeframe).last().ffill()
    ema = close_tf.ewm(span=args.bull_ema, adjust=False).mean()
    if args.require_ema_slope_up:
        bull_tf = (close_tf > ema) & (ema.diff() > 0)
    else:
        bull_tf = close_tf > ema

    bull_5m = bull_tf.reindex(dfx.index, method="ffill").fillna(False)
    df["bull_regime"] = bull_5m.to_numpy(dtype=bool)

    df["spot_eq"] = np.exp(np.cumsum(df["r"].to_numpy()))
    df["strat_eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["excess_r"] = df["strat_r"] - df["r"]
    df["bull_excess_r"] = np.where(df["bull_regime"], df["excess_r"], 0.0)
    df["nonbull_excess_r"] = np.where(~df["bull_regime"], df["excess_r"], 0.0)
    df["bull_excess_eq"] = np.exp(np.cumsum(df["bull_excess_r"].to_numpy()))
    df["nonbull_excess_eq"] = np.exp(np.cumsum(df["nonbull_excess_r"].to_numpy()))
    df["bull_up"] = df["bull_regime"] & (df["r"] > 0)
    df["bull_down"] = df["bull_regime"] & (df["r"] < 0)

    df["bull_up_spot_log"] = np.where(df["bull_up"], df["r"], 0.0)
    df["bull_up_strat_log"] = np.where(df["bull_up"], df["strat_r"], 0.0)
    df["bull_down_spot_log"] = np.where(df["bull_down"], df["r"], 0.0)
    df["bull_down_strat_log"] = np.where(df["bull_down"], df["strat_r"], 0.0)

    df["bull_up_spot_cum"] = np.cumsum(df["bull_up_spot_log"].to_numpy())
    df["bull_up_strat_cum"] = np.cumsum(df["bull_up_strat_log"].to_numpy())
    df["bull_down_spot_cum"] = np.cumsum(df["bull_down_spot_log"].to_numpy())
    df["bull_down_strat_cum"] = np.cumsum(df["bull_down_strat_log"].to_numpy())

    bull = df[df["bull_regime"]]
    nonbull = df[~df["bull_regime"]]

    rows_total = len(df)
    bull_pct = float(df["bull_regime"].mean() * 100.0)

    summary = []
    for name, part in [("Bull", bull), ("Non-Bull", nonbull), ("All", df)]:
        summary.append(
            {
                "segment": name,
                "bars": len(part),
                "avg_weight": float(part["weight"].mean()) if len(part) else np.nan,
                "strat_ret": _safe_ret(part["strat_r"]) if len(part) else np.nan,
                "spot_ret": _safe_ret(part["r"]) if len(part) else np.nan,
                "excess_ret": (_safe_ret(part["strat_r"]) - _safe_ret(part["r"])) if len(part) else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary)

    bull_up_spot_sum = float(df.loc[df["bull_up"], "r"].sum())
    bull_up_strat_sum = float(df.loc[df["bull_up"], "strat_r"].sum())
    bull_down_spot_abs = float(-df.loc[df["bull_down"], "r"].sum())
    bull_down_strat_abs = float(-df.loc[df["bull_down"], "strat_r"].sum())
    bull_up_bars = int(df["bull_up"].sum())
    bull_down_bars = int(df["bull_down"].sum())
    bull_up_capture = _safe_ratio(bull_up_strat_sum, bull_up_spot_sum)
    bull_down_capture = _safe_ratio(bull_down_strat_abs, bull_down_spot_abs)

    capture_df = pd.DataFrame(
        [
            {"metric": "bull_up_bars", "value": bull_up_bars},
            {"metric": "bull_down_bars", "value": bull_down_bars},
            {"metric": "bull_up_capture_log_ratio", "value": bull_up_capture},
            {"metric": "bull_down_loss_capture_ratio", "value": bull_down_capture},
            {"metric": "bull_up_spot_log_sum", "value": bull_up_spot_sum},
            {"metric": "bull_up_strat_log_sum", "value": bull_up_strat_sum},
            {"metric": "bull_down_spot_log_abs_sum", "value": bull_down_spot_abs},
            {"metric": "bull_down_strat_log_abs_sum", "value": bull_down_strat_abs},
            {"metric": "bull_up_spot_log_mean", "value": _safe_ratio(bull_up_spot_sum, max(1, bull_up_bars))},
            {"metric": "bull_up_strat_log_mean", "value": _safe_ratio(bull_up_strat_sum, max(1, bull_up_bars))},
            {"metric": "bull_down_spot_log_abs_mean", "value": _safe_ratio(bull_down_spot_abs, max(1, bull_down_bars))},
            {"metric": "bull_down_strat_log_abs_mean", "value": _safe_ratio(bull_down_strat_abs, max(1, bull_down_bars))},
        ]
    )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Equity: Strategy vs Spot",
            "Weight / Regime Scale / Bull Flags",
            "Cumulative Excess Attribution",
            "Bull Up/Down Capture (Cumulative Log Returns)",
        ],
    )

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["strat_eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["regime_scale"], name="Regime Scale", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_on"], name="Bull Override On", line=dict(color="#17becf")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_regime"].astype(int), name="Bull Regime", line=dict(color="#2ca02c")), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_excess_eq"], name="Bull Excess Eq", line=dict(color="#d62728")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["nonbull_excess_eq"], name="Non-Bull Excess Eq", line=dict(color="#8c564b")), row=3, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_up_spot_cum"], name="Bull Up Spot (cum log)", line=dict(color="#7f7f7f")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_up_strat_cum"], name="Bull Up Strat (cum log)", line=dict(color="#1f77b4")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_down_spot_cum"], name="Bull Down Spot (cum log)", line=dict(color="#d62728", dash="dot")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bull_down_strat_cum"], name="Bull Down Strat (cum log)", line=dict(color="#2ca02c", dash="dot")), row=4, col=1)

    # shade bull regime on top panel
    bull_arr = df["bull_regime"].to_numpy(dtype=bool)
    ts_arr = df["timestamp"].to_numpy()
    start = None
    for i, on in enumerate(bull_arr):
        if on and start is None:
            start = ts_arr[i]
        if (not on or i == len(bull_arr) - 1) and start is not None:
            end = ts_arr[i] if not on else ts_arr[i]
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(44,160,44,0.07)",
                line_width=0,
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(44,160,44,0.05)",
                line_width=0,
                row=4,
                col=1,
            )
            start = None

    header = (
        f"rows={rows_total:,} | bull_pct={bull_pct:.2f}% | "
        f"bull_rule={args.bull_timeframe} close>EMA{args.bull_ema}"
        + (" & EMA slope>0" if args.require_ema_slope_up else "")
    )

    table = summary_df.copy()
    for c in ["avg_weight", "strat_ret", "spot_ret", "excess_ret"]:
        table[c] = table[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    table_html = table.to_html(index=False, border=0)

    cap = capture_df.copy()
    cap["value"] = cap["value"].map(lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x}")
    capture_html = cap.to_html(index=False, border=0)

    fig.update_layout(height=1350, title=f"HMM+ENet Bull Regime Diagnosis<br><sup>{header}</sup>")
    body = fig.to_html(include_plotlyjs="cdn", full_html=False)
    html = (
        "<html><head><meta charset='utf-8'><title>HMM ENet Bull Diagnosis</title></head><body>"
        f"<h3>{header}</h3>"
        "<h4>Segment Summary</h4>"
        f"{table_html}"
        "<h4>Bull Capture Diagnostics</h4>"
        f"{capture_html}"
        f"{body}"
        "</body></html>"
    )
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("wrote", args.out_html)
    print(summary_df.to_string(index=False))
    print(capture_df.to_string(index=False))


if __name__ == "__main__":
    main()
