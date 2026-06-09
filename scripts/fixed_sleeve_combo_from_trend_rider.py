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
    ap = argparse.ArgumentParser(description="Build fixed sleeve allocation from trend-rider daily outputs.")
    ap.add_argument("--in-csv", default="artifacts/backtest/crypto_etf_combo_trend_rider_5y_bf025_bc005.csv")
    ap.add_argument("--alloc-crypto-sleeve", type=float, default=0.30)
    ap.add_argument("--w-col", default="w_crypto_rider")
    ap.add_argument("--cost-col", default="cost_rider")
    ap.add_argument("--out-csv", default="artifacts/backtest/fixed_sleeve_30_70.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/fixed_sleeve_30_70.html")
    ap.add_argument("--summary-csv", default="artifacts/backtest/fixed_sleeve_30_70_summary.csv")
    args = ap.parse_args()

    a = float(args.alloc_crypto_sleeve)
    if a < 0.0 or a > 1.0:
        raise ValueError("alloc-crypto-sleeve must be in [0,1]")

    df = pd.read_csv(args.in_csv)
    required = {"day", "crypto_r_daily", "basket_r_daily", args.w_col, args.cost_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in input: {missing}")

    df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
    df["crypto_r_daily"] = pd.to_numeric(df["crypto_r_daily"], errors="coerce")
    df["basket_r_daily"] = pd.to_numeric(df["basket_r_daily"], errors="coerce")
    df["w_crypto"] = pd.to_numeric(df[args.w_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["cost_crypto"] = pd.to_numeric(df[args.cost_col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["day", "crypto_r_daily", "basket_r_daily"]).sort_values("day")

    df["crypto_sleeve_r"] = df["w_crypto"] * df["crypto_r_daily"] - df["cost_crypto"]
    df["combo_r"] = a * df["crypto_sleeve_r"] + (1.0 - a) * df["basket_r_daily"]
    df["crypto_sleeve_eq"] = np.exp(df["crypto_sleeve_r"].cumsum())
    df["basket_eq"] = np.exp(df["basket_r_daily"].cumsum())
    df["combo_eq"] = np.exp(df["combo_r"].cumsum())
    df["eth_eq"] = np.exp(df["crypto_r_daily"].cumsum())
    for c in ("crypto_sleeve_eq", "basket_eq", "combo_eq", "eth_eq"):
        df[c] = df[c] / float(df[c].iloc[0])

    m_combo = _perf(df["combo_r"])
    m_crypto_sleeve = _perf(df["crypto_sleeve_r"])
    m_basket = _perf(df["basket_r_daily"])
    m_eth = _perf(df["crypto_r_daily"])

    summary = pd.DataFrame(
        [
            {
                "model": "fixed_combo",
                **m_combo,
                "alloc_crypto_sleeve": a,
                "alloc_etf_sleeve": 1.0 - a,
                "avg_w_inside_crypto": float(df["w_crypto"].mean()),
                "effective_crypto_notional": float(a * df["w_crypto"].mean()),
            },
            {
                "model": "crypto_sleeve_only",
                **m_crypto_sleeve,
                "alloc_crypto_sleeve": 1.0,
                "alloc_etf_sleeve": 0.0,
                "avg_w_inside_crypto": float(df["w_crypto"].mean()),
                "effective_crypto_notional": float(df["w_crypto"].mean()),
            },
            {
                "model": "etf_sleeve_only",
                **m_basket,
                "alloc_crypto_sleeve": 0.0,
                "alloc_etf_sleeve": 1.0,
                "avg_w_inside_crypto": float(df["w_crypto"].mean()),
                "effective_crypto_notional": 0.0,
            },
            {
                "model": "eth_spot",
                **m_eth,
                "alloc_crypto_sleeve": 1.0,
                "alloc_etf_sleeve": 0.0,
                "avg_w_inside_crypto": 1.0,
                "effective_crypto_notional": 1.0,
            },
        ]
    )
    summary["window_start"] = str(df["day"].iloc[0])
    summary["window_end"] = str(df["day"].iloc[-1])
    summary["rows"] = int(len(df))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    Path(args.summary_csv).write_text(summary.to_csv(index=False), encoding="utf-8")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity (Normalized)", "Sleeve Weights", "Crypto Internal Weight"],
    )
    fig.add_trace(go.Scatter(x=df["day"], y=df["combo_eq"], name="Fixed Combo Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["day"], y=df["basket_eq"], name="ETF Basket Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["day"], y=df["crypto_sleeve_eq"], name="Crypto Sleeve Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["day"], y=df["eth_eq"], name="ETH Spot Eq"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["day"], y=np.full(len(df), a), name="w_crypto_sleeve"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["day"], y=np.full(len(df), 1.0 - a), name="w_etf_sleeve"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["day"], y=df["w_crypto"], name="w_inside_crypto"), row=3, col=1)
    fig.update_layout(height=1100, title=f"Fixed Sleeve Combo ({a:.0%} Crypto Sleeve / {1.0-a:.0%} ETF Sleeve)")

    html = (
        "<html><head><meta charset='utf-8'><title>Fixed Sleeve Combo</title></head><body>"
        f"<h3>Fixed Sleeve Combo ({a:.0%} Crypto / {1.0-a:.0%} ETF)</h3>"
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
