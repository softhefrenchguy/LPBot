from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_series(path: str, r_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or r_col not in df.columns:
        raise ValueError(f"{path} must have timestamp and {r_col}")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"], utc=True, errors="coerce"),
            "r": pd.to_numeric(df[r_col], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out["r"] = out["r"].fillna(0.0)
    return out


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy()
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    ret = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (bars_per_year / len(x)) - 1.0) if len(x) else float("nan")
    ann_vol = float(np.std(x) * np.sqrt(bars_per_year)) if len(x) else float("nan")
    sharpe = float((np.mean(x) * bars_per_year) / (ann_vol + 1e-12)) if len(x) else float("nan")
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(x) else float("nan")
    return {"ret": ret, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple strategy curves on common window.")
    ap.add_argument("--v5-csv", default="artifacts/paper/hmm_enet_scaled_v5.csv")
    ap.add_argument("--breakout-csv", default="artifacts/paper/breakout_paper.csv")
    ap.add_argument("--funding-csv", default="artifacts/paper/funding_basis_mr_365d_db.csv")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out-html", default="artifacts/paper/alpha_compare_report.html")
    ap.add_argument("--out-csv", default="artifacts/paper/alpha_compare_summary.csv")
    args = ap.parse_args()

    v5 = load_series(args.v5_csv, "strat_r")
    br = load_series(args.breakout_csv, "core_r")
    fu = load_series(args.funding_csv, "strat_r")

    spot = load_series(args.v5_csv, "r")

    # common and window
    start = max(v5["timestamp"].min(), br["timestamp"].min(), fu["timestamp"].min(), spot["timestamp"].min())
    end = min(v5["timestamp"].max(), br["timestamp"].max(), fu["timestamp"].max(), spot["timestamp"].max())
    start = max(start, end - pd.Timedelta(days=int(args.days)))

    def cut(d: pd.DataFrame) -> pd.DataFrame:
        return d[(d["timestamp"] >= start) & (d["timestamp"] <= end)].copy().sort_values("timestamp")

    v5 = cut(v5)
    br = cut(br)
    fu = cut(fu)
    spot = cut(spot)

    merged = spot[["timestamp"]].copy()
    merged = merged.merge(v5.rename(columns={"r": "v5_r"}), on="timestamp", how="inner")
    merged = merged.merge(br.rename(columns={"r": "br_r"}), on="timestamp", how="inner")
    merged = merged.merge(fu.rename(columns={"r": "fu_r"}), on="timestamp", how="inner")
    merged = merged.merge(spot.rename(columns={"r": "spot_r"}), on="timestamp", how="inner")
    merged = merged.sort_values("timestamp")

    merged["v5_eq"] = np.exp(np.cumsum(merged["v5_r"].to_numpy()))
    merged["breakout_eq"] = np.exp(np.cumsum(merged["br_r"].to_numpy()))
    merged["funding_eq"] = np.exp(np.cumsum(merged["fu_r"].to_numpy()))
    merged["spot_eq"] = np.exp(np.cumsum(merged["spot_r"].to_numpy()))

    summary = pd.DataFrame(
        [
            {"model": "v5", **perf(merged["v5_r"])},
            {"model": "breakout", **perf(merged["br_r"])},
            {"model": "funding_basis_db", **perf(merged["fu_r"])},
            {"model": "spot", **perf(merged["spot_r"])},
        ]
    )
    summary["start"] = str(start)
    summary["end"] = str(end)
    summary["rows"] = len(merged)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Aligned {args.days}d Equity"])
    fig.add_trace(go.Scatter(x=merged["timestamp"], y=merged["v5_eq"], name="V5 Eq", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=merged["timestamp"], y=merged["breakout_eq"], name="Breakout Eq", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=merged["timestamp"], y=merged["funding_eq"], name="Funding Eq", line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=merged["timestamp"], y=merged["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")))
    fig.update_layout(height=600, title="Alpha Comparison")

    html = (
        "<html><head><meta charset='utf-8'><title>Alpha Compare</title></head><body>"
        f"<h3>Alpha Comparison ({args.days}d)</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print("wrote", out_csv)
    print("wrote", out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
