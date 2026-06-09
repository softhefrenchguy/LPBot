from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_price(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must include timestamp, close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if int(m.sum()) < 50:
        return float("nan")
    return float(x[m].corr(y[m]))


def _safe_mean(x: pd.Series) -> float:
    if len(x) == 0:
        return float("nan")
    return float(pd.to_numeric(x, errors="coerce").dropna().mean())


def _top_decile_mask(s: pd.Series) -> pd.Series:
    q = s.quantile(0.9)
    return s >= q


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose whether velocity/acceleration predicts future bull runs.")
    p.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    p.add_argument("--spans", default="12,24,48,96", help="EMA spans (bars) for smoothed log-price")
    p.add_argument("--std-window", type=int, default=288, help="Rolling window for feature normalization")
    p.add_argument("--horizons-bars", default="288,864,2016,8640", help="Forward return horizons in bars")
    p.add_argument("--bull-horizon-bars", type=int, default=8640, help="Forward horizon to define bull starts")
    p.add_argument("--bull-threshold", type=float, default=0.20, help="Bull start if fwd return >= threshold")
    p.add_argument("--out-summary-csv", default="artifacts/paper/bull_accel_summary.csv")
    p.add_argument("--out-deciles-csv", default="artifacts/paper/bull_accel_deciles.csv")
    p.add_argument("--out-html", default="artifacts/paper/bull_accel_report.html")
    args = p.parse_args()

    spans = [int(x) for x in args.spans.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons_bars.split(",") if x.strip()]

    df = _load_price(args.price_csv)
    y = np.log(df["close"])

    all_summary_rows: list[dict] = []
    all_decile_rows: list[dict] = []

    for span in spans:
        ema = y.ewm(span=span, adjust=False).mean()
        v = ema.diff()
        a = v.diff()
        denom = v.rolling(int(args.std_window), min_periods=max(20, int(args.std_window // 4))).std()
        z_v = (v / (denom + 1e-12)).replace([np.inf, -np.inf], np.nan)
        z_a = (a / (denom + 1e-12)).replace([np.inf, -np.inf], np.nan)
        score = z_v + 0.5 * z_a

        d = pd.DataFrame({"timestamp": df["timestamp"], "z_v": z_v, "z_a": z_a, "score": score, "close": df["close"]})

        for h in horizons:
            fwd = np.log(d["close"].shift(-h) / d["close"])
            fwd_simple = np.exp(fwd) - 1.0
            d_h = d.copy()
            d_h["fwd"] = fwd_simple
            d_h = d_h.dropna(subset=["fwd", "z_v", "z_a", "score"])
            if d_h.empty:
                continue

            base_rate = float((d_h["fwd"] >= args.bull_threshold).mean()) if h == int(args.bull_horizon_bars) else float("nan")

            for feat_name in ["z_a", "z_v", "score"]:
                feat = d_h[feat_name]
                top = _top_decile_mask(feat)
                bot = feat <= feat.quantile(0.1)
                hit = ((np.sign(feat) == np.sign(d_h["fwd"])).mean())
                top_ret = _safe_mean(d_h.loc[top, "fwd"])
                bot_ret = _safe_mean(d_h.loc[bot, "fwd"])
                spread = top_ret - bot_ret
                ic = _safe_corr(feat, d_h["fwd"])
                top_bull_rate = float((d_h.loc[top, "fwd"] >= args.bull_threshold).mean()) if h == int(args.bull_horizon_bars) else float("nan")
                lift = float(top_bull_rate / (base_rate + 1e-12)) if h == int(args.bull_horizon_bars) else float("nan")

                all_summary_rows.append(
                    {
                        "span": span,
                        "horizon_bars": h,
                        "feature": feat_name,
                        "n": len(d_h),
                        "ic": ic,
                        "hit_rate": float(hit),
                        "top_decile_mean_ret": top_ret,
                        "bot_decile_mean_ret": bot_ret,
                        "decile_spread": spread,
                        "bull_base_rate": base_rate,
                        "bull_top_decile_rate": top_bull_rate,
                        "bull_lift_top_decile": lift,
                    }
                )

            # decile curves on score for charting
            d_h["decile"] = pd.qcut(d_h["score"], 10, labels=False, duplicates="drop")
            g = d_h.groupby("decile")["fwd"].mean().reset_index()
            for _, r in g.iterrows():
                all_decile_rows.append(
                    {
                        "span": span,
                        "horizon_bars": h,
                        "decile": int(r["decile"]),
                        "mean_fwd_ret": float(r["fwd"]),
                    }
                )

    summary = pd.DataFrame(all_summary_rows)
    deciles = pd.DataFrame(all_decile_rows)

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    out_deciles = Path(args.out_deciles_csv)
    out_deciles.parent.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(out_deciles, index=False)

    # Simple HTML report (no hard dependency on plotly)
    best = summary[
        (summary["horizon_bars"] == int(args.bull_horizon_bars)) & (summary["feature"] == "z_a")
    ].sort_values("bull_lift_top_decile", ascending=False)
    best_span = int(best.iloc[0]["span"]) if not best.empty else int(spans[0])

    html = [
        "<html><head><meta charset='utf-8'><title>Bull Acceleration Diagnostics</title></head><body>",
        "<h2>Bull Acceleration Diagnostics</h2>",
        f"<p>price={args.price_csv} | spans={spans} | std_window={args.std_window} | horizons={horizons} | bull_h={args.bull_horizon_bars} | bull_th={args.bull_threshold}</p>",
        "<h3>Summary (top 30 by decile_spread)</h3>",
        summary.sort_values("decile_spread", ascending=False).head(30).round(6).to_html(index=False, border=0),
        f"<h3>Best span by bull-lift (z_a @ h={args.bull_horizon_bars}) = {best_span}</h3>",
    ]

    dplot = deciles[deciles["span"] == best_span].copy()
    if not dplot.empty:
        piv = dplot.pivot_table(index="decile", columns="horizon_bars", values="mean_fwd_ret", aggfunc="mean")
        html.append(piv.round(6).to_html(border=0))
    html.append("</body></html>")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html), encoding="utf-8")

    print("wrote", out_summary)
    print("wrote", out_deciles)
    print("wrote", out_html)
    print(
        summary[
            ["span", "horizon_bars", "feature", "ic", "hit_rate", "decile_spread", "bull_lift_top_decile"]
        ]
        .sort_values(["horizon_bars", "feature", "decile_spread"], ascending=[True, True, False])
        .head(20)
        .round(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
