from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def make_html(
    out_path: Path,
    title: str,
    summary_df: pd.DataFrame,
    deciles_df: pd.DataFrame,
    meta_lines: list[str],
) -> None:
    labels = list(range(10))
    datasets = []
    colors = {
        "1h": "#1f77b4",
        "4h": "#2ca02c",
        "12h": "#ff7f0e",
    }
    for h, g in deciles_df.groupby("horizon", sort=False):
        g = g.sort_values("decile")
        y = g["mean_fwd_logret"].tolist()
        datasets.append(
            {
                "label": h,
                "data": y,
                "borderColor": colors.get(h, "#444"),
                "backgroundColor": colors.get(h, "#444"),
                "borderWidth": 2,
                "pointRadius": 2,
                "fill": False,
            }
        )

    rows_html = []
    for _, r in summary_df.iterrows():
        rows_html.append(
            "<tr>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['ic']:.6f}</td>"
            f"<td>{r['hit_rate']:.4f}</td>"
            f"<td>{r['decile_spread']:.6f}</td>"
            f"<td>{r['q0_mean']:.6f}</td>"
            f"<td>{r['q9_mean']:.6f}</td>"
            "</tr>"
        )

    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta_lines])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    .meta {{ margin-bottom: 8px; color: #444; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: right; }}
    th {{ background: #f7f7f7; text-align: center; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  {meta_html}
  <div class="card">
    <table>
      <tr>
        <th>Horizon</th><th>Rows</th><th>IC</th><th>Hit Rate</th>
        <th>Decile Spread</th><th>Q0 Mean</th><th>Q9 Mean</th>
      </tr>
      {''.join(rows_html)}
    </table>
  </div>
  <div class="card">
    <canvas id="dec" height="120"></canvas>
  </div>
  <script>
    const labels = {json.dumps(labels)};
    const datasets = {json.dumps(datasets)};
    new Chart(document.getElementById('dec'), {{
      type: 'line',
      data: {{ labels, datasets }},
      options: {{
        animation: false,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Signal decile (low -> high)' }} }},
          y: {{ title: {{ display: true, text: 'Mean forward log return' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Perp positioning alpha diagnostics")
    ap.add_argument("--input-csv", default="data/backtest/ETH_perp_features_5m_90d.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/perp_alpha_summary.csv")
    ap.add_argument("--out-deciles-csv", default="artifacts/backtest/perp_alpha_deciles.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/perp_alpha_diagnostics.html")
    ap.add_argument("--z-window-bars", type=int, default=288)  # 1 day on 5m bars
    ap.add_argument("--score-ema-span", type=int, default=12)
    ap.add_argument("--invert-score", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        raise FileNotFoundError(f"missing input csv: {inp}")

    df = pd.read_csv(inp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    for c in ["spot_close", "perp_close", "basis", "funding_rate", "oi_metric", "ret_1h", "oi_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    # Restrict to rows where OI exists (Binance gives only recent history via public endpoint)
    df = df[df["oi_metric"].notna()].copy()
    if len(df) < 1000:
        raise RuntimeError(f"too few rows after OI filter: {len(df)}")

    w = int(args.z_window_bars)
    df["funding_z"] = rolling_z(df["funding_rate"].ffill(), w)
    df["basis_z"] = rolling_z(df["basis"], w)
    df["oi_chg_z"] = rolling_z(df["oi_chg"], w)
    df["ret_1h_z"] = rolling_z(df["ret_1h"], w)

    # Crowding x leverage pressure
    df["long_crowded"] = df["funding_z"] + df["basis_z"]
    df["score_raw"] = -(df["long_crowded"] * df["oi_chg_z"])
    df["score"] = df["score_raw"].ewm(span=int(args.score_ema_span), adjust=False).mean()
    if args.invert_score:
        df["score"] = -df["score"]

    horizons = [("1h", 12), ("4h", 48), ("12h", 144)]
    summary_rows = []
    dec_rows = []

    for h_name, h_bars in horizons:
        d = df.copy()
        d["fwd"] = np.log(d["spot_close"].shift(-h_bars) / d["spot_close"])
        d = d.dropna(subset=["score", "fwd"])
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "fwd"])
        if len(d) < 500:
            continue

        ic = float(d["score"].corr(d["fwd"]))
        hit = float((np.sign(d["score"]) == np.sign(d["fwd"])).mean())

        d["decile"] = pd.qcut(d["score"], 10, labels=False, duplicates="drop")
        dec = d.groupby("decile", as_index=False)["fwd"].mean().rename(columns={"fwd": "mean_fwd_logret"})
        dec["horizon"] = h_name
        dec_rows.append(dec)

        q0 = float(dec.loc[dec["decile"] == dec["decile"].min(), "mean_fwd_logret"].iloc[0])
        q9 = float(dec.loc[dec["decile"] == dec["decile"].max(), "mean_fwd_logret"].iloc[0])
        spread = q9 - q0
        summary_rows.append(
            {
                "horizon": h_name,
                "n": int(len(d)),
                "ic": ic,
                "hit_rate": hit,
                "decile_spread": spread,
                "q0_mean": q0,
                "q9_mean": q9,
            }
        )

    if not summary_rows:
        raise RuntimeError("no valid horizon results")

    summary_df = pd.DataFrame(summary_rows)
    deciles_df = pd.concat(dec_rows, ignore_index=True)

    out_summary = Path(args.out_summary_csv)
    out_dec = Path(args.out_deciles_csv)
    out_html = Path(args.out_html)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_summary, index=False)
    deciles_df.to_csv(out_dec, index=False)

    meta_lines = [
        f"input: {inp}",
        f"rows_used_after_oi_filter: {len(df)}",
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"z_window_bars: {w} | score_ema_span: {args.score_ema_span}",
        f"invert_score: {args.invert_score}",
    ]
    make_html(out_html, "Perp Positioning Alpha Diagnostics", summary_df, deciles_df, meta_lines)

    print(f"wrote {out_summary}")
    print(f"wrote {out_dec}")
    print(f"wrote {out_html}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
