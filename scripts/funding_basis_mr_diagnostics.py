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


def make_html(out_path: Path, summary: pd.DataFrame, deciles: pd.DataFrame, events: pd.DataFrame, meta: list[str]) -> None:
    labels = list(range(10))
    colors = {"1h": "#1f77b4", "4h": "#2ca02c", "12h": "#ff7f0e"}
    datasets = []
    for h, g in deciles.groupby("horizon", sort=False):
        g = g.sort_values("decile")
        datasets.append(
            {
                "label": h,
                "data": g["mean_fwd_logret"].tolist(),
                "borderColor": colors.get(h, "#444"),
                "backgroundColor": colors.get(h, "#444"),
                "borderWidth": 2,
                "pointRadius": 2,
                "fill": False,
            }
        )

    summary_rows = []
    for _, r in summary.iterrows():
        summary_rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['ic']:.6f}</td>"
            f"<td>{r['hit_rate']:.4f}</td>"
            f"<td>{r['decile_spread']:.6f}</td>"
            "</tr>"
        )

    event_rows = []
    for _, r in events.iterrows():
        event_rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td style='text-align:center'>{r['side']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['mean_fwd_logret']:.6f}</td>"
            "</tr>"
        )

    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Funding+Basis Mean Reversion Diagnostics</title>
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
  <h2>Funding + Basis Mean Reversion Diagnostics</h2>
  {meta_html}
  <div class="card">
    <table>
      <tr><th>Horizon</th><th>Rows</th><th>IC</th><th>Hit</th><th>Decile Spread</th></tr>
      {''.join(summary_rows)}
    </table>
  </div>
  <div class="card">
    <table>
      <tr><th>Horizon</th><th>Side</th><th>Events</th><th>Mean Fwd LogRet</th></tr>
      {''.join(event_rows)}
    </table>
  </div>
  <div class="card"><canvas id="dec" height="120"></canvas></div>
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
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Funding+basis mean-reversion diagnostics")
    ap.add_argument("--input-csv", default="data/backtest/ETH_perp_features_5m_90d.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/funding_basis_mr_summary.csv")
    ap.add_argument("--out-deciles-csv", default="artifacts/backtest/funding_basis_mr_deciles.csv")
    ap.add_argument("--out-events-csv", default="artifacts/backtest/funding_basis_mr_events.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/funding_basis_mr_diagnostics.html")
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--event-z-thresh", type=float, default=1.5)
    ap.add_argument("--window-days", type=int, default=90)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["spot_close", "basis", "funding_rate"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["spot_close", "basis", "funding_rate"])

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()
    if len(df) < 1000:
        raise RuntimeError(f"too few rows after window filter: {len(df)}")

    w = int(args.z_window_bars)
    df["basis_z"] = rolling_z(df["basis"], w)
    df["funding_z"] = rolling_z(df["funding_rate"].ffill(), w)

    # Mean reversion score:
    # positive score -> expected up move (too negative funding/basis),
    # negative score -> expected down move (too positive funding/basis).
    df["signal"] = -(0.6 * df["basis_z"] + 0.4 * df["funding_z"])
    df = df.dropna(subset=["signal"]).copy()

    horizons = [("1h", 12), ("4h", 48), ("12h", 144)]
    zthr = float(args.event_z_thresh)
    sum_rows = []
    dec_rows = []
    evt_rows = []

    for h_name, h_bars in horizons:
        d = df.copy()
        d["fwd"] = np.log(d["spot_close"].shift(-h_bars) / d["spot_close"])
        d = d.dropna(subset=["fwd", "signal"])
        if len(d) < 500:
            continue

        ic = float(d["signal"].corr(d["fwd"]))
        hit = float((np.sign(d["signal"]) == np.sign(d["fwd"])).mean())
        d["decile"] = pd.qcut(d["signal"], 10, labels=False, duplicates="drop")
        dec = d.groupby("decile", as_index=False)["fwd"].mean().rename(columns={"fwd": "mean_fwd_logret"})
        dec["horizon"] = h_name
        dec_rows.append(dec)

        q0 = float(dec.loc[dec["decile"] == dec["decile"].min(), "mean_fwd_logret"].iloc[0])
        q9 = float(dec.loc[dec["decile"] == dec["decile"].max(), "mean_fwd_logret"].iloc[0])
        sum_rows.append(
            {
                "horizon": h_name,
                "n": int(len(d)),
                "ic": ic,
                "hit_rate": hit,
                "decile_spread": q9 - q0,
            }
        )

        pos = d[d["signal"] >= zthr]["fwd"]
        neg = d[d["signal"] <= -zthr]["fwd"]
        evt_rows.extend(
            [
                {"horizon": h_name, "side": "signal_up", "n": int(len(pos)), "mean_fwd_logret": float(pos.mean()) if len(pos) else np.nan},
                {"horizon": h_name, "side": "signal_down", "n": int(len(neg)), "mean_fwd_logret": float(neg.mean()) if len(neg) else np.nan},
            ]
        )

    if not sum_rows:
        raise RuntimeError("no valid horizon results")

    summary = pd.DataFrame(sum_rows)
    deciles = pd.concat(dec_rows, ignore_index=True)
    events = pd.DataFrame(evt_rows)

    out_sum = Path(args.out_summary_csv)
    out_dec = Path(args.out_deciles_csv)
    out_evt = Path(args.out_events_csv)
    out_html = Path(args.out_html)
    out_sum.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_sum, index=False)
    deciles.to_csv(out_dec, index=False)
    events.to_csv(out_evt, index=False)

    meta = [
        f"input: {args.input_csv}",
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"rows_used: {len(df)} | z_window_bars: {w} | event_z_thresh: {zthr}",
        "signal = -(0.6*basis_z + 0.4*funding_z)",
    ]
    make_html(out_html, summary, deciles, events, meta)

    print(f"wrote {out_sum}")
    print(f"wrote {out_dec}")
    print(f"wrote {out_evt}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
