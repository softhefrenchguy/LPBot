import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


def _write_html(out_path: Path, title: str, body: str) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h2 {{ margin: 0 0 10px 0; }}
    .meta {{ margin-bottom: 16px; color: #444; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: right; }}
    th {{ background: #f7f7f7; text-align: center; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Breakout paper report generator")
    p.add_argument("--log-csv", default="artifacts/paper/breakout_paper.csv")
    p.add_argument("--out", default="artifacts/paper/breakout_report.html")
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    log_path = Path(args.log_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        if not log_path.exists():
            _write_html(out_path, "Breakout Report", f"<h2>Breakout Paper Report</h2><div class='meta'>log not found: {log_path}</div>")
            if args.once:
                break
            time.sleep(max(1, args.interval_seconds))
            continue

        df = pd.read_csv(log_path)
        if df.empty:
            _write_html(out_path, "Breakout Report", "<h2>Breakout Paper Report</h2><div class='meta'>no rows yet</div>")
            if args.once:
                break
            time.sleep(max(1, args.interval_seconds))
            continue

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.tail(args.max_rows).copy()

        for col in ["close", "weight", "core_r", "eq"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "eq" not in df.columns or df["eq"].isna().all():
            if "core_r" in df.columns:
                r = df["core_r"].fillna(0.0)
                df["eq"] = np.exp(np.cumsum(r))
            else:
                df["eq"] = 1.0

        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
        avg_weight = float(df["weight"].fillna(0.0).mean()) if "weight" in df.columns else 0.0
        time_in_market = float((df.get("weight", 0.0) > 0).mean()) * 100 if "weight" in df.columns else 0.0

        labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in df.columns else list(range(len(df)))
        eq_vals = [round(x, 6) for x in df["eq"].fillna(0.0).tolist()]
        weight_vals = [round(x, 6) for x in df.get("weight", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]
        close_vals = [round(x, 6) for x in df.get("close", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]

        body = f"""
<h2>Breakout Paper Report</h2>
<div class="meta">rows: {len(df)} | last_ts: {last_ts} | avg_weight: {avg_weight:.4f} | time_in_market: {time_in_market:.2f}%</div>
<div class="card">
  <canvas id="eq" height="140"></canvas>
</div>
<div class="card">
  <canvas id="weight" height="120"></canvas>
</div>
<div class="card">
  <canvas id="price" height="140"></canvas>
</div>
<script>
const labels = {json.dumps(labels)};
const eqData = {json.dumps(eq_vals)};
const weightData = {json.dumps(weight_vals)};
const priceData = {json.dumps(close_vals)};

new Chart(document.getElementById('eq'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Equity', data: eqData, borderColor: '#2b8cbe', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
}});
new Chart(document.getElementById('weight'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Weight', data: weightData, borderColor: '#fdae61', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }}, y: {{ min: 0, max: 1 }} }} }}
}});
new Chart(document.getElementById('price'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Price', data: priceData, borderColor: '#31a354', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
}});
</script>
"""
        _write_html(out_path, "Breakout Report", body)

        if args.once:
            break
        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    main()
