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
    p.add_argument("--daily-out-csv", default="artifacts/paper/breakout_daily_summary.csv")
    p.add_argument("--daily-days", type=int, default=14)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    log_path = Path(args.log_csv)
    out_path = Path(args.out)
    daily_out_path = Path(args.daily_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_out_path.parent.mkdir(parents=True, exist_ok=True)

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
        df_all = df.copy()
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

        # Rebase equity to 1.0 for the displayed window
        first_eq = float(df["eq"].iloc[0]) if len(df) else 1.0
        if first_eq != 0:
            df["eq"] = df["eq"] / first_eq

        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
        avg_weight = float(df["weight"].fillna(0.0).mean()) if "weight" in df.columns else 0.0
        time_in_market = float((df.get("weight", 0.0) > 0).mean()) * 100 if "weight" in df.columns else 0.0

        labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in df.columns else list(range(len(df)))
        eq_vals = [round(x, 6) for x in df["eq"].fillna(0.0).tolist()]
        weight_vals = [round(x, 6) for x in df.get("weight", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]
        close_vals = [round(x, 6) for x in df.get("close", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]

        daily_table_html = "<div class='meta'>daily summary unavailable</div>"
        if {"timestamp", "weight", "core_r"}.issubset(df_all.columns):
            full = df_all.copy()
            full["weight"] = pd.to_numeric(full["weight"], errors="coerce").fillna(0.0)
            full["core_r"] = pd.to_numeric(full["core_r"], errors="coerce").fillna(0.0)
            full["weight_prev"] = full["weight"].shift(1).fillna(0.0)
            full["turnover"] = (full["weight"] - full["weight_prev"]).abs()
            full["trade_cost_r"] = -full["turnover"] * (args.trade_cost_bps / 10000.0)
            full["net_r"] = full["core_r"] + full["trade_cost_r"]
            full["active"] = full["weight_prev"] > 0
            full["date"] = full["timestamp"].dt.date

            records = []
            for d, g in full.groupby("date", sort=True):
                gross_eq = np.exp(np.cumsum(g["core_r"]))
                net_eq = np.exp(np.cumsum(g["net_r"]))
                net_peak = np.maximum.accumulate(net_eq)
                net_mdd = float((net_eq / net_peak - 1.0).min()) if len(net_eq) else 0.0
                active = g["active"]
                hit = float((g.loc[active, "core_r"] > 0).mean() * 100.0) if active.any() else np.nan

                records.append(
                    {
                        "date": str(d),
                        "bars": int(len(g)),
                        "daily_return_gross": float(gross_eq.iloc[-1] - 1.0),
                        "daily_return_net": float(net_eq.iloc[-1] - 1.0),
                        "daily_max_dd_net": net_mdd,
                        "turnover": float(g["turnover"].sum()),
                        "avg_weight": float(g["weight"].mean()),
                        "time_in_market_pct": float((g["weight"] > 0).mean() * 100.0),
                        "hit_rate_active_pct": hit,
                    }
                )

            daily = pd.DataFrame(records)
            daily.to_csv(daily_out_path, index=False)
            daily_show = daily.tail(args.daily_days).copy()
            daily_show = daily_show.iloc[::-1]

            rows_html = []
            for _, row in daily_show.iterrows():
                hit_txt = "" if pd.isna(row["hit_rate_active_pct"]) else f"{row['hit_rate_active_pct']:.2f}%"
                rows_html.append(
                    "<tr>"
                    f"<td style='text-align:center'>{row['date']}</td>"
                    f"<td>{int(row['bars'])}</td>"
                    f"<td>{row['daily_return_gross']:.4%}</td>"
                    f"<td>{row['daily_return_net']:.4%}</td>"
                    f"<td>{row['daily_max_dd_net']:.4%}</td>"
                    f"<td>{row['turnover']:.4f}</td>"
                    f"<td>{row['avg_weight']:.4f}</td>"
                    f"<td>{row['time_in_market_pct']:.2f}%</td>"
                    f"<td>{hit_txt}</td>"
                    "</tr>"
                )

            daily_table_html = (
                "<div class='card'>"
                f"<div class='meta'>Daily summary (last {args.daily_days} days) | estimated cost: {args.trade_cost_bps:.2f} bps per unit turnover | csv: {daily_out_path}</div>"
                "<table>"
                "<tr>"
                "<th>Date</th><th>Bars</th><th>Gross Return</th><th>Net Return</th><th>Net Max DD</th>"
                "<th>Turnover</th><th>Avg Weight</th><th>Time In Market</th><th>Hit Rate (active)</th>"
                "</tr>"
                + "".join(rows_html)
                + "</table></div>"
            )

        body = f"""
<h2>Breakout Paper Report</h2>
<div class="meta">rows: {len(df)} | last_ts: {last_ts} | avg_weight: {avg_weight:.4f} | time_in_market: {time_in_market:.2f}%</div>
{daily_table_html}
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
