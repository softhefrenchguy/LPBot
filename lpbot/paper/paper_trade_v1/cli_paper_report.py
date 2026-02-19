from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate HTML report from paper log.")
    p.add_argument("--log-csv", default="artifacts/paper/paper_log.csv")
    p.add_argument("--out", default="artifacts/paper/report.html")
    p.add_argument("--max-rows", type=int, default=5000, help="Max rows to plot (tail).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_csv)
    if not log_path.exists():
        raise SystemExit(f"Missing log csv: {log_path}")

    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if args.max_rows and len(df) > args.max_rows:
        df = df.tail(args.max_rows)

    df["core_r"] = pd.to_numeric(df.get("core_r"), errors="coerce").fillna(0.0)
    df["combined_r"] = pd.to_numeric(df.get("combined_r"), errors="coerce").fillna(0.0)
    df["r"] = pd.to_numeric(df.get("r"), errors="coerce").fillna(0.0)

    df["core_eq"] = np.exp(df["core_r"].cumsum())
    df["combined_eq"] = np.exp(df["combined_r"].cumsum())
    df["spot_eq"] = np.exp(df["r"].cumsum())

    peak = np.maximum.accumulate(df["combined_eq"].values)
    drawdown = df["combined_eq"].values / peak - 1.0

    bar_minutes = 5
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    vol = df["combined_r"].rolling(288, min_periods=30).std() * np.sqrt(bars_per_year)

    data = {
        "ts": [t.isoformat() for t in df["timestamp"]],
        "core_eq": df["core_eq"].tolist(),
        "combined_eq": df["combined_eq"].tolist(),
        "spot_eq": df["spot_eq"].tolist(),
        "price": pd.to_numeric(df.get("close"), errors="coerce").ffill().fillna(0.0).tolist(),
        "drawdown": drawdown.tolist(),
        "vol": vol.fillna(0.0).tolist(),
        "weight": pd.to_numeric(df.get("weight"), errors="coerce").fillna(0.0).tolist(),
        "lp_on": df.get("lp_on", pd.Series([False] * len(df))).astype(bool).tolist(),
        "gate": df.get("gate", pd.Series([False] * len(df))).astype(bool).tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>LPBot Paper Report</title>
  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .chart {{ width: 100%; height: 420px; }}
    .small {{ height: 240px; }}
  </style>
</head>
<body>
  <h2>LPBot Paper Report</h2>
  <div id=\"equity\" class=\"chart\"></div>
  <div id=\"price\" class=\"chart\"></div>
  <div id=\"vol\" class=\"chart small\"></div>
  <div id=\"drawdown\" class=\"chart small\"></div>
  <div id=\"weight\" class=\"chart small\"></div>
  <div id=\"gate\" class=\"chart small\"></div>

<script>
const data = {json.dumps(data)};
const ts = data.ts;

Plotly.newPlot('equity', [
  {{x: ts, y: data.spot_eq, name: 'Spot Eq', line: {{color:'#888'}}}},
  {{x: ts, y: data.core_eq, name: 'Core Eq', line: {{color:'#1f77b4'}}}},
  {{x: ts, y: data.combined_eq, name: 'Combined Eq', line: {{color:'#2ca02c'}}}},
], {{title: 'Equity Curves', margin: {{t: 40}}}});

Plotly.newPlot('price', [
  {{x: ts, y: data.price, name: 'Price', line: {{color:'#17becf'}}}},
], {{title: 'Price', margin: {{t: 40}}}});

Plotly.newPlot('vol', [
  {{x: ts, y: data.vol, name: 'Rolling Vol (ann)', line: {{color:'#bcbd22'}}}},
], {{title: 'Rolling Volatility (ann)', margin: {{t: 40}}}});

Plotly.newPlot('drawdown', [
  {{x: ts, y: data.drawdown, name: 'Drawdown', line: {{color:'#d62728'}}}},
], {{title: 'Drawdown (Combined)', margin: {{t: 40}}}});

Plotly.newPlot('weight', [
  {{x: ts, y: data.weight, name: 'Weight', line: {{color:'#ff7f0e'}}}},
], {{title: 'Weight', margin: {{t: 40}}, yaxis: {{range:[0,1]}}}});

const gate = data.gate.map(v => v ? 1 : 0);
const lp = data.lp_on.map(v => v ? 1 : 0);
Plotly.newPlot('gate', [
  {{x: ts, y: gate, name: 'Risk-Off Gate', line: {{shape:'hv', color:'#d62728'}}}},
  {{x: ts, y: lp, name: 'LP On', line: {{shape:'hv', color:'#9467bd'}}}},
], {{title: 'Gate / LP On', margin: {{t: 40}}, yaxis: {{range:[-0.1,1.1]}}}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
