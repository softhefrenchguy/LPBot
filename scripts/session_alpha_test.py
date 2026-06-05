from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * 12
    eq = np.exp(np.cumsum(x.to_numpy()))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    return {
        "ret": float(eq[-1] - 1.0),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": float((eq / peak - 1.0).min()),
    }


def main():
    ap = argparse.ArgumentParser(description="Session alpha quick test (UTC hour effects)")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--horizon-bars", type=int, default=12, help="forward horizon for hour scoring (12 bars = 1h)")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--top-k-hours", type=int, default=6)
    ap.add_argument("--bottom-k-hours", type=int, default=6)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/session_alpha_summary.csv")
    ap.add_argument("--out-hours-csv", default="artifacts/backtest/session_alpha_hours.csv")
    ap.add_argument("--out-equity-csv", default="artifacts/backtest/session_alpha_equity.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/session_alpha.html")
    args = ap.parse_args()

    p = Path(args.eth_csv)
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")

    df = pd.read_csv(p)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("input needs timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()
    if len(df) < 2000:
        raise RuntimeError(f"too few rows in window: {len(df)}")

    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    h = int(args.horizon_bars)
    df["fwd"] = np.log(df["close"].shift(-h) / df["close"])
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df = df.dropna(subset=["fwd"]).copy()

    n = len(df)
    split = int(n * float(args.train_ratio))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    if len(test) < 500:
        raise RuntimeError("test segment too short")

    hour_train = train.groupby("hour", as_index=False)["fwd"].mean().rename(columns={"fwd": "train_mean_fwd"})
    hour_test = test.groupby("hour", as_index=False)["fwd"].mean().rename(columns={"fwd": "test_mean_fwd"})
    hours = pd.merge(hour_train, hour_test, on="hour", how="outer").fillna(0.0).sort_values("hour")

    ranked = hour_train.sort_values("train_mean_fwd")
    bottom_hours = ranked.head(int(args.bottom_k_hours))["hour"].tolist()
    top_hours = ranked.tail(int(args.top_k_hours))["hour"].tolist()

    test["weight_raw"] = 0.0
    test.loc[test["hour"].isin(top_hours), "weight_raw"] = 1.0
    if args.allow_short:
        test.loc[test["hour"].isin(bottom_hours), "weight_raw"] = -1.0

    test["turnover"] = test["weight_raw"].diff().abs().fillna(0.0)
    test["cost_r"] = -test["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    test["strat_r"] = test["weight_raw"].shift(1).fillna(0.0) * test["r"] + test["cost_r"]
    test["eq"] = np.exp(np.cumsum(test["strat_r"].to_numpy()))
    test["spot_eq"] = np.exp(np.cumsum(test["r"].to_numpy()))
    test["eq"] = test["eq"] / float(test["eq"].iloc[0])
    test["spot_eq"] = test["spot_eq"] / float(test["spot_eq"].iloc[0])

    s = perf(test["strat_r"])
    spot = perf(test["r"])
    time_in_market = float((test["weight_raw"] != 0).mean() * 100.0)
    turnover = float(test["turnover"].sum())

    summary = pd.DataFrame(
        [
            {
                "segment": "test_strategy",
                "rows": int(len(test)),
                "ret": s["ret"],
                "ann_vol": s["ann_vol"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_dd"],
                "time_in_market_pct": time_in_market,
                "turnover": turnover,
                "top_hours": ",".join([str(x) for x in sorted(top_hours)]),
                "bottom_hours": ",".join([str(x) for x in sorted(bottom_hours)]),
            },
            {
                "segment": "test_spot",
                "rows": int(len(test)),
                "ret": spot["ret"],
                "ann_vol": spot["ann_vol"],
                "sharpe": spot["sharpe"],
                "max_dd": spot["max_dd"],
                "time_in_market_pct": 100.0,
                "turnover": 0.0,
                "top_hours": "",
                "bottom_hours": "",
            },
        ]
    )

    out_summary = Path(args.out_summary_csv)
    out_hours = Path(args.out_hours_csv)
    out_eq = Path(args.out_equity_csv)
    out_html = Path(args.out_html)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    hours.to_csv(out_hours, index=False)
    test[["timestamp", "close", "hour", "weight_raw", "r", "strat_r", "eq", "spot_eq"]].to_csv(out_eq, index=False)

    labels = test["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    eq_data = [round(float(x), 6) for x in test["eq"].tolist()]
    spot_data = [round(float(x), 6) for x in test["spot_eq"].tolist()]
    w_data = [round(float(x), 6) for x in test["weight_raw"].tolist()]
    price_data = [round(float(x), 6) for x in test["close"].tolist()]
    hour_labels = [int(x) for x in hours["hour"].tolist()]
    train_hour_vals = [float(x) for x in hours["train_mean_fwd"].tolist()]
    test_hour_vals = [float(x) for x in hours["test_mean_fwd"].tolist()]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Session Alpha Test</title>
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
  <h2>Session Alpha Test (UTC hour)</h2>
  <div class="meta">window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}</div>
  <div class="meta">train rows: {len(train)} | test rows: {len(test)} | allow_short: {args.allow_short}</div>
  <div class="meta">top_hours: {sorted(top_hours)} | bottom_hours: {sorted(bottom_hours) if args.allow_short else 'n/a'}</div>
  <div class="meta">strategy ret: {s['ret']:.2%} | sharpe: {s['sharpe']:.4f} | max_dd: {s['max_dd']:.2%} | time_in_market: {time_in_market:.2f}% | turnover: {turnover:.2f}</div>
  <div class="meta">spot ret(test): {spot['ret']:.2%} | sharpe: {spot['sharpe']:.4f} | max_dd: {spot['max_dd']:.2%}</div>
  <div class="card">
    <table>
      <tr><th>Hour</th><th>Train Mean Fwd</th><th>Test Mean Fwd</th></tr>
      {''.join([f"<tr><td style='text-align:center'>{int(hh)}</td><td>{tv:.6f}</td><td>{sv:.6f}</td></tr>" for hh,tv,sv in zip(hour_labels, train_hour_vals, test_hour_vals)])}
    </table>
  </div>
  <div class="card"><canvas id="hour" height="110"></canvas></div>
  <div class="card"><canvas id="eq" height="110"></canvas></div>
  <div class="card"><canvas id="w" height="90"></canvas></div>
  <div class="card"><canvas id="p" height="110"></canvas></div>
  <script>
    const hourLabels = {json.dumps(hour_labels)};
    const trainHourVals = {json.dumps(train_hour_vals)};
    const testHourVals = {json.dumps(test_hour_vals)};
    const labels = {json.dumps(labels)};
    const eqData = {json.dumps(eq_data)};
    const spotData = {json.dumps(spot_data)};
    const wData = {json.dumps(w_data)};
    const pData = {json.dumps(price_data)};
    const baseOpt = {{ animation: false, plugins: {{ legend: {{ display: true }} }} }};
    new Chart(document.getElementById('hour'), {{
      type: 'line',
      data: {{
        labels: hourLabels,
        datasets: [
          {{ label: 'Train mean fwd', data: trainHourVals, borderColor: '#1f77b4', borderWidth: 2, pointRadius: 2 }},
          {{ label: 'Test mean fwd', data: testHourVals, borderColor: '#ff7f0e', borderWidth: 2, pointRadius: 2 }}
        ]
      }},
      options: {{ ...baseOpt, scales: {{ x: {{ title: {{ display: true, text: 'UTC hour' }} }} }} }}
    }});
    new Chart(document.getElementById('eq'), {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'Strategy Eq (test)', data: eqData, borderColor: '#1f77b4', borderWidth: 2, pointRadius: 0 }},
          {{ label: 'Spot Eq (test)', data: spotData, borderColor: '#7f7f7f', borderWidth: 2, pointRadius: 0 }}
        ]
      }},
      options: {{ ...baseOpt, scales: {{ x: {{ display: false }} }} }}
    }});
    new Chart(document.getElementById('w'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Weight', data: wData, borderColor: '#2ca02c', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ ...baseOpt, scales: {{ x: {{ display: false }}, y: {{ min: -1, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('p'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Price', data: pData, borderColor: '#9467bd', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ ...baseOpt, scales: {{ x: {{ display: false }} }} }}
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_summary}")
    print(f"wrote {out_hours}")
    print(f"wrote {out_eq}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
