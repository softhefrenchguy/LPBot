from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_html(out_path: Path, summary: pd.DataFrame, deciles: pd.DataFrame, events: pd.DataFrame, meta: list[str]) -> None:
    labels = list(range(10))
    colors = {"15m": "#1f77b4", "30m": "#2ca02c", "60m": "#ff7f0e"}
    datasets = []
    for key, g in deciles.groupby(["subset", "horizon"], sort=False):
        subset, horizon = key
        g = g.sort_values("decile")
        datasets.append(
            {
                "label": f"{subset}:{horizon}",
                "data": g["mean_fwd_logret"].tolist(),
                "borderColor": colors.get(horizon, "#444"),
                "backgroundColor": colors.get(horizon, "#444"),
                "borderWidth": 2,
                "pointRadius": 2,
                "fill": False,
                "borderDash": [6, 4] if subset == "chop" else [],
            }
        )

    s_rows = []
    for _, r in summary.iterrows():
        s_rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['subset']}</td>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['ic']:.6f}</td>"
            f"<td>{r['hit_rate']:.4f}</td>"
            f"<td>{r['decile_spread']:.6f}</td>"
            "</tr>"
        )

    e_rows = []
    for _, r in events.iterrows():
        e_rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['subset']}</td>"
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
  <title>Short-Term Reversal Diagnostics</title>
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
  <h2>Short-Term Reversal Diagnostics</h2>
  {meta_html}
  <div class="card">
    <table>
      <tr><th>Subset</th><th>Horizon</th><th>Rows</th><th>IC</th><th>Hit</th><th>Decile Spread</th></tr>
      {''.join(s_rows)}
    </table>
  </div>
  <div class="card">
    <table>
      <tr><th>Subset</th><th>Horizon</th><th>Side</th><th>Events</th><th>Mean Fwd LogRet</th></tr>
      {''.join(e_rows)}
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
    ap = argparse.ArgumentParser(description="Short-term reversal alpha diagnostics")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--lookback-bars", type=int, default=6, help="return accumulation bars for reversal signal")
    ap.add_argument("--mode", choices=["reversal", "momentum"], default="reversal")
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--trend-fast-bars", type=int, default=48)
    ap.add_argument("--trend-slow-bars", type=int, default=288)
    ap.add_argument("--trend-max", type=float, default=0.01, help="max abs(log(ema_fast/ema_slow)) for chop subset")
    ap.add_argument("--event-z-thresh", type=float, default=1.5)
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/reversal_summary.csv")
    ap.add_argument("--out-deciles-csv", default="artifacts/backtest/reversal_deciles.csv")
    ap.add_argument("--out-events-csv", default="artifacts/backtest/reversal_events.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/reversal_diagnostics.html")
    args = ap.parse_args()

    eth_path = Path(args.eth_csv)
    if not eth_path.exists():
        raise FileNotFoundError(f"missing file: {eth_path}")

    df = pd.read_csv(eth_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("input must have timestamp and close columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()
    if len(df) < 1000:
        raise RuntimeError(f"too few rows in window: {len(df)}")

    lb = int(args.lookback_bars)
    zw = int(args.z_window_bars)
    df["r"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_lb"] = df["r"].rolling(lb, min_periods=lb).sum()
    vol = df["r"].rolling(zw, min_periods=max(20, zw // 4)).std()
    base_sig = df["ret_lb"] / (vol * np.sqrt(lb) + 1e-12)
    df["signal"] = -base_sig if args.mode == "reversal" else base_sig

    fast = int(args.trend_fast_bars)
    slow = int(args.trend_slow_bars)
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["trend_strength"] = np.abs(np.log(df["ema_fast"] / df["ema_slow"]))
    df["is_chop"] = df["trend_strength"] <= float(args.trend_max)

    df = df.dropna(subset=["signal"]).copy()

    subsets = {
        "all": df,
        "chop": df[df["is_chop"]].copy(),
    }
    horizons = [("15m", 3), ("30m", 6), ("60m", 12)]
    zthr = float(args.event_z_thresh)

    sum_rows = []
    dec_rows = []
    evt_rows = []

    for subset_name, base in subsets.items():
        if len(base) < 500:
            continue
        for h_name, h_bars in horizons:
            d = base.copy()
            d["fwd"] = np.log(d["close"].shift(-h_bars) / d["close"])
            d = d.dropna(subset=["signal", "fwd"])
            if len(d) < 300:
                continue

            ic = float(d["signal"].corr(d["fwd"]))
            hit = float((np.sign(d["signal"]) == np.sign(d["fwd"])).mean())
            d["decile"] = pd.qcut(d["signal"], 10, labels=False, duplicates="drop")
            dec = d.groupby("decile", as_index=False)["fwd"].mean().rename(columns={"fwd": "mean_fwd_logret"})
            dec["horizon"] = h_name
            dec["subset"] = subset_name
            dec_rows.append(dec)

            q0 = float(dec.loc[dec["decile"] == dec["decile"].min(), "mean_fwd_logret"].iloc[0])
            q9 = float(dec.loc[dec["decile"] == dec["decile"].max(), "mean_fwd_logret"].iloc[0])
            sum_rows.append(
                {
                    "subset": subset_name,
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
                    {
                        "subset": subset_name,
                        "horizon": h_name,
                        "side": "signal_up",
                        "n": int(len(pos)),
                        "mean_fwd_logret": float(pos.mean()) if len(pos) else np.nan,
                    },
                    {
                        "subset": subset_name,
                        "horizon": h_name,
                        "side": "signal_down",
                        "n": int(len(neg)),
                        "mean_fwd_logret": float(neg.mean()) if len(neg) else np.nan,
                    },
                ]
            )

    if not sum_rows:
        raise RuntimeError("no valid results")

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
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"rows_used: {len(df)} | lookback_bars: {lb} | z_window_bars: {zw}",
        f"mode: {args.mode}",
        f"chop filter: abs(log(ema{fast}/ema{slow})) <= {args.trend_max}",
    ]
    make_html(out_html, summary, deciles, events, meta)

    print(f"wrote {out_sum}")
    print(f"wrote {out_dec}")
    print(f"wrote {out_evt}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
