from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")
    if "close" not in df.columns:
        raise ValueError(f"{path} missing close column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")[["timestamp", "close"]]


def _make_html(out_path: Path, summary: pd.DataFrame, deciles: pd.DataFrame, events: pd.DataFrame, meta: list[str]) -> None:
    labels = list(range(10))
    colors = {"15m": "#1f77b4", "30m": "#2ca02c", "60m": "#ff7f0e"}
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

    sum_rows = []
    for _, r in summary.iterrows():
        sum_rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['ic']:.6f}</td>"
            f"<td>{r['hit_rate']:.4f}</td>"
            f"<td>{r['decile_spread']:.6f}</td>"
            "</tr>"
        )

    evt_rows = []
    for _, r in events.iterrows():
        evt_rows.append(
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
  <title>BTC Spillover Diagnostics</title>
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
  <h2>BTC Spillover Diagnostics (ETH target)</h2>
  {meta_html}
  <div class="card">
    <table>
      <tr><th>Horizon</th><th>Rows</th><th>IC</th><th>Hit</th><th>Decile Spread</th></tr>
      {''.join(sum_rows)}
    </table>
  </div>
  <div class="card">
    <table>
      <tr><th>Horizon</th><th>Side</th><th>Events</th><th>Mean Fwd LogRet</th></tr>
      {''.join(evt_rows)}
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
          x: {{ title: {{ display: true, text: 'BTC impulse decile (low -> high)' }} }},
          y: {{ title: {{ display: true, text: 'ETH mean forward log return' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="BTC -> ETH spillover diagnostics")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/backtest/BTCUSDC_5m.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/btc_spill_summary.csv")
    ap.add_argument("--out-deciles-csv", default="artifacts/backtest/btc_spill_deciles.csv")
    ap.add_argument("--out-events-csv", default="artifacts/backtest/btc_spill_events.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/btc_spill_diagnostics.html")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--impulse-bars", type=int, default=3, help="BTC return accumulation bars (5m bars)")
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--event-z-thresh", type=float, default=1.5)
    args = ap.parse_args()

    eth = _load_price(Path(args.eth_csv)).rename(columns={"close": "eth_close"})
    btc = _load_price(Path(args.btc_csv)).rename(columns={"close": "btc_close"})

    df = pd.merge(eth, btc, on="timestamp", how="inner").sort_values("timestamp")
    if len(df) < 500:
        raise RuntimeError("not enough merged rows")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=args.window_days)
    df = df[df["timestamp"] >= start].copy()

    df["eth_r"] = np.log(df["eth_close"] / df["eth_close"].shift(1))
    df["btc_r"] = np.log(df["btc_close"] / df["btc_close"].shift(1))
    df = df.dropna(subset=["eth_r", "btc_r"]).copy()

    k = int(args.impulse_bars)
    zw = int(args.z_window_bars)
    df["btc_impulse"] = df["btc_r"].rolling(k, min_periods=k).sum()
    mu = df["btc_impulse"].rolling(zw, min_periods=max(20, zw // 4)).mean()
    sd = df["btc_impulse"].rolling(zw, min_periods=max(20, zw // 4)).std()
    df["signal"] = (df["btc_impulse"] - mu) / (sd + 1e-12)
    df = df.dropna(subset=["signal"]).copy()

    horizons = [("15m", 3), ("30m", 6), ("60m", 12)]
    sum_rows = []
    dec_rows = []
    evt_rows = []
    zthr = float(args.event_z_thresh)

    for h_name, h_bars in horizons:
        d = df.copy()
        d["fwd"] = np.log(d["eth_close"].shift(-h_bars) / d["eth_close"])
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
                {
                    "horizon": h_name,
                    "side": "btc_impulse_up",
                    "n": int(len(pos)),
                    "mean_fwd_logret": float(pos.mean()) if len(pos) else np.nan,
                },
                {
                    "horizon": h_name,
                    "side": "btc_impulse_down",
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
        f"rows_used: {len(df)} | impulse_bars: {k} | z_window_bars: {zw} | event_z_th: {zthr}",
    ]
    _make_html(out_html, summary, deciles, events, meta)

    print(f"wrote {out_sum}")
    print(f"wrote {out_dec}")
    print(f"wrote {out_evt}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
