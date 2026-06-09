from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def apply_step_cap(target: pd.Series, max_dw: float) -> pd.Series:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, x in enumerate(target.fillna(0.0).to_numpy()):
        lo = prev - max_dw
        hi = prev + max_dw
        v = min(max(x, lo), hi)
        out[i] = v
        prev = v
    return pd.Series(out, index=target.index)


def write_html(df: pd.DataFrame, out_path: Path, title: str, meta: list[str]) -> None:
    labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    eq_data = [round(float(x), 6) for x in df["eq"].tolist()]
    spot_eq_data = [round(float(x), 6) for x in df["spot_eq"].tolist()]
    w_data = [round(float(x), 6) for x in df["weight"].tolist()]
    s_data = [round(float(x), 6) for x in df["signal"].fillna(0.0).tolist()]
    chop_data = [1 if bool(x) else 0 for x in df["is_chop"].tolist()]
    p_data = [round(float(x), 6) for x in df["close"].tolist()]
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])

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
  </style>
</head>
<body>
  <h2>{title}</h2>
  {meta_html}
  <div class="card"><canvas id="eq" height="110"></canvas></div>
  <div class="card"><canvas id="w" height="100"></canvas></div>
  <div class="card"><canvas id="s" height="100"></canvas></div>
  <div class="card"><canvas id="c" height="90"></canvas></div>
  <div class="card"><canvas id="p" height="110"></canvas></div>
  <script>
    const labels = {json.dumps(labels)};
    const eqData = {json.dumps(eq_data)};
    const spotEqData = {json.dumps(spot_eq_data)};
    const wData = {json.dumps(w_data)};
    const sData = {json.dumps(s_data)};
    const cData = {json.dumps(chop_data)};
    const pData = {json.dumps(p_data)};
    const baseOpt = {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }};
    new Chart(document.getElementById('eq'), {{
      type: 'line',
      data: {{ labels, datasets: [
        {{ label: 'Strategy Eq', data: eqData, borderColor: '#1f77b4', borderWidth: 2, pointRadius: 0 }},
        {{ label: 'Spot Eq', data: spotEqData, borderColor: '#7f7f7f', borderWidth: 2, pointRadius: 0 }}
      ] }},
      options: baseOpt
    }});
    new Chart(document.getElementById('w'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Weight', data: wData, borderColor: '#ff7f0e', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ ...baseOpt, scales: {{ ...baseOpt.scales, y: {{ min: -1, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('s'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Signal', data: sData, borderColor: '#2ca02c', borderWidth: 2, pointRadius: 0 }}] }},
      options: baseOpt
    }});
    new Chart(document.getElementById('c'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Is Chop', data: cData, borderColor: '#9467bd', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ ...baseOpt, scales: {{ ...baseOpt.scales, y: {{ min: 0, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('p'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Price', data: pData, borderColor: '#31a354', borderWidth: 2, pointRadius: 0 }}] }},
      options: baseOpt
    }});
  </script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Momentum in chop regime backtest")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--lookback-bars", type=int, default=6)
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--mode", choices=["momentum", "reversal"], default="momentum")
    ap.add_argument("--subset", choices=["all", "chop"], default="chop")
    ap.add_argument("--trend-fast-bars", type=int, default=48)
    ap.add_argument("--trend-slow-bars", type=int, default=288)
    ap.add_argument("--trend-max", type=float, default=0.01)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--position-scale", type=float, default=2.0)
    ap.add_argument("--deadband", type=float, default=0.25)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--out-csv", default="artifacts/backtest/momentum_chop_backtest.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/momentum_chop_backtest.html")
    args = ap.parse_args()

    eth = Path(args.eth_csv)
    if not eth.exists():
        raise FileNotFoundError(f"missing file: {eth}")

    df = pd.read_csv(eth)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("eth csv needs timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()
    if len(df) < 1000:
        raise RuntimeError(f"too few rows: {len(df)}")

    lb = int(args.lookback_bars)
    zw = int(args.z_window_bars)
    df["r"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_lb"] = df["r"].rolling(lb, min_periods=lb).sum()
    vol = df["r"].rolling(zw, min_periods=max(20, zw // 4)).std()
    base_sig = df["ret_lb"] / (vol * np.sqrt(lb) + 1e-12)
    df["signal"] = base_sig if args.mode == "momentum" else -base_sig

    fast = int(args.trend_fast_bars)
    slow = int(args.trend_slow_bars)
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["trend_strength"] = np.abs(np.log(df["ema_fast"] / df["ema_slow"]))
    df["is_chop"] = df["trend_strength"] <= float(args.trend_max)

    df = df.dropna(subset=["signal"]).copy()

    raw_w = (df["signal"] / float(args.position_scale)).clip(-float(args.max_weight), float(args.max_weight))
    raw_w = raw_w.where(raw_w.abs() >= float(args.deadband), 0.0)
    if args.subset == "chop":
        raw_w = raw_w.where(df["is_chop"], 0.0)
    if not args.allow_short:
        raw_w = raw_w.clip(lower=0.0)

    df["weight_raw"] = raw_w.fillna(0.0)
    df["weight"] = apply_step_cap(df["weight_raw"], max_dw=float(args.max_dw_per_bar))

    df["turnover"] = df["weight"].diff().abs().fillna(0.0)
    df["cost_r"] = -df["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["r"].fillna(0.0) + df["cost_r"]
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["r"].fillna(0.0).to_numpy()))
    df["eq"] = df["eq"] / float(df["eq"].iloc[0])
    df["spot_eq"] = df["spot_eq"] / float(df["spot_eq"].iloc[0])

    bars_per_year = 365 * 24 * 12
    ret = float(df["eq"].iloc[-1] - 1.0)
    ann_vol = float(np.std(df["strat_r"].to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(df["strat_r"].to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    peak = np.maximum.accumulate(df["eq"].to_numpy())
    mdd = float((df["eq"].to_numpy() / peak - 1.0).min())
    avg_w = float(df["weight"].mean())
    tim = float((df["weight"] != 0).mean() * 100.0)
    turnover = float(df["turnover"].sum())
    spot_ret = float(df["spot_eq"].iloc[-1] - 1.0)

    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    meta = [
        f"rows: {len(df)}",
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"mode: {args.mode} | subset: {args.subset} | allow_short: {args.allow_short}",
        f"ret: {ret:.4f} | ann_vol: {ann_vol:.4f} | sharpe: {sharpe:.4f} | max_dd: {mdd:.4f}",
        f"avg_weight: {avg_w:.4f} | time_in_market: {tim:.2f}% | turnover: {turnover:.4f}",
        f"spot_return_same_window: {spot_ret:.4f}",
    ]
    write_html(df.tail(3000).copy(), out_html, "Momentum/Chop Backtest", meta)

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(
        f"rows={len(df)} ret={ret:.4f} ann_vol={ann_vol:.4f} sharpe={sharpe:.4f} max_dd={mdd:.4f} "
        f"avg_weight={avg_w:.4f} time_in_market={tim:.2f}% turnover={turnover:.4f} spot_ret={spot_ret:.4f}"
    )


if __name__ == "__main__":
    main()
