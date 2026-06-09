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
    w_data = [round(float(x), 6) for x in df["weight"].tolist()]
    score_data = [round(float(x), 6) for x in df["score_n"].fillna(0.0).tolist()]
    price_data = [round(float(x), 6) for x in df["spot_close"].tolist()]
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
  <div class="card"><canvas id="p" height="110"></canvas></div>
  <script>
    const labels = {json.dumps(labels)};
    const eqData = {json.dumps(eq_data)};
    const wData = {json.dumps(w_data)};
    const sData = {json.dumps(score_data)};
    const pData = {json.dumps(price_data)};

    new Chart(document.getElementById('eq'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Equity', data: eqData, borderColor: '#1f77b4', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
    }});
    new Chart(document.getElementById('w'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Weight', data: wData, borderColor: '#ff7f0e', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }}, y: {{ min: -1, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('s'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Score Z', data: sData, borderColor: '#2ca02c', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
    }});
    new Chart(document.getElementById('p'), {{
      type: 'line',
      data: {{ labels, datasets: [{{ label: 'Spot Price', data: pData, borderColor: '#7f7f7f', borderWidth: 2, pointRadius: 0 }}] }},
      options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Quick backtest for perp positioning score")
    ap.add_argument("--input-csv", default="data/backtest/ETH_perp_features_5m_90d.csv")
    ap.add_argument("--out-csv", default="artifacts/backtest/perp_alpha_backtest.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/perp_alpha_backtest.html")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--score-ema-span", type=int, default=12)
    ap.add_argument("--invert-score", action="store_true")
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--position-scale", type=float, default=2.0)
    ap.add_argument("--deadband", type=float, default=0.25)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--allow-short", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["spot_close", "basis", "funding_rate", "oi_metric", "ret_1h", "oi_chg"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["oi_metric"].notna()].copy()
    if len(df) < 1000:
        raise RuntimeError(f"too few rows after oi filter: {len(df)}")

    w = int(args.z_window_bars)
    df["funding_z"] = rolling_z(df["funding_rate"].ffill(), w)
    df["basis_z"] = rolling_z(df["basis"], w)
    df["oi_chg_z"] = rolling_z(df["oi_chg"], w)
    df["ret_1h_z"] = rolling_z(df["ret_1h"], w)
    df["long_crowded"] = df["funding_z"] + df["basis_z"]
    df["score_raw"] = -(df["long_crowded"] * df["oi_chg_z"])
    df["score"] = df["score_raw"].ewm(span=int(args.score_ema_span), adjust=False).mean()
    if args.invert_score:
        df["score"] = -df["score"]

    score_sd = df["score"].rolling(w, min_periods=max(20, w // 4)).std()
    df["score_n"] = df["score"] / (score_sd + 1e-12)
    raw_weight = (df["score_n"] / float(args.position_scale)).clip(-float(args.max_weight), float(args.max_weight))
    raw_weight = raw_weight.where(raw_weight.abs() >= float(args.deadband), 0.0)
    if not args.allow_short:
        raw_weight = raw_weight.clip(lower=0.0)

    df["weight_raw"] = raw_weight.fillna(0.0)
    df["weight"] = apply_step_cap(df["weight_raw"], max_dw=float(args.max_dw_per_bar))

    df["r"] = np.log(df["spot_close"] / df["spot_close"].shift(1)).fillna(0.0)
    df["turnover"] = df["weight"].diff().abs().fillna(0.0)
    df["cost_r"] = -df["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["r"] + df["cost_r"]
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    bars_per_year = 365 * 24 * 12
    ret = float(df["eq"].iloc[-1] - 1.0)
    ann_vol = float(np.std(df["strat_r"].to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(df["strat_r"].to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    peak = np.maximum.accumulate(df["eq"].to_numpy())
    mdd = float((df["eq"].to_numpy() / peak - 1.0).min())
    avg_w = float(df["weight"].mean())
    tim = float((df["weight"] != 0).mean() * 100.0)
    turnover = float(df["turnover"].sum())

    html_out = Path(args.out_html)
    meta = [
        f"rows: {len(df)}",
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"invert_score: {args.invert_score} | allow_short: {args.allow_short}",
        f"total_return: {ret:.4f} | ann_vol: {ann_vol:.4f} | sharpe: {sharpe:.4f} | max_dd: {mdd:.4f}",
        f"avg_weight: {avg_w:.4f} | time_in_market: {tim:.2f}% | turnover: {turnover:.4f}",
    ]
    write_html(df.tail(3000).copy(), html_out, "Perp Positioning Backtest", meta)

    print(f"wrote {out}")
    print(f"wrote {html_out}")
    print(
        f"rows={len(df)} total_return={ret:.4f} ann_vol={ann_vol:.4f} sharpe={sharpe:.4f} "
        f"max_dd={mdd:.4f} avg_weight={avg_w:.4f} time_in_market={tim:.2f}% turnover={turnover:.4f}"
    )


if __name__ == "__main__":
    main()
