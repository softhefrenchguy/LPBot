from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path} must have timestamp,close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")[["timestamp", "close"]]


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


def perf(log_r: pd.Series) -> dict:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * 12
    eq = np.exp(np.cumsum(x.to_numpy()))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    return {"ret": float(eq[-1] - 1.0), "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": float((eq / peak - 1.0).min())}


def write_diag_html(out_path: Path, summary: pd.DataFrame, deciles: pd.DataFrame, meta: list[str]) -> None:
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
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td style='text-align:center'>{r['horizon']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['ic']:.6f}</td>"
            f"<td>{r['hit_rate']:.4f}</td>"
            f"<td>{r['decile_spread']:.6f}</td>"
            "</tr>"
        )
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ETH/BTC Relative Strength Diagnostics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: right; }}
th {{ background: #f7f7f7; text-align: center; }}
</style></head><body>
<h2>ETH/BTC Relative Strength Diagnostics</h2>
{meta_html}
<div class="card"><table>
<tr><th>Horizon</th><th>Rows</th><th>IC</th><th>Hit</th><th>Decile Spread</th></tr>
{''.join(rows)}
</table></div>
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
      y: {{ title: {{ display: true, text: 'ETH mean forward log return' }} }}
    }}
  }}
}});
</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def write_bt_html(out_path: Path, df: pd.DataFrame, meta: list[str]) -> None:
    labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    eq = [round(float(x), 6) for x in df["eq"].tolist()]
    spot = [round(float(x), 6) for x in df["spot_eq"].tolist()]
    w = [round(float(x), 6) for x in df["weight"].tolist()]
    s = [round(float(x), 6) for x in df["signal"].fillna(0.0).tolist()]
    p = [round(float(x), 6) for x in df["eth_close"].tolist()]
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ETH/BTC Relative Strength Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
</style></head><body>
<h2>ETH/BTC Relative Strength Backtest</h2>
{meta_html}
<div class="card"><canvas id="eq" height="110"></canvas></div>
<div class="card"><canvas id="w" height="90"></canvas></div>
<div class="card"><canvas id="s" height="90"></canvas></div>
<div class="card"><canvas id="p" height="110"></canvas></div>
<script>
const labels = {json.dumps(labels)};
const eqData = {json.dumps(eq)};
const spotData = {json.dumps(spot)};
const wData = {json.dumps(w)};
const sData = {json.dumps(s)};
const pData = {json.dumps(p)};
const base = {{ animation:false, plugins:{{legend:{{display:true}}}}, scales:{{x:{{display:false}}}} }};
new Chart(document.getElementById('eq'), {{ type:'line', data:{{labels,datasets:[
{{label:'Strategy Eq',data:eqData,borderColor:'#1f77b4',borderWidth:2,pointRadius:0}},
{{label:'Spot Eq',data:spotData,borderColor:'#7f7f7f',borderWidth:2,pointRadius:0}}
]}}, options:base }});
new Chart(document.getElementById('w'), {{ type:'line', data:{{labels,datasets:[{{label:'Weight',data:wData,borderColor:'#ff7f0e',borderWidth:2,pointRadius:0}}]}}, options:{{...base, scales:{{...base.scales,y:{{min:-1,max:1}}}}}} }});
new Chart(document.getElementById('s'), {{ type:'line', data:{{labels,datasets:[{{label:'Signal',data:sData,borderColor:'#2ca02c',borderWidth:2,pointRadius:0}}]}}, options:base }});
new Chart(document.getElementById('p'), {{ type:'line', data:{{labels,datasets:[{{label:'ETH Price',data:pData,borderColor:'#9467bd',borderWidth:2,pointRadius:0}}]}}, options:base }});
</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="ETH/BTC relative strength diagnostics + backtest")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/backtest/BTCUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--lookback-bars", type=int, default=12)
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--signal-mode", choices=["momentum", "reversal"], default="momentum")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--position-scale", type=float, default=2.0)
    ap.add_argument("--deadband", type=float, default=0.25)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/relstr_summary.csv")
    ap.add_argument("--out-deciles-csv", default="artifacts/backtest/relstr_deciles.csv")
    ap.add_argument("--out-diag-html", default="artifacts/backtest/relstr_diagnostics.html")
    ap.add_argument("--out-bt-csv", default="artifacts/backtest/relstr_backtest.csv")
    ap.add_argument("--out-bt-html", default="artifacts/backtest/relstr_backtest.html")
    args = ap.parse_args()

    eth = load_price(Path(args.eth_csv)).rename(columns={"close": "eth_close"})
    btc = load_price(Path(args.btc_csv)).rename(columns={"close": "btc_close"})
    df = pd.merge(eth, btc, on="timestamp", how="inner").sort_values("timestamp")
    if len(df) < 1000:
        raise RuntimeError("not enough merged rows")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()

    lb = int(args.lookback_bars)
    zw = int(args.z_window_bars)
    df["eth_r"] = np.log(df["eth_close"] / df["eth_close"].shift(1))
    df["btc_r"] = np.log(df["btc_close"] / df["btc_close"].shift(1))
    df = df.dropna(subset=["eth_r", "btc_r"]).copy()
    rel = (df["eth_r"] - df["btc_r"]).rolling(lb, min_periods=lb).sum()
    sig = rolling_z(rel, zw)
    if args.signal_mode == "reversal":
        sig = -sig
    df["signal"] = sig
    df = df.dropna(subset=["signal"]).copy()

    # Diagnostics
    horizons = [("1h", 12), ("4h", 48), ("12h", 144)]
    sum_rows = []
    dec_rows = []
    for hname, hb in horizons:
        d = df.copy()
        d["fwd"] = np.log(d["eth_close"].shift(-hb) / d["eth_close"])
        d = d.dropna(subset=["fwd", "signal"])
        if len(d) < 500:
            continue
        ic = float(d["signal"].corr(d["fwd"]))
        hit = float((np.sign(d["signal"]) == np.sign(d["fwd"])).mean())
        d["decile"] = pd.qcut(d["signal"], 10, labels=False, duplicates="drop")
        dec = d.groupby("decile", as_index=False)["fwd"].mean().rename(columns={"fwd": "mean_fwd_logret"})
        dec["horizon"] = hname
        dec_rows.append(dec)
        q0 = float(dec.loc[dec["decile"] == dec["decile"].min(), "mean_fwd_logret"].iloc[0])
        q9 = float(dec.loc[dec["decile"] == dec["decile"].max(), "mean_fwd_logret"].iloc[0])
        sum_rows.append({"horizon": hname, "n": int(len(d)), "ic": ic, "hit_rate": hit, "decile_spread": q9 - q0})

    if not sum_rows:
        raise RuntimeError("no diagnostics rows")
    summary = pd.DataFrame(sum_rows)
    deciles = pd.concat(dec_rows, ignore_index=True)

    out_sum = Path(args.out_summary_csv)
    out_dec = Path(args.out_deciles_csv)
    out_diag = Path(args.out_diag_html)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)
    deciles.to_csv(out_dec, index=False)
    diag_meta = [
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"signal_mode: {args.signal_mode} | lookback_bars: {lb} | z_window_bars: {zw}",
    ]
    write_diag_html(out_diag, summary, deciles, diag_meta)

    # Backtest
    raw_w = (df["signal"] / float(args.position_scale)).clip(-float(args.max_weight), float(args.max_weight))
    raw_w = raw_w.where(raw_w.abs() >= float(args.deadband), 0.0)
    if not args.allow_short:
        raw_w = raw_w.clip(lower=0.0)
    df["weight_raw"] = raw_w.fillna(0.0)
    df["weight"] = apply_step_cap(df["weight_raw"], float(args.max_dw_per_bar))
    df["turnover"] = df["weight"].diff().abs().fillna(0.0)
    df["cost_r"] = -df["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["eth_r"] + df["cost_r"]
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["eth_r"].to_numpy()))
    df["eq"] = df["eq"] / float(df["eq"].iloc[0])
    df["spot_eq"] = df["spot_eq"] / float(df["spot_eq"].iloc[0])

    st = perf(df["strat_r"])
    sp = perf(df["eth_r"])
    avg_w = float(df["weight"].mean())
    tim = float((df["weight"] != 0).mean() * 100.0)
    turnover = float(df["turnover"].sum())

    out_bt_csv = Path(args.out_bt_csv)
    out_bt_html = Path(args.out_bt_html)
    df[["timestamp", "eth_close", "btc_close", "signal", "weight_raw", "weight", "eth_r", "strat_r", "eq", "spot_eq"]].to_csv(out_bt_csv, index=False)
    bt_meta = [
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"mode: {args.signal_mode} | allow_short: {args.allow_short}",
        f"ret: {st['ret']:.4f} | ann_vol: {st['ann_vol']:.4f} | sharpe: {st['sharpe']:.4f} | max_dd: {st['max_dd']:.4f}",
        f"avg_weight: {avg_w:.4f} | time_in_market: {tim:.2f}% | turnover: {turnover:.4f}",
        f"spot_ret_same_window: {sp['ret']:.4f} | spot_sharpe: {sp['sharpe']:.4f} | spot_max_dd: {sp['max_dd']:.4f}",
    ]
    write_bt_html(out_bt_html, df.tail(3000).copy(), bt_meta)

    print(f"wrote {out_sum}")
    print(f"wrote {out_dec}")
    print(f"wrote {out_diag}")
    print(f"wrote {out_bt_csv}")
    print(f"wrote {out_bt_html}")
    print(summary.to_string(index=False))
    print(
        f"backtest ret={st['ret']:.4f} ann_vol={st['ann_vol']:.4f} sharpe={st['sharpe']:.4f} "
        f"max_dd={st['max_dd']:.4f} avg_weight={avg_w:.4f} time_in_market={tim:.2f}% turnover={turnover:.4f}"
    )


if __name__ == "__main__":
    main()
