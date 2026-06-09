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


def step_cap(target: pd.Series, max_dw: float) -> pd.Series:
    out = np.zeros(len(target), dtype=float)
    prev = 1.0
    for i, x in enumerate(target.fillna(0.0).to_numpy()):
        lo = prev - max_dw
        hi = prev + max_dw
        v = min(max(float(x), lo), hi)
        out[i] = v
        prev = v
    return pd.Series(out, index=target.index)


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


def write_html(out_path: Path, df: pd.DataFrame, meta: list[str]) -> None:
    labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    eq = [round(float(x), 6) for x in df["eq"].tolist()]
    spot = [round(float(x), 6) for x in df["spot_eq"].tolist()]
    w = [round(float(x), 6) for x in df["weight"].tolist()]
    rvz = [round(float(x), 6) for x in df["rv_z"].fillna(0.0).tolist()]
    vovz = [round(float(x), 6) for x in df["vov_z"].fillna(0.0).tolist()]
    p = [round(float(x), 6) for x in df["close"].tolist()]
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Vol-of-Vol Filter Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
</style></head><body>
<h2>Vol-of-Vol Filter Backtest</h2>
{meta_html}
<div class="card"><canvas id="eq" height="110"></canvas></div>
<div class="card"><canvas id="w" height="90"></canvas></div>
<div class="card"><canvas id="z" height="110"></canvas></div>
<div class="card"><canvas id="p" height="110"></canvas></div>
<script>
const labels = {json.dumps(labels)};
const eqData = {json.dumps(eq)};
const spotData = {json.dumps(spot)};
const wData = {json.dumps(w)};
const rvzData = {json.dumps(rvz)};
const vovzData = {json.dumps(vovz)};
const pData = {json.dumps(p)};
const base = {{ animation:false, plugins:{{legend:{{display:true}}}}, scales:{{x:{{display:false}}}} }};
new Chart(document.getElementById('eq'), {{ type:'line', data:{{labels,datasets:[
{{label:'Strategy Eq',data:eqData,borderColor:'#1f77b4',borderWidth:2,pointRadius:0}},
{{label:'Spot Eq',data:spotData,borderColor:'#7f7f7f',borderWidth:2,pointRadius:0}}
]}}, options:base }});
new Chart(document.getElementById('w'), {{ type:'line', data:{{labels,datasets:[{{label:'Weight',data:wData,borderColor:'#ff7f0e',borderWidth:2,pointRadius:0}}]}}, options:{{...base, scales:{{...base.scales, y:{{min:0,max:1}}}}}} }});
new Chart(document.getElementById('z'), {{ type:'line', data:{{labels,datasets:[
{{label:'rv_z',data:rvzData,borderColor:'#2ca02c',borderWidth:2,pointRadius:0}},
{{label:'vov_z',data:vovzData,borderColor:'#9467bd',borderWidth:2,pointRadius:0}}
]}}, options:base }});
new Chart(document.getElementById('p'), {{ type:'line', data:{{labels,datasets:[{{label:'Price',data:pData,borderColor:'#8c564b',borderWidth:2,pointRadius:0}}]}}, options:base }});
</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Vol-of-vol de-risk/re-risk filter backtest")
    ap.add_argument("--eth-csv", default="data/backtest/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--rv-window-bars", type=int, default=72)
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--rv-off-z", type=float, default=1.0)
    ap.add_argument("--rv-on-z", type=float, default=0.2)
    ap.add_argument("--vov-off-z", type=float, default=1.0)
    ap.add_argument("--vov-on-z", type=float, default=0.2)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.20)
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/vov_summary.csv")
    ap.add_argument("--out-equity-csv", default="artifacts/backtest/vov_equity.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/vov_backtest.html")
    args = ap.parse_args()

    p = Path(args.eth_csv)
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")
    df = pd.read_csv(p)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("input needs timestamp, close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()
    if len(df) < 2000:
        raise RuntimeError(f"too few rows in window: {len(df)}")

    rvw = int(args.rv_window_bars)
    zw = int(args.z_window_bars)
    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["rv"] = df["r"].rolling(rvw, min_periods=max(20, rvw // 4)).std()
    df["vov"] = df["rv"].diff().abs()
    df["rv_z"] = rolling_z(df["rv"], zw)
    df["vov_z"] = rolling_z(df["vov"], zw)
    df = df.dropna(subset=["rv_z", "vov_z"]).copy()

    state = 1  # 1 risk-on, 0 risk-off
    t = np.zeros(len(df), dtype=float)
    for i, (rvz, vovz) in enumerate(zip(df["rv_z"].to_numpy(), df["vov_z"].to_numpy())):
        if state == 1 and (rvz > float(args.rv_off_z) or vovz > float(args.vov_off_z)):
            state = 0
        elif state == 0 and (rvz < float(args.rv_on_z) and vovz < float(args.vov_on_z)):
            state = 1
        t[i] = float(state)
    df["target_weight"] = t
    df["weight"] = step_cap(df["target_weight"], max_dw=float(args.max_dw_per_bar))
    df["turnover"] = df["weight"].diff().abs().fillna(0.0)
    df["cost_r"] = -df["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["r"] + df["cost_r"]
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["r"].to_numpy()))
    df["eq"] = df["eq"] / float(df["eq"].iloc[0])
    df["spot_eq"] = df["spot_eq"] / float(df["spot_eq"].iloc[0])

    st = perf(df["strat_r"])
    sp = perf(df["r"])
    tim = float((df["weight"] > 0).mean() * 100.0)
    avg_w = float(df["weight"].mean())
    turnover = float(df["turnover"].sum())

    summary = pd.DataFrame(
        [
            {
                "segment": "strategy",
                "rows": int(len(df)),
                "ret": st["ret"],
                "ann_vol": st["ann_vol"],
                "sharpe": st["sharpe"],
                "max_dd": st["max_dd"],
                "avg_weight": avg_w,
                "time_in_market_pct": tim,
                "turnover": turnover,
            },
            {
                "segment": "spot",
                "rows": int(len(df)),
                "ret": sp["ret"],
                "ann_vol": sp["ann_vol"],
                "sharpe": sp["sharpe"],
                "max_dd": sp["max_dd"],
                "avg_weight": 1.0,
                "time_in_market_pct": 100.0,
                "turnover": 0.0,
            },
        ]
    )

    out_sum = Path(args.out_summary_csv)
    out_eq = Path(args.out_equity_csv)
    out_html = Path(args.out_html)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)
    df[
        [
            "timestamp",
            "close",
            "r",
            "rv",
            "vov",
            "rv_z",
            "vov_z",
            "target_weight",
            "weight",
            "turnover",
            "strat_r",
            "eq",
            "spot_eq",
        ]
    ].to_csv(out_eq, index=False)

    meta = [
        f"window: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
        f"rv_window: {rvw} bars | z_window: {zw} bars",
        f"off thresholds: rv_z>{args.rv_off_z}, vov_z>{args.vov_off_z}",
        f"on thresholds: rv_z<{args.rv_on_z}, vov_z<{args.vov_on_z}",
        f"ret: {st['ret']:.4f} | ann_vol: {st['ann_vol']:.4f} | sharpe: {st['sharpe']:.4f} | max_dd: {st['max_dd']:.4f}",
        f"avg_weight: {avg_w:.4f} | time_in_market: {tim:.2f}% | turnover: {turnover:.2f}",
        f"spot_ret_same_window: {sp['ret']:.4f} | spot_sharpe: {sp['sharpe']:.4f} | spot_max_dd: {sp['max_dd']:.4f}",
    ]
    write_html(out_html, df.tail(3000).copy(), meta)

    print(f"wrote {out_sum}")
    print(f"wrote {out_eq}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
