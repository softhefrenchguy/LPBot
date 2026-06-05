from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must include timestamp and close columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def _compute_sigma_ann(r: pd.Series, bar_minutes: int, vol_window: int) -> pd.Series:
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    sigma = r.rolling(vol_window, min_periods=vol_window).std()
    return sigma * np.sqrt(bars_per_year)


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * 12
    eq = np.exp(np.cumsum(x.to_numpy()))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    return {"ret": float(eq[-1] - 1.0), "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": float((eq / peak - 1.0).min())}


def write_html(out_path: Path, df: pd.DataFrame, meta: list[str]) -> None:
    labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    # Rebase displayed curves to 1.0 at the first visible point for fair visual comparison.
    eq_br_s = pd.to_numeric(df["eq_breakout"], errors="coerce").astype(float)
    eq_combo_s = pd.to_numeric(df["eq_combo"], errors="coerce").astype(float)
    eq_spot_s = pd.to_numeric(df["eq_spot"], errors="coerce").astype(float)
    eq_br_base = float(eq_br_s.iloc[0]) if len(eq_br_s) else 1.0
    eq_combo_base = float(eq_combo_s.iloc[0]) if len(eq_combo_s) else 1.0
    eq_spot_base = float(eq_spot_s.iloc[0]) if len(eq_spot_s) else 1.0
    eq_br = [round(float(x / (eq_br_base + 1e-12)), 6) for x in eq_br_s.tolist()]
    eq_combo = [round(float(x / (eq_combo_base + 1e-12)), 6) for x in eq_combo_s.tolist()]
    eq_spot = [round(float(x / (eq_spot_base + 1e-12)), 6) for x in eq_spot_s.tolist()]
    w_br = [round(float(x), 6) for x in df["weight_breakout"].tolist()]
    w_combo = [round(float(x), 6) for x in df["weight_combo"].tolist()]
    filt = [round(float(x), 6) for x in df["funding_filter"].tolist()]
    fsig = [round(float(x), 6) for x in df["funding_signal"].fillna(0.0).tolist()]
    p = [round(float(x), 6) for x in df["close"].tolist()]
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Breakout + Funding Filter Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
</style></head><body>
<h2>Breakout + Funding Filter Backtest</h2>
{meta_html}
<div class="card"><canvas id="eq" height="120"></canvas></div>
<div class="card"><canvas id="w" height="110"></canvas></div>
<div class="card"><canvas id="f" height="100"></canvas></div>
<div class="card"><canvas id="p" height="110"></canvas></div>
<script>
const labels = {json.dumps(labels)};
const eqBr = {json.dumps(eq_br)};
const eqCo = {json.dumps(eq_combo)};
const eqSp = {json.dumps(eq_spot)};
const wBr = {json.dumps(w_br)};
const wCo = {json.dumps(w_combo)};
const filt = {json.dumps(filt)};
const fsig = {json.dumps(fsig)};
const pData = {json.dumps(p)};
const base = {{ animation:false, plugins:{{legend:{{display:true}}}}, scales:{{x:{{display:false}}}} }};
new Chart(document.getElementById('eq'), {{ type:'line', data:{{labels,datasets:[
{{label:'Breakout Eq',data:eqBr,borderColor:'#1f77b4',borderWidth:2,pointRadius:0}},
{{label:'Combo Eq',data:eqCo,borderColor:'#2ca02c',borderWidth:2,pointRadius:0}},
{{label:'Spot Eq',data:eqSp,borderColor:'#7f7f7f',borderWidth:2,pointRadius:0}}
]}}, options:base }});
new Chart(document.getElementById('w'), {{ type:'line', data:{{labels,datasets:[
{{label:'Breakout Weight',data:wBr,borderColor:'#1f77b4',borderWidth:2,pointRadius:0}},
{{label:'Combo Weight',data:wCo,borderColor:'#2ca02c',borderWidth:2,pointRadius:0}}
]}}, options:{{...base, scales:{{...base.scales, y:{{min:0,max:1}}}}}} }});
new Chart(document.getElementById('f'), {{ type:'line', data:{{labels,datasets:[
{{label:'Funding Filter',data:filt,borderColor:'#ff7f0e',borderWidth:2,pointRadius:0}},
{{label:'Funding Signal',data:fsig,borderColor:'#9467bd',borderWidth:2,pointRadius:0}}
]}}, options:base }});
new Chart(document.getElementById('p'), {{ type:'line', data:{{labels,datasets:[{{label:'Price',data:pData,borderColor:'#8c564b',borderWidth:2,pointRadius:0}}]}}, options:base }});
</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Combine breakout core with funding+basis filter")
    p.add_argument("--price-5m", default="data/backtest/ETHUSDC_5m.csv")
    p.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_90d.csv")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--bar-minutes", type=int, default=5)

    # breakout params
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--compress-window", type=int, default=96)
    p.add_argument("--compress-quantile", type=float, default=0.50)
    p.add_argument("--hold-bars", type=int, default=24)
    p.add_argument("--trend-ema", type=int, default=0, help="If >0, only allow breakout gate when close > EMA(trend_ema)")
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)

    # funding filter params
    p.add_argument("--z-window-bars", type=int, default=288)
    p.add_argument("--basis-weight", type=float, default=0.6)
    p.add_argument("--funding-weight", type=float, default=0.4)
    p.add_argument("--signal-ema-span", type=int, default=12)
    p.add_argument("--filt-on-thresh", type=float, default=0.75)
    p.add_argument("--filt-off-thresh", type=float, default=0.25)
    p.add_argument("--funding-min-hold-bars", type=int, default=0, help="Minimum bars to keep funding filter ON once triggered")

    # costs
    p.add_argument("--trade-cost-bps", type=float, default=5.0)

    p.add_argument("--out-csv", default="artifacts/backtest/combo_breakout_funding.csv")
    p.add_argument("--out-html", default="artifacts/backtest/combo_breakout_funding.html")
    args = p.parse_args()

    price = _load_price(Path(args.price_5m))
    end = price["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    price = price[price["timestamp"] >= start].copy()
    price = price.reset_index(drop=True)
    if len(price) < 500:
        raise RuntimeError(f"too few price rows in window: {len(price)}")

    close = price["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    # Breakout baseline (same logic as breakout_backtest, no trend filter)
    rolling_high = close.rolling(args.lookback, min_periods=args.lookback).max()
    rolling_low = close.rolling(args.lookback, min_periods=args.lookback).min()
    rolling_range = (rolling_high - rolling_low) / rolling_low
    range_pct = rolling_range.rolling(args.compress_window, min_periods=args.compress_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    compress = range_pct <= args.compress_quantile
    breakout = (close > rolling_high.shift(1)) & compress.shift(1)
    breakdown = (close < rolling_low.shift(1)) & compress.shift(1)

    sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=args.w_max)

    gate_on = np.zeros(len(price), dtype=bool)
    hold = 0
    for i in range(len(price)):
        if bool(breakdown.iat[i]):
            hold = 0
            gate_on[i] = False
            continue
        if bool(breakout.iat[i]):
            hold = max(int(args.hold_bars), 0)
        if hold > 0:
            gate_on[i] = True
            hold -= 1
        else:
            gate_on[i] = bool(breakout.iat[i])

    trend_ok = pd.Series(True, index=price.index)
    if int(args.trend_ema) > 0:
        ema = close.ewm(span=int(args.trend_ema), adjust=False).mean()
        trend_ok = (close > ema).fillna(False)
        gate_on = gate_on & trend_ok.to_numpy(dtype=bool)

    weight_breakout = weight_raw * gate_on.astype(float)

    # Funding/basis filter
    fpath = Path(args.funding_csv)
    if not fpath.exists():
        raise FileNotFoundError(f"missing funding csv: {fpath}")
    f = pd.read_csv(fpath)
    req = {"timestamp", "basis", "funding_rate"}
    if not req.issubset(set(f.columns)):
        raise ValueError(f"funding csv missing columns: {req - set(f.columns)}")
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    f["basis"] = pd.to_numeric(f["basis"], errors="coerce")
    f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
    f = f.dropna(subset=["timestamp", "basis", "funding_rate"]).sort_values("timestamp")

    base = pd.merge(price[["timestamp", "close"]], f[["timestamp", "basis", "funding_rate"]], on="timestamp", how="left").sort_values("timestamp")
    base["basis"] = base["basis"].ffill()
    base["funding_rate"] = base["funding_rate"].ffill()

    zw = int(args.z_window_bars)
    basis_z = rolling_z(base["basis"], zw)
    funding_z = rolling_z(base["funding_rate"], zw)
    sig = -(float(args.basis_weight) * basis_z + float(args.funding_weight) * funding_z)
    sig = sig.ewm(span=int(args.signal_ema_span), adjust=False).mean()

    fstate = np.zeros(len(base), dtype=float)
    st = 0.0
    on_th = float(args.filt_on_thresh)
    off_th = float(args.filt_off_thresh)
    min_hold = max(int(args.funding_min_hold_bars), 0)
    on_age = 0
    for i, s in enumerate(sig.fillna(0.0).to_numpy()):
        if st == 0.0 and s >= on_th:
            st = 1.0
            on_age = 1
        elif st == 1.0 and s <= off_th:
            if on_age >= min_hold:
                st = 0.0
                on_age = 0
            else:
                on_age += 1
        elif st == 1.0:
            on_age += 1
        fstate[i] = st

    # Combined weight
    weight_combo = weight_breakout * fstate

    # Returns and costs
    cost_k = float(args.trade_cost_bps) / 10000.0
    to_br = pd.Series(weight_breakout).diff().abs().fillna(0.0)
    to_co = pd.Series(weight_combo).diff().abs().fillna(0.0)
    ret_breakout = pd.Series(weight_breakout).shift(1).fillna(0.0) * r + (-to_br * cost_k)
    ret_combo = pd.Series(weight_combo).shift(1).fillna(0.0) * r + (-to_co * cost_k)
    ret_spot = r

    eq_breakout = np.exp(np.cumsum(ret_breakout.to_numpy()))
    eq_combo = np.exp(np.cumsum(ret_combo.to_numpy()))
    eq_spot = np.exp(np.cumsum(ret_spot.to_numpy()))
    eq_breakout = eq_breakout / float(eq_breakout[0])
    eq_combo = eq_combo / float(eq_combo[0])
    eq_spot = eq_spot / float(eq_spot[0])

    out = pd.DataFrame(
        {
            "timestamp": price["timestamp"],
            "close": close,
            "r": r,
            "breakout": breakout.astype(bool),
            "breakdown": breakdown.astype(bool),
            "gate_on": gate_on.astype(bool),
            "trend_ok": trend_ok.astype(bool),
            "weight_breakout": weight_breakout,
            "funding_signal": sig,
            "funding_filter": fstate,
            "weight_combo": weight_combo,
            "ret_breakout": ret_breakout,
            "ret_combo": ret_combo,
            "ret_spot": ret_spot,
            "eq_breakout": eq_breakout,
            "eq_combo": eq_combo,
            "eq_spot": eq_spot,
        }
    )

    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    pb = perf(out["ret_breakout"])
    pc = perf(out["ret_combo"])
    ps = perf(out["ret_spot"])
    tim_br = float((out["weight_breakout"] > 0).mean() * 100.0)
    tim_co = float((out["weight_combo"] > 0).mean() * 100.0)
    avg_br = float(out["weight_breakout"].mean())
    avg_co = float(out["weight_combo"].mean())
    to_br_sum = float(out["weight_breakout"].diff().abs().fillna(0.0).sum())
    to_co_sum = float(out["weight_combo"].diff().abs().fillna(0.0).sum())

    meta = [
        f"window: {out['timestamp'].iloc[0]} -> {out['timestamp'].iloc[-1]}",
        f"breakout params: lookback={args.lookback}, compress_window={args.compress_window}, compress_q={args.compress_quantile}, hold={args.hold_bars}, target_vol={args.target_vol}",
        f"breakout trend guard: trend_ema={args.trend_ema} (0=disabled)",
        f"funding filter: on>={args.filt_on_thresh}, off<={args.filt_off_thresh}, min_hold={args.funding_min_hold_bars}, signal=-(0.6*basis_z+0.4*funding_z) ema={args.signal_ema_span}",
        f"Breakout net: ret={pb['ret']:.4f}, ann_vol={pb['ann_vol']:.4f}, sharpe={pb['sharpe']:.4f}, max_dd={pb['max_dd']:.4f}, avg_w={avg_br:.4f}, tim={tim_br:.2f}%, turnover={to_br_sum:.2f}",
        f"Combo net: ret={pc['ret']:.4f}, ann_vol={pc['ann_vol']:.4f}, sharpe={pc['sharpe']:.4f}, max_dd={pc['max_dd']:.4f}, avg_w={avg_co:.4f}, tim={tim_co:.2f}%, turnover={to_co_sum:.2f}",
        f"Spot: ret={ps['ret']:.4f}, ann_vol={ps['ann_vol']:.4f}, sharpe={ps['sharpe']:.4f}, max_dd={ps['max_dd']:.4f}",
    ]
    write_html(out_html, out.tail(3000).copy(), meta)

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(f"breakout_ret={pb['ret']:.4f} combo_ret={pc['ret']:.4f} spot_ret={ps['ret']:.4f}")
    print(f"breakout_sharpe={pb['sharpe']:.4f} combo_sharpe={pc['sharpe']:.4f} spot_sharpe={ps['sharpe']:.4f}")
    print(f"breakout_mdd={pb['max_dd']:.4f} combo_mdd={pc['max_dd']:.4f} spot_mdd={ps['max_dd']:.4f}")


if __name__ == "__main__":
    main()
