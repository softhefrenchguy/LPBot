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
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _compute_sigma_ann(r: pd.Series, bar_minutes: int, vol_window: int) -> pd.Series:
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    sigma = r.rolling(vol_window, min_periods=vol_window).std()
    return sigma * np.sqrt(bars_per_year)


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def _regime_state(
    close: pd.Series,
    ts: pd.Series,
    bar_minutes: int,
    timeframe: str,
    on_ema: int,
    off_ema: int,
    slope_ema: int,
    n_on: int,
    n_off: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if on_ema <= 0 or off_ema <= 0:
        on = np.ones(len(close), dtype=bool)
        off = np.zeros(len(close), dtype=bool)
        state = np.ones(len(close), dtype=bool)
        return state, on, off

    tf_map = {"same": f"{bar_minutes}min", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = tf_map[timeframe]
    tmp = pd.DataFrame({"close": close.to_numpy()}, index=ts)
    n_on = max(int(n_on), 1)
    n_off = max(int(n_off), 1)

    def _run_state(on_c: np.ndarray, off_c: np.ndarray) -> np.ndarray:
        st = False
        on_count = 0
        off_count = 0
        out = np.zeros(len(on_c), dtype=bool)
        for i in range(len(on_c)):
            if not st:
                on_count = on_count + 1 if on_c[i] else 0
                if on_count >= n_on:
                    st = True
                    off_count = 0
            else:
                off_count = off_count + 1 if off_c[i] else 0
                if off_count >= n_off:
                    st = False
                    on_count = 0
            out[i] = st
        return out

    if tf == f"{bar_minutes}min":
        tclose = tmp["close"]
        on_level = tclose.ewm(span=on_ema, adjust=False).mean()
        off_level = tclose.ewm(span=off_ema, adjust=False).mean()
        slope_base = tclose.ewm(span=slope_ema, adjust=False).mean()
        slope = slope_base.diff()
        on_cond = ((tclose > on_level) & (slope > 0)).fillna(False).to_numpy(dtype=bool)
        off_cond = ((tclose < off_level) & (slope < 0)).fillna(False).to_numpy(dtype=bool)
        state = _run_state(on_cond, off_cond)
        return state, on_cond, off_cond

    # Use completed HTF bars only; run state machine on HTF bars, then map back to base bars.
    htf_close = tmp["close"].resample(tf).last().shift(1).ffill()
    on_level = htf_close.ewm(span=on_ema, adjust=False).mean()
    off_level = htf_close.ewm(span=off_ema, adjust=False).mean()
    slope_base = htf_close.ewm(span=slope_ema, adjust=False).mean()
    slope = slope_base.diff()
    on_cond_htf = ((htf_close > on_level) & (slope > 0)).fillna(False).to_numpy(dtype=bool)
    off_cond_htf = ((htf_close < off_level) & (slope < 0)).fillna(False).to_numpy(dtype=bool)
    state_htf = _run_state(on_cond_htf, off_cond_htf)

    htf_idx = htf_close.index
    state = pd.Series(state_htf.astype(int), index=htf_idx).reindex(tmp.index, method="ffill").fillna(0).to_numpy(dtype=bool)
    on_cond = pd.Series(on_cond_htf.astype(int), index=htf_idx).reindex(tmp.index, method="ffill").fillna(0).to_numpy(dtype=bool)
    off_cond = pd.Series(off_cond_htf.astype(int), index=htf_idx).reindex(tmp.index, method="ffill").fillna(0).to_numpy(dtype=bool)
    return state, on_cond, off_cond


def _perf(log_r: pd.Series, bar_minutes: int) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
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


def _write_html(out_path: Path, df: pd.DataFrame, meta: list[str]) -> None:
    labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    cols = {
        "eq_combined": "#2ca02c",
        "eq_on": "#1f77b4",
        "eq_off": "#ff7f0e",
        "eq_spot": "#7f7f7f",
        "w_combined": "#2ca02c",
        "w_on": "#1f77b4",
        "w_off": "#ff7f0e",
        "regime_on": "#9467bd",
        "regime_on_cond": "#bcbd22",
        "regime_off_cond": "#e377c2",
        "gate_breakout": "#d62728",
        "funding_filter": "#17becf",
        "funding_signal": "#8c564b",
        "close": "#111111",
    }

    data = {}
    for k in cols:
        vals = pd.to_numeric(df[k], errors="coerce").fillna(0.0).tolist()
        data[k] = [round(float(x), 6) for x in vals]

    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Two-Sleeve Regime Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
</style></head><body>
<h2>Two-Sleeve Regime Backtest</h2>
{meta_html}
<div class="card"><canvas id="eq" height="120"></canvas></div>
<div class="card"><canvas id="w" height="110"></canvas></div>
<div class="card"><canvas id="state" height="100"></canvas></div>
<div class="card"><canvas id="sig" height="100"></canvas></div>
<div class="card"><canvas id="px" height="110"></canvas></div>
<script>
const labels = {json.dumps(labels)};
const d = {json.dumps(data)};
const base = {{ animation:false, plugins:{{legend:{{display:true}}}}, scales:{{x:{{display:false}}}} }};
new Chart(document.getElementById('eq'), {{ type:'line', data:{{labels,datasets:[
{{label:'Combined Eq',data:d.eq_combined,borderColor:'{cols["eq_combined"]}',borderWidth:2,pointRadius:0}},
{{label:'Risk-On Eq',data:d.eq_on,borderColor:'{cols["eq_on"]}',borderWidth:2,pointRadius:0}},
{{label:'Risk-Off Eq',data:d.eq_off,borderColor:'{cols["eq_off"]}',borderWidth:2,pointRadius:0}},
{{label:'Spot Eq',data:d.eq_spot,borderColor:'{cols["eq_spot"]}',borderWidth:2,pointRadius:0}}
]}} , options:base }});
new Chart(document.getElementById('w'), {{ type:'line', data:{{labels,datasets:[
{{label:'Combined Weight',data:d.w_combined,borderColor:'{cols["w_combined"]}',borderWidth:2,pointRadius:0}},
{{label:'Risk-On Weight',data:d.w_on,borderColor:'{cols["w_on"]}',borderWidth:2,pointRadius:0}},
{{label:'Risk-Off Weight',data:d.w_off,borderColor:'{cols["w_off"]}',borderWidth:2,pointRadius:0}}
]}} , options:{{...base, scales:{{...base.scales, y:{{min:0,max:1}}}}}} }});
new Chart(document.getElementById('state'), {{ type:'line', data:{{labels,datasets:[
{{label:'Regime On',data:d.regime_on,borderColor:'{cols["regime_on"]}',borderWidth:2,pointRadius:0}},
{{label:'Regime On Cond',data:d.regime_on_cond,borderColor:'{cols["regime_on_cond"]}',borderWidth:2,pointRadius:0}},
{{label:'Regime Off Cond',data:d.regime_off_cond,borderColor:'{cols["regime_off_cond"]}',borderWidth:2,pointRadius:0}},
{{label:'Breakout Gate',data:d.gate_breakout,borderColor:'{cols["gate_breakout"]}',borderWidth:2,pointRadius:0}},
{{label:'Funding Filter',data:d.funding_filter,borderColor:'{cols["funding_filter"]}',borderWidth:2,pointRadius:0}}
]}} , options:{{...base, scales:{{...base.scales, y:{{min:0,max:1}}}}}} }});
new Chart(document.getElementById('sig'), {{ type:'line', data:{{labels,datasets:[
{{label:'Funding Signal',data:d.funding_signal,borderColor:'{cols["funding_signal"]}',borderWidth:2,pointRadius:0}}
]}} , options:base }});
new Chart(document.getElementById('px'), {{ type:'line', data:{{labels,datasets:[
{{label:'Price',data:d.close,borderColor:'{cols["close"]}',borderWidth:2,pointRadius:0}}
]}} , options:base }});
</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Two-sleeve regime backtest: risk-on trend sleeve + risk-off breakout/funding sleeve")
    p.add_argument("--price-5m", default="data/ETHUSDC_5m.csv")
    p.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--bar-minutes", type=int, default=5)

    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)

    p.add_argument("--regime-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--regime-ema", type=int, default=200, help="Legacy fallback EMA if on/off EMA not set")
    p.add_argument("--regime-on-ema", type=int, default=100, help="Fast EMA level used to turn bull regime ON")
    p.add_argument("--regime-off-ema", type=int, default=200, help="Slow EMA level used to turn bull regime OFF")
    p.add_argument("--regime-slope-ema", type=int, default=50, help="EMA used for slope sign in regime conditions")
    p.add_argument("--regime-n-on", type=int, default=3, help="Persistence bars required to switch regime ON")
    p.add_argument("--regime-n-off", type=int, default=3, help="Persistence bars required to switch regime OFF")
    p.add_argument("--min-weight-on", type=float, default=0.35, help="Minimum risk-on sleeve weight while bull regime is ON")
    p.add_argument("--riskoff-scale", type=float, default=1.0)

    p.add_argument("--lookback", type=int, default=144)
    p.add_argument("--compress-window", type=int, default=96)
    p.add_argument("--compress-quantile", type=float, default=0.60)
    p.add_argument("--hold-bars", type=int, default=36)
    p.add_argument("--filt-on-thresh", type=float, default=0.75)
    p.add_argument("--filt-off-thresh", type=float, default=0.25)
    p.add_argument("--funding-min-hold-bars", type=int, default=6)
    p.add_argument("--signal-ema-span", type=int, default=12)
    p.add_argument("--z-window-bars", type=int, default=288)
    p.add_argument("--basis-weight", type=float, default=0.6)
    p.add_argument("--funding-weight", type=float, default=0.4)

    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--out-csv", default="artifacts/paper/two_sleeve_backtest.csv")
    p.add_argument("--out-html", default="artifacts/paper/two_sleeve_backtest.html")
    args = p.parse_args()

    price = _load_price(Path(args.price_5m))
    end = price["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    price = price[(price["timestamp"] >= start) & (price["timestamp"] <= end)].reset_index(drop=True)
    if len(price) < 500:
        raise RuntimeError(f"too few price rows in window: {len(price)}")

    close = price["close"]
    ts = price["timestamp"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=args.w_max)

    # Sleeve A: risk-on trend sleeve (faster ON, slower OFF, with persistence)
    on_ema = int(args.regime_on_ema) if int(args.regime_on_ema) > 0 else int(args.regime_ema)
    off_ema = int(args.regime_off_ema) if int(args.regime_off_ema) > 0 else int(args.regime_ema)
    slope_ema = int(args.regime_slope_ema) if int(args.regime_slope_ema) > 0 else max(2, on_ema // 2)
    regime_on, regime_on_cond, regime_off_cond = _regime_state(
        close=close,
        ts=ts,
        bar_minutes=args.bar_minutes,
        timeframe=args.regime_timeframe,
        on_ema=on_ema,
        off_ema=off_ema,
        slope_ema=slope_ema,
        n_on=args.regime_n_on,
        n_off=args.regime_n_off,
    )
    w_on = np.where(regime_on, np.maximum(weight_raw.to_numpy(), float(args.min_weight_on)), 0.0)
    w_on = pd.Series(w_on).clip(lower=0.0, upper=args.w_max).to_numpy()

    # Sleeve B: risk-off breakout + funding filter
    rolling_high = close.rolling(args.lookback, min_periods=args.lookback).max()
    rolling_low = close.rolling(args.lookback, min_periods=args.lookback).min()
    rolling_range = (rolling_high - rolling_low) / rolling_low
    range_pct = rolling_range.rolling(args.compress_window, min_periods=args.compress_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    compress = range_pct <= args.compress_quantile
    breakout = (close > rolling_high.shift(1)) & compress.shift(1)
    breakdown = (close < rolling_low.shift(1)) & compress.shift(1)
    gate_breakout = np.zeros(len(price), dtype=bool)
    hold = 0
    for i in range(len(price)):
        if bool(breakdown.iat[i]):
            hold = 0
            gate_breakout[i] = False
            continue
        if bool(breakout.iat[i]):
            hold = max(int(args.hold_bars), 0)
        if hold > 0:
            gate_breakout[i] = True
            hold -= 1
        else:
            gate_breakout[i] = bool(breakout.iat[i])
    w_breakout = weight_raw * gate_breakout.astype(float)

    f = pd.read_csv(args.funding_csv)
    need = {"timestamp", "basis", "funding_rate"}
    if not need.issubset(set(f.columns)):
        raise ValueError(f"funding csv missing columns: {need - set(f.columns)}")
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    f["basis"] = pd.to_numeric(f["basis"], errors="coerce")
    f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
    f = f.dropna(subset=["timestamp", "basis", "funding_rate"]).sort_values("timestamp")

    base = pd.merge(price[["timestamp", "close"]], f[["timestamp", "basis", "funding_rate"]], on="timestamp", how="left").sort_values("timestamp")
    base["basis"] = base["basis"].ffill()
    base["funding_rate"] = base["funding_rate"].ffill()
    base = base.dropna(subset=["basis", "funding_rate"]).reset_index(drop=True)
    if len(base) != len(price):
        # align to price when there are leading NaNs from funding file coverage
        keep_ts = set(base["timestamp"].tolist())
        mask = price["timestamp"].isin(keep_ts)
        price = price[mask].reset_index(drop=True)
        close = price["close"]
        ts = price["timestamp"]
        r = np.log(close / close.shift(1)).fillna(0.0)
        sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
        weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=args.w_max)
        regime_on, regime_on_cond, regime_off_cond = _regime_state(
            close=close,
            ts=ts,
            bar_minutes=args.bar_minutes,
            timeframe=args.regime_timeframe,
            on_ema=on_ema,
            off_ema=off_ema,
            slope_ema=slope_ema,
            n_on=args.regime_n_on,
            n_off=args.regime_n_off,
        )
        w_on = np.where(regime_on, np.maximum(weight_raw.to_numpy(), float(args.min_weight_on)), 0.0)
        w_on = pd.Series(w_on).clip(lower=0.0, upper=args.w_max).to_numpy()
        rolling_high = close.rolling(args.lookback, min_periods=args.lookback).max()
        rolling_low = close.rolling(args.lookback, min_periods=args.lookback).min()
        rolling_range = (rolling_high - rolling_low) / rolling_low
        range_pct = rolling_range.rolling(args.compress_window, min_periods=args.compress_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        compress = range_pct <= args.compress_quantile
        breakout = (close > rolling_high.shift(1)) & compress.shift(1)
        breakdown = (close < rolling_low.shift(1)) & compress.shift(1)
        gate_breakout = np.zeros(len(price), dtype=bool)
        hold = 0
        for i in range(len(price)):
            if bool(breakdown.iat[i]):
                hold = 0
                gate_breakout[i] = False
                continue
            if bool(breakout.iat[i]):
                hold = max(int(args.hold_bars), 0)
            if hold > 0:
                gate_breakout[i] = True
                hold -= 1
            else:
                gate_breakout[i] = bool(breakout.iat[i])
        w_breakout = weight_raw * gate_breakout.astype(float)
        base = pd.merge(price[["timestamp", "close"]], f[["timestamp", "basis", "funding_rate"]], on="timestamp", how="left").sort_values("timestamp")
        base["basis"] = base["basis"].ffill()
        base["funding_rate"] = base["funding_rate"].ffill()
        base = base.dropna(subset=["basis", "funding_rate"]).reset_index(drop=True)

    basis_z = _rolling_z(base["basis"], args.z_window_bars)
    funding_z = _rolling_z(base["funding_rate"], args.z_window_bars)
    funding_signal = -(args.basis_weight * basis_z + args.funding_weight * funding_z)
    funding_signal = funding_signal.ewm(span=args.signal_ema_span, adjust=False).mean().fillna(0.0)

    funding_filter = np.zeros(len(price), dtype=float)
    st = 0.0
    on_age = 0
    for i, s in enumerate(funding_signal.to_numpy()):
        if st == 0.0 and s >= args.filt_on_thresh:
            st = 1.0
            on_age = 1
        elif st == 1.0 and s <= args.filt_off_thresh:
            if on_age >= int(args.funding_min_hold_bars):
                st = 0.0
                on_age = 0
            else:
                on_age += 1
        elif st == 1.0:
            on_age += 1
        funding_filter[i] = st

    w_off = w_breakout * funding_filter

    # Regime switch
    w_combined = np.where(regime_on, w_on, w_off * float(args.riskoff_scale))
    w_combined = pd.Series(w_combined).clip(lower=0.0, upper=args.w_max).to_numpy()

    k = float(args.trade_cost_bps) / 10000.0
    to_on = pd.Series(w_on).diff().abs().fillna(0.0)
    to_off = pd.Series(w_off).diff().abs().fillna(0.0)
    to_comb = pd.Series(w_combined).diff().abs().fillna(0.0)

    ret_on = pd.Series(w_on).shift(1).fillna(0.0) * r + (-to_on * k)
    ret_off = pd.Series(w_off).shift(1).fillna(0.0) * r + (-to_off * k)
    ret_comb = pd.Series(w_combined).shift(1).fillna(0.0) * r + (-to_comb * k)
    ret_spot = r

    eq_on = np.exp(np.cumsum(ret_on.to_numpy()))
    eq_off = np.exp(np.cumsum(ret_off.to_numpy()))
    eq_comb = np.exp(np.cumsum(ret_comb.to_numpy()))
    eq_spot = np.exp(np.cumsum(ret_spot.to_numpy()))
    eq_on /= float(eq_on[0] + 1e-12)
    eq_off /= float(eq_off[0] + 1e-12)
    eq_comb /= float(eq_comb[0] + 1e-12)
    eq_spot /= float(eq_spot[0] + 1e-12)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "r": r,
            "regime_on": regime_on.astype(int),
            "regime_on_cond": regime_on_cond.astype(int),
            "regime_off_cond": regime_off_cond.astype(int),
            "breakout": breakout.astype(int),
            "breakdown": breakdown.astype(int),
            "gate_breakout": gate_breakout.astype(int),
            "funding_signal": funding_signal,
            "funding_filter": funding_filter,
            "w_on": w_on,
            "w_off": w_off,
            "w_combined": w_combined,
            "ret_on": ret_on,
            "ret_off": ret_off,
            "ret_combined": ret_comb,
            "ret_spot": ret_spot,
            "eq_on": eq_on,
            "eq_off": eq_off,
            "eq_combined": eq_comb,
            "eq_spot": eq_spot,
        }
    )

    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    po = _perf(ret_on, args.bar_minutes)
    pf = _perf(ret_off, args.bar_minutes)
    pc = _perf(ret_comb, args.bar_minutes)
    ps = _perf(ret_spot, args.bar_minutes)
    meta = [
        f"window: {out['timestamp'].iloc[0]} -> {out['timestamp'].iloc[-1]} | rows={len(out)}",
        f"regime: tf={args.regime_timeframe}, on_ema={on_ema}, off_ema={off_ema}, slope_ema={slope_ema}, n_on={args.regime_n_on}, n_off={args.regime_n_off}, min_w_on={args.min_weight_on} | riskoff_scale={args.riskoff_scale}",
        f"risk-on sleeve: ret={po['ret']:.4f}, sharpe={po['sharpe']:.4f}, mdd={po['max_dd']:.4f}, avg_w={out['w_on'].mean():.4f}, tim={(out['w_on']>0).mean()*100:.2f}%",
        f"risk-off sleeve: ret={pf['ret']:.4f}, sharpe={pf['sharpe']:.4f}, mdd={pf['max_dd']:.4f}, avg_w={out['w_off'].mean():.4f}, tim={(out['w_off']>0).mean()*100:.2f}%",
        f"combined: ret={pc['ret']:.4f}, sharpe={pc['sharpe']:.4f}, mdd={pc['max_dd']:.4f}, avg_w={out['w_combined'].mean():.4f}, tim={(out['w_combined']>0).mean()*100:.2f}%",
        f"spot: ret={ps['ret']:.4f}, sharpe={ps['sharpe']:.4f}, mdd={ps['max_dd']:.4f}",
    ]
    _write_html(out_html, out.tail(3000).copy(), meta)

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(f"combined_ret={pc['ret']:.4f} combined_sharpe={pc['sharpe']:.4f} combined_mdd={pc['max_dd']:.4f}")
    print(f"risk_on_ret={po['ret']:.4f} risk_off_ret={pf['ret']:.4f} spot_ret={ps['ret']:.4f}")


if __name__ == "__main__":
    main()
