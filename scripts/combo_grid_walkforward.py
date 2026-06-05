from __future__ import annotations

import argparse
import itertools
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


def parse_list(s: str, cast=float):
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def sigma_ann(r: pd.Series, bar_minutes: int, vol_window: int) -> pd.Series:
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return r.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(bars_per_year)


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


def simulate_combo(
    df: pd.DataFrame,
    bar_minutes: int,
    lookback: int,
    compress_window: int,
    compress_quantile: float,
    hold_bars: int,
    target_vol: float,
    w_max: float,
    vol_window: int,
    z_window_bars: int,
    basis_weight: float,
    funding_weight: float,
    signal_ema_span: int,
    filt_on_thresh: float,
    filt_off_thresh: float,
    funding_min_hold_bars: int,
    trend_ema: int,
    trade_cost_bps: float,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    r = out["r"]

    rolling_high = close.rolling(lookback, min_periods=lookback).max()
    rolling_low = close.rolling(lookback, min_periods=lookback).min()
    rolling_range = (rolling_high - rolling_low) / rolling_low
    range_pct = rolling_range.rolling(compress_window, min_periods=compress_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    compress = range_pct <= compress_quantile
    breakout = (close > rolling_high.shift(1)) & compress.shift(1)
    breakdown = (close < rolling_low.shift(1)) & compress.shift(1)

    sig_ann = sigma_ann(r, bar_minutes, vol_window)
    weight_raw = (target_vol / sig_ann).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=w_max)

    gate = np.zeros(len(out), dtype=bool)
    hold = 0
    for i in range(len(out)):
        if bool(breakdown.iat[i]):
            hold = 0
            gate[i] = False
            continue
        if bool(breakout.iat[i]):
            hold = max(int(hold_bars), 0)
        if hold > 0:
            gate[i] = True
            hold -= 1
        else:
            gate[i] = bool(breakout.iat[i])
    trend_ok = pd.Series(True, index=out.index)
    if int(trend_ema) > 0:
        ema = close.ewm(span=int(trend_ema), adjust=False).mean()
        trend_ok = (close > ema).fillna(False)
        gate = gate & trend_ok.to_numpy(dtype=bool)
    w_breakout = weight_raw * gate.astype(float)

    basis_z = rolling_z(out["basis"], z_window_bars)
    funding_z = rolling_z(out["funding_rate"], z_window_bars)
    sig = -(basis_weight * basis_z + funding_weight * funding_z)
    sig = sig.ewm(span=int(signal_ema_span), adjust=False).mean().fillna(0.0)
    fstate = np.zeros(len(out), dtype=float)
    st = 0.0
    min_hold = max(int(funding_min_hold_bars), 0)
    on_age = 0
    for i, s in enumerate(sig.to_numpy()):
        if st == 0.0 and s >= filt_on_thresh:
            st = 1.0
            on_age = 1
        elif st == 1.0 and s <= filt_off_thresh:
            if on_age >= min_hold:
                st = 0.0
                on_age = 0
            else:
                on_age += 1
        elif st == 1.0:
            on_age += 1
        fstate[i] = st

    w_combo = w_breakout * fstate
    k = trade_cost_bps / 10000.0
    to_combo = pd.Series(w_combo).diff().abs().fillna(0.0)
    to_break = pd.Series(w_breakout).diff().abs().fillna(0.0)
    ret_combo = pd.Series(w_combo).shift(1).fillna(0.0) * r + (-to_combo * k)
    ret_break = pd.Series(w_breakout).shift(1).fillna(0.0) * r + (-to_break * k)
    ret_spot = r

    out["weight_breakout"] = w_breakout
    out["trend_ok"] = trend_ok.astype(bool)
    out["funding_signal"] = sig
    out["funding_filter"] = fstate
    out["weight_combo"] = w_combo
    out["ret_combo"] = ret_combo
    out["ret_breakout"] = ret_break
    out["ret_spot"] = ret_spot
    return out


def split_metrics(sim: pd.DataFrame, n_splits: int) -> list[dict]:
    n = len(sim)
    split = max(1, n // n_splits)
    out = []
    for i in range(n_splits):
        a = i * split
        b = n if i == n_splits - 1 else min(n, (i + 1) * split)
        s = sim.iloc[a:b].copy()
        if len(s) < 50:
            continue
        pc = perf(s["ret_combo"])
        ps = perf(s["ret_spot"])
        pb = perf(s["ret_breakout"])
        out.append(
            {
                "split_idx": i,
                "rows": int(len(s)),
                "combo_ret": pc["ret"],
                "combo_sharpe": pc["sharpe"],
                "combo_mdd": pc["max_dd"],
                "spot_ret": ps["ret"],
                "breakout_ret": pb["ret"],
                "ret_excess_vs_spot": pc["ret"] - ps["ret"],
                "ret_excess_vs_breakout": pc["ret"] - pb["ret"],
                "combo_turnover": float(s["weight_combo"].diff().abs().fillna(0.0).sum()),
                "combo_tim": float((s["weight_combo"] > 0).mean() * 100.0),
            }
        )
    return out


def write_top_html(out_path: Path, top: pd.DataFrame, meta: list[str]) -> None:
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    rows = []
    for _, r in top.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td>"
            f"<td>{int(r['lookback'])}</td>"
            f"<td>{int(r['compress_window'])}</td>"
            f"<td>{r['compress_quantile']:.2f}</td>"
            f"<td>{int(r['hold_bars'])}</td>"
            f"<td>{int(r['trend_ema'])}</td>"
            f"<td>{r['filt_on']:.2f}</td>"
            f"<td>{r['filt_off']:.2f}</td>"
            f"<td>{int(r['funding_min_hold_bars'])}</td>"
            f"<td>{r['avg_combo_ret']:.2%}</td>"
            f"<td>{r['avg_excess_spot']:.2%}</td>"
            f"<td>{r['avg_excess_breakout']:.2%}</td>"
            f"<td>{r['avg_combo_sharpe']:.3f}</td>"
            f"<td>{r['worst_combo_mdd']:.2%}</td>"
            f"<td>{r['avg_turnover']:.2f}</td>"
            f"<td>{int(r['splits_beating_spot'])}/{int(r['splits_total'])}</td>"
            f"<td>{int(r['splits_pos_combo_ret'])}/{int(r['splits_total'])}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Combo Grid Top 10</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.meta {{ margin-bottom: 8px; color: #444; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: right; }}
th {{ background: #f7f7f7; text-align: center; }}
</style></head><body>
<h2>Breakout + Funding Filter Grid (Walk-Forward) Top 10</h2>
{meta_html}
<div class="card"><table>
<tr>
<th>Rank</th><th>Lookback</th><th>CompWin</th><th>CompQ</th><th>Hold</th><th>Trend EMA</th><th>On</th><th>Off</th><th>FMinHold</th>
<th>Avg Combo Ret</th><th>Avg Excess vs Spot</th><th>Avg Excess vs Breakout</th>
<th>Avg Sharpe</th><th>Worst MDD</th><th>Avg Turnover</th><th>Beat Spot Splits</th><th>Pos Ret Splits</th>
</tr>
{''.join(rows)}
</table></div></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Grid search with walk-forward splits for combo model")
    p.add_argument("--price-5m", default="data/backtest/ETHUSDC_5m.csv")
    p.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_90d.csv")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--start-ts", default="", help="Optional UTC timestamp override start, e.g. 2025-01-01")
    p.add_argument("--end-ts", default="", help="Optional UTC timestamp override end, e.g. 2025-03-31")
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--z-window-bars", type=int, default=288)
    p.add_argument("--basis-weight", type=float, default=0.6)
    p.add_argument("--funding-weight", type=float, default=0.4)
    p.add_argument("--signal-ema-span", type=int, default=12)
    p.add_argument("--lookback-list", default="96,144")
    p.add_argument("--compress-window-list", default="96,144")
    p.add_argument("--compress-quantile-list", default="0.40,0.50,0.60")
    p.add_argument("--hold-bars-list", default="12,24,36")
    p.add_argument("--trend-ema-list", default="0,100")
    p.add_argument("--filt-on-list", default="0.60,0.75,0.90")
    p.add_argument("--filt-off-list", default="0.00,0.25,0.40")
    p.add_argument("--funding-min-hold-bars-list", default="0,6,12")
    p.add_argument("--out-results-csv", default="artifacts/paper/combo_grid_results.csv")
    p.add_argument("--out-top10-html", default="artifacts/paper/combo_grid_top10.html")
    args = p.parse_args()

    price = _load_price(Path(args.price_5m))
    if str(args.end_ts).strip():
        end = pd.to_datetime(args.end_ts, utc=True, errors="coerce")
    else:
        end = price["timestamp"].iloc[-1]
    if pd.isna(end):
        raise ValueError(f"invalid end-ts: {args.end_ts}")
    if str(args.start_ts).strip():
        start = pd.to_datetime(args.start_ts, utc=True, errors="coerce")
    else:
        start = end - pd.Timedelta(days=int(args.window_days))
    if pd.isna(start):
        raise ValueError(f"invalid start-ts: {args.start_ts}")
    if start >= end:
        raise ValueError(f"start must be before end: start={start} end={end}")
    price = price[price["timestamp"] >= start].copy()
    price = price[price["timestamp"] <= end].copy()

    fund = pd.read_csv(args.funding_csv)
    need = {"timestamp", "basis", "funding_rate"}
    if not need.issubset(set(fund.columns)):
        raise ValueError(f"funding csv missing columns: {need - set(fund.columns)}")
    fund["timestamp"] = pd.to_datetime(fund["timestamp"], utc=True, errors="coerce")
    fund["basis"] = pd.to_numeric(fund["basis"], errors="coerce")
    fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="coerce")
    fund = fund.dropna(subset=["timestamp", "basis", "funding_rate"]).sort_values("timestamp")

    df = pd.merge(price[["timestamp", "close"]], fund[["timestamp", "basis", "funding_rate"]], on="timestamp", how="left").sort_values("timestamp")
    df["basis"] = df["basis"].ffill()
    df["funding_rate"] = df["funding_rate"].ffill()
    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df = df.dropna(subset=["basis", "funding_rate"]).reset_index(drop=True)
    if len(df) < 1000:
        raise RuntimeError(f"too few rows after merge: {len(df)}")

    lookbacks = parse_list(args.lookback_list, int)
    cws = parse_list(args.compress_window_list, int)
    cqs = parse_list(args.compress_quantile_list, float)
    holds = parse_list(args.hold_bars_list, int)
    trend_emas = parse_list(args.trend_ema_list, int)
    ons = parse_list(args.filt_on_list, float)
    offs = parse_list(args.filt_off_list, float)
    fmins = parse_list(args.funding_min_hold_bars_list, int)
    grid = list(itertools.product(lookbacks, cws, cqs, holds, trend_emas, ons, offs, fmins))

    rows = []
    for (lookback, cw, cq, hold, trend_ema, on, off, fmin) in grid:
        if off > on:
            continue
        sim = simulate_combo(
            df=df,
            bar_minutes=args.bar_minutes,
            lookback=lookback,
            compress_window=cw,
            compress_quantile=cq,
            hold_bars=hold,
            target_vol=args.target_vol,
            w_max=args.w_max,
            vol_window=args.vol_window,
            z_window_bars=args.z_window_bars,
            basis_weight=args.basis_weight,
            funding_weight=args.funding_weight,
            signal_ema_span=args.signal_ema_span,
            filt_on_thresh=on,
            filt_off_thresh=off,
            funding_min_hold_bars=fmin,
            trend_ema=trend_ema,
            trade_cost_bps=args.trade_cost_bps,
        )
        sm = split_metrics(sim, int(args.n_splits))
        if not sm:
            continue
        s = pd.DataFrame(sm)
        rows.append(
            {
                "lookback": lookback,
                "compress_window": cw,
                "compress_quantile": cq,
                "hold_bars": hold,
                "trend_ema": int(trend_ema),
                "filt_on": on,
                "filt_off": off,
                "funding_min_hold_bars": int(fmin),
                "avg_combo_ret": float(s["combo_ret"].mean()),
                "avg_excess_spot": float(s["ret_excess_vs_spot"].mean()),
                "avg_excess_breakout": float(s["ret_excess_vs_breakout"].mean()),
                "avg_combo_sharpe": float(s["combo_sharpe"].mean()),
                "worst_combo_mdd": float(s["combo_mdd"].min()),
                "avg_turnover": float(s["combo_turnover"].mean()),
                "splits_beating_spot": int((s["ret_excess_vs_spot"] > 0).sum()),
                "splits_pos_combo_ret": int((s["combo_ret"] > 0).sum()),
                "splits_total": int(len(s)),
                "combo_ret_std": float(s["combo_ret"].std(ddof=0)),
            }
        )

    if not rows:
        raise RuntimeError("no grid results")
    res = pd.DataFrame(rows)
    # Ranking: prefer excess return and robustness, penalize turnover.
    res["score"] = (
        res["avg_excess_spot"]
        + 0.5 * res["avg_excess_breakout"]
        - 0.1 * res["combo_ret_std"].fillna(0.0)
        - 0.0002 * res["avg_turnover"]
    )
    res = res.sort_values(["score", "avg_combo_ret"], ascending=[False, False]).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)

    out_csv = Path(args.out_results_csv)
    out_html = Path(args.out_top10_html)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    top = res.head(10).copy()
    meta = [
        f"window_days={args.window_days} | n_splits={args.n_splits} | bars={len(df)}",
        f"grid_size={len(grid)} (valid={len(res)})",
        "score = avg_excess_spot + 0.5*avg_excess_breakout - 0.1*combo_ret_std - 0.0002*avg_turnover",
    ]
    write_top_html(out_html, top, meta)

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(
        top[
            [
                "rank",
                "lookback",
                "compress_window",
                "compress_quantile",
                "hold_bars",
                "trend_ema",
                "filt_on",
                "filt_off",
                "funding_min_hold_bars",
                "avg_combo_ret",
                "avg_excess_spot",
                "avg_combo_sharpe",
                "worst_combo_mdd",
                "splits_beating_spot",
                "splits_pos_combo_ret",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
