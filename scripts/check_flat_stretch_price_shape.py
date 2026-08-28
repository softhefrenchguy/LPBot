from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _segments(flat: np.ndarray) -> list[tuple[int, int]]:
    n = len(flat)
    if not flat.any():
        return []
    diff = np.diff(flat.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if flat[-1] else end_candidates
    return list(zip(starts.tolist(), ends.tolist()))


def _max_dd(path: np.ndarray) -> float:
    path = path[np.isfinite(path)]
    if len(path) < 2:
        return 0.0
    running_max = np.maximum.accumulate(path)
    return float((path / running_max - 1.0).min())


def _max_rally(path: np.ndarray) -> float:
    path = path[np.isfinite(path)]
    if len(path) < 2:
        return 0.0
    running_min = np.minimum.accumulate(path)
    return float((path / running_min - 1.0).max())


def _categorize(net_pct: float, max_dd_pct: float, max_rally_pct: float) -> str:
    if net_pct <= -5.0 and max_rally_pct < 10.0:
        return "persistent_bear"
    if net_pct >= 5.0 and max_dd_pct > -10.0:
        return "missed_rally"
    if max_dd_pct <= -15.0 and max_rally_pct >= 15.0:
        return "failed_trend_whipsaw"
    return "choppy_range_bound"


def main() -> int:
    ap = argparse.ArgumentParser(description="Characterize price-action shape during every historical flat/no-trade stretch, not just the long ones -- persistent decline vs choppy range vs failed-trend whipsaw vs missed rally.")
    ap.add_argument("--daily", default="artifacts/backtest/paper_window_fresh/validated_crypto_daily.csv")
    ap.add_argument("--min-length", type=int, default=1)
    ap.add_argument("--out", default="artifacts/backtest/flat_stretch_price_shape.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.daily, low_memory=False)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    for col in ["alloc_eth", "alloc_btc", "eth_close", "btc_close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    flat = ((d["alloc_eth"].abs() < 1e-9) & (d["alloc_btc"].abs() < 1e-9)).to_numpy()
    eth_close = d["eth_close"].to_numpy()
    eth_regime = d["eth_regime"].astype(str).to_numpy()
    btc_regime = d["btc_regime"].astype(str).to_numpy()
    day = d["day"]
    total_days = len(d)

    segs = [(s, e) for s, e in _segments(flat) if (e - s + 1) >= args.min_length]

    print("=" * 130)
    print("PRICE-ACTION SHAPE DURING FLAT/NO-TRADE STRETCHES (all stretches, not just 100+ day ones)")
    print("=" * 130)
    print(f"{'Start':11} {'End':11} {'Days':>5} {'Net%':>8} {'MaxDD%':>8} {'MaxRally%':>10} {'%BEAR':>6} {'%BULL':>6} {'%CHOP/CHOP':>10}  Category")
    print("-" * 130)

    rows = []
    for s, e in segs:
        length = e - s + 1
        start_date, end_date = day.iloc[s].date(), day.iloc[e].date()
        ongoing = e == total_days - 1
        path = eth_close[s:e + 1]
        net_pct = (path[-1] / path[0] - 1.0) * 100 if np.isfinite(path[0]) and path[0] > 0 and np.isfinite(path[-1]) else np.nan
        max_dd_pct = _max_dd(path) * 100
        max_rally_pct = _max_rally(path) * 100

        combo = pd.Series([f"{a}/{b}" for a, b in zip(eth_regime[s:e+1], btc_regime[s:e+1])])
        n = len(combo)
        pct_bear = float(combo.str.contains("BEAR").mean() * 100)
        pct_bull = float(combo.str.contains("BULL").mean() * 100)
        pct_chop_chop = float((combo == "CHOP/CHOP").mean() * 100)

        category = _categorize(net_pct, max_dd_pct, max_rally_pct) if np.isfinite(net_pct) else "unknown"
        tag = " [ONGOING]" if ongoing else ""
        print(f"{str(start_date):11} {str(end_date):11} {length:>5} {net_pct:>+7.1f}% {max_dd_pct:>+7.1f}% {max_rally_pct:>+9.1f}% {pct_bear:>5.0f}% {pct_bull:>5.0f}% {pct_chop_chop:>9.0f}%  {category}{tag}")

        rows.append({
            "start": str(start_date), "end": str(end_date), "length_days": length, "ongoing": ongoing,
            "eth_net_pct": net_pct, "eth_max_dd_pct": max_dd_pct, "eth_max_rally_pct": max_rally_pct,
            "pct_days_bear_involved": pct_bear, "pct_days_bull_involved": pct_bull, "pct_days_chop_chop": pct_chop_chop,
            "category": category,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print()
    print("=" * 130)
    print("AGGREGATE (all stretches, unweighted count vs day-weighted -- a few long stretches dominate total flat-time)")
    print("=" * 130)
    n_stretches = len(out_df)
    by_count = out_df["category"].value_counts()
    by_days = out_df.groupby("category")["length_days"].sum()
    total_flat_days = int(out_df["length_days"].sum())
    print(f"{'Category':22} {'# stretches':>12} {'% of stretches':>15} {'total days':>11} {'% of flat-days':>15}")
    for cat in sorted(set(list(by_count.index) + list(by_days.index))):
        n_c = int(by_count.get(cat, 0))
        d_c = int(by_days.get(cat, 0))
        print(f"{cat:22} {n_c:>12} {n_c/n_stretches*100:>14.1f}% {d_c:>11} {d_c/total_flat_days*100:>14.1f}%")

    print()
    persistent_bear_days = int(by_days.get("persistent_bear", 0))
    print(f"Persistent-bear days as share of ALL flat time: {persistent_bear_days}/{total_flat_days} ({persistent_bear_days/total_flat_days*100:.1f}%)")
    non_bear_days = total_flat_days - persistent_bear_days
    print(f"Everything else (choppy/whipsaw/missed-rally) as share of ALL flat time: {non_bear_days}/{total_flat_days} ({non_bear_days/total_flat_days*100:.1f}%)")

    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
