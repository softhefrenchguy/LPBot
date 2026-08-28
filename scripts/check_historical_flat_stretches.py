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


def _fwd_return(close: np.ndarray, from_idx: int, days: int) -> float:
    to_idx = from_idx + days
    if to_idx >= len(close) or not np.isfinite(close[from_idx]) or close[from_idx] <= 0:
        return np.nan
    return float(close[to_idx] / close[from_idx] - 1.0)


def _fwd_maxdd(close: np.ndarray, from_idx: int, days: int) -> float:
    to_idx = min(from_idx + days, len(close) - 1)
    window = close[from_idx:to_idx + 1]
    window = window[np.isfinite(window)]
    if len(window) < 2:
        return np.nan
    running_max = np.maximum.accumulate(window)
    dd = window / running_max - 1.0
    return float(dd.min())


def main() -> int:
    ap = argparse.ArgumentParser(description="Historical analysis of consecutive flat-position stretches: how often, how costly, and did caution pay off.")
    ap.add_argument("--daily", default="artifacts/backtest/paper_window_fresh/validated_crypto_daily.csv")
    ap.add_argument("--min-length", type=int, default=100)
    ap.add_argument("--out", default="artifacts/backtest/historical_flat_stretches.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.daily, low_memory=False)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    for col in ["alloc_eth", "alloc_btc", "eth_close", "btc_close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    flat = ((d["alloc_eth"].abs() < 1e-9) & (d["alloc_btc"].abs() < 1e-9)).to_numpy()
    btc_close = d["btc_close"].to_numpy()
    eth_close = d["eth_close"].to_numpy()
    eth_regime = d["eth_regime"].astype(str).to_numpy()
    btc_regime = d["btc_regime"].astype(str).to_numpy()
    day = d["day"]

    print("=" * 115)
    print("HISTORICAL FLAT-STRETCH ANALYSIS")
    print(f"Data: {day.min().date()} to {day.max().date()} ({len(d)} days) -- full available history; 2018 excluded, ETHUSDC/BTCUSDC only listed on Binance from 2018-12-15")
    print("=" * 115)

    total_days = len(d)
    flat_days = int(flat.sum())
    print(f"\nOverall: flat {flat_days}/{total_days} days ({flat_days/total_days*100:.1f}%) across the full available history.")

    all_segs = _segments(flat)
    lengths = [e - s + 1 for s, e in all_segs]
    print(f"Total flat stretches (any length): {len(all_segs)}")
    if lengths:
        current_len = lengths[-1] if (all_segs and all_segs[-1][1] == total_days - 1) else None
        pct_rank = float((np.array(lengths) < 159).mean() * 100) if lengths else np.nan
        print(f"Stretch length distribution: min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}")
        print(f"The current (ongoing) 159-day stretch is longer than {pct_rank:.0f}% of all historical flat stretches.")

    long_segs = [(s, e) for s, e in all_segs if (e - s + 1) >= args.min_length]
    print(f"\nFlat stretches >= {args.min_length} days: {len(long_segs)} (including the current ongoing one, if it qualifies)")
    print("-" * 115)

    rows = []
    for s, e in long_segs:
        length = e - s + 1
        start_date, end_date = day.iloc[s].date(), day.iloc[e].date()
        is_ongoing = e == total_days - 1
        btc_during = btc_close[e] / btc_close[s] - 1.0 if np.isfinite(btc_close[s]) and btc_close[s] > 0 else np.nan
        eth_during = eth_close[e] / eth_close[s] - 1.0 if np.isfinite(eth_close[s]) and eth_close[s] > 0 else np.nan

        # regime mix during the stretch
        regime_combo = pd.Series([f"{a}/{b}" for a, b in zip(eth_regime[s:e+1], btc_regime[s:e+1])]).value_counts()
        had_bull_blip = bool(any(("BULL" in c) for c in regime_combo.index))
        n_bear_days = int(sum(v for c, v in regime_combo.items() if "BEAR" in c))
        n_bull_days = int(sum(v for c, v in regime_combo.items() if "BULL" in c))
        n_chop_chop = int(regime_combo.get("CHOP/CHOP", 0))

        # what happened next: fixed-horizon forward returns/drawdowns from the END of the stretch
        fwd = {}
        for horizon in [30, 60, 90]:
            fwd[f"btc_fwd_{horizon}d_ret"] = _fwd_return(btc_close, e, horizon)
            fwd[f"btc_fwd_{horizon}d_maxdd"] = _fwd_maxdd(btc_close, e, horizon)

        # gap until the strategy actually re-entered (any sleeve), and that trade's early return
        next_entry_idx = None
        for j in range(e + 1, total_days):
            if abs(d["alloc_eth"].iloc[j]) > 1e-9 or abs(d["alloc_btc"].iloc[j]) > 1e-9:
                next_entry_idx = j
                break
        days_to_next_entry = (next_entry_idx - e) if next_entry_idx is not None else np.nan
        gap_btc_ret = (btc_close[next_entry_idx] / btc_close[e] - 1.0) if next_entry_idx is not None and np.isfinite(btc_close[e]) and btc_close[e] > 0 else np.nan
        post_entry_30d_ret = _fwd_return(btc_close, next_entry_idx, 30) if next_entry_idx is not None else np.nan

        row = {
            "start": str(start_date), "end": str(end_date), "length_days": length, "ongoing": is_ongoing,
            "btc_return_during_pct": btc_during * 100 if np.isfinite(btc_during) else np.nan,
            "eth_return_during_pct": eth_during * 100 if np.isfinite(eth_during) else np.nan,
            "regime_chop_chop_days": n_chop_chop, "regime_bear_days": n_bear_days, "regime_bull_days": n_bull_days, "had_bull_blip": had_bull_blip,
            "days_to_next_entry": days_to_next_entry,
            "gap_end_to_next_entry_btc_ret_pct": gap_btc_ret * 100 if np.isfinite(gap_btc_ret) else np.nan,
            "post_entry_30d_btc_ret_pct": post_entry_30d_ret * 100 if np.isfinite(post_entry_30d_ret) else np.nan,
        }
        for horizon in [30, 60, 90]:
            row[f"fwd_{horizon}d_btc_ret_pct"] = fwd[f"btc_fwd_{horizon}d_ret"] * 100 if np.isfinite(fwd[f"btc_fwd_{horizon}d_ret"]) else np.nan
            row[f"fwd_{horizon}d_btc_maxdd_pct"] = fwd[f"btc_fwd_{horizon}d_maxdd"] * 100 if np.isfinite(fwd[f"btc_fwd_{horizon}d_maxdd"]) else np.nan
        rows.append(row)

        tag = " [ONGOING -- current paper-trading stretch]" if is_ongoing else ""
        print(f"{start_date} -> {end_date}  ({length}d){tag}")
        print(f"  During stretch: BTC {btc_during*100:+.2f}%  ETH {eth_during*100:+.2f}%")
        print(f"  Regime mix: CHOP/CHOP={n_chop_chop}d  BEAR-involved={n_bear_days}d  BULL-involved={n_bull_days}d  (mixed-signal: {had_bull_blip})")
        if not is_ongoing:
            print(f"  Forward from stretch end: 30d {row['fwd_30d_btc_ret_pct']:+.2f}% (maxdd {row['fwd_30d_btc_maxdd_pct']:.2f}%)  "
                  f"60d {row['fwd_60d_btc_ret_pct']:+.2f}% (maxdd {row['fwd_60d_btc_maxdd_pct']:.2f}%)  "
                  f"90d {row['fwd_90d_btc_ret_pct']:+.2f}% (maxdd {row['fwd_90d_btc_maxdd_pct']:.2f}%)")
            if next_entry_idx is not None:
                print(f"  Strategy re-entered {days_to_next_entry}d later ({day.iloc[next_entry_idx].date()}); BTC moved {gap_btc_ret*100:+.2f}% during that extra gap; "
                      f"first 30d of the new trade: BTC {post_entry_30d_ret:+.2f}%" if np.isfinite(post_entry_30d_ret) else "")
        print()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    # verdict: for CLOSED stretches only, tally avoided-drawdown vs pure-missed-opportunity
    closed = out_df[~out_df["ongoing"]].copy()
    if not closed.empty:
        DD_THRESHOLD = -10.0
        closed["avoided_drawdown"] = closed["fwd_90d_btc_maxdd_pct"] <= DD_THRESHOLD
        closed["pure_missed_rally"] = (closed["fwd_90d_btc_ret_pct"] > 0) & (~closed["avoided_drawdown"])
        n_avoided = int(closed["avoided_drawdown"].sum())
        n_missed = int(closed["pure_missed_rally"].sum())
        n_closed = len(closed)
        print("=" * 115)
        print(f"VERDICT across {n_closed} closed >= {args.min_length}-day flat stretches (90d forward window, {DD_THRESHOLD:.0f}% drawdown threshold):")
        print(f"  Followed by a drawdown of {DD_THRESHOLD:.0f}%+ within 90 days (caution would have paid off): {n_avoided}/{n_closed}")
        print(f"  Pure missed rally, no offsetting drawdown avoided: {n_missed}/{n_closed}")
        avg_fwd_ret = closed["fwd_90d_btc_ret_pct"].mean()
        print(f"  Average 90d forward BTC return after these stretches ended: {avg_fwd_ret:+.2f}%")
        if n_avoided > n_missed:
            print("  -> Historically, caution has paid off more often than not: these long flat stretches were more often")
            print("     followed by real drawdowns than by clean, undisturbed rallies.")
        elif n_missed > n_avoided:
            print("  -> Historically, these long flat stretches were more often pure missed opportunity than protection --")
            print("     evidence the entry threshold/confirm-day requirement may be too conservative.")
        else:
            print("  -> Split evenly; no strong historical lean either way.")
    else:
        print("No CLOSED >= {}-day stretches to evaluate (only the current ongoing one qualifies).".format(args.min_length))

    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
