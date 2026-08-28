from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_forex_optimised import _stats
from backtest_overlay_strategies import _mean_reversion_overlay


def _download_fng(cache_csv: Path, refresh: bool) -> pd.DataFrame:
    if cache_csv.exists() and not refresh:
        d = pd.read_csv(cache_csv)
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        return d
    url = "https://api.alternative.me/fng/?limit=3000&format=json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    rows = []
    for rec in payload["data"]:
        rows.append({
            "date": pd.to_datetime(int(rec["timestamp"]), unit="s", utc=True).floor("D"),
            "value": int(rec["value"]),
            "classification": rec["value_classification"],
        })
    d = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(cache_csv, index=False)
    return d


def _segment_stats(active: np.ndarray, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (starts, ends, trade_returns) for contiguous True runs of `active`."""
    n = len(active)
    if not active.any():
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    diff = np.diff(active.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if active[-1] else end_candidates
    log_ret = np.log1p(np.clip(ret, -0.999999, None))
    cum0 = np.concatenate(([0.0], np.cumsum(log_ret)))
    trade_returns = np.expm1(cum0[ends + 1] - cum0[starts])
    return starts, ends, trade_returns


def test_a_standalone(base: pd.DataFrame, cost_bps: float) -> dict:
    n = len(base)
    fng = base["fng_value"].to_numpy()
    eth_ret = base["eth_spot_return"].fillna(0.0).to_numpy()
    target = np.zeros(n, dtype=int)
    active = 0
    for i in range(n):
        if active == 0 and fng[i] < 25:
            active = 1
        elif active == 1 and fng[i] > 50:
            active = 0
        target[i] = active
    weight_exec = np.roll(target, 1).astype(float); weight_exec[0] = 0.0
    prev_w = np.roll(weight_exec, 1); prev_w[0] = 0.0
    cost = np.abs(weight_exec - prev_w) * (cost_bps / 10000.0)
    strat = weight_exec * eth_ret - cost

    starts, ends, trade_returns = _segment_stats(np.abs(weight_exec) > 1e-12, strat)
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan
    st = _stats(pd.Series(strat))
    return {"strategy_return": strat, "trades": int(len(trade_returns)), "win_rate": win_rate, **st}


def _apply_conviction(base: pd.DataFrame, gross_cap: float) -> pd.DataFrame:
    x = base.copy()
    fng = x["fng_value"]
    mult = pd.Series(np.select([fng < 30, fng > 70], [1.2, 0.8], default=1.0), index=x.index)
    active = (x["alloc_eth"] > 0) | (x["alloc_btc"] > 0)
    x["fng_conviction_mult"] = np.where(active, mult, 1.0)
    x.loc[active, "alloc_eth"] = x.loc[active, "alloc_eth"] * mult.loc[active]
    x.loc[active, "alloc_btc"] = x.loc[active, "alloc_btc"] * mult.loc[active]
    gross = x["alloc_eth"] + x["alloc_btc"]
    over = gross > gross_cap
    x.loc[over, "alloc_eth"] = x.loc[over, "alloc_eth"] * gross_cap / gross.loc[over]
    x.loc[over, "alloc_btc"] = x.loc[over, "alloc_btc"] * gross_cap / gross.loc[over]
    return x


def _apply_gate(base: pd.DataFrame, allow_below: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero out alloc_eth/alloc_btc for any trade whose entry day has fng_value >= allow_below.
    Returns (modified_base, blocked_trades_df)."""
    x = base.copy()
    fng = x["fng_value"].to_numpy()
    blocked_rows = []
    for leg, off_col, alloc_col, ret_col in [("eth", "eth_off_active", "alloc_eth", "eth_strategy_return"), ("btc", "btc_off_active", "alloc_btc", "btc_strategy_return")]:
        active = x[off_col].fillna(0).to_numpy().astype(bool)
        starts, ends, trade_returns = _segment_stats(active, x[ret_col].fillna(0.0).to_numpy())
        alloc = x[alloc_col].to_numpy().copy()
        for s, e, r in zip(starts, ends, trade_returns):
            if fng[s] >= allow_below:
                alloc[s:e + 1] = 0.0
                blocked_rows.append({
                    "leg": leg, "entry_date": pd.Timestamp(x["day"].iloc[s]).date().isoformat(),
                    "exit_date": pd.Timestamp(x["day"].iloc[e]).date().isoformat(),
                    "fng_on_entry": int(fng[s]), "days_held": int(e - s + 1), "would_be_return_pct": float(r * 100.0),
                })
        x[alloc_col] = alloc
    return x, pd.DataFrame(blocked_rows)


def _recompute_main_return(x: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.Series:
    x = x.copy()
    x["combined_return"] = x["alloc_eth"] * x["eth_strategy_return"] + x["alloc_btc"] * x["btc_strategy_return"] + x["gold_strategy_return"]
    mr = _mean_reversion_overlay(x, gross_cap=gross_cap, cost_bps=cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    return x["combined_return"] + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fear & Greed Index: standalone signal, conviction modifier, and entry gate tests.")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised", help="Cache dir holding validated_crypto_daily.csv")
    ap.add_argument("--out-fng-data", default="data/fear_greed_daily.csv")
    ap.add_argument("--out-summary", default="artifacts/backtest/fear_greed_summary.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--refresh-fng", action="store_true")
    args = ap.parse_args()

    fng = _download_fng(Path(args.out_fng_data), bool(args.refresh_fng))
    print(f"Fear & Greed data: {len(fng)} days, {fng['date'].min().date()} to {fng['date'].max().date()}")

    base = pd.read_csv(Path(args.work_dir) / "validated_crypto_daily.csv", low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    base = base.merge(fng.rename(columns={"date": "day", "value": "fng_value"})[["day", "fng_value", "classification"]], on="day", how="left")
    base["fng_value"] = base["fng_value"].ffill()
    base = base.dropna(subset=["fng_value"]).reset_index(drop=True)
    print(f"Merged sample: {len(base)} days ({base['day'].min().date()} to {base['day'].max().date()})")

    main_return_baseline = _recompute_main_return(base, args.gross_cap, args.cost_bps)
    main_stats = _stats(main_return_baseline)
    print(f"Baseline main strategy (recomputed, sanity check): Sharpe {main_stats['sharpe']:.3f}")

    # ---- Test 1A: standalone ----
    a = test_a_standalone(base, args.cost_bps)
    corr_a = float(pd.Series(a["strategy_return"]).corr(main_return_baseline))
    print("=" * 80); print("TEST 1A: STANDALONE F&G REGIME SIGNAL (ETH long/flat)"); print("=" * 80)
    print(f"Trades: {a['trades']}  Win rate: {a['win_rate']*100:.1f}%  Raw Sharpe: {a['raw_sharpe']:.3f}  CAGR: {a['cagr']*100:.1f}%  Corr to main: {corr_a:.3f}")

    # ---- Test 1B: conviction modifier ----
    base_b = _apply_conviction(base, args.gross_cap)
    main_return_b = _recompute_main_return(base_b, args.gross_cap, args.cost_bps)
    stats_b = _stats(main_return_b)
    print("=" * 80); print("TEST 1B: F&G CONVICTION MODIFIER (1.2x fear / 0.8x greed on eth+btc legs)"); print("=" * 80)
    print(f"Main alone:     Sharpe {main_stats['sharpe']:.3f}")
    print(f"Main + F&G mod: Sharpe {stats_b['sharpe']:.3f}")
    print(f"Improvement: {stats_b['sharpe']-main_stats['sharpe']:+.3f}")

    # ---- Test 1C: entry gate ----
    base_c, blocked = _apply_gate(base, allow_below=50.0)
    main_return_c = _recompute_main_return(base_c, args.gross_cap, args.cost_bps)
    stats_c = _stats(main_return_c)
    n_blocked = len(blocked)
    blocked_avg_ret = float(blocked["would_be_return_pct"].mean()) if n_blocked else np.nan
    blocked_win_rate = float((blocked["would_be_return_pct"] > 0).mean()) if n_blocked else np.nan
    was_blocking_good = bool(blocked_avg_ret < 0) if n_blocked else None
    print("=" * 80); print("TEST 1C: F&G ENTRY GATE (block new entries when F&G >= 50)"); print("=" * 80)
    print(f"Main alone:      Sharpe {main_stats['sharpe']:.3f}")
    print(f"Main + F&G gate: Sharpe {stats_c['sharpe']:.3f}")
    print(f"Entries blocked: {n_blocked}")
    if n_blocked:
        print(blocked.to_string(index=False))
        print(f"Blocked entries avg return: {blocked_avg_ret:.2f}%  (win rate if taken: {blocked_win_rate*100:.1f}%)")
    print(f"Was blocking them good? {'YES' if was_blocking_good else 'NO' if was_blocking_good is not None else 'N/A'}")

    print("=" * 80)
    candidates = {"A": a["raw_sharpe"], "B": stats_b["sharpe"] - main_stats["sharpe"], "C": stats_c["sharpe"] - main_stats["sharpe"]}
    best_app = max(candidates, key=lambda k: candidates[k])
    if candidates[best_app] <= 0:
        best_app = "NONE"
    decision = "IMPLEMENT" if best_app != "NONE" and candidates.get(best_app, 0) > 0.1 else "RESEARCH" if best_app != "NONE" else "SKIP"
    print(f"Best F&G application: {best_app}")
    print(f"Decision: {decision}")

    pd.DataFrame([
        {"test": "A_standalone", "trades": a["trades"], "win_rate": a["win_rate"], "raw_sharpe": a["raw_sharpe"], "sharpe": a["sharpe"], "cagr": a["cagr"], "corr_to_main": corr_a},
        {"test": "B_conviction_modifier", "main_only_sharpe": main_stats["sharpe"], "with_mod_sharpe": stats_b["sharpe"], "improvement": stats_b["sharpe"] - main_stats["sharpe"]},
        {"test": "C_entry_gate", "main_only_sharpe": main_stats["sharpe"], "with_gate_sharpe": stats_c["sharpe"], "improvement": stats_c["sharpe"] - main_stats["sharpe"], "entries_blocked": n_blocked, "blocked_avg_return_pct": blocked_avg_ret, "blocking_was_good": was_blocking_good},
    ]).to_csv(args.out_summary, index=False)
    print(f"\nSaved: {args.out_fng_data}")
    print(f"Saved: {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
