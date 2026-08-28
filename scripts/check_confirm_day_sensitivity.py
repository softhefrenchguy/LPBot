from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_overlay_strategies import _mean_reversion_overlay
from backtest_forex_optimised import _stats


VARIANTS = {
    "baseline_3_5": {"eth_confirm": 3, "btc_confirm": 5, "daily": "artifacts/backtest/paper_window_fresh/validated_crypto_daily.csv"},
    "loose_2_3": {"eth_confirm": 2, "btc_confirm": 3, "daily": "artifacts/backtest/confirm_sensitivity/loose_2_3_daily.csv"},
    "loose_1_2": {"eth_confirm": 1, "btc_confirm": 2, "daily": "artifacts/backtest/confirm_sensitivity/loose_1_2_daily.csv"},
}

HISTORICAL_STRETCHES = [
    ("2019_2020", "2019-08-26", "2020-02-02"),
    ("2022_2023", "2022-06-17", "2023-04-03"),
    ("current", "2026-02-03", None),  # None end = ongoing, use data's last day
]


def _load_with_overlay(path: str, gross_cap: float, cost_bps: float) -> pd.DataFrame:
    base = pd.read_csv(path, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    base = base.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    if "gold_strategy_return" not in base.columns:
        base["gold_strategy_return"] = 0.0
    mr = _mean_reversion_overlay(base, gross_cap=gross_cap, cost_bps=cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    base["main_return"] = base["combined_return"] + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    return base


def _entries(off_active: np.ndarray) -> np.ndarray:
    """Indices where off_active transitions 0 -> 1."""
    a = off_active.astype(np.int8)
    diff = np.diff(a, prepend=np.int8(0))
    return np.flatnonzero(diff == 1)


def _segment_return(strategy_return: np.ndarray, off_active: np.ndarray, entry_idx: int) -> tuple[int, float]:
    """From entry_idx, walk forward while off_active stays 1; return (exit_idx, compounded_return)."""
    n = len(off_active)
    e = entry_idx
    while e + 1 < n and off_active[e + 1]:
        e += 1
    seg = strategy_return[entry_idx:e + 1]
    seg = np.where(np.isfinite(seg), seg, 0.0)
    ret = float(np.prod(1.0 + seg) - 1.0)
    return e, ret


def main() -> int:
    ap = argparse.ArgumentParser(description="Confirm-day sensitivity: does loosening the entry confirm-day requirement help or hurt, on both headline stats and specific flat-stretch entries.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out-headline", default="artifacts/backtest/confirm_sensitivity_headline.csv")
    ap.add_argument("--out-stretch-entries", default="artifacts/backtest/confirm_sensitivity_stretch_entries.csv")
    ap.add_argument("--out-false-starts", default="artifacts/backtest/confirm_sensitivity_false_starts.csv")
    args = ap.parse_args()

    data = {}
    for name, spec in VARIANTS.items():
        d = _load_with_overlay(spec["daily"], args.gross_cap, args.cost_bps)
        data[name] = d

    print("=" * 115)
    print("STEP 1: HEADLINE STATS, 2019-2024 (full stack, same cost model, only confirm-days changed)")
    print("=" * 115)
    headline_rows = []
    for name, spec in VARIANTS.items():
        d = data[name]
        window = d[(d["day"] >= pd.to_datetime(args.start, utc=True)) & (d["day"] <= pd.to_datetime(args.end, utc=True))]
        st = _stats(window["main_return"])
        n_eth_entries = len(_entries(window["eth_off_active"].fillna(0).to_numpy()))
        n_btc_entries = len(_entries(window["btc_off_active"].fillna(0).to_numpy()))
        headline_rows.append({"variant": name, "eth_confirm": spec["eth_confirm"], "btc_confirm": spec["btc_confirm"], "sharpe": st["sharpe"], "cagr": st["cagr"], "maxdd": st["maxdd"], "ann_vol": st["ann_vol"], "eth_entries": n_eth_entries, "btc_entries": n_btc_entries})
        print(f"{name:15} (ETH={spec['eth_confirm']}d BTC={spec['btc_confirm']}d): Sharpe {st['sharpe']:.3f}  CAGR {st['cagr']*100:.1f}%  MaxDD {st['maxdd']*100:.1f}%  ETH entries {n_eth_entries}  BTC entries {n_btc_entries}")
    headline_df = pd.DataFrame(headline_rows)
    headline_df.to_csv(args.out_headline, index=False)
    baseline_sharpe = headline_df.loc[headline_df["variant"] == "baseline_3_5", "sharpe"].iloc[0]
    print()
    for _, r in headline_df.iterrows():
        if r["variant"] != "baseline_3_5":
            print(f"{r['variant']}: {r['sharpe']-baseline_sharpe:+.3f} Sharpe vs baseline")
    print()

    print("=" * 115)
    print("STEP 2: HISTORICAL + CURRENT STRETCH -- WOULD A LOOSER THRESHOLD HAVE ENTERED EARLIER?")
    print("=" * 115)
    stretch_rows = []
    for stretch_name, start_str, end_str in HISTORICAL_STRETCHES:
        print(f"\n--- {stretch_name} (flat from {start_str}{' to ' + end_str if end_str else ', ongoing'}) ---")
        for name, spec in VARIANTS.items():
            d = data[name]
            stretch_start = pd.to_datetime(start_str, utc=True)
            after = d[d["day"] >= stretch_start].reset_index(drop=True)
            eth_active = after["eth_off_active"].fillna(0).to_numpy()
            btc_active = after["btc_off_active"].fillna(0).to_numpy()
            eth_strat_ret = after["eth_strategy_return"].fillna(0).to_numpy()
            btc_strat_ret = after["btc_strategy_return"].fillna(0).to_numpy()

            eth_entry_idxs = _entries(eth_active)
            btc_entry_idxs = _entries(btc_active)
            first_eth = int(eth_entry_idxs[0]) if len(eth_entry_idxs) else None
            first_btc = int(btc_entry_idxs[0]) if len(btc_entry_idxs) else None

            candidates = []
            if first_eth is not None:
                exit_idx, ret = _segment_return(eth_strat_ret, eth_active, first_eth)
                candidates.append(("ETH", after["day"].iloc[first_eth].date(), after["day"].iloc[exit_idx].date(), ret))
            if first_btc is not None:
                exit_idx, ret = _segment_return(btc_strat_ret, btc_active, first_btc)
                candidates.append(("BTC", after["day"].iloc[first_btc].date(), after["day"].iloc[exit_idx].date(), ret))

            for leg, entry_date, exit_date, ret in candidates:
                print(f"  {name:15} {leg}: first entry after stretch start = {entry_date} -> {exit_date}, trade return {ret*100:+.2f}%")
                stretch_rows.append({"stretch": stretch_name, "variant": name, "leg": leg, "entry_date": str(entry_date), "exit_date": str(exit_date), "trade_return_pct": ret * 100})
            if not candidates:
                print(f"  {name:15}: no entry yet (still flat through end of available data)")
                stretch_rows.append({"stretch": stretch_name, "variant": name, "leg": None, "entry_date": None, "exit_date": None, "trade_return_pct": None})
    stretch_df = pd.DataFrame(stretch_rows)
    stretch_df.to_csv(args.out_stretch_entries, index=False)
    print()

    print("=" * 115)
    print("STEP 3: OVERALL FALSE-START RATE (2019-2024) -- entries a looser threshold makes that baseline never confirms nearby")
    print("=" * 115)
    baseline_window = data["baseline_3_5"]
    baseline_window = baseline_window[(baseline_window["day"] >= pd.to_datetime(args.start, utc=True)) & (baseline_window["day"] <= pd.to_datetime(args.end, utc=True))].reset_index(drop=True)

    false_start_rows = []
    for name in ["loose_2_3", "loose_1_2"]:
        d = data[name]
        window = d[(d["day"] >= pd.to_datetime(args.start, utc=True)) & (d["day"] <= pd.to_datetime(args.end, utc=True))].reset_index(drop=True)
        for leg, active_col, ret_col in [("ETH", "eth_off_active", "eth_strategy_return"), ("BTC", "btc_off_active", "btc_strategy_return")]:
            loose_active = window[active_col].fillna(0).to_numpy()
            loose_ret = window[ret_col].fillna(0).to_numpy()
            base_active = baseline_window[active_col].fillna(0).to_numpy()
            base_entry_dates = set(window["day"].iloc[_entries(base_active)].dt.date) if len(base_active) == len(loose_active) else set()

            for entry_idx in _entries(loose_active):
                entry_date = window["day"].iloc[entry_idx].date()
                # "additional" = no baseline entry (this leg) within +/-15 days
                near_baseline = any(abs((entry_date - bd).days) <= 15 for bd in base_entry_dates)
                if not near_baseline:
                    exit_idx, ret = _segment_return(loose_ret, loose_active, entry_idx)
                    exit_date = window["day"].iloc[exit_idx].date()
                    false_start_rows.append({"variant": name, "leg": leg, "entry_date": str(entry_date), "exit_date": str(exit_date), "days_held": exit_idx - entry_idx + 1, "trade_return_pct": ret * 100})

    false_starts_df = pd.DataFrame(false_start_rows, columns=["variant", "leg", "entry_date", "exit_date", "days_held", "trade_return_pct"])
    false_starts_df.to_csv(args.out_false_starts, index=False)
    for name in ["loose_2_3", "loose_1_2"]:
        sub = false_starts_df[false_starts_df["variant"] == name]
        n = len(sub)
        avg_ret = sub["trade_return_pct"].mean() if n else np.nan
        win_rate = (sub["trade_return_pct"] > 0).mean() if n else np.nan
        total_ret = sub["trade_return_pct"].sum() if n else 0.0
        print(f"{name}: {n} additional entries (no baseline counterpart within 15 days)")
        if n:
            print(f"  Win rate: {win_rate*100:.1f}%  Average return: {avg_ret:+.2f}%  Aggregate summed return: {total_ret:+.2f}%")
    print()

    print("=" * 115)
    print("SUMMARY")
    print("=" * 115)
    for _, r in headline_df.iterrows():
        if r["variant"] != "baseline_3_5":
            n_fs = len(false_starts_df[false_starts_df["variant"] == r["variant"]])
            fs_avg = false_starts_df.loc[false_starts_df["variant"] == r["variant"], "trade_return_pct"].mean() if n_fs else np.nan
            verdict = "NET POSITIVE" if r["sharpe"] > baseline_sharpe else "NET NEGATIVE"
            print(f"{r['variant']}: Sharpe {r['sharpe']-baseline_sharpe:+.3f} vs baseline, {n_fs} additional entries (avg return {fs_avg:+.2f}%) -> {verdict}")

    print(f"\nSaved: {args.out_headline}")
    print(f"Saved: {args.out_stretch_entries}")
    print(f"Saved: {args.out_false_starts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
