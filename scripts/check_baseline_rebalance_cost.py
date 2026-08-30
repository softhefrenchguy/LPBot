from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_continuous_regime_integrations import _load_base, _main_return
from backtest_mode_a_validation import _alloc_turnover
from backtest_forex_optimised import _stats


PRODUCTION_REFERENCE_SHARPE = 1.510  # PRE-alloc-turnover-cost-fix headline, kept here only as a
# before/after comparison point for this verification check -- NOT the current accepted reference.
# See backtest_walkforward_production.py / README.md for the current one.


def main() -> int:
    ap = argparse.ArgumentParser(description="VERIFICATION check (not a cost-gap adjustment anymore): the alloc_eth/alloc_btc turnover cost gap this script used to quantify and apply as a post-hoc adjustment is now fixed at the source in backtest_eth_btc_portfolio.py (alloc_turnover_cost is computed and subtracted from combined_return directly, and _main_return recomputes+subtracts it too). This script now just confirms that fix actually closes the gap it used to report.")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--cost-bps", type=float, default=60.0)
    ap.add_argument("--out", default="artifacts/backtest/baseline_rebalance_cost_check.csv")
    args = ap.parse_args()

    d = _load_base(Path(args.work_dir), args.start, args.end)

    # The NEW, properly-costed baseline -- alloc_turnover_cost is now built into this.
    ret_fixed = _main_return(d, args.gross_cap, args.cost_bps)
    stats_fixed = _stats(ret_fixed)

    # Reconstruct what the OLD (pre-fix) "uncosted" reconstruction would have been, by adding
    # the alloc-turnover cost back -- this reproduces the exact number this script used to
    # report as "uncosted", to confirm the fix's magnitude matches what was previously measured.
    turnover = _alloc_turnover(d["alloc_eth"], d["alloc_btc"])
    rebalance_days = turnover > 1e-9
    years = d["day"].dt.year
    per_year = pd.DataFrame({"year": years, "rebalance_days": rebalance_days.astype(int)}).groupby("year").sum()
    alloc_turnover_cost = turnover * (args.cost_bps / 10000.0)
    ret_old_uncosted = ret_fixed + alloc_turnover_cost.values
    stats_old_uncosted = _stats(ret_old_uncosted)
    total_cost_pct = float(alloc_turnover_cost.sum() * 100)

    print("=" * 100)
    print("BASELINE ALLOC-TURNOVER COST -- FIX VERIFICATION (previously a gap-quantification check)")
    print("=" * 100)
    print("alloc_eth/alloc_btc's own day-to-day change (driven by the vol filter's vol_multiplier and")
    print("asymmetric sizing's conviction multiplier, both of which vary continuously, not just by the")
    print("underlying weight_exec trend signal) is now charged a transaction cost at the source in")
    print("backtest_eth_btc_portfolio.py, instead of being a silent gap discovered only by this check.")
    print()
    print("Alloc-turnover rebalance days per year (alloc_eth or alloc_btc actually changed, day to day):")
    print(per_year.to_string())
    print(f"Average: {per_year['rebalance_days'].mean():.1f} days/year")
    print()
    print(f"Total alloc-turnover cost now charged @ {args.cost_bps:.0f}bps: {total_cost_pct:.3f}% cumulative drag over {args.start[:4]}-{args.end[:4]}")
    print()
    print(f"{'':45} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"{'Baseline, WITHOUT alloc-turnover cost (old, pre-fix)':45} {stats_old_uncosted['sharpe']:>8.3f} {stats_old_uncosted['cagr']*100:>7.1f}% {stats_old_uncosted['maxdd']*100:>7.1f}%")
    print(f"{'Baseline, WITH alloc-turnover cost (current, fixed at source)':45} {stats_fixed['sharpe']:>8.3f} {stats_fixed['cagr']*100:>7.1f}% {stats_fixed['maxdd']*100:>7.1f}%")
    print()
    print(f"Pre-fix headline (for reference only, now superseded): {PRODUCTION_REFERENCE_SHARPE:.3f}")
    print(f"Verification: reconstructed pre-fix Sharpe = {stats_old_uncosted['sharpe']:.3f} (should match the pre-fix headline above)")
    print(f"Current, properly-costed Sharpe: {stats_fixed['sharpe']:.3f}  (delta vs pre-fix {stats_fixed['sharpe']-PRODUCTION_REFERENCE_SHARPE:+.3f})")
    print("=" * 100)

    pd.DataFrame([
        {"config": "pre_fix_without_alloc_turnover_cost", "sharpe": stats_old_uncosted["sharpe"], "cagr": stats_old_uncosted["cagr"], "maxdd": stats_old_uncosted["maxdd"]},
        {"config": "current_with_alloc_turnover_cost_at_source", "sharpe": stats_fixed["sharpe"], "cagr": stats_fixed["cagr"], "maxdd": stats_fixed["maxdd"], "total_cost_pct": total_cost_pct},
    ]).to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
