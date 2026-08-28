from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_continuous_regime_integrations import _load_base, _main_return
from backtest_mode_a_validation import _alloc_turnover
from backtest_forex_optimised import _stats


PRODUCTION_REFERENCE_SHARPE = 1.660  # corrected uncosted headline (was 1.762 before the Sep2022-Mar2023 Binance USDC-pair data-gap fix)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantify the alloc_eth/alloc_btc turnover cost gap in the PRODUCTION BASELINE itself (not Mode A) -- same gap, applied to vol-filter/asymmetric-sizing driven rebalancing that already exists in the accepted 1.660 reference.")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out", default="artifacts/backtest/baseline_rebalance_cost_check.csv")
    args = ap.parse_args()

    d = _load_base(Path(args.work_dir), args.start, args.end)
    ret_uncosted = _main_return(d, args.gross_cap, args.cost_bps)
    stats_uncosted = _stats(ret_uncosted)

    turnover = _alloc_turnover(d["alloc_eth"], d["alloc_btc"])
    rebalance_days = turnover > 1e-9
    years = d["day"].dt.year
    per_year = pd.DataFrame({"year": years, "rebalance_days": rebalance_days.astype(int)}).groupby("year").sum()

    additional_cost = turnover * (args.cost_bps / 10000.0)
    ret_costed = ret_uncosted - additional_cost.values
    stats_costed = _stats(ret_costed)
    total_cost_pct = float(additional_cost.sum() * 100)

    print("=" * 100)
    print("BASELINE ALLOC-TURNOVER COST GAP CHECK")
    print("=" * 100)
    print("Same gap as Mode A's: alloc_eth/alloc_btc's own day-to-day change (driven here by the vol filter's")
    print("vol_multiplier and asymmetric sizing's conviction multiplier, both of which vary continuously, not just")
    print("by the underlying weight_exec trend signal) is never charged a transaction cost in the production pipeline.")
    print("This is pre-existing and affects the accepted 1.660 reference itself, not something Mode A introduced.")
    print()
    print("Uncosted rebalance days per year (alloc_eth or alloc_btc actually changed, day to day):")
    print(per_year.to_string())
    print(f"Average: {per_year['rebalance_days'].mean():.1f} days/year")
    print()
    print(f"Total additional cost if fully priced @ {args.cost_bps:.0f}bps: {total_cost_pct:.3f}% cumulative drag over {args.start[:4]}-{args.end[:4]}")
    print()
    print(f"{'':45} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"{'Baseline, as currently reported (uncosted)':45} {stats_uncosted['sharpe']:>8.3f} {stats_uncosted['cagr']*100:>7.1f}% {stats_uncosted['maxdd']*100:>7.1f}%")
    print(f"{'Baseline, with alloc-turnover fully costed':45} {stats_costed['sharpe']:>8.3f} {stats_costed['cagr']*100:>7.1f}% {stats_costed['maxdd']*100:>7.1f}%")
    print()
    print(f"Reference (accepted, unchanged) headline: {PRODUCTION_REFERENCE_SHARPE:.3f}")
    print(f"True (fully-costed) baseline: {stats_costed['sharpe']:.3f}  (delta {stats_costed['sharpe']-PRODUCTION_REFERENCE_SHARPE:+.3f})")
    print("=" * 100)

    pd.DataFrame([
        {"config": "baseline_uncosted_alloc_turnover", "sharpe": stats_uncosted["sharpe"], "cagr": stats_uncosted["cagr"], "maxdd": stats_uncosted["maxdd"]},
        {"config": "baseline_fully_costed_alloc_turnover", "sharpe": stats_costed["sharpe"], "cagr": stats_costed["cagr"], "maxdd": stats_costed["maxdd"], "total_cost_pct": total_cost_pct},
    ]).to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
