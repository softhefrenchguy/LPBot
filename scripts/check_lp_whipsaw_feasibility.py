from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNISWAP_V3_LAUNCH = pd.Timestamp("2021-05-05", tz="UTC")

# Fee APY scenarios, anchored to real (live-checked) DeFiLlama data for the main
# Ethereum Uniswap v3 USDC-WETH pool: apyBase (24h) ~11%, apyBase7d ~28% on the
# largest ($107M TVL) pool at time of check. Conservative/typical/aggressive
# scenarios span a defensible range around that, not fabricated precision.
FEE_APY_SCENARIOS = {"conservative_8pct": 0.08, "typical_15pct": 0.15, "aggressive_25pct": 0.25}

# IL concentration scenarios: il_k=0.125 is the exact small-move Taylor constant for
# a FULL-RANGE (Uniswap v2-equivalent) position (IL ~= -r^2/8). A concentrated v3
# position amplifies IL roughly in proportion to how much narrower the range is than
# full range; 0.5 (~4x full-range) matches a moderately tight range, consistent with
# the il_k values already swept in lpbot/overlays/lp_overlay_v1's cached artifacts
# (artifacts/lp_overlay_backtest_k_*.csv tested 0.5-3.0).
IL_K_SCENARIOS = {"full_range": 0.125, "moderate_concentration": 0.5}


def _exact_v2_il(price_ratio: float) -> float:
    """Exact constant-product IL formula (not the small-move approximation) -- more
    accurate for stretches with large net price moves."""
    if price_ratio <= 0 or not np.isfinite(price_ratio):
        return np.nan
    return 2.0 * np.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate LP fee income vs impermanent loss for each failed_trend_whipsaw stretch, using each stretch's actual net entry-to-exit price move.")
    ap.add_argument("--shape-csv", default="artifacts/backtest/flat_stretch_price_shape.csv")
    ap.add_argument("--out", default="artifacts/backtest/lp_whipsaw_feasibility.csv")
    args = ap.parse_args()

    shapes = pd.read_csv(args.shape_csv)
    whipsaw = shapes[shapes["category"] == "failed_trend_whipsaw"].copy()
    whipsaw["start"] = pd.to_datetime(whipsaw["start"], utc=True)
    whipsaw["end"] = pd.to_datetime(whipsaw["end"], utc=True)

    print("=" * 130)
    print("LP FEE vs IMPERMANENT LOSS -- FAILED-TREND-WHIPSAW STRETCHES")
    print("Fee APY scenarios anchored to live-checked DeFiLlama data (Ethereum USDC-WETH v3, largest pool: apyBase ~11%, apyBase7d ~28%)")
    print("IL: exact constant-product formula on each stretch's ACTUAL net entry-to-exit price move (not internal swings)")
    print("=" * 130)

    rows = []
    for _, r in whipsaw.iterrows():
        length_days = int(r["length_days"])
        years = length_days / 365.25
        net_pct = float(r["eth_net_pct"])
        price_ratio = 1.0 + net_pct / 100.0
        pre_v3 = bool(r["start"] < UNISWAP_V3_LAUNCH)

        il_full = _exact_v2_il(price_ratio)
        il_conc = il_full * (IL_K_SCENARIOS["moderate_concentration"] / IL_K_SCENARIOS["full_range"])

        print(f"\n{r['start'].date()} -> {r['end'].date()} ({length_days}d, ETH net {net_pct:+.1f}%){' [PRE-UNISWAP-V3, Sep2016-style v2 pool only]' if pre_v3 else ''}")
        print(f"  IL (exact, full-range/v2-equivalent): {il_full*100:.2f}%   IL (moderately concentrated, ~4x amplified): {il_conc*100:.2f}%")
        for scen_name, apy in FEE_APY_SCENARIOS.items():
            fee_income_full = apy * years
            fee_income_conc = apy * years * (IL_K_SCENARIOS["moderate_concentration"] / IL_K_SCENARIOS["full_range"]) ** 0.5
            # concentrated positions earn proportionally more fee too (same capital,
            # narrower range = higher effective liquidity density at the trading price),
            # approximated here as scaling with sqrt of the IL amplification factor --
            # a standard rule-of-thumb relationship, not an exact identity.
            net_full = fee_income_full + il_full
            net_conc = fee_income_conc + il_conc
            print(f"  @ {scen_name:20} fee APY {apy*100:>4.0f}%: full-range net {net_full*100:+.2f}%   concentrated net {net_conc*100:+.2f}%")
            rows.append({
                "start": str(r["start"].date()), "end": str(r["end"].date()), "length_days": length_days, "eth_net_pct": net_pct, "pre_uniswap_v3": pre_v3,
                "fee_scenario": scen_name, "fee_apy": apy,
                "il_full_range_pct": il_full * 100, "fee_income_full_range_pct": fee_income_full * 100, "net_full_range_pct": net_full * 100,
                "il_concentrated_pct": il_conc * 100, "fee_income_concentrated_pct": fee_income_conc * 100, "net_concentrated_pct": net_conc * 100,
                "trend_strategy_return_pct": 0.0,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print()
    print("=" * 130)
    print("SUMMARY (net LP return vs trend strategy's ~0% during these stretches)")
    print("=" * 130)
    for scen_name in FEE_APY_SCENARIOS:
        sub = out_df[out_df["fee_scenario"] == scen_name]
        print(f"{scen_name:20}  full-range avg net: {sub['net_full_range_pct'].mean():+.2f}%   concentrated avg net: {sub['net_concentrated_pct'].mean():+.2f}%   "
              f"(range: full {sub['net_full_range_pct'].min():+.1f}% to {sub['net_full_range_pct'].max():+.1f}%)")
    n_negative_full_typical = int((out_df[out_df["fee_scenario"] == "typical_15pct"]["net_full_range_pct"] < 0).sum())
    n_total = int((out_df["fee_scenario"] == "typical_15pct").sum())
    print(f"\nAt the 'typical 15% APY' scenario, full-range LP would have been net negative in {n_negative_full_typical}/{n_total} whipsaw stretches.")

    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
