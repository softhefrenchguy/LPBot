from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_continuous_regime_integrations import _load_base, _main_return  # noqa: E402
from backtest_overlay_strategies import _mean_reversion_overlay  # noqa: E402
from backtest_forex_optimised import _stats  # noqa: E402


PRODUCTION_REFERENCE_SHARPE_UNCOSTED = 1.660
PRODUCTION_REFERENCE_SHARPE_COSTED = 1.528

UNISWAP_V3_LAUNCH = pd.Timestamp("2021-05-05", tz="UTC")
REAL_FEE_DATA_START = pd.Timestamp("2024-04-01", tz="UTC")  # subgraph volumeUSD coverage begins here (verified this session)

# Fee rate: full-range, computed from REAL Graph subgraph data (mainnet WETH/USDC 0.05%, Apr-Dec 2024).
# Used for BOTH the verified (2024-04+) period AND as the best-available anchor for the unverified
# 2019-2023 period, since it's the closest verified data point chronologically -- NOT because it's
# assumed representative of those years. Real 2025-2026 data (2.0-6.7% ann.) is LOWER, suggesting a
# declining-yield trend over time, which if anything argues 2019-2023 could have been HIGHER than
# this anchor, not lower -- so this is a conservative-to-neutral choice, not an optimistic one.
FEE_RATE_ANN = 0.1030
IL_K_FULL_RANGE = 0.125  # exact full-range Taylor constant (bot_20 now defaults to full-range mints)

LP_VOL_ON = 0.57      # recalibrated this session (was 0.25, miscalibrated)
LP_SCALE = 0.5
LP_WEIGHT_MAX = 0.5
LP_MIN_ON_BARS = 6
LP_COOLDOWN_BARS = 12

SLIPPAGE_BPS = 15.0            # matches bot_20's SWAP_SLIPPAGE_BPS standard
GAS_UNITS_ROUND_TRIP = 1_000_000  # full-range mint (2 approvals + mint) + exit (collect+decrease+collect+burn), mainnet
GAS_PRICE_GWEI = 20.0           # "typical", not peak-congestion, mainnet assumption -- flagged as a simplification
LP_REFERENCE_POSITION_USD = 100_000.0  # assumed notional LP capital at full weight, for converting gas $ into a bps drag


def _build_lp_on(raw_on: pd.Series, stable_on: pd.Series, cooldown_bars: int) -> pd.Series:
    """Same stability/cooldown state machine as lp_overlay_v1.overlay._build_lp_on."""
    lp_on = np.zeros(len(raw_on), dtype=bool)
    cooldown = 0
    is_on = False
    for i in range(len(raw_on)):
        raw = bool(raw_on.iat[i])
        stable = bool(stable_on.iat[i])
        if is_on and not raw:
            is_on = False
            cooldown = cooldown_bars
        if cooldown > 0:
            cooldown -= 1
            is_on = False
        elif not is_on and stable:
            is_on = True
        lp_on[i] = is_on
    return pd.Series(lp_on, index=raw_on.index)


def compute_lp_overlay(d: pd.DataFrame, weight: pd.Series, fee_rate_ann: float, il_k: float,
                        lp_vol_on: float, lp_scale: float, lp_weight_max: float,
                        lp_min_on_bars: int, lp_cooldown_bars: int, gate: pd.Series) -> pd.DataFrame:
    """Same formula as lp_overlay_v1.compute_lp_overlay_returns (daily bar), but self-contained
    here so `weight` (capacity, after the CHOP overlay's own claim) stays a per-day Series with
    no library changes needed."""
    close = pd.to_numeric(d["eth_close"], errors="coerce")
    r = np.log(close / close.shift(1))
    sigma = pd.to_numeric(d["rolling_vol_20d"], errors="coerce")

    raw_on = (~gate.astype(bool)) & (weight > 0) & (sigma < lp_vol_on)
    stable_on = raw_on.rolling(lp_min_on_bars, min_periods=lp_min_on_bars).min().fillna(False).astype(bool)
    lp_on = _build_lp_on(raw_on, stable_on, lp_cooldown_bars)

    lp_weight = (weight * lp_scale).clip(lower=0.0, upper=lp_weight_max) * lp_on.astype(float)
    fee_per_day = fee_rate_ann / 365.25
    lp_weight_lag = lp_weight.shift(1).fillna(0.0)
    fee_r = lp_weight_lag * fee_per_day
    il_r = -lp_weight_lag * il_k * (r ** 2)

    # Transition (gas + slippage) cost: charged once on the day LP turns on (entry) and once on
    # the day it turns off (exit), sized as a bps-of-position drag from a real mainnet gas estimate.
    eth_price = close
    gas_cost_usd = GAS_UNITS_ROUND_TRIP * GAS_PRICE_GWEI * 1e-9 * eth_price
    gas_bps_of_position = gas_cost_usd / LP_REFERENCE_POSITION_USD
    turn_on = lp_on & ~lp_on.shift(1, fill_value=False)
    turn_off = ~lp_on & lp_on.shift(1, fill_value=False)
    transition_cost = pd.Series(0.0, index=d.index)
    transition_cost[turn_on] = (gas_bps_of_position[turn_on] / 2.0) + (SLIPPAGE_BPS / 10000.0)
    transition_cost[turn_off] = (gas_bps_of_position[turn_off] / 2.0) + (SLIPPAGE_BPS / 10000.0)

    overlay_r = fee_r + il_r - transition_cost

    out = d[["day"]].copy()
    out["lp_on"] = lp_on
    out["lp_weight"] = lp_weight
    out["fee_r"] = fee_r
    out["il_r"] = il_r
    out["transition_cost"] = transition_cost
    out["overlay_r"] = overlay_r
    out["turn_on"] = turn_on
    out["turn_off"] = turn_off
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LP integrated into the actual portfolio backtest as a genuine mode -- not an isolated whipsaw-stretch comparison. Substitutes LP fee/IL return for idle cash on gate+vol-eligible flat days, respecting the mean-reversion (CHOP) overlay's own capital claim so the two don't double-count the same idle capacity.")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out-daily", default="artifacts/backtest/lp_full_stack_daily.csv")
    ap.add_argument("--out-yearly", default="artifacts/backtest/lp_full_stack_yearly.csv")
    args = ap.parse_args()

    d = _load_base(Path(args.work_dir), args.start, args.end)
    d["eth_close"] = pd.to_numeric(d["eth_close"], errors="coerce")
    d["rolling_vol_20d"] = pd.to_numeric(d["rolling_vol_20d"], errors="coerce")

    baseline_ret = _main_return(d, args.gross_cap, args.cost_bps)
    baseline_stats = _stats(baseline_ret)

    print("=" * 115)
    print("LP FULL-STACK INTEGRATION -- baseline vs baseline+LP, same corrected 2019-2024 data")
    print("=" * 115)
    print(f"Baseline (this run, corrected data): Sharpe={baseline_stats['sharpe']:.3f}  CAGR={baseline_stats['cagr']*100:.1f}%  MaxDD={baseline_stats['maxdd']*100:.1f}%")
    print(f"Production reference (accepted): uncosted={PRODUCTION_REFERENCE_SHARPE_UNCOSTED:.3f}  fully-costed={PRODUCTION_REFERENCE_SHARPE_COSTED:.3f}")
    print()

    # --- Capacity accounting: respect the mean-reversion (CHOP) overlay's own claim on idle capital ---
    mr = _mean_reversion_overlay(d, gross_cap=args.gross_cap, cost_bps=args.cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr_weight_exec = pd.to_numeric(mr["mr_weight_exec"], errors="coerce").fillna(0.0)
    mr_remaining_cap = pd.to_numeric(mr["mr_remaining_cap"], errors="coerce").fillna(0.0)
    lp_capacity_shared = (mr_remaining_cap - mr_weight_exec).clip(lower=0.0)
    lp_capacity_naive = pd.Series(1.0, index=d.index)  # what capacity would look like if LP ignored the CHOP overlay entirely

    gate = (d["eth_off_active"].fillna(0).astype(bool)) | (d["btc_off_active"].fillna(0).astype(bool))
    pre_v3 = d["day"] < UNISWAP_V3_LAUNCH
    gate_effective = gate | pre_v3

    vol_eligible = d["rolling_vol_20d"] < LP_VOL_ON
    lp_eligible_days = (~gate_effective) & vol_eligible
    overlap_days = lp_eligible_days & (mr_weight_exec > 0)
    print("INTERACTION-EFFECT CHECK (LP eligibility vs mean-reversion/CHOP overlay's own idle-capital claim):")
    print(f"  LP-eligible days (gate off, vol OK, v3 exists): {int(lp_eligible_days.sum())}/{len(d)} ({lp_eligible_days.mean()*100:.1f}%)")
    print(f"  Of those, days the CHOP overlay ALSO already claims capital: {int(overlap_days.sum())} ({(overlap_days.sum()/max(1,lp_eligible_days.sum()))*100:.1f}% of LP-eligible days)")
    if overlap_days.sum() > 0:
        avg_cap_used_by_mr = float(mr_weight_exec[overlap_days].mean())
        print(f"  On overlapping days, CHOP overlay claims avg {avg_cap_used_by_mr*100:.1f}% of gross_cap -- naive LP sizing (ignoring this) would double-count that capital.")
    print("  Mode A (state sizing multiplier) check: it only rescales alloc_eth/alloc_btc where alloc>0 already --")
    print("  structurally disjoint from LP-eligible days (both alloc_eth=alloc_btc=0 there), confirmed on this data:")
    mode_a_active_mask = (d["alloc_eth"] > 0) | (d["alloc_btc"] > 0)
    print(f"  Mode A's own active mask overlaps LP-eligible days on {int((mode_a_active_mask & lp_eligible_days).sum())} days (expect 0).")
    print()

    results = {}
    for label, weight in [
        ("naive (ignores CHOP overlay's capital claim)", lp_capacity_naive.where(lp_eligible_days, 0.0)),
        ("capacity-aware (respects CHOP overlay's claim)", lp_capacity_shared.where(lp_eligible_days, 0.0)),
    ]:
        lp = compute_lp_overlay(d, weight, FEE_RATE_ANN, IL_K_FULL_RANGE, LP_VOL_ON, LP_SCALE, LP_WEIGHT_MAX,
                                 LP_MIN_ON_BARS, LP_COOLDOWN_BARS, gate_effective)
        combined = baseline_ret.reset_index(drop=True) + lp["overlay_r"].reset_index(drop=True)
        combined.index = baseline_ret.index
        st = _stats(combined)
        results[label] = {"stats": st, "lp": lp, "combined": combined}
        print(f"[{label}]")
        print(f"  LP on: {int(lp['lp_on'].sum())}/{len(lp)} days ({lp['lp_on'].mean()*100:.1f}%) | transitions: {int(lp['turn_on'].sum())}")
        print(f"  Cumulative: fee={lp['fee_r'].sum()*100:.2f}%  IL={lp['il_r'].sum()*100:.2f}%  transition_cost={-lp['transition_cost'].sum()*100:.2f}%  net_overlay={lp['overlay_r'].sum()*100:.2f}%")
        print(f"  Full-stack (baseline+LP): Sharpe={st['sharpe']:.3f} (delta {st['sharpe']-baseline_stats['sharpe']:+.3f})  CAGR={st['cagr']*100:.1f}%  MaxDD={st['maxdd']*100:.1f}% (delta {(st['maxdd']-baseline_stats['maxdd'])*100:+.1f}pp)")
        print()

    chosen = results["capacity-aware (respects CHOP overlay's claim)"]
    lp = chosen["lp"]
    combined = chosen["combined"]

    print("=" * 115)
    print("REAL vs ASSUMED FEE-DATA CONFIDENCE SPLIT")
    print("=" * 115)
    real_mask = d["day"] >= REAL_FEE_DATA_START
    lp_on_real = lp.loc[real_mask.reset_index(drop=True).values, "lp_on"]
    lp_on_assumed = lp.loc[(~real_mask).reset_index(drop=True).values, "lp_on"]
    fee_real = lp.loc[real_mask.reset_index(drop=True).values, "fee_r"].sum()
    fee_assumed = lp.loc[(~real_mask).reset_index(drop=True).values, "fee_r"].sum()
    print(f"Days on real Graph fee data (>= {REAL_FEE_DATA_START.date()}): {int(real_mask.sum())}/{len(d)} ({real_mask.mean()*100:.1f}% of backtest)")
    print(f"  LP-on days in that window: {int(lp_on_real.sum())}, cumulative fee contribution: {fee_real*100:.3f}%")
    print(f"Days on the assumed/extrapolated fee rate (2019 to {REAL_FEE_DATA_START.date()}): {int((~real_mask).sum())}/{len(d)}")
    print(f"  LP-on days in that window: {int(lp_on_assumed.sum())}, cumulative fee contribution: {fee_assumed*100:.3f}%")
    print(f"  -> {(fee_assumed/(fee_real+fee_assumed)*100 if (fee_real+fee_assumed)!=0 else 0):.1f}% of LP's total fee return in this backtest rests on the UNVERIFIED assumption, not real data.")
    print()

    print("=" * 115)
    print("DRAWDOWN CHARACTER: is LP's own MaxDD contribution a materially different kind of risk?")
    print("=" * 115)
    combined_eq = (1.0 + combined.fillna(0.0)).cumprod()
    combined_dd = combined_eq / combined_eq.cummax() - 1.0
    worst_dd_idx = combined_dd.idxmin()
    worst_dd_date = d["day"].iloc[worst_dd_idx] if worst_dd_idx < len(d) else None
    baseline_eq = (1.0 + baseline_ret.fillna(0.0)).cumprod()
    baseline_dd = baseline_eq / baseline_eq.cummax() - 1.0
    print(f"Full-stack worst drawdown: {combined_dd.min()*100:.1f}% (around {worst_dd_date})")
    print(f"Baseline worst drawdown:   {baseline_dd.min()*100:.1f}%")
    # Contribution of LP's own IL specifically during the worst full-stack drawdown's build-up window
    dd_start = max(0, worst_dd_idx - 30)
    il_during_worst_dd = lp["il_r"].iloc[dd_start:worst_dd_idx + 1].sum()
    fee_during_worst_dd = lp["fee_r"].iloc[dd_start:worst_dd_idx + 1].sum()
    print(f"LP's own fee/IL contribution in the 30 days leading into that drawdown: fee={fee_during_worst_dd*100:.2f}%  IL={il_during_worst_dd*100:.2f}%")
    print()

    print("=" * 115)
    print("WALK-FORWARD: year by year")
    print("=" * 115)
    d_years = pd.to_datetime(d["day"]).dt.year
    rows = []
    for yr in sorted(d_years.unique()):
        mask = (d_years == yr).to_numpy()
        base_yr = baseline_ret[mask]
        comb_yr = combined[mask]
        real_frac = float(real_mask[mask].mean())
        st_base = _stats(base_yr) if len(base_yr) > 20 else {"sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan}
        st_comb = _stats(comb_yr) if len(comb_yr) > 20 else {"sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan}
        confidence = "REAL DATA" if real_frac >= 0.99 else ("PARTIAL (real from Apr)" if 0 < real_frac < 0.99 else "UNTESTABLE -- assumed fee rate only, no real volume data this year")
        rows.append({
            "year": yr, "baseline_sharpe": st_base["sharpe"], "fullstack_sharpe": st_comb["sharpe"],
            "delta_sharpe": (st_comb["sharpe"] - st_base["sharpe"]) if pd.notna(st_comb["sharpe"]) and pd.notna(st_base["sharpe"]) else np.nan,
            "baseline_maxdd_pct": st_base["maxdd"] * 100 if pd.notna(st_base["maxdd"]) else np.nan,
            "fullstack_maxdd_pct": st_comb["maxdd"] * 100 if pd.notna(st_comb["maxdd"]) else np.nan,
            "lp_on_days": int(lp["lp_on"][mask].sum()),
            "fee_data_confidence": confidence,
        })
    yearly = pd.DataFrame(rows)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print(yearly.to_string(index=False))
    yearly.to_csv(args.out_yearly, index=False)
    print()

    out_daily = d[["day"]].copy()
    out_daily["baseline_return"] = baseline_ret.values
    out_daily["fullstack_return"] = combined.values
    out_daily["lp_on"] = lp["lp_on"].values
    out_daily["lp_fee_r"] = lp["fee_r"].values
    out_daily["lp_il_r"] = lp["il_r"].values
    out_daily["lp_transition_cost"] = lp["transition_cost"].values
    out_daily.to_csv(args.out_daily, index=False)

    print("=" * 115)
    print("HEADLINE SUMMARY")
    print("=" * 115)
    st_full = results["capacity-aware (respects CHOP overlay's claim)"]["stats"]
    print(f"{'':45} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"{'Baseline (corrected data, this run)':45} {baseline_stats['sharpe']:>8.3f} {baseline_stats['cagr']*100:>7.1f}% {baseline_stats['maxdd']*100:>7.1f}%")
    print(f"{'Full-stack w/ LP (capacity-aware)':45} {st_full['sharpe']:>8.3f} {st_full['cagr']*100:>7.1f}% {st_full['maxdd']*100:>7.1f}%")
    print(f"{'Delta':45} {st_full['sharpe']-baseline_stats['sharpe']:>+8.3f} {(st_full['cagr']-baseline_stats['cagr'])*100:>+7.1f}% {(st_full['maxdd']-baseline_stats['maxdd'])*100:>+7.1f}pp")
    print()
    print(f"Saved: {args.out_daily}")
    print(f"Saved: {args.out_yearly}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
