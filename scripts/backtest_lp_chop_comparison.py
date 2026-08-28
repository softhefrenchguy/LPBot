from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_continuous_regime_integrations import _load_base  # noqa: E402
from backtest_overlay_strategies import _mean_reversion_overlay  # noqa: E402
from backtest_forex_optimised import _stats  # noqa: E402
from backtest_lp_full_stack import (  # noqa: E402
    UNISWAP_V3_LAUNCH, FEE_RATE_ANN, IL_K_FULL_RANGE, LP_VOL_ON, LP_SCALE, LP_WEIGHT_MAX,
    LP_MIN_ON_BARS, LP_COOLDOWN_BARS, compute_lp_overlay,
)

WORK_DIR = Path("artifacts/backtest/forex_optimised")
START, END = "2019-01-01", "2024-12-31"
GROSS_CAP, COST_BPS = 0.8, 20.0


def main() -> int:
    d = _load_base(WORK_DIR, START, END)
    d["eth_close"] = pd.to_numeric(d["eth_close"], errors="coerce")
    d["rolling_vol_20d"] = pd.to_numeric(d["rolling_vol_20d"], errors="coerce")

    combined_no_overlay = d["alloc_eth"] * d["eth_strategy_return"] + d["alloc_btc"] * d["btc_strategy_return"] + d["gold_strategy_return"]

    mr = _mean_reversion_overlay(d, gross_cap=GROSS_CAP, cost_bps=COST_BPS, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr_weight_exec = pd.to_numeric(mr["mr_weight_exec"], errors="coerce").fillna(0.0)
    mr_remaining_cap = pd.to_numeric(mr["mr_remaining_cap"], errors="coerce").fillna(0.0)
    mr_return = pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    mr_z = pd.to_numeric(mr["mr_z_score"], errors="coerce")

    gate = (d["eth_off_active"].fillna(0).astype(bool)) | (d["btc_off_active"].fillna(0).astype(bool))
    pre_v3 = d["day"] < UNISWAP_V3_LAUNCH
    gate_effective = gate | pre_v3
    vol_eligible = d["rolling_vol_20d"] < LP_VOL_ON
    lp_eligible_days = (~gate_effective) & vol_eligible

    def lp_with_weight(weight: pd.Series):
        return compute_lp_overlay(d, weight, FEE_RATE_ANN, IL_K_FULL_RANGE, LP_VOL_ON, LP_SCALE, LP_WEIGHT_MAX,
                                   LP_MIN_ON_BARS, LP_COOLDOWN_BARS, gate_effective)

    print("=" * 115)
    print("1) STANDALONE COMPARISON -- CHOP alone vs LP alone (full capacity, no competitor) vs neither")
    print("=" * 115)

    neither = combined_no_overlay
    st_neither = _stats(neither)

    chop_alone = combined_no_overlay + mr_return
    st_chop = _stats(chop_alone)

    lp_full_weight = lp_eligible_days.astype(float)  # LP alone: no CHOP competing, full 1.0 capacity when eligible
    lp_alone_overlay = lp_with_weight(lp_full_weight)
    lp_alone = combined_no_overlay.reset_index(drop=True) + lp_alone_overlay["overlay_r"].reset_index(drop=True)
    lp_alone.index = combined_no_overlay.index
    st_lp = _stats(lp_alone)

    print(f"{'':40} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'delta_vs_neither':>18}")
    print(f"{'Neither (no CHOP, no LP)':40} {st_neither['sharpe']:>8.3f} {st_neither['cagr']*100:>7.1f}% {st_neither['maxdd']*100:>7.1f}% {'--':>18}")
    print(f"{'CHOP alone (current baseline)':40} {st_chop['sharpe']:>8.3f} {st_chop['cagr']*100:>7.1f}% {st_chop['maxdd']*100:>7.1f}% {st_chop['sharpe']-st_neither['sharpe']:>+18.3f}")
    print(f"{'LP alone (full capacity)':40} {st_lp['sharpe']:>8.3f} {st_lp['cagr']*100:>7.1f}% {st_lp['maxdd']*100:>7.1f}% {st_lp['sharpe']-st_neither['sharpe']:>+18.3f}")
    print()
    chop_standalone_delta = st_chop["sharpe"] - st_neither["sharpe"]
    lp_standalone_delta = st_lp["sharpe"] - st_neither["sharpe"]
    print(f"CHOP's own standalone contribution: {chop_standalone_delta:+.3f} Sharpe")
    print(f"LP's own standalone contribution (full capacity, uncontested): {lp_standalone_delta:+.3f} Sharpe")
    print(f"LP-alone on-days: {int(lp_alone_overlay['lp_on'].sum())}/{len(d)} ({lp_alone_overlay['lp_on'].mean()*100:.1f}%) -- vs {int(lp_eligible_days.sum())} eligible days (full capacity means every eligible day gets max lp_weight, not throttled by CHOP)")
    print()

    print("=" * 115)
    print("2) CAPITAL-SHARING SENSITIVITY -- explicit split rules for the shared leftover capacity")
    print("=" * 115)

    variants = {}

    # (a) CHOP-only, LP excluded entirely -- this IS the current accepted baseline
    variants["a_CHOP_only_LP_excluded (baseline)"] = chop_alone

    # (b) LP-only, CHOP excluded entirely
    variants["b_LP_only_CHOP_excluded"] = lp_alone

    # (c) CHOP-first, LP gets leftover (last turn's "capacity-aware" default)
    lp_cap_leftover = (mr_remaining_cap - mr_weight_exec).clip(lower=0.0).where(lp_eligible_days, 0.0)
    lp_c = lp_with_weight(lp_cap_leftover)
    variants["c_CHOP_first_LP_leftover (prior default)"] = combined_no_overlay.reset_index(drop=True) + mr_return.reset_index(drop=True) + lp_c["overlay_r"].reset_index(drop=True)
    variants["c_CHOP_first_LP_leftover (prior default)"].index = combined_no_overlay.index

    # (d) LP-first, CHOP gets leftover (symmetry check)
    lp_first_weight = lp_eligible_days.astype(float)  # LP takes its full slice first (same as "alone")
    lp_d = lp_with_weight(lp_first_weight)
    lp_weight_used = lp_d["lp_weight"]  # actual capital LP claims once stability/cooldown is applied
    mr_remaining_after_lp = (mr_remaining_cap - lp_weight_used.reset_index(drop=True).clip(lower=0.0)).clip(lower=0.0)
    mr_scale_d = (mr_remaining_after_lp / mr_remaining_cap.reset_index(drop=True)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 1.0)
    mr_return_scaled_d = mr_return.reset_index(drop=True) * mr_scale_d
    variants["d_LP_first_CHOP_leftover"] = combined_no_overlay.reset_index(drop=True) + mr_return_scaled_d + lp_d["overlay_r"].reset_index(drop=True)
    variants["d_LP_first_CHOP_leftover"].index = combined_no_overlay.index

    # (e) CHOP capped at 50% of its current claim, LP gets the rest of the leftover
    mr_weight_half = mr_weight_exec * 0.5
    mr_return_half = mr_return * 0.5  # linear: turnover and cost both scale linearly with a uniform weight scale
    lp_cap_e = (mr_remaining_cap - mr_weight_half).clip(lower=0.0).where(lp_eligible_days, 0.0)
    lp_e = lp_with_weight(lp_cap_e)
    variants["e_CHOP_capped_50pct_LP_rest"] = combined_no_overlay.reset_index(drop=True) + mr_return_half.reset_index(drop=True) + lp_e["overlay_r"].reset_index(drop=True)
    variants["e_CHOP_capped_50pct_LP_rest"].index = combined_no_overlay.index

    # (f) Proportional split by each's own standalone Sharpe contribution
    chop_frac = max(chop_standalone_delta, 0.0)
    lp_frac = max(lp_standalone_delta, 0.0)
    total_frac = chop_frac + lp_frac
    if total_frac > 0:
        chop_share = chop_frac / total_frac
        lp_share = lp_frac / total_frac
    else:
        chop_share, lp_share = 1.0, 0.0
    mr_weight_f = mr_weight_exec * chop_share
    mr_return_f = mr_return * chop_share
    lp_cap_f = (mr_remaining_cap * lp_share).clip(lower=0.0).where(lp_eligible_days, 0.0)
    lp_f = lp_with_weight(lp_cap_f)
    variants[f"f_proportional_by_standalone_Sharpe (CHOP {chop_share*100:.0f}% / LP {lp_share*100:.0f}%)"] = combined_no_overlay.reset_index(drop=True) + mr_return_f.reset_index(drop=True) + lp_f["overlay_r"].reset_index(drop=True)
    variants[f"f_proportional_by_standalone_Sharpe (CHOP {chop_share*100:.0f}% / LP {lp_share*100:.0f}%)"].index = combined_no_overlay.index

    # (g) Day-by-day: whichever trigger condition is "more clearly met" gets that day's leftover capacity
    z_entry_g, lp_vol_on_g = -1.5, LP_VOL_ON
    chop_severity = (z_entry_g - mr_z).clip(lower=0.0) / abs(z_entry_g)  # how far below the entry z-threshold, normalized
    chop_severity = chop_severity.fillna(0.0)
    lp_severity = ((lp_vol_on_g - d["rolling_vol_20d"]) / lp_vol_on_g).clip(lower=0.0).fillna(0.0)
    chop_wins_g = chop_severity >= lp_severity
    mr_weight_g = mr_weight_exec.where(chop_wins_g.reset_index(drop=True).values, 0.0)
    mr_return_g = mr_return.where(chop_wins_g.reset_index(drop=True).values, 0.0)
    lp_cap_g = mr_remaining_cap.where(~chop_wins_g.reset_index(drop=True).values, 0.0).where(lp_eligible_days, 0.0)
    lp_g = lp_with_weight(lp_cap_g)
    variants["g_day_by_day_severity_winner"] = combined_no_overlay.reset_index(drop=True) + mr_return_g.reset_index(drop=True) + lp_g["overlay_r"].reset_index(drop=True)
    variants["g_day_by_day_severity_winner"].index = combined_no_overlay.index

    print(f"{'variant':55} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'vs_a_baseline':>15} {'vs_c_prior':>12}")
    st_a = _stats(variants["a_CHOP_only_LP_excluded (baseline)"])
    st_c = _stats(variants["c_CHOP_first_LP_leftover (prior default)"])
    rows = []
    for name, ret in variants.items():
        st = _stats(ret)
        rows.append({"variant": name, "sharpe": st["sharpe"], "cagr": st["cagr"], "maxdd": st["maxdd"],
                      "delta_vs_a": st["sharpe"] - st_a["sharpe"], "delta_vs_c": st["sharpe"] - st_c["sharpe"]})
        print(f"{name:55} {st['sharpe']:>8.3f} {st['cagr']*100:>7.1f}% {st['maxdd']*100:>7.1f}% {st['sharpe']-st_a['sharpe']:>+15.3f} {st['sharpe']-st_c['sharpe']:>+12.3f}")
    pd.DataFrame(rows).to_csv("artifacts/backtest/lp_chop_capital_sharing_variants.csv", index=False)
    print()
    best = max(rows, key=lambda r: r["sharpe"])
    print(f"Best variant: {best['variant']} (Sharpe {best['sharpe']:.3f})")
    print(f"Beats (a) CHOP-only baseline (1.660)? {best['sharpe'] > st_a['sharpe']}")
    print(f"Beats (c) CHOP-first/LP-leftover (1.652)? {best['sharpe'] > st_c['sharpe']}")
    print()

    print("=" * 115)
    print("3) OVERLAP-DAY INSPECTION -- are CHOP and LP triggering off the same or different conditions?")
    print("=" * 115)
    overlap_days = lp_eligible_days & (mr_weight_exec > 0)
    print(f"Overlap days: {int(overlap_days.sum())} / {int(lp_eligible_days.sum())} LP-eligible days ({(overlap_days.sum()/max(1,lp_eligible_days.sum()))*100:.1f}%)")
    print()
    if overlap_days.sum() > 0:
        detail = pd.DataFrame({
            "day": d["day"][overlap_days.values].dt.date.values,
            "eth_regime": d["eth_regime"][overlap_days.values].values,
            "mr_z_score": mr_z[overlap_days.values].values,
            "mr_daily_ret_pct": (pd.to_numeric(mr["mr_daily_ret"], errors="coerce")[overlap_days.values] * 100).values,
            "rolling_vol_20d": d["rolling_vol_20d"][overlap_days.values].values,
            "lp_vol_on_threshold": LP_VOL_ON,
            "gate_off_active": gate[overlap_days.values].values,
        })
        pd.set_option("display.width", 180)
        pd.set_option("display.max_columns", 20)
        print(detail.to_string(index=False))
        detail.to_csv("artifacts/backtest/lp_chop_overlap_days_detail.csv", index=False)
        print()

    # Correlation of "severity" signals across ALL lp-eligible days (not just the 15 overlap ones) --
    # are the two triggers picking up the same underlying condition (quiet/choppy market) or different ones?
    elig = lp_eligible_days.values
    chop_sev_elig = chop_severity[elig]
    lp_sev_elig = lp_severity[elig]
    valid = chop_sev_elig.notna() & lp_sev_elig.notna()
    corr = float(chop_sev_elig[valid].corr(lp_sev_elig[valid])) if valid.sum() > 2 else float("nan")
    print(f"Correlation of CHOP-severity vs LP-severity across all {int(elig.sum())} LP-eligible days: {corr:.3f}")
    chop_active_frac_on_elig = float((mr_weight_exec[elig] > 0).mean())
    print(f"CHOP overlay is active (mr_weight_exec>0) on {chop_active_frac_on_elig*100:.1f}% of ALL LP-eligible days (not just the 6% overlap on THIS run's specific transitions -- recomputed fresh here).")
    print(f"Of LP-eligible days, {(d['eth_regime'][elig]=='CHOP').mean()*100:.1f}% are also in eth_regime=='CHOP' (LP's gate doesn't require CHOP regime specifically, so this measures true overlap in classification, not just active-capital overlap).")
    print()
    print("Saved: artifacts/backtest/lp_chop_capital_sharing_variants.csv")
    print("Saved: artifacts/backtest/lp_chop_overlap_days_detail.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
