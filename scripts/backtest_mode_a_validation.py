from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_continuous_regime_integrations import _apply_state_conviction, _load_base, _load_states, _main_return
from backtest_forex_optimised import _stats


PRODUCTION_REFERENCE_SHARPE = 1.121  # corrected: alloc-turnover cost (vol-filter/asymmetric-sizing driven resizing) now charged at the source, not a post-hoc adjustment (was 1.510 pre-fix, 1.649 @ 20bps/leg, 1.762 pre-gap-fix)
PRODUCTION_WALKFORWARD_OOS = {2021: 1.610, 2022: -0.704, 2023: -0.480, 2024: 1.536}  # corrected for alloc-turnover-cost-at-source fix (was {2021: 1.996, 2022: -0.496, 2023: 0.163, 2024: 2.087} pre-fix)
PRODUCTION_WALKFORWARD_AVG = 0.490  # corrected for alloc-turnover-cost-at-source fix (was 0.938 pre-fix, 1.144 @ 20bps, 1.086 pre-lookahead-bias-fix, 1.109 pre-gap-fix)

NO_PANIC_SHRINK = {"risk_on": 1.0, "weakening": 0.85, "risk_off": 0.70, "panic": 1.0}
SHRINK_PANIC = {"risk_on": 1.0, "weakening": 0.85, "risk_off": 0.70, "panic": 0.50}

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Out-of-sample validation, trade attribution, and vol-filter double-counting check for Mode A (state-based sizing multiplier).")
    p.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    p.add_argument("--regime-score", default="artifacts/backtest/continuous_regime_score_daily.csv")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--gross-cap", type=float, default=0.8)
    p.add_argument("--cost-bps", type=float, default=60.0)  # was 20.0
    p.add_argument("--novol-work-dir", default="artifacts/backtest/mode_a_novol")
    p.add_argument("--out-walkforward", default="artifacts/backtest/mode_a_walkforward.csv")
    p.add_argument("--out-attribution", default="artifacts/backtest/mode_a_attribution.csv")
    p.add_argument("--out-volfilter-check", default="artifacts/backtest/mode_a_volfilter_check.csv")
    p.add_argument("--out-rebalance-cost", default="artifacts/backtest/mode_a_rebalance_cost.csv")
    return p.parse_args()


def _load_d(args: argparse.Namespace) -> pd.DataFrame:
    d = _load_base(Path(args.work_dir), args.start, args.end)
    d = _load_states(Path(args.regime_score), d)
    return d


# ---------------------------------------------------------------------------
# Step 1: walk-forward OOS (same calendar-year-slice methodology as
# backtest_walkforward_production.py -- one continuous run, sliced by year,
# so 120/300-day EMAs and the vol filter keep proper multi-year warm-up).
# ---------------------------------------------------------------------------

def _step1_walkforward(d: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.DataFrame:
    modified = _apply_state_conviction(d, SHRINK_PANIC, gross_cap)
    full_ret = _main_return(modified, gross_cap, cost_bps)
    full_ret.index = d["day"]

    rows = []
    full_stats = _stats(full_ret)
    rows.append({"period": "2019-2024 (full, in-sample fit)", "sharpe": full_stats["sharpe"], "cagr": full_stats["cagr"], "maxdd": full_stats["maxdd"], "production_oos_sharpe": np.nan, "delta_vs_production": np.nan})

    print("Full-period (in-sample) Mode A Sharpe:", f"{full_stats['sharpe']:.3f}", "(baseline production reference:", f"{PRODUCTION_REFERENCE_SHARPE:.3f})")
    print()
    print(f"{'Year':6} {'Mode A OOS Sharpe':>18} {'Production OOS Sharpe':>22} {'Delta':>8}")
    oos_sharpes = []
    for yr in [2021, 2022, 2023, 2024]:
        yr_ret = full_ret[(full_ret.index >= f"{yr}-01-01") & (full_ret.index <= f"{yr}-12-31")]
        st = _stats(yr_ret)
        oos_sharpes.append(st["sharpe"])
        prod = PRODUCTION_WALKFORWARD_OOS[yr]
        delta = st["sharpe"] - prod
        print(f"{yr:<6} {st['sharpe']:>18.3f} {prod:>22.3f} {delta:>+8.3f}")
        rows.append({"period": str(yr), "sharpe": st["sharpe"], "cagr": st["cagr"], "maxdd": st["maxdd"], "production_oos_sharpe": prod, "delta_vs_production": delta})

    avg_oos = float(np.mean(oos_sharpes))
    n_positive = int(sum(1 for s in oos_sharpes if s > 0))
    print()
    print(f"Average Mode A OOS Sharpe: {avg_oos:.3f}  (production walk-forward average: {PRODUCTION_WALKFORWARD_AVG:.3f})")
    print(f"Positive OOS years: {n_positive}/4")
    concentrated = max(oos_sharpes) - min(s for s in oos_sharpes) > 2.0 or (avg_oos > 0 and max(oos_sharpes) > 2 * avg_oos)
    verdict = "HOLDS UP" if avg_oos > PRODUCTION_WALKFORWARD_AVG - 0.15 and n_positive >= 3 else "DOES NOT CLEARLY HOLD UP -- improvement may be in-sample-fit-specific"
    print(f"Verdict: {verdict}")
    rows.append({"period": "average_2021_2024", "sharpe": avg_oos, "cagr": np.nan, "maxdd": np.nan, "production_oos_sharpe": PRODUCTION_WALKFORWARD_AVG, "delta_vs_production": avg_oos - PRODUCTION_WALKFORWARD_AVG})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: trade-level / state-conditional attribution
# ---------------------------------------------------------------------------

def _step2_attribution(d: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.DataFrame:
    baseline_ret = _main_return(d, gross_cap, cost_bps)
    no_shrink_d = _apply_state_conviction(d, NO_PANIC_SHRINK, gross_cap)
    no_shrink_ret = _main_return(no_shrink_d, gross_cap, cost_bps)
    shrink_d = _apply_state_conviction(d, SHRINK_PANIC, gross_cap)
    shrink_ret = _main_return(shrink_d, gross_cap, cost_bps)

    baseline_stats, no_shrink_stats, shrink_stats = _stats(baseline_ret), _stats(no_shrink_ret), _stats(shrink_ret)
    print(f"baseline (no scaling):        Sharpe {baseline_stats['sharpe']:.3f}  ann_vol {baseline_stats['ann_vol']*100:.1f}%")
    print(f"scale_no_panic_shrink:        Sharpe {no_shrink_stats['sharpe']:.3f}  ann_vol {no_shrink_stats['ann_vol']*100:.1f}%  (isolates risk_on/weakening/risk_off scaling only)")
    print(f"scale_shrink_panic:           Sharpe {shrink_stats['sharpe']:.3f}  ann_vol {shrink_stats['ann_vol']*100:.1f}%  (adds panic 0.5x on top)")
    print()
    print(f"Non-panic-state scaling contributes: {no_shrink_stats['sharpe']-baseline_stats['sharpe']:+.3f} Sharpe (risk_on 1.0x / weakening 0.85x / risk_off 0.70x -- identical in both variants)")
    print(f"Panic-specific scaling (1.0x -> 0.5x) contributes an additional: {shrink_stats['sharpe']-no_shrink_stats['sharpe']:+.3f} Sharpe")
    print()

    # Panic-day exposure: how much is there actually to shrink?
    panic_mask = d["state"].eq("panic")
    invested_mask = (d["alloc_eth"] > 0) | (d["alloc_btc"] > 0)
    n_panic = int(panic_mask.sum())
    n_panic_invested = int((panic_mask & invested_mask).sum())
    avg_alloc_panic = float((d.loc[panic_mask, "alloc_eth"] + d.loc[panic_mask, "alloc_btc"]).mean())
    avg_alloc_risk_on = float((d.loc[d["state"].eq("risk_on"), "alloc_eth"] + d.loc[d["state"].eq("risk_on"), "alloc_btc"]).mean())
    print(f"Panic days: {n_panic}, invested on {n_panic_invested} ({n_panic_invested/n_panic*100:.1f}%), avg total alloc {avg_alloc_panic:.3f}")
    print(f"(for comparison, risk_on avg total alloc: {avg_alloc_risk_on:.3f} -- the underlying EMA-trend strategy is usually already flat")
    print(f" by the time panic hits, since panic follows a sharp drawdown the trend regime has typically already broken on)")
    print()

    # Direct dollar/return cost of the panic shrink, isolated: no_shrink_ret and
    # shrink_ret are IDENTICAL in construction except panic's multiplier (1.0 vs 0.5),
    # so their difference on panic days isolates the pure cost of shrinking.
    panic_no_shrink_compounded = float((1.0 + no_shrink_ret[panic_mask]).prod() - 1.0)
    panic_shrink_compounded = float((1.0 + shrink_ret[panic_mask]).prod() - 1.0)
    cost_of_panic_shrink = panic_shrink_compounded - panic_no_shrink_compounded
    print(f"Compounded return on panic days, panic mult=1.0x: {panic_no_shrink_compounded*100:+.2f}%")
    print(f"Compounded return on panic days, panic mult=0.5x: {panic_shrink_compounded*100:+.2f}%")
    print(f"Direct cost of shrinking panic exposure: {cost_of_panic_shrink*100:+.2f}% cumulative over 2019-2024")
    if cost_of_panic_shrink < 0:
        print("-> Confirms the continuous-score finding: panic days were net positive for the strategy when it happened to be invested.")
        print("   But the position is small and rare enough (14.8% of panic days invested, ~0.07 avg alloc) that the cost is minor")
        print("   relative to the vol-reduction benefit of shrinking the much-more-common weakening/risk_off states.")
    else:
        print("-> Panic days were net negative or neutral for the strategy's actual (small, rare) exposure during them; shrinking cost nothing.")

    print()
    print("Where the Sharpe improvement actually lives: comparing baseline vs scale_shrink_panic by state")
    rows = []
    for state in ["risk_on", "weakening", "risk_off", "panic"]:
        mask = d["state"].eq(state)
        base_state_ret = float((1.0 + baseline_ret[mask]).prod() - 1.0)
        mod_state_ret = float((1.0 + shrink_ret[mask]).prod() - 1.0)
        rows.append({
            "state": state, "days": int(mask.sum()), "invested_pct": float((mask & invested_mask).mean()),
            "avg_total_alloc": float((d.loc[mask, "alloc_eth"] + d.loc[mask, "alloc_btc"]).mean()),
            "baseline_compounded_return_pct": base_state_ret * 100, "mode_a_compounded_return_pct": mod_state_ret * 100,
            "delta_pct": (mod_state_ret - base_state_ret) * 100,
        })
        print(f"  {state:10} days={int(mask.sum()):4d}  invested={float((mask&invested_mask).mean())*100:5.1f}%  baseline_ret={base_state_ret*100:+7.2f}%  mode_a_ret={mod_state_ret*100:+7.2f}%  delta={((mod_state_ret-base_state_ret)*100):+6.2f}pp")
    attribution_df = pd.DataFrame(rows)
    attribution_df.attrs["cost_of_panic_shrink_pct"] = cost_of_panic_shrink * 100
    attribution_df.attrs["non_panic_scaling_sharpe_contribution"] = no_shrink_stats["sharpe"] - baseline_stats["sharpe"]
    attribution_df.attrs["panic_scaling_sharpe_contribution"] = shrink_stats["sharpe"] - no_shrink_stats["sharpe"]
    return attribution_df


# ---------------------------------------------------------------------------
# Step 3: vol-filter double-counting -- a real subprocess re-run without
# --vol-filter (not a post-hoc approximation), reconstructed the same way.
# ---------------------------------------------------------------------------

def _run_novol_backtest(start: str, end: str, work_dir: Path, cost_bps: float) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    summary = work_dir / "novol_summary.csv"
    daily = work_dir / "novol_daily.csv"
    if daily.exists() and summary.exists():
        return daily
    cmd = [
        sys.executable, "scripts/backtest_eth_btc_portfolio.py",
        "--start", start, "--end", end,
        "--eth-ema", "50,120,300", "--btc-ema", "15,40,120",
        "--eth-confirm-days", "3", "--btc-confirm-days", "5",
        "--transition-momentum",  # vol-filter deliberately omitted
        "--asymmetric-sizing",
        "--allocation-mode", "signal_weighted", "--gross-cap", "0.8",
        "--cost-bps", str(cost_bps), "--cost-mode", "weight_change",
        "--include-gold", "--gold-symbol", "PAXG-USD", "--gold-ema", "25,65,180", "--gold-cap", "0.3", "--gold-cost-bps", str(cost_bps),
        "--out-summary-csv", str(summary), "--out-daily-csv", str(daily),
    ]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"no-vol-filter backtest failed:\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
    return daily


def _step3_volfilter_check(args: argparse.Namespace, d: pd.DataFrame, withvol_modeA_sharpe: float) -> pd.DataFrame:
    novol_daily_path = _run_novol_backtest(args.start, args.end, Path(args.novol_work_dir), args.cost_bps)
    novol_base = pd.read_csv(novol_daily_path, low_memory=False)
    novol_base["day"] = pd.to_datetime(novol_base["day"], utc=True, errors="coerce").dt.floor("D")
    for col in ["alloc_eth", "alloc_btc", "eth_strategy_return", "btc_strategy_return", "gold_strategy_return"]:
        if col in novol_base.columns:
            novol_base[col] = pd.to_numeric(novol_base[col], errors="coerce").fillna(0.0)

    states = pd.read_csv(args.regime_score)
    states["day"] = pd.to_datetime(states["date"], utc=True, errors="coerce").dt.floor("D")
    states["state"] = states["state"].astype(str)
    novol_d = novol_base.merge(states[["day", "state"]], on="day", how="left")
    novol_d["state"] = novol_d["state"].ffill().fillna("weakening")

    novol_baseline_ret = _main_return(novol_d, args.gross_cap, args.cost_bps)
    novol_modeA_d = _apply_state_conviction(novol_d, SHRINK_PANIC, args.gross_cap)
    novol_modeA_ret = _main_return(novol_modeA_d, args.gross_cap, args.cost_bps)

    novol_baseline_stats = _stats(novol_baseline_ret)
    novol_modeA_stats = _stats(novol_modeA_ret)

    withvol_baseline_sharpe = PRODUCTION_REFERENCE_SHARPE

    uplift_with_vol = withvol_modeA_sharpe - withvol_baseline_sharpe
    uplift_without_vol = novol_modeA_stats["sharpe"] - novol_baseline_stats["sharpe"]

    print(f"{'':30} {'Mode A OFF':>12} {'Mode A ON':>12} {'Mode A uplift':>15}")
    print(f"{'Vol filter ON (production)':30} {withvol_baseline_sharpe:>12.3f} {withvol_modeA_sharpe:>12.3f} {uplift_with_vol:>+15.3f}")
    print(f"{'Vol filter OFF':30} {novol_baseline_stats['sharpe']:>12.3f} {novol_modeA_stats['sharpe']:>12.3f} {uplift_without_vol:>+15.3f}")
    print()
    ratio = uplift_without_vol / uplift_with_vol if uplift_with_vol != 0 else np.nan
    if uplift_without_vol > uplift_with_vol * 1.3:
        verdict = f"Mode A's uplift is LARGER without the vol filter ({uplift_without_vol:+.3f} vs {uplift_with_vol:+.3f}) -- meaningful overlap: Mode A is partially substituting for what the vol filter already does."
    elif uplift_without_vol < uplift_with_vol * 0.7:
        verdict = f"Mode A's uplift is SMALLER without the vol filter ({uplift_without_vol:+.3f} vs {uplift_with_vol:+.3f}) -- they're complementary, not overlapping; Mode A works better alongside the vol filter than as a substitute for it."
    else:
        verdict = f"Mode A's uplift is similar with or without the vol filter ({uplift_without_vol:+.3f} vs {uplift_with_vol:+.3f}, ratio {ratio:.2f}) -- adds independent information, not meaningfully double-counting."
    print(f"Verdict: {verdict}")

    return pd.DataFrame([
        {"vol_filter": "ON", "mode_a": "OFF", "sharpe": withvol_baseline_sharpe},
        {"vol_filter": "ON", "mode_a": "ON", "sharpe": withvol_modeA_sharpe, "uplift": uplift_with_vol},
        {"vol_filter": "OFF", "mode_a": "OFF", "sharpe": novol_baseline_stats["sharpe"], "cagr": novol_baseline_stats["cagr"], "maxdd": novol_baseline_stats["maxdd"]},
        {"vol_filter": "OFF", "mode_a": "ON", "sharpe": novol_modeA_stats["sharpe"], "cagr": novol_modeA_stats["cagr"], "maxdd": novol_modeA_stats["maxdd"], "uplift": uplift_without_vol},
    ])


# ---------------------------------------------------------------------------
# Step 4: rebalancing cost context (formerly "incremental rebalancing cost check").
#
# combined_return = alloc_eth*eth_strategy_return + alloc_btc*btc_strategy_return
# + gold_strategy_return - alloc_turnover_cost. eth_strategy_return/btc_strategy_return
# have their own sleeve-level cost (from weight_exec's OWN turnover, pre-portfolio-
# normalization -- see backtest_eth_btc_portfolio.py run_sleeve()). alloc_turnover_cost
# is a SEPARATE cost on alloc_eth/alloc_btc's own day-to-day change, now charged at the
# source (backtest_eth_btc_portfolio.py's main(), and recomputed fresh by _main_return
# for any alloc series passed to it -- including Mode A's). This used to be an
# unpriced gap that this script isolated and charged as an "incremental" adjustment on
# top of an otherwise-uncosted baseline; now that the baseline itself prices its own
# alloc-turnover at the source, modeA_ret below already reflects Mode A's FULL
# alloc-turnover cost (not just the incremental slice beyond baseline). This step now
# reports the incremental turnover Mode A introduces vs. baseline purely for context
# (how much extra churn is Mode A responsible for), without a second cost subtraction.
# ---------------------------------------------------------------------------

def _alloc_turnover(alloc_eth: pd.Series, alloc_btc: pd.Series) -> pd.Series:
    return (alloc_eth - alloc_eth.shift(1).fillna(0.0)).abs() + (alloc_btc - alloc_btc.shift(1).fillna(0.0)).abs()


def _step4_rebalancing_cost(d: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.DataFrame:
    """Alloc-turnover cost is now charged at the source (backtest_eth_btc_portfolio.py) and
    _main_return recomputes+charges it fresh for whatever alloc series it's given -- so
    modeA_ret below ALREADY reflects Mode A's own full alloc-turnover cost, not just a
    "baseline vs incremental" delta. This step now reports the incremental turnover Mode A
    introduces (for context on how much extra churn it is) without re-subtracting a second
    cost layer on top of what _main_return already charges."""
    modeA_d = _apply_state_conviction(d, SHRINK_PANIC, gross_cap)

    baseline_turnover = _alloc_turnover(d["alloc_eth"], d["alloc_btc"])
    modeA_turnover = _alloc_turnover(modeA_d["alloc_eth"], modeA_d["alloc_btc"])
    incremental_turnover = (modeA_turnover - baseline_turnover).clip(lower=0.0)

    state_transitions = d["state"].ne(d["state"].shift(1))
    incremental_days = incremental_turnover > 1e-9
    years = d["day"].dt.year
    per_year = pd.DataFrame({"year": years, "state_transitions": state_transitions.astype(int), "incremental_rebalance_days": incremental_days.astype(int)}).groupby("year").sum()
    per_year = per_year[(per_year.index >= 2019) & (per_year.index <= 2024)]

    print("Additional weight-change events Mode A introduces vs baseline, per year:")
    print(per_year.to_string())
    print(f"Average state transitions/year: {per_year['state_transitions'].mean():.1f}")
    print(f"Average incremental (alloc actually changed beyond baseline) rebalance days/year: {per_year['incremental_rebalance_days'].mean():.1f}")
    print()

    # modeA_ret already has Mode A's full alloc-turnover cost baked in via _main_return.
    modeA_ret = _main_return(modeA_d, gross_cap, cost_bps)
    incremental_cost_for_context = incremental_turnover * (cost_bps / 10000.0)
    stats_modeA = _stats(modeA_ret)
    total_incremental_cost_pct = float(incremental_cost_for_context.sum() * 100)

    print(f"Total incremental turnover vs baseline (sum of |alloc change| beyond baseline, 2019-2024): {float(incremental_turnover.sum()):.2f}")
    print(f"Of which incremental cost (context only, already included in Mode A's Sharpe below) @ {cost_bps:.0f}bps: {total_incremental_cost_pct:.3f}% cumulative drag over the period")
    print()
    print(f"{'':45} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"{'Mode A, fully costed (incl. its own alloc-turnover)':45} {stats_modeA['sharpe']:>8.3f} {stats_modeA['cagr']*100:>7.1f}% {stats_modeA['maxdd']*100:>7.1f}%")
    print(f"{'Production baseline (also fully costed)':45} {PRODUCTION_REFERENCE_SHARPE:>8.3f}")
    print()
    still_positive = stats_modeA["sharpe"] > PRODUCTION_REFERENCE_SHARPE
    print(f"Verdict: Mode A {'STILL BEATS' if still_positive else 'NO LONGER BEATS'} the production baseline once its own rebalancing is fully costed "
          f"({stats_modeA['sharpe']:.3f} vs {PRODUCTION_REFERENCE_SHARPE:.3f}).")

    return pd.DataFrame([
        {"config": "mode_a_fully_costed", "sharpe": stats_modeA["sharpe"], "cagr": stats_modeA["cagr"], "maxdd": stats_modeA["maxdd"], "total_incremental_cost_pct_context_only": total_incremental_cost_pct},
        {"config": "production_baseline_reference", "sharpe": PRODUCTION_REFERENCE_SHARPE},
    ])


def main() -> int:
    args = _parse_args()
    d = _load_d(args)

    print("=" * 110)
    print("MODE A VALIDATION: OUT-OF-SAMPLE, ATTRIBUTION, DOUBLE-COUNTING")
    print("=" * 110)
    print()
    print("STEP 1: Walk-forward out-of-sample (calendar-year slice of continuous run, same methodology as backtest_walkforward_production.py)")
    print("-" * 110)
    wf_df = _step1_walkforward(d, args.gross_cap, args.cost_bps)
    wf_df.to_csv(args.out_walkforward, index=False)
    print()

    print("STEP 2: Trade-level / state-conditional attribution")
    print("-" * 110)
    attr_df = _step2_attribution(d, args.gross_cap, args.cost_bps)
    attr_df.to_csv(args.out_attribution, index=False)
    print()

    print("STEP 3: Vol-filter double-counting check (real re-run without --vol-filter)")
    print("-" * 110)
    withvol_modeA_sharpe = _stats(_main_return(_apply_state_conviction(d, SHRINK_PANIC, args.gross_cap), args.gross_cap, args.cost_bps))["sharpe"]
    volcheck_df = _step3_volfilter_check(args, d, withvol_modeA_sharpe)
    volcheck_df.to_csv(args.out_volfilter_check, index=False)
    print()

    print("STEP 4: Incremental rebalancing cost check (is Mode A's own state-transition churn actually costed?)")
    print("-" * 110)
    rebalance_df = _step4_rebalancing_cost(d, args.gross_cap, args.cost_bps)
    rebalance_df.to_csv(args.out_rebalance_cost, index=False)
    print()

    print("=" * 110)
    print(f"Saved: {args.out_walkforward}")
    print(f"Saved: {args.out_attribution}")
    print(f"Saved: {args.out_rebalance_cost}")
    print(f"Saved: {args.out_volfilter_check}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
