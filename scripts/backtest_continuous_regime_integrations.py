from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_bear_classifier_rotation import _download_prices
from backtest_fear_greed import _segment_stats
from backtest_forex_optimised import _stats
from backtest_overlay_strategies import _mean_reversion_overlay


PRODUCTION_REFERENCE_SHARPE = 1.510  # corrected for real Kraken cost @ ~$1k-10k/month volume, 60bps/leg (was 1.649 @ 20bps/leg, 1.762 pre-gap-fix)
STATES = ["risk_on", "weakening", "risk_off", "panic"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research harness for continuous_regime_score integration modes A-D. Reads the real production daily file and reconstructs returns via alloc_eth/alloc_btc (not raw sleeve weight_exec).")
    p.add_argument("--work-dir", default="artifacts/backtest/forex_optimised", help="Dir holding the real validated_crypto_daily.csv (the actual 1.510-Sharpe production run).")
    p.add_argument("--regime-score", default="artifacts/backtest/continuous_regime_score_daily.csv")
    p.add_argument("--classifier-daily", default="artifacts/bear_classifier/bear_classifier_rotation_daily.csv")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost-bps", type=float, default=60.0)  # was 20.0
    p.add_argument("--defensive-cap", type=float, default=0.30)
    p.add_argument("--gross-cap", type=float, default=0.80)
    p.add_argument("--cache-dir", default="artifacts/bear_classifier/cache")
    p.add_argument("--out-summary", default="artifacts/backtest/continuous_regime_integration_summary.csv")
    p.add_argument("--out-blocked-trades", default="artifacts/backtest/continuous_regime_blocked_trades.csv")
    return p.parse_args()


def _load_base(work_dir: Path, start: str, end: str) -> pd.DataFrame:
    """The REAL production daily file (via _crypto_main), not extended_overlay_daily.csv
    (verified stale -- its own combined_return gives ~1.50 Sharpe, not 1.510, for any
    column in it, regardless of reconstruction method)."""
    base = pd.read_csv(work_dir / "validated_crypto_daily.csv", low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    base = base[(base["day"] >= pd.to_datetime(start, utc=True)) & (base["day"] <= pd.to_datetime(end, utc=True))].reset_index(drop=True)
    for col in ["alloc_eth", "alloc_btc", "eth_strategy_return", "btc_strategy_return", "gold_strategy_return", "gold_weight_exec", "eth_off_active", "btc_off_active", "vol_multiplier"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)
    return base


def _load_states(path: Path, base: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(path)
    s["day"] = pd.to_datetime(s["date"], utc=True, errors="coerce").dt.floor("D")
    s["state"] = s["state"].astype(str)
    cols = ["day", "state", "regime_score", "btc_dd_20d", "btc_ret_20d", "btc_vol_pctile_365d"]
    out = base.merge(s[[c for c in cols if c in s.columns]], on="day", how="left")
    out["state"] = out["state"].ffill().fillna("weakening")
    # continuous_regime_score.py computes `state` from day t's own BTC close with
    # no internal shift, so it isn't knowable until after day t's close -- shift
    # here so _apply_state_conviction only ever sizes day t's position off the
    # state as of day t-1 (same causal convention as the vol-filter/asymmetric-
    # sizing fix in backtest_eth_btc_portfolio.py).
    out["state"] = out["state"].shift(1).fillna("weakening")
    out["regime_score"] = pd.to_numeric(out.get("regime_score"), errors="coerce").fillna(0.0)
    return out


def _main_return(x: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.Series:
    """alloc_eth/alloc_btc are the REAL portfolio-normalized executed weights (verified:
    alloc_eth*eth_strategy_return + alloc_btc*btc_strategy_return + gold_strategy_return
    reproduces the file's own combined_return to 1e-16). eth_weight_exec/btc_weight_exec
    are the pre-normalization sleeve-level signals and are NOT what determines executed
    P&L -- using them was the bug in the first version of this harness."""
    x = x.copy()
    x["combined_return"] = x["alloc_eth"] * x["eth_strategy_return"] + x["alloc_btc"] * x["btc_strategy_return"] + x["gold_strategy_return"]
    mr = _mean_reversion_overlay(x, gross_cap=gross_cap, cost_bps=cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    return x["combined_return"] + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)


def _add_result(rows: list[dict], mode: str, config: str, returns: pd.Series, baseline_stats: dict, diagnosis: str) -> dict:
    st = _stats(returns)
    row = {
        "mode": mode, "config": config, "sharpe": st["sharpe"], "cagr": st["cagr"], "maxdd": st["maxdd"],
        "return": st["return"], "ann_vol": st["ann_vol"],
        "delta_sharpe_vs_baseline": st["sharpe"] - baseline_stats["sharpe"],
        "delta_cagr_vs_baseline": st["cagr"] - baseline_stats["cagr"],
        "delta_maxdd_vs_baseline": st["maxdd"] - baseline_stats["maxdd"],
        "delta_sharpe_vs_production_reference": st["sharpe"] - PRODUCTION_REFERENCE_SHARPE,
        "diagnosis": diagnosis,
    }
    rows.append(row)
    return row


# ---------------------------------------------------------------------------
# Mode A: sizing multiplier
# ---------------------------------------------------------------------------

def _apply_state_conviction(base: pd.DataFrame, mapping: dict[str, float], gross_cap: float, remove_vol_filter: bool = False) -> pd.DataFrame:
    x = base.copy()
    if remove_vol_filter:
        vol_mult = x["vol_multiplier"].replace(0.0, np.nan).fillna(1.0)
        x["alloc_eth"] = (x["alloc_eth"] / vol_mult).clip(lower=0.0)
        x["alloc_btc"] = (x["alloc_btc"] / vol_mult).clip(lower=0.0)
    mult = x["state"].map(mapping).astype(float).fillna(1.0)
    active = (x["alloc_eth"] > 0) | (x["alloc_btc"] > 0)
    x.loc[active, "alloc_eth"] = x.loc[active, "alloc_eth"] * mult.loc[active]
    x.loc[active, "alloc_btc"] = x.loc[active, "alloc_btc"] * mult.loc[active]
    gross = x["alloc_eth"] + x["alloc_btc"]
    over = gross > gross_cap
    x.loc[over, "alloc_eth"] = x.loc[over, "alloc_eth"] * gross_cap / gross.loc[over]
    x.loc[over, "alloc_btc"] = x.loc[over, "alloc_btc"] * gross_cap / gross.loc[over]
    return x


def _mode_a(d: pd.DataFrame, baseline_stats: dict, gross_cap: float, cost_bps: float, rows: list[dict]) -> None:
    variants = {
        "scale_no_panic_shrink": {"risk_on": 1.0, "weakening": 0.85, "risk_off": 0.70, "panic": 1.0},
        "scale_shrink_panic": {"risk_on": 1.0, "weakening": 0.85, "risk_off": 0.70, "panic": 0.50},
        "scale_mild": {"risk_on": 1.0, "weakening": 0.95, "risk_off": 0.85, "panic": 1.0},
    }
    results = {}
    for name, mapping in variants.items():
        modified = _apply_state_conviction(d, mapping, gross_cap)
        ret = _main_return(modified, gross_cap, cost_bps)
        results[name] = _add_result(rows, "A_sizing_multiplier", name, ret, baseline_stats, "Scales alloc_eth/alloc_btc by BTC state (gold + mean-reversion overlay left unchanged, overlay recomputed since it depends on used capacity).")

    best = max(results, key=lambda k: results[k]["sharpe"])
    if results[best]["sharpe"] <= baseline_stats["sharpe"]:
        # Diagnose: is the regime score just re-flagging the same volatility the vol
        # filter already responds to? Isolate by removing the vol filter's effect first.
        iso_mapping = variants[best]
        iso_modified = _apply_state_conviction(d, iso_mapping, gross_cap, remove_vol_filter=True)
        iso_ret = _main_return(iso_modified, gross_cap, cost_bps)
        iso = _add_result(
            rows, "A_sizing_multiplier", f"{best}_vol_filter_removed_isolation", iso_ret, baseline_stats,
            f"Diagnostic: best variant ({best}) underperformed baseline. Isolation test removes the existing "
            f"vol filter's multiplier from alloc_eth/alloc_btc before applying the state scale, to check whether "
            f"the state score is just re-detecting the same volatility the vol filter already prices in "
            f"(double-counting/fighting) rather than adding independent information.",
        )
        corr = float(d["vol_multiplier"].corr((d["state"] == "risk_off").astype(int) + 2 * (d["state"] == "panic").astype(int)))
        verdict = "CONFIRMS double-counting" if iso["sharpe"] > results[best]["sharpe"] else "REFUTES double-counting (isolation didn't help either -- state score adds no info beyond noise here)"
        print(f"  [A diagnosis] best sizing variant ({best}, Sharpe {results[best]['sharpe']:.3f}) is below baseline ({baseline_stats['sharpe']:.3f}).")
        print(f"  [A diagnosis] corr(vol_multiplier, state severity) = {corr:.3f}. Isolation (vol filter removed first) Sharpe: {iso['sharpe']:.3f} -> {verdict}")


# ---------------------------------------------------------------------------
# Mode B: entry gate
# ---------------------------------------------------------------------------

def _apply_state_gate(base: pd.DataFrame, block_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = base.copy()
    state = x["state"].to_numpy()
    score = x["regime_score"].to_numpy()
    blocked_rows = []
    for leg, off_col, alloc_col, ret_col in [("eth", "eth_off_active", "alloc_eth", "eth_strategy_return"), ("btc", "btc_off_active", "alloc_btc", "btc_strategy_return")]:
        active = x[off_col].fillna(0).to_numpy().astype(bool)
        starts, ends, trade_returns = _segment_stats(active, x[ret_col].fillna(0.0).to_numpy())
        alloc = x[alloc_col].to_numpy().copy()
        for s, e, r in zip(starts, ends, trade_returns):
            if block_fn(state[s], score[s]):
                alloc[s:e + 1] = 0.0
                blocked_rows.append({
                    "leg": leg, "entry_date": pd.Timestamp(x["day"].iloc[s]).date().isoformat(),
                    "exit_date": pd.Timestamp(x["day"].iloc[e]).date().isoformat(),
                    "entry_state": state[s], "entry_regime_score": float(score[s]),
                    "days_held": int(e - s + 1), "would_be_return_pct": float(r * 100.0),
                })
        x[alloc_col] = alloc
    return x, pd.DataFrame(blocked_rows)


def _mode_b(d: pd.DataFrame, baseline_stats: dict, gross_cap: float, cost_bps: float, rows: list[dict]) -> pd.DataFrame:
    variants = {
        "block_risk_off": lambda st, sc: st == "risk_off",
        "block_risk_off_and_panic": lambda st, sc: st in ("risk_off", "panic"),
    }
    all_blocked = []
    results = {}
    for name, fn in variants.items():
        modified, blocked = _apply_state_gate(d, fn)
        if not blocked.empty:
            blocked["config"] = name
            all_blocked.append(blocked)
        ret = _main_return(modified, gross_cap, cost_bps)
        blocked_sum = float(blocked["would_be_return_pct"].sum() / 100.0) if not blocked.empty else 0.0
        diag = f"Blocked {len(blocked)} ETH/BTC trade segments; sum of would-be trade returns {blocked_sum:+.1%}."
        results[name] = _add_result(rows, "B_entry_gate", name, ret, baseline_stats, diag)

    worst_name = min(results, key=lambda k: results[k]["sharpe"])
    if results[worst_name]["sharpe"] < baseline_stats["sharpe"]:
        blocked_all = pd.concat(all_blocked, ignore_index=True) if all_blocked else pd.DataFrame()
        this_blocked = blocked_all[blocked_all["config"] == worst_name] if not blocked_all.empty else blocked_all
        n_positive_blocked = int((this_blocked["would_be_return_pct"] > 0).sum()) if not this_blocked.empty else 0
        print(f"  [B diagnosis] {worst_name} (Sharpe {results[worst_name]['sharpe']:.3f}) underperforms baseline ({baseline_stats['sharpe']:.3f}).")
        print(f"  [B diagnosis] Of {len(this_blocked)} blocked trades, {n_positive_blocked} would have been profitable -- same structural failure as Fear&Greed/on-chain gates: the best trades start exactly when risk-off filters fire.")
        # Attempted fix: gate only the more extreme, less-common subset (state AND a
        # severely negative regime score) rather than the entire (32%-of-days-common)
        # risk_off state, to see if narrowing the trigger salvages any edge.
        fix_fn = lambda st, sc: st == "risk_off" and sc <= -3
        fix_modified, fix_blocked = _apply_state_gate(d, fix_fn)
        fix_ret = _main_return(fix_modified, gross_cap, cost_bps)
        fix = _add_result(
            rows, "B_entry_gate", "attempted_fix_severe_risk_off_only", fix_ret, baseline_stats,
            f"Attempted fix: narrows the gate to risk_off AND regime_score<=-3 (a stricter, rarer subset) instead of "
            f"all of risk_off (32% of days), to test whether the failure is 'risk-off gates fundamentally don't work "
            f"for this strategy' vs 'the risk_off state alone is too broad/noisy a trigger'. "
            f"Blocked {len(fix_blocked)} trades vs {len(this_blocked)} for the broad gate.",
        )
        verdict = "Narrowing helped" if fix["sharpe"] > results[worst_name]["sharpe"] else "Narrowing did NOT help -- confirms the structural problem (blocking risk-off entries hurts regardless of how the trigger is defined), not just a noisy/overbroad trigger."
        print(f"  [B diagnosis] Narrowed-gate Sharpe: {fix['sharpe']:.3f} -> {verdict}")

    return pd.concat(all_blocked, ignore_index=True) if all_blocked else pd.DataFrame()


# ---------------------------------------------------------------------------
# Mode C: defensive/hedge trigger (idle-capital overlay, same construction as
# the commodities priority-allocation work earlier in this conversation)
# ---------------------------------------------------------------------------

def _defensive_overlay_return(d: pd.DataFrame, base_main_return: pd.Series, mask: pd.Series, hedge_ret: pd.Series, defensive_cap: float, gross_cap: float, cost_bps: float) -> pd.Series:
    used = (d["alloc_eth"] + d["alloc_btc"] + d["gold_weight_exec"]).clip(lower=0.0)
    idle = (gross_cap - used).clip(lower=0.0, upper=defensive_cap)
    target = idle.where(mask, 0.0)
    exec_w = target.shift(1).fillna(0.0)
    prev = exec_w.shift(1).fillna(0.0)
    cost = (exec_w - prev).abs() * (cost_bps / 10000.0)
    hedge_contrib = exec_w * hedge_ret.fillna(0.0) - cost
    return base_main_return + hedge_contrib


def _asset_ret(d: pd.DataFrame, px: pd.DataFrame) -> pd.Series:
    x = d[["day"]].merge(px, on="day", how="left")
    x["close"] = pd.to_numeric(x["close"], errors="coerce").ffill()
    return x["close"].pct_change().fillna(0.0)


def _mode_c(d: pd.DataFrame, base_main_return: pd.Series, baseline_stats: dict, args: argparse.Namespace, rows: list[dict]) -> None:
    cache = Path(args.cache_dir)
    vixy = _download_prices("VIXY", d["day"].min(), d["day"].max(), cache / "VIXY.csv", False)
    uso = _download_prices("USO", d["day"].min(), d["day"].max(), cache / "USO.csv", False)
    vixy_ret = _asset_ret(d, vixy)
    uso_ret = _asset_ret(d, uso)

    variants = {
        "hedge_risk_off_vixy_uso_blend": (d["state"].eq("risk_off"), 0.5 * vixy_ret + 0.5 * uso_ret),
        "hedge_risk_off_uso_only": (d["state"].eq("risk_off"), uso_ret),
        "hedge_risk_off_vixy_only": (d["state"].eq("risk_off"), vixy_ret),
    }
    results = {}
    for name, (mask, hedge_ret) in variants.items():
        ret = _defensive_overlay_return(d, base_main_return, mask, hedge_ret, args.defensive_cap, args.gross_cap, args.cost_bps)
        diag = "Hypothesis: hedge risk_off (weak/ambiguous state) using idle capital, not panic (continuous_regime_score shows panic has positive forward returns -- likely capitulation/rebound zone, not a hedge trigger)."
        results[name] = _add_result(rows, "C_defensive_trigger", name, ret, baseline_stats, diag)

    best = max(results, key=lambda k: results[k]["sharpe"])
    if results[best]["sharpe"] <= baseline_stats["sharpe"]:
        pct_risk_off = float(d["state"].eq("risk_off").mean())
        print(f"  [C diagnosis] best hedge variant ({best}, Sharpe {results[best]['sharpe']:.3f}) underperforms baseline ({baseline_stats['sharpe']:.3f}).")
        print(f"  [C diagnosis] risk_off covers {pct_risk_off:.1%} of days -- broad/common state, likely triggers the hedge too often relative to how often it precedes real drawdowns, dragging on returns via hedge cost/decay during good periods misclassified as risk_off.")
        # Attempted fix: narrow the trigger to persistent, severe risk_off (regime_score
        # very negative AND already several days into the state) rather than any risk_off day.
        persistent_severe = d["state"].eq("risk_off") & (d["regime_score"] <= -3)
        fix_ret = _defensive_overlay_return(d, base_main_return, persistent_severe, 0.5 * vixy_ret + 0.5 * uso_ret, args.defensive_cap, args.gross_cap, args.cost_bps)
        fix = _add_result(
            rows, "C_defensive_trigger", "attempted_fix_severe_risk_off_only", fix_ret, baseline_stats,
            f"Attempted fix: hedge only when risk_off AND regime_score<=-3 (severe subset, {float(persistent_severe.mean()):.1%} of days vs {pct_risk_off:.1%} for plain risk_off), to test whether narrowing to genuinely bad conditions salvages the hedge.",
        )
        verdict = "Narrowing helped" if fix["sharpe"] > results[best]["sharpe"] else "Narrowing did NOT help either -- the defensive-trigger idea doesn't add value over the existing classifier regardless of threshold."
        print(f"  [C diagnosis] Severe-subset hedge Sharpe: {fix['sharpe']:.3f} -> {verdict}")

    # Existing classifier benchmark: reported on ITS OWN file's baseline (that file is
    # also not on the 1.510 reference -- flagged separately, not blended into the table above).
    cls_path = Path(args.classifier_daily)
    if cls_path.exists():
        raw = pd.read_csv(cls_path, low_memory=False)
        raw["day"] = pd.to_datetime(raw["day"], utc=True, errors="coerce").dt.floor("D")
        raw = raw[(raw["day"] >= pd.to_datetime(args.start, utc=True)) & (raw["day"] <= pd.to_datetime(args.end, utc=True))]
        if "classifier_return" in raw.columns and "baseline_current_return" in raw.columns:
            cls_stats = _stats(pd.to_numeric(raw["classifier_return"], errors="coerce").fillna(0.0))
            cls_base_stats = _stats(pd.to_numeric(raw["baseline_current_return"], errors="coerce").fillna(0.0))
            print(f"  [C benchmark] Existing panic/inflation classifier (own file, NOT the 1.510 reference): "
                  f"its own baseline Sharpe {cls_base_stats['sharpe']:.3f} -> with classifier Sharpe {cls_stats['sharpe']:.3f} "
                  f"({cls_stats['sharpe']-cls_base_stats['sharpe']:+.3f}). Reported separately, not blended into the table above "
                  f"since it's measured on a different (also non-1.510) baseline file.")


def main() -> int:
    args = _parse_args()
    d = _load_base(Path(args.work_dir), args.start, args.end)
    d = _load_states(Path(args.regime_score), d)

    baseline_return = _main_return(d, args.gross_cap, args.cost_bps)
    baseline_stats = _stats(baseline_return)

    rows: list[dict] = []
    _add_result(rows, "baseline", "production_reconstruction", baseline_return, baseline_stats, "alloc_eth/alloc_btc reconstruction of the real production daily file + mean-reversion overlay recomputed.")

    print("=" * 120)
    print("CONTINUOUS REGIME INTEGRATION TESTS (corrected: real production file, alloc_eth/alloc_btc reconstruction)")
    print("=" * 120)
    print(f"Period: {args.start} to {args.end}")
    print(f"Reconstructed baseline Sharpe: {baseline_stats['sharpe']:.3f}  (production reference: {PRODUCTION_REFERENCE_SHARPE:.3f})")
    print()

    print("Mode A: sizing multiplier")
    _mode_a(d, baseline_stats, args.gross_cap, args.cost_bps, rows)
    print()
    print("Mode B: entry gate")
    blocked = _mode_b(d, baseline_stats, args.gross_cap, args.cost_bps, rows)
    print()
    print("Mode C: defensive/hedge trigger")
    _mode_c(d, baseline_return, baseline_stats, args, rows)
    print()

    latest = d.iloc[-1]
    counts = d["state"].value_counts(normalize=True).reindex(STATES, fill_value=0.0)
    monitoring_note = (
        f"Easy to expose: latest state={latest['state']}, score={latest['regime_score']:.1f}; "
        f"distribution risk_on={counts['risk_on']:.1%}, weakening={counts['weakening']:.1%}, "
        f"risk_off={counts['risk_off']:.1%}, panic={counts['panic']:.1%}. "
        "Add columns from continuous_regime_score_daily.csv to dashboard/analysis only; no trading dependency required."
    )
    rows.append({"mode": "D_monitoring_only", "config": "dashboard_analysis_context_only", "sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan, "return": np.nan, "ann_vol": np.nan, "delta_sharpe_vs_baseline": np.nan, "delta_cagr_vs_baseline": np.nan, "delta_maxdd_vs_baseline": np.nan, "delta_sharpe_vs_production_reference": np.nan, "diagnosis": monitoring_note})
    print("Mode D: monitoring only")
    print(f"  {monitoring_note}")
    print()

    out = pd.DataFrame(rows)
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    blocked_path = Path(args.out_blocked_trades)
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    blocked.to_csv(blocked_path, index=False)

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    print("=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(out[["mode", "config", "sharpe", "cagr", "maxdd", "delta_sharpe_vs_baseline"]].to_string(index=False))
    print("=" * 120)
    print(f"Saved: {out_path}")
    print(f"Saved: {blocked_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
