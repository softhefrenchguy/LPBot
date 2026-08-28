from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns  # noqa: E402


def _transitions(on: np.ndarray) -> int:
    return int(np.sum(np.diff(on.astype(np.int8)) == 1))


def _il_cost_for_window(sub: pd.DataFrame, lp_col: str, il_k: float) -> float:
    r = sub["r"].fillna(0.0).to_numpy()
    lp_on = sub[lp_col].to_numpy()
    lp_weight_lag = np.roll(lp_on.astype(float), 1)
    lp_weight_lag[0] = 0.0
    il_r = -lp_weight_lag * il_k * (r**2)
    return float(il_r.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare LP exit-gate variants: current confirm-day off_active gate vs continuous_regime_score panic-state gate.")
    ap.add_argument("--daily", default="artifacts/backtest/paper_window_fresh/validated_crypto_daily_gapfixed.csv")
    ap.add_argument("--regime-score-csv", default="artifacts/backtest/continuous_regime_score_daily.csv")
    ap.add_argument("--crash-events-csv", default="artifacts/backtest/eth_crash_events.csv")
    ap.add_argument("--lp-vol-on", type=float, default=None, help="Defaults to calibrated median rolling_vol_20d")
    ap.add_argument("--lp-min-on-bars", type=int, default=6)
    ap.add_argument("--lp-cooldown-bars", type=int, default=12)
    ap.add_argument("--il-k", type=float, default=0.5)
    ap.add_argument("--fee-rate-ann", type=float, default=0.15)
    ap.add_argument("--out", default="artifacts/backtest/lp_panic_gate_comparison.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.daily, low_memory=False)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    regime = pd.read_csv(args.regime_score_csv)
    regime["day"] = pd.to_datetime(regime["date"], utc=True, errors="coerce").dt.floor("D")
    d = d.merge(regime[["day", "state"]].rename(columns={"state": "regime_state"}), on="day", how="left")
    d["regime_state"] = d["regime_state"].fillna("weakening")

    eth_active = d["eth_off_active"].fillna(0).astype(bool)
    btc_active = d["btc_off_active"].fillna(0).astype(bool)
    current_gate = eth_active | btc_active
    panic = d["regime_state"].eq("panic")

    lp_vol_on = args.lp_vol_on if args.lp_vol_on is not None else float(pd.to_numeric(d["rolling_vol_20d"], errors="coerce").median())

    print("=" * 115)
    print("LP EXIT-GATE COMPARISON: current confirm-day off_active gate vs continuous_regime_score panic-state gate")
    print("=" * 115)
    print(f"lp_vol_on (calibrated, median rolling_vol_20d) = {lp_vol_on:.3f}")
    print(f"current_gate=True (blocks LP, confirmed uptrend in either sleeve): {int(current_gate.sum())}/{len(d)} days ({current_gate.mean()*100:.1f}%)")
    print(f"panic=True (continuous_regime_score panic state): {int(panic.sum())}/{len(d)} days ({panic.mean()*100:.1f}%)")
    overlap = (current_gate & panic).sum()
    print(f"Overlap (current_gate AND panic both True): {int(overlap)} days")
    print()
    print("KEY STRUCTURAL FINDING (verified by reading lp_overlay_v1/overlay.py):")
    print("  raw_on = (~gate) & (weight>0) & (sigma_ann_smooth < lp_vol_on)  -- gate=True BLOCKS LP.")
    print("  current_gate = eth_off_active | btc_off_active is True only when a LONG position is CONFIRMED (this is a")
    print("  long-only strategy). During a crash/bear stretch, off_active is already 0/False (no confirmed long) --")
    print("  so current_gate is essentially never the thing blocking LP during a crash. It blocks LP during confirmed")
    print("  BULL runs, not during crashes. The only channel that can currently turn LP off during a crash is the")
    print("  volatility filter (sigma_ann_smooth < lp_vol_on), which uses a 20-day ROLLING vol -- itself lagged.")
    print("  This means 'swap the slow gate for a fast panic gate' is not a literal apples-to-apples swap: the")
    print("  confirm-day gate was never functioning as a crash-exit trigger in the first place. Testing both a literal")
    print("  swap (gate <- panic only) and a realistic augmentation (gate <- current_gate OR panic) below.")
    print()

    lp_input_base = pd.DataFrame({
        "timestamp": d["day"],
        "close": d["eth_close"],
        "weight": 1.0,
        "sigma_ann_smooth": pd.to_numeric(d["rolling_vol_20d"], errors="coerce"),
    })

    variants = {
        "A_current (off_active gate only, as-is today)": current_gate,
        "B_panic_only (literal swap: gate <- panic state)": panic,
        "C_current_or_panic (augmented: block on either)": current_gate | panic,
    }

    results = {}
    for label, gate in variants.items():
        lp_input = lp_input_base.copy()
        lp_input["gate"] = gate.to_numpy()
        out = compute_lp_overlay_returns(
            lp_input, bar_minutes=1440, fee_rate_ann=args.fee_rate_ann, il_k=args.il_k,
            lp_vol_on=lp_vol_on, lp_scale=1.0, lp_weight_max=1.0,
            lp_min_on_bars=args.lp_min_on_bars, lp_cooldown_bars=args.lp_cooldown_bars,
        )
        out["day"] = pd.to_datetime(d["day"].to_numpy(), utc=True)
        results[label] = out
        lp_on_pct = float(out["lp_on"].mean() * 100)
        n_trans = _transitions(out["lp_on"].to_numpy())
        total_il = float(out["il_r"].sum())
        total_fee = float(out["fee_r"].sum())
        print(f"{label}")
        print(f"  LP on: {lp_on_pct:.1f}% of days | on/off transitions (full history): {n_trans} | cumulative IL: {total_il*100:.2f}% | cumulative fee: {total_fee*100:.2f}%")

    print()
    print("=" * 115)
    print("1) CRASH-EXPOSURE COMPARISON (per detected crash event, +/- 3 day buffer around flagged window)")
    print("=" * 115)
    crashes = pd.read_csv(args.crash_events_csv)
    crashes["start"] = pd.to_datetime(crashes["start"], utc=True)
    crashes["end"] = pd.to_datetime(crashes["end"], utc=True)
    crashes = crashes.sort_values("worst_ret5").head(8)  # 8 most severe

    rows = []
    for _, ev in crashes.iterrows():
        w_start = ev["start"] - pd.Timedelta(days=3)
        w_end = ev["end"] + pd.Timedelta(days=3)
        row = {"start": ev["start"].date(), "end": ev["end"].date(), "worst_ret5_pct": ev["worst_ret5"] * 100}
        days_on = {}
        il_cost = {}
        for label, out in results.items():
            sub = out[(out["day"] >= w_start) & (out["day"] <= w_end)]
            days_on[label] = int(sub["lp_on"].sum())
            il_cost[label] = _il_cost_for_window(sub, "lp_on", args.il_k)
        row["days_on_A_current"] = days_on["A_current (off_active gate only, as-is today)"]
        row["days_on_B_panic_only"] = days_on["B_panic_only (literal swap: gate <- panic state)"]
        row["days_on_C_current_or_panic"] = days_on["C_current_or_panic (augmented: block on either)"]
        row["il_cost_pct_A_current"] = il_cost["A_current (off_active gate only, as-is today)"] * 100
        row["il_cost_pct_B_panic_only"] = il_cost["B_panic_only (literal swap: gate <- panic state)"] * 100
        row["il_cost_pct_C_current_or_panic"] = il_cost["C_current_or_panic (augmented: block on either)"] * 100
        row["days_saved_C_vs_A"] = row["days_on_A_current"] - row["days_on_C_current_or_panic"]
        rows.append(row)

    crash_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(crash_df.to_string(index=False))
    crash_df.to_csv("artifacts/backtest/lp_panic_gate_crash_exposure.csv", index=False)

    print()
    print("=" * 115)
    print("2) FLAPPING / CHURN CHECK (full history 2019-2026)")
    print("=" * 115)
    for label, out in results.items():
        n_trans = _transitions(out["lp_on"].to_numpy())
        print(f"{label}: {n_trans} on/off transitions over {len(out)} days")

    print()
    print("=" * 115)
    print("3) NET EFFECT: cumulative fee vs IL over full history, each variant")
    print("=" * 115)
    for label, out in results.items():
        total_fee = float(out["fee_r"].sum()) * 100
        total_il = float(out["il_r"].sum()) * 100
        net = total_fee + total_il
        print(f"{label}: fee={total_fee:.2f}%  IL={total_il:.2f}%  net={net:.2f}%  (assumed flat fee_rate_ann={args.fee_rate_ann*100:.0f}% -- same assumption across variants, isolates the gate's effect)")

    out_all = pd.concat([out.assign(variant=label) for label, out in results.items()], ignore_index=True)
    out_all.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print(f"Saved: artifacts/backtest/lp_panic_gate_crash_exposure.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
