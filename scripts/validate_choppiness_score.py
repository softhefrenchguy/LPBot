from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


SPOT_CHECKS = [
    ("mt_gox_grind (known whipsaw)", "2019-08-26", "2020-02-02", "CHOPPY expected"),
    ("2020_2021_bull (clean trend)", "2020-10-01", "2021-04-14", "TRENDING expected"),
    ("2022_h1_whipsaw (known whipsaw)", "2022-03-02", "2022-05-30", "CHOPPY expected"),
    ("2022_2023_whipsaw (known whipsaw)", "2022-06-17", "2023-04-03", "CHOPPY expected"),
    ("2023_2024_recovery", "2023-10-01", "2024-03-14", "TRENDING expected (per earlier continuous_regime_score check, this was actually mixed/weakening -- interesting cross-check)"),
    ("2024_bull_continuation (clean trend)", "2024-10-01", "2024-12-15", "TRENDING expected"),
    ("current_stretch (known whipsaw)", "2026-02-03", "2026-08-26", "CHOPPY expected"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate choppiness_score.py: spot-check known whipsaw/trending periods, and compare against continuous_regime_score to confirm it's a different axis (trend clarity) not a relabeled direction signal.")
    ap.add_argument("--choppiness-csv", default="artifacts/backtest/choppiness_score_daily.csv")
    ap.add_argument("--regime-score-csv", default="artifacts/backtest/continuous_regime_score_daily.csv")
    ap.add_argument("--out-spot-checks", default="artifacts/backtest/choppiness_spot_checks.csv")
    ap.add_argument("--out-comparison", default="artifacts/backtest/choppiness_vs_regime_score_comparison.csv")
    args = ap.parse_args()

    chop = pd.read_csv(args.choppiness_csv)
    chop["day"] = pd.to_datetime(chop["timestamp"], utc=True, errors="coerce").dt.floor("D")

    print("=" * 110)
    print("SPOT CHECKS: does choppiness_score read CHOPPY during known whipsaw periods and TRENDING during clean trends?")
    print("=" * 110)
    rows = []
    for name, start, end, expectation in SPOT_CHECKS:
        sub = chop[(chop["day"] >= pd.to_datetime(start, utc=True)) & (chop["day"] <= pd.to_datetime(end, utc=True))]
        if sub.empty:
            continue
        dist = sub["state"].value_counts(normalize=True).reindex(["TRENDING", "NEUTRAL", "CHOPPY"], fill_value=0.0) * 100
        dominant = dist.idxmax()
        print(f"{name:45} ({start} to {end}): dominant={dominant:9}  TRENDING={dist['TRENDING']:.0f}% NEUTRAL={dist['NEUTRAL']:.0f}% CHOPPY={dist['CHOPPY']:.0f}%  [{expectation}]")
        rows.append({"check": name, "start": start, "end": end, "dominant_state": dominant, "pct_trending": dist["TRENDING"], "pct_neutral": dist["NEUTRAL"], "pct_choppy": dist["CHOPPY"], "expectation": expectation})
    pd.DataFrame(rows).to_csv(args.out_spot_checks, index=False)

    print()
    print("=" * 110)
    print("COMPARISON: choppiness_score (trend clarity) vs continuous_regime_score (directional risk)")
    print("=" * 110)
    try:
        regime = pd.read_csv(args.regime_score_csv)
        regime["day"] = pd.to_datetime(regime["date"], utc=True, errors="coerce").dt.floor("D")
        merged = chop.merge(regime[["day", "state"]].rename(columns={"state": "regime_state"}), on="day", how="inner")

        cross = pd.crosstab(merged["state"], merged["regime_state"], normalize="index") * 100
        print("Row = choppiness state, columns = continuous_regime_score state (% of choppiness-state days in each regime state):")
        print(cross.round(1).to_string())
        cross.to_csv(args.out_comparison)

        print()
        # If choppiness were just relabeled direction, CHOPPY days would cluster heavily
        # in one single regime_state (e.g. all risk_off). Check how concentrated they are.
        choppy_rows = merged[merged["state"] == "CHOPPY"]
        if not choppy_rows.empty:
            choppy_regime_dist = choppy_rows["regime_state"].value_counts(normalize=True) * 100
            max_concentration = float(choppy_regime_dist.max())
            print(f"CHOPPY days spread across regime_score states: {choppy_regime_dist.round(1).to_dict()}")
            print(f"Max concentration in a single regime_score state: {max_concentration:.1f}%")
            verdict = "Genuinely different axis -- CHOPPY spans multiple regime_score states, not just one." if max_concentration < 60 else "Some overlap -- CHOPPY days lean toward one regime_score state; check whether it's adding real information."
            print(verdict)

        trending_rows = merged[merged["state"] == "TRENDING"]
        if not trending_rows.empty:
            trending_regime_dist = trending_rows["regime_state"].value_counts(normalize=True) * 100
            print(f"\nTRENDING days spread across regime_score states: {trending_regime_dist.round(1).to_dict()}")
    except FileNotFoundError:
        print(f"regime-score-csv not found at {args.regime_score_csv} -- skipping comparison.")

    print(f"\nSaved: {args.out_spot_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
