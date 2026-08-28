from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


"""
Rules-based major bear type classifier.

Inputs must come from scripts/prepare_bear_classifier_data.py. The classifier
uses only no-look-ahead columns available at bear_start:
- vix_at_start
- vix_trend_approx_20d_at_start
- cpi_yoy_known_at_start
- cpi_yoy_trend_3m_known_at_start
- credit_spread_known_at_start
- credit_spread_trend_20obs_known_at_start

The hindsight winner column is used only for scoring after classification, not
for rule assignment or threshold tuning.
"""


PREFERRED_ASSETS = {
    "INFLATION_BEAR": {"energy", "usd"},
    "PANIC_BEAR": {"vix", "gold"},
    "GEOPOLITICAL_BEAR": {"gold"},
    "AMBIGUOUS": set(),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify major BEAR periods into macro bear types.")
    p.add_argument("--input", default="artifacts/bear_classifier/bear_period_classifier_inputs.csv")
    p.add_argument("--out", default="artifacts/bear_classifier/bear_period_classifications.csv")
    return p.parse_args()


def _score_row(row: pd.Series) -> tuple[str, str, bool]:
    vix = float(row["vix_at_start"])
    vix_trend = float(row["vix_trend_approx_20d_at_start"])
    cpi = float(row["cpi_yoy_known_at_start"])
    cpi_trend = float(row["cpi_yoy_trend_3m_known_at_start"])
    credit = float(row["credit_spread_known_at_start"])
    credit_trend = float(row["credit_spread_trend_20obs_known_at_start"])

    panic_score = 0
    inflation_score = 0
    geopolitical_score = 0
    reasons: list[str] = []

    # Panic is visible in market stress immediately: high VIX, a sharp VIX jump,
    # and/or fast credit spread widening.
    if vix >= 25:
        panic_score += 2
        reasons.append(f"VIX elevated at {vix:.1f}")
    elif vix >= 20:
        panic_score += 1
        reasons.append(f"VIX moderately high at {vix:.1f}")
    if vix_trend >= 8:
        panic_score += 2
        reasons.append(f"VIX 20d trend spiking +{vix_trend:.1f}")
    elif vix_trend >= 4:
        panic_score += 1
        reasons.append(f"VIX 20d trend rising +{vix_trend:.1f}")
    if credit_trend >= 0.30:
        panic_score += 1
        reasons.append(f"credit spreads widening +{credit_trend:.2f}")
    if vix >= 20 and credit_trend >= 0.25:
        panic_score += 1
        reasons.append("moderate/high VIX plus widening credit stress")

    # Inflation bears are macro-slow: high/rising CPI with comparatively calmer
    # VIX at the start, before equity-style panic necessarily appears.
    if cpi >= 0.055:
        inflation_score += 2
        reasons.append(f"CPI high at {cpi * 100:.1f}%")
    elif cpi >= 0.04:
        inflation_score += 1
        reasons.append(f"CPI elevated at {cpi * 100:.1f}%")
    if cpi_trend >= 0.0075:
        inflation_score += 2
        reasons.append(f"CPI rising 3m +{cpi_trend * 100:.1f}pp")
    elif cpi_trend >= 0.0025:
        inflation_score += 1
        reasons.append(f"CPI drifting up 3m +{cpi_trend * 100:.1f}pp")
    if vix < 25:
        inflation_score += 1
        reasons.append("VIX not yet panic-high")

    # Geopolitical/stress-shock bucket: credit widening or moderate VIX stress
    # without enough CPI evidence for an inflation bear and without full panic.
    if credit >= 0.95:
        geopolitical_score += 1
        reasons.append(f"credit spread level high at {credit:.2f}")
    if credit_trend >= 0.15:
        geopolitical_score += 2
        reasons.append(f"credit spread trend widening +{credit_trend:.2f}")
    elif credit_trend >= 0.05:
        geopolitical_score += 1
        reasons.append(f"credit spread trend mildly widening +{credit_trend:.2f}")
    if 16 <= vix < 25:
        geopolitical_score += 1
        reasons.append(f"moderate VIX stress at {vix:.1f}")
    if cpi < 0.04 or cpi_trend <= 0:
        geopolitical_score += 1
        reasons.append("no clear CPI acceleration")

    scores = {
        "PANIC_BEAR": panic_score,
        "INFLATION_BEAR": inflation_score,
        "GEOPOLITICAL_BEAR": geopolitical_score,
    }

    # Economic priority: high/rising CPI is treated as an inflation bear unless
    # VIX is already at crisis levels. Otherwise obvious panic dominates.
    if cpi >= 0.055 and cpi_trend > 0 and vix < 35:
        assigned = "INFLATION_BEAR"
    elif panic_score >= 4:
        assigned = "PANIC_BEAR"
    elif inflation_score >= 4 and panic_score < 4:
        assigned = "INFLATION_BEAR"
    elif geopolitical_score >= 3:
        assigned = "GEOPOLITICAL_BEAR"
    elif panic_score >= 3:
        assigned = "PANIC_BEAR"
    elif inflation_score >= 3:
        assigned = "INFLATION_BEAR"
    else:
        assigned = "AMBIGUOUS"

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    borderline = (
        assigned == "AMBIGUOUS"
        or len(sorted_scores) > 1
        and sorted_scores[0][1] - sorted_scores[1][1] <= 1
        and sorted_scores[1][1] >= 3
    )
    reason = (
        f"panic={panic_score}, inflation={inflation_score}, geopolitical={geopolitical_score}; "
        + "; ".join(reasons)
    )
    return assigned, reason, bool(borderline)


def _hit(row: pd.Series) -> bool:
    assigned = str(row["assigned_bear_type"])
    winner = str(row["best_hindsight_asset_return_over_bear_period"]).lower()
    return winner in PREFERRED_ASSETS.get(assigned, set())


def main() -> int:
    args = _parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input file: {inp}. Run scripts/prepare_bear_classifier_data.py first.")

    d = pd.read_csv(inp)
    major = d[d["bear_scale"].eq("major_bear")].copy()
    if major.empty:
        raise SystemExit("No major_bear rows found in input.")

    assignments = major.apply(_score_row, axis=1, result_type="expand")
    major["assigned_bear_type"] = assignments[0]
    major["classification_reason"] = assignments[1]
    major["ambiguous_or_borderline"] = assignments[2]
    major["preferred_defensive_assets_for_assigned_type"] = major["assigned_bear_type"].map(
        lambda x: ",".join(sorted(PREFERRED_ASSETS.get(x, set()))) or ""
    )
    major["hindsight_asset_match"] = major.apply(_hit, axis=1)

    out_cols = [
        "bear_start",
        "bear_end",
        "duration_days",
        "underlying_peak_to_trough_pct",
        "vix_at_start",
        "vix_trend_approx_20d_at_start",
        "cpi_yoy_known_at_start",
        "cpi_yoy_trend_3m_known_at_start",
        "credit_spread_known_at_start",
        "credit_spread_trend_20obs_known_at_start",
        "assigned_bear_type",
        "ambiguous_or_borderline",
        "classification_reason",
        "preferred_defensive_assets_for_assigned_type",
        "best_hindsight_asset_return_over_bear_period",
        "hindsight_asset_match",
        "all_inputs_no_lookahead",
    ]
    out = major[out_cols].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    hit_rate = float(out["hindsight_asset_match"].mean())
    non_amb = out[~out["assigned_bear_type"].eq("AMBIGUOUS")]
    non_amb_hit_rate = float(non_amb["hindsight_asset_match"].mean()) if not non_amb.empty else np.nan

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 220)
    print("=" * 120)
    print("MAJOR BEAR CLASSIFICATIONS")
    print("=" * 120)
    print(f"Input: {inp}")
    print(f"Rows classified: {len(out)}")
    print(f"Saved: {out_path}")
    print(f"Overall hit rate: {hit_rate * 100:.1f}%")
    print(f"Non-ambiguous hit rate: {non_amb_hit_rate * 100:.1f}%" if np.isfinite(non_amb_hit_rate) else "Non-ambiguous hit rate: n/a")
    print()
    print(
        out[
            [
                "bear_start",
                "bear_end",
                "duration_days",
                "underlying_peak_to_trough_pct",
                "vix_at_start",
                "vix_trend_approx_20d_at_start",
                "cpi_yoy_known_at_start",
                "cpi_yoy_trend_3m_known_at_start",
                "credit_spread_trend_20obs_known_at_start",
                "assigned_bear_type",
                "ambiguous_or_borderline",
                "best_hindsight_asset_return_over_bear_period",
                "hindsight_asset_match",
            ]
        ].to_string(index=False)
    )
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
