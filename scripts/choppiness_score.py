from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from regime_classifier_v2 import smooth_short_islands  # noqa: E402 -- reuse the exact same persistence-filter used for BULL/BEAR/CHOP


CHOPPY_THRESHOLD = 61.8   # standard Dreiss CI convention (Fibonacci-derived)
TRENDING_THRESHOLD = 38.2  # standard Dreiss CI convention


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """Dreiss Choppiness Index: CI = 100 * log10( sum(true_range, n) / (max_high(n) - min_low(n)) ) / log10(n).
    High CI (near 100) = a lot of back-and-forth relative to net range -> choppy/oscillating.
    Low CI (near 0) = price moved in close to a straight line -> trending.
    n=20 matches this project's existing 20-day convention (rolling_vol_20d, dd_20d, etc.), not the
    more common technical-analysis default of n=14 -- noted as a deliberate consistency choice."""
    tr = _true_range(high, low, close)
    tr_sum = tr.rolling(n, min_periods=n).sum()
    range_n = high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min()
    ci = 100.0 * np.log10((tr_sum / range_n.replace(0.0, np.nan))) / np.log10(n)
    return ci.clip(lower=0.0, upper=100.0)


def _build_state(ci: pd.Series, min_persistence: int) -> pd.Series:
    raw = np.select([ci > CHOPPY_THRESHOLD, ci < TRENDING_THRESHOLD], ["CHOPPY", "TRENDING"], default="NEUTRAL")
    raw_series = pd.Series(raw, index=ci.index)
    smoothed = smooth_short_islands(raw_series, min_persistence)
    # 1-day lag, same convention as regime_classifier_v2's regime_v2, to avoid lookahead
    return smoothed.shift(1).fillna("NEUTRAL")


def _build_state_percentile(ci: pd.Series, min_persistence: int, window: int = 252, choppy_pct: float = 0.75, trending_pct: float = 0.25) -> pd.Series:
    """The standard 61.8/38.2 CI thresholds are calibrated for traditional-market
    instruments; on ETH they land near the extremes of the observed range (median 49.7,
    range 23.7-71.8 here) rather than the center, so >86% of days fall in NEUTRAL and the
    score rarely commits either way. Same fix already used elsewhere in this codebase for
    the same reason (continuous_regime_score's vol percentile, backtest_forex_optimised's
    _vol_multiplier): rank against the asset's own trailing distribution instead of a fixed
    absolute level."""
    pct_rank = ci.rolling(window, min_periods=max(20, window // 4)).apply(lambda x: float((x <= x[-1]).mean()), raw=True)
    raw = np.select([pct_rank > choppy_pct, pct_rank < trending_pct], ["CHOPPY", "TRENDING"], default="NEUTRAL")
    raw_series = pd.Series(raw, index=ci.index)
    smoothed = smooth_short_islands(raw_series, min_persistence)
    return smoothed.shift(1).fillna("NEUTRAL")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dreiss Choppiness Index, computed as a continuous daily score with regime_classifier_v2-style persistence smoothing. Research-only, standalone -- distinct axis from continuous_regime_score (trend clarity/oscillation vs directional risk state).")
    p.add_argument("--ohlc-csv", default="data/ETHUSDC_daily_ohlc_full.csv", help="Real daily OHLC with true high/low (fetched fresh from Binance -- NOT a |close-close_prev| proxy, since real high/low was easy to add).")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--min-persistence", type=int, default=5)
    p.add_argument("--out-daily", default="artifacts/backtest/choppiness_score_daily.csv")
    p.add_argument("--out-spot-checks", default="artifacts/backtest/choppiness_score_spot_checks.csv")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    d = pd.read_csv(args.ohlc_csv)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["ci"] = choppiness_index(d["high"], d["low"], d["close"], n=args.window)
    d["state_standard"] = _build_state(d["ci"], args.min_persistence)
    d["state"] = _build_state_percentile(d["ci"], args.min_persistence)

    Path(args.out_daily).parent.mkdir(parents=True, exist_ok=True)
    d[["timestamp", "close", "ci", "state_standard", "state"]].to_csv(args.out_daily, index=False)

    print("=" * 100)
    print("CHOPPINESS SCORE (Dreiss CI, daily, real high/low, 20d window)")
    print("=" * 100)
    print(f"Coverage: {d['timestamp'].min().date()} to {d['timestamp'].max().date()} ({len(d)} days)")
    print(f"CI stats: median={d['ci'].median():.1f}  mean={d['ci'].mean():.1f}  min={d['ci'].min():.1f}  max={d['ci'].max():.1f}")
    dist_std = d["state_standard"].value_counts(normalize=True) * 100
    dist_pct = d["state"].value_counts(normalize=True) * 100
    print(f"State distribution (standard 61.8/38.2 thresholds): {dist_std.round(1).to_dict()}")
    print(f"State distribution (percentile-calibrated, 252d rolling): {dist_pct.round(1).to_dict()}")

    print(f"\nSaved: {args.out_daily}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
