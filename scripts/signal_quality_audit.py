from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _stale_run_stats(s: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(s, errors="coerce")
    # A stale bar is equal to previous non-null value.
    same = v.eq(v.shift(1)) & v.notna() & v.shift(1).notna()
    runs: list[int] = []
    cur = 0
    for x in same.to_numpy():
        if bool(x):
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        return 0.0, 0.0
    arr = np.asarray(runs, dtype=float)
    return float(np.percentile(arr, 95)), float(arr.max())


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit signal data quality for predictive features.")
    ap.add_argument("--input-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--features", default="basis,funding_rate,oi_chg,oi_metric")
    ap.add_argument("--out-csv", default="artifacts/backtest/signal_quality_audit.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "timestamp" not in df.columns:
        raise ValueError("input csv must contain timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Global timestamp health
    dupes = int(df["timestamp"].duplicated().sum())
    monotonic = bool(df["timestamp"].is_monotonic_increasing)
    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    median_sec = float(deltas.median()) if len(deltas) else np.nan

    rows: list[dict[str, float | int | str | bool]] = []
    feats = [x.strip() for x in args.features.split(",") if x.strip()]
    for f in feats:
        if f not in df.columns:
            rows.append(
                {
                    "feature": f,
                    "exists": False,
                    "rows": len(df),
                    "coverage_pct": 0.0,
                    "nonzero_pct": 0.0,
                    "std": np.nan,
                    "stale_run_p95_bars": np.nan,
                    "stale_run_max_bars": np.nan,
                }
            )
            continue

        s = pd.to_numeric(df[f], errors="coerce")
        cov = float(s.notna().mean() * 100.0)
        nz = float((s.fillna(0.0) != 0.0).mean() * 100.0)
        std = float(s.std()) if s.notna().any() else np.nan
        p95, mx = _stale_run_stats(s)
        rows.append(
            {
                "feature": f,
                "exists": True,
                "rows": len(df),
                "coverage_pct": cov,
                "nonzero_pct": nz,
                "std": std,
                "stale_run_p95_bars": p95,
                "stale_run_max_bars": mx,
            }
        )

    out = pd.DataFrame(rows)
    out["timestamp_monotonic"] = monotonic
    out["timestamp_dupes"] = dupes
    out["bar_interval_sec_median"] = median_sec

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
