#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STALE_HOURS="${MAX_STALE_HOURS:-5.5}"

cd "$REPO_ROOT"

docker-compose exec -T lpbot python /app/scripts/download_perp_features.py \
  --spot-symbol ETHUSDC \
  --perp-symbol ETHUSDT \
  --interval 5m \
  --days 30 \
  --out data/backtest/ETH_perp_features_5m_live.csv \
  --refresh-cache

docker-compose exec -T lpbot python - \
  --history-csv data/backtest/ETH_perp_features_5m_6y.csv \
  --live-csv data/backtest/ETH_perp_features_5m_live.csv \
  < scripts/merge_perp_live_into_history.py

docker-compose exec -T lpbot python - --max-stale-hours "$MAX_STALE_HOURS" <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def latest_age_hours(path: Path) -> tuple[pd.Timestamp, float]:
    df = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise SystemExit(f"No valid timestamps in {path}")
    latest = ts.max()
    age_h = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600.0
    return latest, age_h


parser = argparse.ArgumentParser()
parser.add_argument("--max-stale-hours", type=float, required=True)
args = parser.parse_args()

for raw in [
    "data/backtest/ETH_perp_features_5m_live.csv",
    "data/backtest/ETH_perp_features_5m_6y.csv",
]:
    latest, age_h = latest_age_hours(Path(raw))
    print(f"{raw}: latest={latest} age_h={age_h:.2f}")
    if age_h > args.max_stale_hours:
        raise SystemExit(f"{raw} stale after refresh: age_h={age_h:.2f} > {args.max_stale_hours:.2f}")
PY
