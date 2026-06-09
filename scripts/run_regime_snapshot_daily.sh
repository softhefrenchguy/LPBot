#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USE_DOCKER="${USE_DOCKER:-1}"
PYTHON_BIN="${PYTHON_EXE:-python3}"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR" "$REPO_ROOT/artifacts/paper_trade"

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T lpbot python - \
      --classifier-csv artifacts/backtest/regime_classifier_v2_daily.csv \
      --snapshot-csv artifacts/paper_trade/regime_snapshot_immutable.csv \
      --regime-col regime_v2 \
      --checks-log-csv artifacts/paper_trade/daily_checks_log.csv \
      --backfill-from-checks \
      --today-only \
      < scripts/regime_classifier_daily_snapshot.py
  ) >> "$LOG_DIR/regime_snapshot.log" 2>&1
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/regime_classifier_daily_snapshot.py" \
    --classifier-csv "$REPO_ROOT/artifacts/backtest/regime_classifier_v2_daily.csv" \
    --snapshot-csv "$REPO_ROOT/artifacts/paper_trade/regime_snapshot_immutable.csv" \
    --regime-col regime_v2 \
    --checks-log-csv "$REPO_ROOT/artifacts/paper_trade/daily_checks_log.csv" \
    --backfill-from-checks \
    --today-only \
    >> "$LOG_DIR/regime_snapshot.log" 2>&1
fi
