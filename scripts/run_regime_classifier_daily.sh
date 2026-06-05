#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USE_DOCKER="${USE_DOCKER:-1}"
PYTHON_BIN="${PYTHON_EXE:-python3}"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T lpbot python - \
      --eth-csv data/ETHUSDC_5m.csv \
      --btc-csv data/BTCUSDC_5m.csv \
      --out-daily-csv artifacts/backtest/regime_classifier_v2_daily.csv \
      --out-summary-csv artifacts/backtest/regime_classifier_v2_checks.csv \
      --out-transition-csv artifacts/backtest/regime_classifier_v2_transition.csv \
      --skip-html \
      < scripts/regime_classifier_v2.py
  ) >> "$LOG_DIR/regime_classifier.log" 2>&1
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/regime_classifier_v2.py" \
    --eth-csv "$REPO_ROOT/data/ETHUSDC_5m.csv" \
    --btc-csv "$REPO_ROOT/data/BTCUSDC_5m.csv" \
    --out-daily-csv "$REPO_ROOT/artifacts/backtest/regime_classifier_v2_daily.csv" \
    --out-summary-csv "$REPO_ROOT/artifacts/backtest/regime_classifier_v2_checks.csv" \
    --out-transition-csv "$REPO_ROOT/artifacts/backtest/regime_classifier_v2_transition.csv" \
    --skip-html \
    >> "$LOG_DIR/regime_classifier.log" 2>&1
fi
