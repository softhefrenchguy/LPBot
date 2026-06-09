#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
USE_DOCKER="${USE_DOCKER:-1}"
PYTHON_BIN="${PYTHON_EXE:-python3}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

mkdir -p "$LOG_DIR" "$REPO_ROOT/artifacts/markets" "$REPO_ROOT/data/markets"

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T lpbot python - \
      --out-csv artifacts/markets/market_tracker.csv \
      --data-dir data/markets \
      < scripts/market_tracker.py
  ) 2>&1 | tee -a "$LOG_DIR/market_tracker.log"
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/market_tracker.py" \
    --out-csv "$REPO_ROOT/artifacts/markets/market_tracker.csv" \
    --data-dir "$REPO_ROOT/data/markets" \
    2>&1 | tee -a "$LOG_DIR/market_tracker.log"
fi
