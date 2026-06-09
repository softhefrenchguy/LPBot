#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USE_DOCKER="${USE_DOCKER:-1}"
PYTHON_BIN="${PYTHON_EXE:-python3}"
LOG_DIR="$REPO_ROOT/logs"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

FUNDING_CSV="${FUNDING_CSV:-data/backtest/ETH_perp_features_5m_live.csv}"
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL-}"
FUNDING_MAX_AGE_HOURS="${FUNDING_MAX_AGE_HOURS:-9}"

mkdir -p "$LOG_DIR"

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T \
      -e DISCORD_WEBHOOK_URL="$DISCORD_WEBHOOK_URL" \
      -e FUNDING_CSV="$FUNDING_CSV" \
      -e FUNDING_MAX_AGE_HOURS="$FUNDING_MAX_AGE_HOURS" \
      lpbot python - \
        --funding-csv "$FUNDING_CSV" \
        --max-age-hours "$FUNDING_MAX_AGE_HOURS" \
        < scripts/check_feed_health.py
  ) >> "$LOG_DIR/feed_health.log" 2>&1
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/check_feed_health.py" \
    --funding-csv "$REPO_ROOT/$FUNDING_CSV" \
    --discord-webhook "$DISCORD_WEBHOOK_URL" \
    --max-age-hours "$FUNDING_MAX_AGE_HOURS" \
    >> "$LOG_DIR/feed_health.log" 2>&1
fi
