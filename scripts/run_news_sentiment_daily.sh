#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
USE_DOCKER="${USE_DOCKER:-1}"
PYTHON_BIN="${PYTHON_EXE:-python3}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

mkdir -p "$LOG_DIR" "$REPO_ROOT/artifacts/news"

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY-}" lpbot python - \
      --model "$CLAUDE_MODEL" \
      --log-csv artifacts/news/news_log.csv \
      --debug-out artifacts/news/news_last_input.json \
      < scripts/news_sentiment.py
  ) 2>&1 | tee -a "$LOG_DIR/news_sentiment.log"
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/news_sentiment.py" \
    --model "$CLAUDE_MODEL" \
    --log-csv "$REPO_ROOT/artifacts/news/news_log.csv" \
    --debug-out "$REPO_ROOT/artifacts/news/news_last_input.json" \
    2>&1 | tee -a "$LOG_DIR/news_sentiment.log"
fi
