#!/usr/bin/env bash
# Dead-man's-switch for the daily pipeline. Runs directly on the host -- deliberately does NOT
# go through `docker-compose exec` into the lpbot container, because the one failure mode this
# exists to catch is exactly "that container is broken" (2026-10-03/04: a corrupted file inside
# it silently took down price feeds, news, and the live execution check for ~1.5 days, with zero
# alert, until the user noticed Discord had gone quiet). Checking from the host and alerting via
# a direct curl to the webhook means this stays alive even when the container doesn't.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

WEBHOOK="${DISCORD_WEBHOOK_URL-}"
TODAY_FILE="$REPO_ROOT/artifacts/paper_trade/daily_check_$(date -u +%Y%m%d).csv"
TS="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

send_alert() {
  local msg="$1"
  echo "$TS ALERT: $msg" >> "$LOG_DIR/heartbeat.log"
  if [[ -n "$WEBHOOK" ]]; then
    curl -s -X POST -H "Content-Type: application/json" \
      -d "{\"content\": \"\xe2\x9a\xa0\xef\xb8\x8f HEARTBEAT ALERT: $msg\"}" \
      "$WEBHOOK" > /dev/null 2>&1 || true
  fi
}

ok=1

if [[ ! -f "$TODAY_FILE" ]]; then
  send_alert "Today's paper check never completed -- no daily_check file found for $(date -u +%Y-%m-%d). The daily pipeline likely failed silently before reaching Discord."
  ok=0
fi

if command -v docker >/dev/null 2>&1; then
  STATE="$(docker inspect -f '{{.State.Status}}' lpbot_lpbot_1 2>/dev/null || echo 'unknown')"
  if [[ "$STATE" != "running" ]]; then
    send_alert "lpbot container is not running (state=$STATE). It may be crash-looping."
    ok=0
  fi
fi

if [[ "$ok" -eq 1 ]]; then
  echo "$TS heartbeat OK" >> "$LOG_DIR/heartbeat.log"
fi
