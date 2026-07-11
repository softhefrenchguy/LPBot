#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
echo "=== Daily Claude Analysis $(date -Is) ==="

# Use a one-off container so the analysis job does not depend on the long-running
# lpbot service container being healthy at exactly 08:05.
docker-compose run --rm -T --no-deps lpbot python scripts/analyse_daily.py "$@"
