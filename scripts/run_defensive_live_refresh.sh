#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
PERP_CSV="${PERP_CSV:-data/backtest/ETH_perp_features_5m_live.csv}"
WINDOW_DAYS="${WINDOW_DAYS:-60}"
WINDOW_END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$LOG_DIR" "$REPO_ROOT/artifacts/paper_trade"

(
  cd "$REPO_ROOT"
  docker-compose exec -T lpbot python - \
    --price-csv data/ETHUSDC_5m.csv \
    --perp-csv "$PERP_CSV" \
    --window-days "$WINDOW_DAYS" \
    --window-end-ts "$WINDOW_END_TS" \
    --wf-train-days 21 \
    --wf-test-days 7 \
    --wf-step-days 7 \
    --long-thresholds 0.80 \
    --short-thresholds 0.20 \
    --hold-bars-list 24 \
    --tail-infer-unseen \
    --no-shorts \
    --out-preds-csv artifacts/paper_trade/defensive_live_preds.csv \
    --out-sweep-csv artifacts/paper_trade/defensive_live_sweep.csv \
    --out-best-csv artifacts/paper_trade/defensive_live.csv \
    --out-html artifacts/paper_trade/defensive_live.html \
    < scripts/direction_event_model_v1.py
) >> "$LOG_DIR/defensive_live_refresh.log" 2>&1
