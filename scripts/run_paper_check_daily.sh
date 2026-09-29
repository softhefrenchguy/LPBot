#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_EXE:-python3}"
USE_DOCKER="${USE_DOCKER:-1}"
REFRESH_DEFENSE_BEFORE_CHECK="${REFRESH_DEFENSE_BEFORE_CHECK:-1}"
LOG_DIR="$REPO_ROOT/logs"
TODAY="$(date +%Y%m%d)"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

# Optional environment file (supports DISCORD_WEBHOOK_URL, PYTHON_EXE, etc.)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL-}"
PAPER_START_DATE="${PAPER_START_DATE-}"
FUNDING_CSV="${FUNDING_CSV:-data/backtest/ETH_perp_features_5m_live.csv}"
REGIME_CSV="${REGIME_CSV:-artifacts/paper_trade/regime_snapshot_immutable.csv}"
PAXG_CSV="${PAXG_CSV:-data/paxg_daily.csv}"
BTC_DAILY_CSV="${BTC_DAILY_CSV:-data/btc_daily.csv}"
BTC_PERP_CSV="${BTC_PERP_CSV:-data/btc_perp_features.csv}"
EXTRA_ARGS=("$@")

PAPER_START_ARGS=()
if [[ -n "$PAPER_START_DATE" ]]; then
  PAPER_START_ARGS=(--paper-start-date "$PAPER_START_DATE")
fi

mkdir -p "$LOG_DIR" "$REPO_ROOT/artifacts/paper_trade"

echo "=== Paper Trade Check $TS ===" | tee -a "$LOG_DIR/paper_check_cron.log"

# Refresh ETH price input used by the checklist. This is intentionally
# lightweight enough for daily cron and uses ETHUSDT as live proxy because
# Binance ETHUSDC spot history is stale/discontinued.
if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  echo "Refreshing ETHUSDC price feed..." | tee -a "$LOG_DIR/paper_check_cron.log"
  (
    cd "$REPO_ROOT"
    docker-compose exec -T lpbot python scripts/download_recent_binance_klines.py \
      --symbol ETHUSDT \
      --interval 5m \
      --out data/ETHUSDC_5m.csv \
      --limit 1000 \
      --days 500 \
      --timestamp-format iso \
      --count-1m-rows 5 \
      --overwrite
  ) 2>&1 | tee -a "$LOG_DIR/paper_check_cron.log" || true
fi

# Refresh market tracker artifact (best-effort). Primary schedule should run at 07:45.
if [[ -x "$REPO_ROOT/scripts/run_market_tracker_daily.sh" ]]; then
  echo "Refreshing market tracker artifact..." | tee -a "$LOG_DIR/paper_check_cron.log"
  "$REPO_ROOT/scripts/run_market_tracker_daily.sh" 2>&1 | tee -a "$LOG_DIR/paper_check_cron.log" || true
fi

# Refresh daily macro-news sentiment (best-effort). Primary schedule should run at 07:50.
if [[ -x "$REPO_ROOT/scripts/run_news_sentiment_daily.sh" ]]; then
  echo "Refreshing news sentiment artifact..." | tee -a "$LOG_DIR/paper_check_cron.log"
  "$REPO_ROOT/scripts/run_news_sentiment_daily.sh" 2>&1 | tee -a "$LOG_DIR/paper_check_cron.log" || true
fi

if [[ "$REFRESH_DEFENSE_BEFORE_CHECK" == "1" ]]; then
  if [[ -x "$REPO_ROOT/scripts/run_defensive_live_refresh.sh" ]]; then
    echo "Refreshing defensive live artifact..." | tee -a "$LOG_DIR/paper_check_cron.log"
    set +e
    "$REPO_ROOT/scripts/run_defensive_live_refresh.sh" >> "$LOG_DIR/paper_check_cron.log" 2>&1
    def_refresh_rc=$?
    set -e
    if [[ "$def_refresh_rc" -ne 0 ]]; then
      echo "WARNING: defensive live refresh failed (exit=$def_refresh_rc); using previous defensive artifact" \
        | tee -a "$LOG_DIR/paper_check_cron.log"
    fi
  else
    echo "WARNING: scripts/run_defensive_live_refresh.sh not found/executable; skipping defensive refresh" \
      | tee -a "$LOG_DIR/paper_check_cron.log"
  fi
fi

if [[ "$USE_DOCKER" == "1" ]] && command -v docker-compose >/dev/null 2>&1; then
  (
    cd "$REPO_ROOT"
    docker-compose exec -T \
      -e DISCORD_WEBHOOK_URL="$DISCORD_WEBHOOK_URL" \
      -e LIVE_TRADING_ENABLED="${LIVE_TRADING_ENABLED:-}" \
      -e LIVE_TRADING_CONFIRM="${LIVE_TRADING_CONFIRM:-}" \
      lpbot python - \
      --price-csv data/ETHUSDC_5m.csv \
      --funding-csv "$FUNDING_CSV" \
      --regime-csv "$REGIME_CSV" \
      --regime-col regime_v2 \
      --offense-log-csv artifacts/backtest/offense_trend_follow_v1a_6y_ema21_55_144.csv \
      --defense-live-mode artifact \
      --defense-log-csv artifacts/paper_trade/defensive_live.csv \
      --combined-log-csv artifacts/backtest/combined_offtf_ema21_55_144_defv1_routerA_6y.csv \
      --paxg-csv "$PAXG_CSV" \
      --btc-daily-csv "$BTC_DAILY_CSV" \
      --btc-perp-csv "$BTC_PERP_CSV" \
      --out-dir artifacts/paper_trade \
      "${PAPER_START_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      < scripts/paper_trade_checklist.py
  ) 2>&1 | tee -a "$LOG_DIR/paper_check_cron.log"
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/paper_trade_checklist.py" \
    --price-csv "$REPO_ROOT/data/ETHUSDC_5m.csv" \
    --funding-csv "$REPO_ROOT/$FUNDING_CSV" \
    --regime-csv "$REPO_ROOT/$REGIME_CSV" \
    --regime-col regime_v2 \
    --offense-log-csv "$REPO_ROOT/artifacts/backtest/offense_trend_follow_v1a_6y_ema21_55_144.csv" \
    --defense-live-mode artifact \
    --defense-log-csv "$REPO_ROOT/artifacts/paper_trade/defensive_live.csv" \
    --combined-log-csv "$REPO_ROOT/artifacts/backtest/combined_offtf_ema21_55_144_defv1_routerA_6y.csv" \
    --paxg-csv "$REPO_ROOT/$PAXG_CSV" \
    --btc-daily-csv "$REPO_ROOT/$BTC_DAILY_CSV" \
    --btc-perp-csv "$REPO_ROOT/$BTC_PERP_CSV" \
    --discord-webhook "$DISCORD_WEBHOOK_URL" \
    --out-dir "$REPO_ROOT/artifacts/paper_trade" \
    "${PAPER_START_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/paper_check_cron.log"
fi

DAILY_CHECK="$REPO_ROOT/artifacts/paper_trade/daily_check_${TODAY}.csv"
if [[ -f "$DAILY_CHECK" ]]; then
  status_col="$(head -1 "$DAILY_CHECK" | tr ',' '\n' | nl -ba | awk '$2=="status"{print $1}')"
  STATUS=""
  if [[ -n "${status_col:-}" ]]; then
    STATUS="$(tail -1 "$DAILY_CHECK" | cut -d',' -f"$status_col" | tr -d '\r\n')"
  fi
  if [[ "$STATUS" == "STOP" ]]; then
    echo "*** STOP CONDITION TRIGGERED - review $DAILY_CHECK ***" | tee -a "$LOG_DIR/paper_check_cron.log"
  fi
fi
