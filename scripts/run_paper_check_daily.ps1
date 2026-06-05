Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Repository root (one level above scripts/)
$REPO_ROOT = Split-Path -Parent $PSScriptRoot

# Python executable override:
# 1) set $env:PYTHON_EXE
# 2) fallback to "python" on PATH
$PYTHON = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$PAXG_CSV = if ($env:PAXG_CSV) { $env:PAXG_CSV } else { "$REPO_ROOT\\data\\paxg_daily.csv" }
$BTC_DAILY_CSV = if ($env:BTC_DAILY_CSV) { $env:BTC_DAILY_CSV } else { "$REPO_ROOT\\data\\btc_daily.csv" }
$BTC_PERP_CSV = if ($env:BTC_PERP_CSV) { $env:BTC_PERP_CSV } else { "$REPO_ROOT\\data\\btc_perp_features.csv" }

$TIMESTAMP = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "=== Paper Trade Check $TIMESTAMP ===" -ForegroundColor Cyan

& $PYTHON "$REPO_ROOT\scripts\paper_trade_checklist.py" `
    --price-csv "$REPO_ROOT\data\ETHUSDC_5m.csv" `
    --funding-csv "$REPO_ROOT\data\backtest\ETH_perp_features_5m_live.csv" `
    --regime-csv "$REPO_ROOT\artifacts\paper_trade\regime_snapshot_immutable.csv" `
    --regime-col regime_v2 `
    --offense-log-csv "$REPO_ROOT\artifacts\backtest\offense_trend_follow_v1a_6y_ema21_55_144.csv" `
    --defense-live-mode artifact `
    --defense-log-csv "$REPO_ROOT\artifacts\paper_trade\defensive_live.csv" `
    --combined-log-csv "$REPO_ROOT\artifacts\backtest\combined_offtf_ema21_55_144_defv1_routerA_6y.csv" `
    --paxg-csv "$PAXG_CSV" `
    --btc-daily-csv "$BTC_DAILY_CSV" `
    --btc-perp-csv "$BTC_PERP_CSV" `
    --out-dir "$REPO_ROOT\artifacts\paper_trade"

if ($LASTEXITCODE -eq 0) {
    Write-Host "CHECK COMPLETE" -ForegroundColor Green
} else {
    Write-Host "CHECK FAILED - review output above" -ForegroundColor Red
    exit $LASTEXITCODE
}

# STOP status visibility in task logs
$TODAY = Get-Date -Format "yyyyMMdd"
$LOG_FILE = "$REPO_ROOT\artifacts\paper_trade\daily_check_$TODAY.csv"

if (Test-Path $LOG_FILE) {
    $status = (Import-Csv $LOG_FILE | Select-Object -Last 1).status
    if ($status -eq "STOP") {
        Write-Host ""
        Write-Host "*** STOP CONDITION TRIGGERED ***" -ForegroundColor Red
        Write-Host "Review artifacts/paper_trade/daily_check_$TODAY.csv" -ForegroundColor Red
    }
}
