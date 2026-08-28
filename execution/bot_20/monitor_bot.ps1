# ============================
# 📊 LIVE MONITOR FOR bot_20
# Auto-refresh: every 60 seconds
# ============================

cd "C:\Users\Sofien\Desktop\bot_20"

while ($true) {
    Clear-Host
    Write-Host "📘 bot_20 — LP CYCLE HISTORY (refreshing every 60 seconds)
" -ForegroundColor Cyan
    python history_viewer.py
    Start-Sleep -Seconds 60
}
