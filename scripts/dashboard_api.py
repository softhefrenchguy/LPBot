from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(os.getenv("LPBOT_ROOT", "/app" if Path("/app").exists() else ".")).resolve()
PAPER_DIR = APP_ROOT / "artifacts" / "paper_trade"
MARKETS_DIR = APP_ROOT / "artifacts" / "markets"
NEWS_DIR = APP_ROOT / "artifacts" / "news"
DASHBOARD_DIST = APP_ROOT / "dashboard" / "dist"

DAILY_LOG = Path(os.getenv("DASHBOARD_DAILY_LOG", PAPER_DIR / "daily_checks_log.csv"))
TRADES_LOG = Path(os.getenv("DASHBOARD_TRADES_LOG", PAPER_DIR / "completed_trades.csv"))
MARKETS_LOG = Path(os.getenv("DASHBOARD_MARKETS_LOG", MARKETS_DIR / "market_tracker.csv"))
NEWS_LOG = Path(os.getenv("DASHBOARD_NEWS_LOG", NEWS_DIR / "news_log.csv"))

app = FastAPI(title="LPBot Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://89.167.65.103:3000",
        "http://89.167.65.103:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    if hasattr(value, "item"):
        return _clean(value.item(), default)
    return value


def _float(row: pd.Series | dict[str, Any], *cols: str, default: float = 0.0) -> float:
    for c in cols:
        if c in row:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v) and math.isfinite(float(v)):
                return float(v)
    return default


def _str(row: pd.Series | dict[str, Any], *cols: str, default: str = "") -> str:
    for c in cols:
        if c in row:
            v = _clean(row[c])
            if v is not None:
                return str(v)
    return default


def _bool(row: pd.Series | dict[str, Any], *cols: str, default: bool = False) -> bool:
    for c in cols:
        if c in row:
            v = _clean(row[c])
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"true", "1", "yes", "y"}
    return default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _latest_daily() -> pd.Series:
    df = _read_csv(DAILY_LOG)
    if df.empty:
        return pd.Series(dtype=object)
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.tail(1).iloc[0]


def _pct_value(v: float) -> float:
    return float(v) if math.isfinite(float(v)) else 0.0


def _status_from_row(row: pd.Series) -> dict[str, Any]:
    if row.empty:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "status": "UNKNOWN",
            "healthy": False,
            "flags": "No data",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    eth_weight = _float(row, "combined_weight", default=0.0)
    btc_weight = _float(row, "btc_combined_weight", default=0.0)
    return {
        "date": _str(row, "date"),
        "status": _str(row, "status", default="UNKNOWN"),
        "healthy": not _bool(row, "any_flag", default=False),
        "flags": _str(row, "flags_text", "flag_reasons", default="No flags"),
        "eth_price": _float(row, "eth_price"),
        "eth_24h_pct": _float(row, "eth_24h_pct"),
        "eth_regime": _str(row, "regime", default="NA"),
        "eth_regime_days": _float(row, "regime_days"),
        "eth_weight": eth_weight,
        "btc_price": _float(row, "btc_price"),
        "btc_24h_pct": _float(row, "btc_24h_pct"),
        "btc_regime": _str(row, "btc_regime", "regime", default="NA"),
        "btc_weight": btc_weight,
        "capital_deployed": min(1.0, max(0.0, eth_weight + btc_weight)),
        "vol_regime": _str(row, "vol_regime", default="NA"),
        "vol_percentile": _float(row, "vol_percentile"),
        "conviction": _str(row, "conviction_bucket", default="NA"),
        "btc_conviction": _str(row, "btc_conviction_bucket", default="NA"),
        "dd_20d": _float(row, "dd_20d"),
        "days_live": int(_float(row, "days_live", default=0)),
        "last_updated": _str(row, "timestamp_utc", default=datetime.now(timezone.utc).isoformat()),
        "strategy_return": _float(row, "portfolio_strategy_ret", "cum_strategy_ret"),
        "excess": _float(row, "portfolio_excess_vs_basket", "excess"),
        "peak_dd": _float(row, "portfolio_peak_dd", "peak_dd"),
        "eth_spot": _float(row, "eth_sleeve_spot_ret", "cum_spot_ret"),
        "btc_spot": _float(row, "btc_sleeve_spot_ret"),
        "basket_spot": _float(row, "portfolio_spot_ret"),
        "eth_ema_fast": int(_float(row, "eth_ema_fast", default=21)),
        "eth_ema_mid": int(_float(row, "eth_ema_mid", default=55)),
        "eth_ema_slow": int(_float(row, "eth_ema_slow", default=144)),
        "ema21": _float(row, "ema21"),
        "ema55": _float(row, "ema55"),
        "ema144": _float(row, "ema144"),
        "stack_aligned": _bool(row, "stack_aligned"),
        "stack_aligned_days": _float(row, "stack_aligned_days"),
        "btc_ema15": _float(row, "btc_ema15", "btc_ema21"),
        "btc_ema40": _float(row, "btc_ema40", "btc_ema55"),
        "btc_ema120": _float(row, "btc_ema120", "btc_ema144"),
        "btc_stack_aligned": _bool(row, "btc_stack_aligned"),
        "btc_stack_aligned_days": _float(row, "btc_stack_aligned_days"),
        "gold_price": _float(row, "gold_price"),
        "gold_24h_pct": _float(row, "gold_24h_pct"),
        "gold_ema21": _float(row, "gold_ema21"),
        "gold_ema55": _float(row, "gold_ema55"),
        "gold_ema144": _float(row, "gold_ema144"),
        "gold_flat_bear_gate": _bool(row, "gold_flat_bear_gate"),
        "gold_condition_met": _bool(row, "gold_condition_met"),
        "gold_position_active": _bool(row, "gold_position_active"),
        "dvol_atm_iv_30d": _float(row, "dvol_atm_iv_30d", default=float("nan")),
        "dvol_iv_percentile": _float(row, "dvol_iv_percentile", default=float("nan")),
        "dvol_options_vol_regime": _str(row, "dvol_options_vol_regime", default="NA"),
        "dvol_rv_vol_regime": _str(row, "dvol_rv_vol_regime", default="NA"),
        "dvol_agreement": _str(row, "dvol_agreement", default="NA"),
        "dvol_term_slope": _float(row, "dvol_term_slope", default=float("nan")),
        "dvol_history_days": int(_float(row, "dvol_history_days", default=0)),
        "dvol_insufficient_history": _bool(row, "dvol_insufficient_history", default=True),
    }


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return _status_from_row(_latest_daily())


@app.get("/api/performance")
def api_performance() -> list[dict[str, Any]]:
    df = _read_csv(DAILY_LOG)
    if df.empty:
        return []
    if "date" in df.columns:
        df = df.sort_values("date")
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        strategy = _float(r, "portfolio_strategy_ret", "cum_strategy_ret")
        eth_spot = _float(r, "eth_sleeve_spot_ret", "cum_spot_ret")
        btc_spot = _float(r, "btc_sleeve_spot_ret")
        basket = _float(r, "portfolio_spot_ret", default=(eth_spot + btc_spot) / 2.0 if btc_spot else eth_spot)
        rows.append({
            "date": _str(r, "date"),
            "strategy_ret": strategy,
            "eth_spot": eth_spot,
            "btc_spot": btc_spot,
            "combined": basket,
            "excess": _float(r, "portfolio_excess_vs_basket", "excess"),
            "peak_dd": _float(r, "portfolio_peak_dd", "peak_dd"),
        })
    return rows


@app.get("/api/trades")
def api_trades() -> list[dict[str, Any]]:
    df = _read_csv(TRADES_LOG)
    if df.empty:
        return []
    if "entry_date" in df.columns:
        df = df.sort_values("entry_date")
    out = []
    for _, r in df.iterrows():
        out.append({
            "entry_date": _str(r, "entry_date"),
            "entry_price": _float(r, "entry_price"),
            "exit_date": _str(r, "exit_date"),
            "exit_price": _float(r, "exit_price"),
            "return_pct": _float(r, "return_pct"),
            "days_held": int(_float(r, "days_held", default=0)),
            "exit_reason": _str(r, "exit_reason"),
        })
    return out


@app.get("/api/markets")
def api_markets() -> dict[str, dict[str, Any]]:
    df = _read_csv(MARKETS_LOG)
    if df.empty:
        return {}
    date_col = "date" if "date" in df.columns else None
    asset_col = next((c for c in ["asset", "name", "symbol"] if c in df.columns), None)
    if not asset_col:
        return {}
    if date_col:
        latest_date = df[date_col].astype(str).max()
        df = df[df[date_col].astype(str) == latest_date]
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        asset = _str(r, asset_col, default="UNKNOWN")
        out[asset] = {
            "price": _float(r, "price", "latest_price", "close"),
            "pct": _float(r, "pct_change", "pct", "change_pct", "daily_pct"),
            "regime": _str(r, "trend", "regime", default="NA"),
            "stack_aligned": _bool(r, "stack_aligned"),
        }
    return out


@app.get("/api/news")
def api_news() -> list[dict[str, Any]]:
    df = _read_csv(NEWS_LOG)
    if df.empty:
        return []
    if "date" in df.columns:
        df = df.sort_values("date").tail(7)
    out = []
    for _, r in df.iterrows():
        out.append({
            "date": _str(r, "date"),
            "major_event": _bool(r, "major_event"),
            "severity": _str(r, "severity", default="low"),
            "affected": _str(r, "affected", default=""),
            "direction": _str(r, "direction", default="mixed"),
            "summary": _str(r, "summary", default="No major macro events"),
            "articles_scanned": int(_float(r, "articles_scanned", default=0)),
        })
    return out


def _binance_price(symbol: str) -> float | None:
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = api_status()
            eth_live = _binance_price("ETHUSDC") or _binance_price("ETHUSDT")
            btc_live = _binance_price("BTCUSDC") or _binance_price("BTCUSDT")
            payload["live_eth_price"] = eth_live
            payload["live_btc_price"] = btc_live
            payload["ws_updated"] = datetime.now(timezone.utc).isoformat()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        return


if DASHBOARD_DIST.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_DIST / "assets"), name="assets")


@app.get("/", response_model=None)
def dashboard_index():
    index = DASHBOARD_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "LPBot dashboard API running", "docs": "/docs"}


@app.get("/{path:path}", response_model=None)
def dashboard_spa(path: str):
    index = DASHBOARD_DIST / "index.html"
    if index.exists() and not path.startswith("api/"):
        return FileResponse(index)
    return {"message": "Not found"}
