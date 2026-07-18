from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False

try:
    import krakenex
except Exception:  # pragma: no cover - optional dependency
    krakenex = None  # type: ignore[assignment]


BASE_CURRENCY = "EUR"
BASE_SYMBOL = "€"
ETH_PAIR = "ETHEUR"
BTC_PAIR = "XBTEUR"
MIN_TRADE = 10.0
MAX_SINGLE_TRADE = 5000.0
MAX_DAILY_LOSS = 500.0
MIN_CASH_RESERVE_PCT = 0.10
SLIPPAGE_WARN_PCT = 0.02
EXECUTION_LOG = Path("artifacts/live_trades/execution_log.csv")
DAILY_PNL_LOG = Path("artifacts/live_trades/daily_pnl.csv")


class KrakenExecutionError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_env() -> None:
    load_dotenv()


def _api() -> Any:
    _load_env()
    if krakenex is None:
        raise KrakenExecutionError("krakenex is not installed. Run: pip install krakenex")
    api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()
    api = krakenex.API()
    if api_key and api_secret:
        api.key = api_key
        api.secret = api_secret
    return api


def _private_query(method: str, data: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any]:
    api = _api()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = api.query_private(method, data or {})
            errs = resp.get("error") or []
            if errs:
                err_text = ";".join(str(e) for e in errs)
                if "Rate limit" in err_text or "EGeneral:Too many requests" in err_text:
                    time.sleep(5.0)
                    continue
                raise KrakenExecutionError(err_text)
            result = resp.get("result")
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(5.0)
    raise KrakenExecutionError(f"Kraken private {method} failed: {last_error}")


def _public_query(method: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any]:
    api = _api()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = api.query_public(method, params or {})
            errs = resp.get("error") or []
            if errs:
                err_text = ";".join(str(e) for e in errs)
                if "Rate limit" in err_text or "EGeneral:Too many requests" in err_text:
                    time.sleep(5.0)
                    continue
                raise KrakenExecutionError(err_text)
            result = resp.get("result")
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(5.0)
    raise KrakenExecutionError(f"Kraken public {method} failed: {last_error}")


def _send_discord(message: str) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    payload = json.dumps({"content": message}).encode("utf-8")
    req = request.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=10.0):
            pass
    except (error.HTTPError, Exception):
        return


def _asset_aliases(asset: str) -> list[str]:
    a = asset.upper()
    if a == "BTC":
        return ["XXBT", "XBT", "BTC"]
    if a == "ETH":
        return ["XETH", "ETH"]
    if a == "EUR":
        return ["ZEUR", "EUR"]
    return [a]


def _balance_amount(bal: dict[str, Any], asset: str) -> float:
    for key in _asset_aliases(asset):
        if key in bal:
            try:
                return float(bal[key])
            except Exception:
                return 0.0
    return 0.0


def _resolve_pair(asset: str) -> str:
    asset = asset.upper()
    if asset == "ETH":
        return ETH_PAIR
    if asset == "BTC":
        return BTC_PAIR
    raise KrakenExecutionError(f"No Kraken {BASE_CURRENCY} pair configured for {asset}")


def get_current_price(asset: str) -> float:
    pair = _resolve_pair(asset)
    ticker = _public_query("Ticker", {"pair": pair})
    if not ticker:
        raise KrakenExecutionError(f"No ticker returned for {asset}")
    row = next(iter(ticker.values()))
    bid = float(row["b"][0])
    ask = float(row["a"][0])
    return (bid + ask) / 2.0


def get_account_balance() -> dict[str, float]:
    bal = _private_query("Balance")
    eur = _balance_amount(bal, BASE_CURRENCY)
    eth = _balance_amount(bal, "ETH")
    btc = _balance_amount(bal, "BTC")
    eth_price = get_current_price("ETH")
    btc_price = get_current_price("BTC")
    total = eur + eth * eth_price + btc * btc_price
    return {BASE_CURRENCY: eur, "ETH": eth, "BTC": btc, "total_eur": total}


def _log_execution(row: dict[str, Any]) -> None:
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp",
        "base_currency",
        "asset",
        "action",
        "target_weight",
        "actual_weight",
        "eur_amount",
        "asset_amount",
        "fill_price",
        "expected_price",
        "slippage_bps",
        "fee_eur",
        "order_id",
        "status",
        "dry_run",
        "error_message",
    ]
    exists = EXECUTION_LOG.exists()
    if exists:
        try:
            header = EXECUTION_LOG.open("r", encoding="utf-8").readline().strip().split(",")
        except Exception:
            header = []
        if header and header != cols:
            archive = EXECUTION_LOG.with_name(f"{EXECUTION_LOG.stem}_legacy_non_eur_{int(time.time())}{EXECUTION_LOG.suffix}")
            EXECUTION_LOG.rename(archive)
            exists = False
    with EXECUTION_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def _latest_daily_loss_eur() -> float:
    if not DAILY_PNL_LOG.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(DAILY_PNL_LOG.open("r", encoding="utf-8")))
    except Exception:
        return 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    loss = 0.0
    for row in rows:
        if str(row.get("date", "")) == today:
            try:
                loss += float(row.get("realised_unrealised_loss_eur", 0.0))
            except Exception:
                pass
    return loss


def calculate_position_size(
    asset: str,
    target_weight: float,
    total_capital_eur: float,
    current_balance: dict[str, float] | None = None,
) -> dict[str, Any]:
    bal = current_balance if current_balance is not None else get_account_balance()
    price = get_current_price(asset)
    asset_amt = float(bal[asset.upper()])
    current_eur = asset_amt * price
    target_weight = max(0.0, min(float(target_weight), 0.8))
    target_eur = target_weight * float(total_capital_eur)
    delta_eur = target_eur - current_eur
    if abs(delta_eur) < MIN_TRADE:
        action = "HOLD"
    else:
        action = "BUY" if delta_eur > 0 else "SELL"
    amount = abs(delta_eur) / price if price > 0 else 0.0
    current_weight = current_eur / total_capital_eur if total_capital_eur > 0 else 0.0
    return {
        "action": action,
        "asset": asset.upper(),
        "eur_amount": abs(delta_eur),
        "asset_amount": amount,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "price": price,
    }


def _require_live_confirmation(asset: str, action: str, asset_amount: float, eur_amount: float) -> None:
    if os.getenv("LIVE_TRADING_CONFIRM", "").strip().lower() == "yes":
        return
    print("ABOUT TO PLACE REAL ORDER")
    print(f"Asset: {asset}")
    print(f"Action: {action}")
    print(f"Amount: {asset_amount:.10f} {asset}")
    print(f"Value: ~{BASE_SYMBOL}{eur_amount:.2f}")
    ans = input("Confirm? (yes/no): ").strip().lower()
    if ans != "yes":
        raise KrakenExecutionError("Real order aborted by confirmation prompt")


def place_market_order(asset: str, action: str, asset_amount: float, dry_run: bool = True) -> dict[str, Any]:
    asset = asset.upper()
    action = action.lower()
    expected_price = get_current_price(asset)
    eur_amount = float(asset_amount) * expected_price
    if dry_run:
        result = {
            "order_id": f"DRYRUN-{int(time.time())}",
            "status": "filled",
            "filled_price": expected_price,
            "filled_amount": float(asset_amount),
            "fee_eur": 0.0,
            "timestamp": _now_iso(),
            "dry_run": True,
        }
        print(f"DRY RUN: would {action.upper()} {asset_amount:.10f} {asset} at ~{BASE_SYMBOL}{expected_price:.2f}, value {BASE_SYMBOL}{eur_amount:.2f}")
        return result

    if os.getenv("LIVE_TRADING_ENABLED", "").strip().lower() != "true":
        raise KrakenExecutionError("LIVE_TRADING_ENABLED is not true; refusing real order")
    if eur_amount > MAX_SINGLE_TRADE:
        raise KrakenExecutionError(f"Order {BASE_SYMBOL}{eur_amount:.2f} exceeds hard max single trade {BASE_SYMBOL}{MAX_SINGLE_TRADE:.2f}")
    if _latest_daily_loss_eur() > MAX_DAILY_LOSS:
        raise KrakenExecutionError(f"Daily loss guard exceeded {BASE_SYMBOL}{MAX_DAILY_LOSS:.2f}; refusing trade")
    _require_live_confirmation(asset, action, float(asset_amount), eur_amount)

    pair = _resolve_pair(asset)
    resp = _private_query(
        "AddOrder",
        {
            "pair": pair,
            "type": action,
            "ordertype": "market",
            "volume": f"{float(asset_amount):.10f}",
        },
    )
    txid = resp.get("txid", [])
    order_id = txid[0] if isinstance(txid, list) and txid else str(txid)
    fill = confirm_order_filled(order_id)
    actual_price = float(fill["filled_price"])
    if expected_price > 0 and abs(actual_price / expected_price - 1.0) > SLIPPAGE_WARN_PCT:
        _send_discord(f"SLIPPAGE WARNING: {asset} {action} moved from {BASE_SYMBOL}{expected_price:.2f} to {BASE_SYMBOL}{actual_price:.2f}")
    return {
        "order_id": order_id,
        "status": "filled",
        "filled_price": actual_price,
        "filled_amount": float(fill["filled_amount"]),
        "fee_eur": float(fill.get("fee_eur", 0.0)),
        "timestamp": _now_iso(),
        "dry_run": False,
    }


def confirm_order_filled(order_id: str, timeout_seconds: int = 30) -> dict[str, float]:
    deadline = time.time() + int(timeout_seconds)
    while time.time() < deadline:
        result = _private_query("QueryOrders", {"txid": order_id})
        row = result.get(order_id)
        if isinstance(row, dict) and row.get("status") == "closed":
            vol = float(row.get("vol_exec", 0.0))
            cost = float(row.get("cost", 0.0))
            fee = float(row.get("fee", 0.0))
            price = cost / vol if vol > 0 else 0.0
            return {"filled_price": price, "filled_amount": vol, "fee_eur": fee}
        time.sleep(2.0)
    _send_discord(f"LIVE TRADE FAILED: order {order_id} not filled within {timeout_seconds}s. Manual review required.")
    raise KrakenExecutionError(f"ORDER NOT FILLED: {order_id}")


def execute_strategy_signal(
    eth_target_weight: float,
    btc_target_weight: float,
    total_capital_eur: float | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    start = time.time()
    try:
        pre = get_account_balance()
    except Exception:
        if not dry_run:
            raise
        fallback_capital = float(total_capital_eur or os.getenv("LIVE_DRY_RUN_CAPITAL_EUR", "1000"))
        pre = {BASE_CURRENCY: fallback_capital, "ETH": 0.0, "BTC": 0.0, "total_eur": fallback_capital}
    capital = float(total_capital_eur or pre["total_eur"])
    reserve_cap = max(0.0, 1.0 - MIN_CASH_RESERVE_PCT)
    eth_target_weight = min(float(eth_target_weight), reserve_cap)
    btc_target_weight = min(float(btc_target_weight), reserve_cap)
    if eth_target_weight + btc_target_weight > reserve_cap:
        scale = reserve_cap / (eth_target_weight + btc_target_weight)
        eth_target_weight *= scale
        btc_target_weight *= scale

    plans = [
        calculate_position_size("ETH", eth_target_weight, capital, current_balance=pre),
        calculate_position_size("BTC", btc_target_weight, capital, current_balance=pre),
    ]
    sells = [p for p in plans if p["action"] == "SELL"]
    buys = [p for p in plans if p["action"] == "BUY"]
    holds = [p for p in plans if p["action"] == "HOLD"]
    results: dict[str, Any] = {"ETH": None, "BTC": None}

    for plan in sells + buys:
        asset = str(plan["asset"])
        try:
            result = place_market_order(asset, str(plan["action"]).lower(), float(plan["asset_amount"]), dry_run=dry_run)
            expected = float(plan["price"])
            filled = float(result["filled_price"])
            slippage_bps = ((filled / expected) - 1.0) * 10000.0 if expected > 0 else 0.0
            _log_execution(
                {
                    "timestamp": _now_iso(),
                    "base_currency": BASE_CURRENCY,
                    "asset": asset,
                    "action": plan["action"],
                    "target_weight": plan["target_weight"],
                    "actual_weight": plan["current_weight"],
                    "eur_amount": plan["eur_amount"],
                    "asset_amount": plan["asset_amount"],
                    "fill_price": filled,
                    "expected_price": expected,
                    "slippage_bps": slippage_bps,
                    "fee_eur": result["fee_eur"],
                    "order_id": result["order_id"],
                    "status": result["status"],
                    "dry_run": dry_run,
                    "error_message": "",
                }
            )
            results[asset] = result
        except Exception as exc:
            _log_execution(
                {
                    "timestamp": _now_iso(),
                    "base_currency": BASE_CURRENCY,
                    "asset": asset,
                    "action": plan["action"],
                    "target_weight": plan["target_weight"],
                    "actual_weight": plan["current_weight"],
                    "eur_amount": plan["eur_amount"],
                    "asset_amount": plan["asset_amount"],
                    "fill_price": "",
                    "expected_price": plan.get("price", ""),
                    "slippage_bps": "",
                    "fee_eur": "",
                    "order_id": "",
                    "status": "error",
                    "dry_run": dry_run,
                    "error_message": str(exc),
                }
            )
            _send_discord(f"LIVE TRADE FAILED: {asset} {plan['action']} - {exc}. Manual review required.")
            results[asset] = {"status": "error", "error": str(exc), "dry_run": dry_run}

    for plan in holds:
        _log_execution(
            {
                "timestamp": _now_iso(),
                "base_currency": BASE_CURRENCY,
                "asset": plan["asset"],
                "action": "HOLD",
                "target_weight": plan["target_weight"],
                "actual_weight": plan["current_weight"],
                "eur_amount": 0.0,
                "asset_amount": 0.0,
                "fill_price": "",
                "expected_price": plan.get("price", ""),
                "slippage_bps": 0.0,
                "fee_eur": 0.0,
                "order_id": "",
                "status": "hold",
                "dry_run": dry_run,
                "error_message": "",
            }
        )
        results[str(plan["asset"])] = {"status": "hold", "dry_run": dry_run}

    post = pre if dry_run else get_account_balance()
    fees = sum(float((r or {}).get("fee_eur", 0.0)) for r in results.values() if isinstance(r, dict))
    report = {
        "eth_trade": results.get("ETH"),
        "btc_trade": results.get("BTC"),
        "pre_trade_balance": pre,
        "post_trade_balance": post,
        "base_currency": BASE_CURRENCY,
        "total_fees_eur": fees,
        "execution_time_ms": int((time.time() - start) * 1000),
        "dry_run": dry_run,
    }
    if not dry_run:
        _send_discord(_format_trade_report(report))
    return report


def _format_trade_report(report: dict[str, Any]) -> str:
    lines = ["LIVE TRADE EXECUTED"]
    for asset_key in ["eth_trade", "btc_trade"]:
        trade = report.get(asset_key)
        asset = asset_key.split("_")[0].upper()
        if not isinstance(trade, dict):
            continue
        if trade.get("status") == "hold":
            lines.append(f"{asset}: HOLD")
        elif trade.get("status") == "error":
            lines.append(f"{asset}: ERROR {trade.get('error')}")
        else:
            lines.append(
                f"{asset}: filled {float(trade.get('filled_amount', 0.0)):.8f} at {BASE_SYMBOL}{float(trade.get('filled_price', 0.0)):.2f}"
            )
    lines.append(f"Fees: {BASE_SYMBOL}{float(report.get('total_fees_eur', 0.0)):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("execution_kraken.py is a library. Run scripts/test_execution_kraken.py for tests.")
