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


MIN_TRADE_USD = 10.0
MAX_SINGLE_TRADE_GBP = 5000.0
MAX_DAILY_LOSS_GBP = 500.0
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
    if a == "USD":
        return ["ZUSD", "USD"]
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
    desired = "XBTUSD" if asset.upper() == "BTC" else f"{asset.upper()}USD"
    pairs = _public_query("AssetPairs")
    candidates: list[tuple[str, dict[str, Any]]] = []
    for pair_name, meta in pairs.items():
        if not isinstance(meta, dict):
            continue
        alt = str(meta.get("altname", ""))
        ws = str(meta.get("wsname", ""))
        base = str(meta.get("base", ""))
        quote = str(meta.get("quote", ""))
        text = f"{pair_name} {alt} {ws} {base} {quote}".upper()
        if asset.upper() == "BTC":
            asset_match = "XBT" in text or "XXBT" in text or "BTC" in text
        else:
            asset_match = asset.upper() in text or f"X{asset.upper()}" in text
        quote_match = "USD" in quote.upper() or "ZUSD" in quote.upper() or "USD" in alt.upper() or "USD" in ws.upper()
        if asset_match and quote_match:
            candidates.append((pair_name, meta))
    for pair_name, meta in candidates:
        if str(meta.get("altname", "")).upper() == desired:
            return pair_name
    if candidates:
        return candidates[0][0]
    raise KrakenExecutionError(f"No Kraken USD pair found for {asset}")


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
    usd = _balance_amount(bal, "USD")
    eth = _balance_amount(bal, "ETH")
    btc = _balance_amount(bal, "BTC")
    eth_price = get_current_price("ETH")
    btc_price = get_current_price("BTC")
    total = usd + eth * eth_price + btc * btc_price
    return {"USD": usd, "ETH": eth, "BTC": btc, "total_usd": total}


def _log_execution(row: dict[str, Any]) -> None:
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp",
        "asset",
        "action",
        "target_weight",
        "actual_weight",
        "usd_amount",
        "asset_amount",
        "fill_price",
        "expected_price",
        "slippage_bps",
        "fee_usd",
        "order_id",
        "status",
        "dry_run",
        "error_message",
    ]
    exists = EXECUTION_LOG.exists()
    with EXECUTION_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def _latest_daily_loss_gbp() -> float:
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
                loss += float(row.get("realised_unrealised_loss_gbp", 0.0))
            except Exception:
                pass
    return loss


def calculate_position_size(
    asset: str,
    target_weight: float,
    total_capital_usd: float,
    current_balance: dict[str, float] | None = None,
) -> dict[str, Any]:
    bal = current_balance if current_balance is not None else get_account_balance()
    price = get_current_price(asset)
    asset_amt = float(bal[asset.upper()])
    current_usd = asset_amt * price
    target_weight = max(0.0, min(float(target_weight), 0.8))
    target_usd = target_weight * float(total_capital_usd)
    delta_usd = target_usd - current_usd
    if abs(delta_usd) < MIN_TRADE_USD:
        action = "HOLD"
    else:
        action = "BUY" if delta_usd > 0 else "SELL"
    amount = abs(delta_usd) / price if price > 0 else 0.0
    current_weight = current_usd / total_capital_usd if total_capital_usd > 0 else 0.0
    return {
        "action": action,
        "asset": asset.upper(),
        "usd_amount": abs(delta_usd),
        "asset_amount": amount,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "price": price,
    }


def _require_live_confirmation(asset: str, action: str, asset_amount: float, usd_amount: float) -> None:
    if os.getenv("LIVE_TRADING_CONFIRM", "").strip().lower() == "yes":
        return
    print("ABOUT TO PLACE REAL ORDER")
    print(f"Asset: {asset}")
    print(f"Action: {action}")
    print(f"Amount: {asset_amount:.10f} {asset}")
    print(f"Value: ~${usd_amount:.2f}")
    ans = input("Confirm? (yes/no): ").strip().lower()
    if ans != "yes":
        raise KrakenExecutionError("Real order aborted by confirmation prompt")


def place_market_order(asset: str, action: str, asset_amount: float, dry_run: bool = True) -> dict[str, Any]:
    asset = asset.upper()
    action = action.lower()
    expected_price = get_current_price(asset)
    usd_amount = float(asset_amount) * expected_price
    if dry_run:
        result = {
            "order_id": f"DRYRUN-{int(time.time())}",
            "status": "filled",
            "filled_price": expected_price,
            "filled_amount": float(asset_amount),
            "fee_usd": 0.0,
            "timestamp": _now_iso(),
            "dry_run": True,
        }
        print(f"DRY RUN: would {action.upper()} {asset_amount:.10f} {asset} at ~${expected_price:.2f}, value ${usd_amount:.2f}")
        return result

    if os.getenv("LIVE_TRADING_ENABLED", "").strip().lower() != "true":
        raise KrakenExecutionError("LIVE_TRADING_ENABLED is not true; refusing real order")
    if usd_amount > MAX_SINGLE_TRADE_GBP:
        raise KrakenExecutionError(f"Order ${usd_amount:.2f} exceeds hard max single trade {MAX_SINGLE_TRADE_GBP:.2f}")
    if _latest_daily_loss_gbp() > MAX_DAILY_LOSS_GBP:
        raise KrakenExecutionError(f"Daily loss guard exceeded {MAX_DAILY_LOSS_GBP:.2f}; refusing trade")
    _require_live_confirmation(asset, action, float(asset_amount), usd_amount)

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
        _send_discord(f"SLIPPAGE WARNING: {asset} {action} moved from ${expected_price:.2f} to ${actual_price:.2f}")
    return {
        "order_id": order_id,
        "status": "filled",
        "filled_price": actual_price,
        "filled_amount": float(fill["filled_amount"]),
        "fee_usd": float(fill.get("fee_usd", 0.0)),
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
            return {"filled_price": price, "filled_amount": vol, "fee_usd": fee}
        time.sleep(2.0)
    _send_discord(f"LIVE TRADE FAILED: order {order_id} not filled within {timeout_seconds}s. Manual review required.")
    raise KrakenExecutionError(f"ORDER NOT FILLED: {order_id}")


def execute_strategy_signal(
    eth_target_weight: float,
    btc_target_weight: float,
    total_capital_usd: float | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    start = time.time()
    try:
        pre = get_account_balance()
    except Exception:
        if not dry_run:
            raise
        fallback_capital = float(total_capital_usd or os.getenv("LIVE_DRY_RUN_CAPITAL_USD", "1000"))
        pre = {"USD": fallback_capital, "ETH": 0.0, "BTC": 0.0, "total_usd": fallback_capital}
    capital = float(total_capital_usd or pre["total_usd"])
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
                    "asset": asset,
                    "action": plan["action"],
                    "target_weight": plan["target_weight"],
                    "actual_weight": plan["current_weight"],
                    "usd_amount": plan["usd_amount"],
                    "asset_amount": plan["asset_amount"],
                    "fill_price": filled,
                    "expected_price": expected,
                    "slippage_bps": slippage_bps,
                    "fee_usd": result["fee_usd"],
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
                    "asset": asset,
                    "action": plan["action"],
                    "target_weight": plan["target_weight"],
                    "actual_weight": plan["current_weight"],
                    "usd_amount": plan["usd_amount"],
                    "asset_amount": plan["asset_amount"],
                    "fill_price": "",
                    "expected_price": plan.get("price", ""),
                    "slippage_bps": "",
                    "fee_usd": "",
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
                "asset": plan["asset"],
                "action": "HOLD",
                "target_weight": plan["target_weight"],
                "actual_weight": plan["current_weight"],
                "usd_amount": 0.0,
                "asset_amount": 0.0,
                "fill_price": "",
                "expected_price": plan.get("price", ""),
                "slippage_bps": 0.0,
                "fee_usd": 0.0,
                "order_id": "",
                "status": "hold",
                "dry_run": dry_run,
                "error_message": "",
            }
        )
        results[str(plan["asset"])] = {"status": "hold", "dry_run": dry_run}

    post = pre if dry_run else get_account_balance()
    fees = sum(float((r or {}).get("fee_usd", 0.0)) for r in results.values() if isinstance(r, dict))
    report = {
        "eth_trade": results.get("ETH"),
        "btc_trade": results.get("BTC"),
        "pre_trade_balance": pre,
        "post_trade_balance": post,
        "total_fees_usd": fees,
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
                f"{asset}: filled {float(trade.get('filled_amount', 0.0)):.8f} at ${float(trade.get('filled_price', 0.0)):.2f}"
            )
    lines.append(f"Fees: ${float(report.get('total_fees_usd', 0.0)):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("execution_kraken.py is a library. Run scripts/test_execution_kraken.py for tests.")
