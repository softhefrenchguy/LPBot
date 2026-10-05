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
# These were left at placeholder values sized for a much bigger account (a single trade could never
# hit EUR5000 and a EUR500 daily loss guard only trips after losing essentially the whole account
# against the real ~EUR500 balance this is actually running on). Scaled down to the real account size
# and made env-overridable so they can be raised later (e.g. once more capital is moved in) without
# another code change + deploy.
MAX_SINGLE_TRADE = float(os.getenv("MAX_SINGLE_TRADE_EUR", "450.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS_EUR", "50.0"))
MIN_CASH_RESERVE_PCT = 0.10
SLIPPAGE_WARN_PCT = 0.02
EXECUTION_LOG = Path("artifacts/live_trades/execution_log.csv")
DAILY_PNL_LOG = Path("artifacts/live_trades/daily_pnl.csv")
# Kraken's lowest fee tier (this account's current tier, <$2,500 30-day volume) charges 0.80%
# taker vs 0.40% maker -- exactly double. Every order placed here was always "ordertype": "market",
# i.e. always taker, for no reason tied to urgency (there is no code path that needs an
# immediate fill for risk-control reasons -- a STOP day skips placing any order at all, see
# execution_skipped_stop_active in paper_trade_checklist.py). Default on; env-overridable to
# fall back to plain market orders without a code change if anything looks wrong in production.
USE_POST_ONLY_ORDERS = os.getenv("USE_POST_ONLY_ORDERS", "true").strip().lower() == "true"
POST_ONLY_TIMEOUT_SECONDS = float(os.getenv("POST_ONLY_TIMEOUT_SECONDS", "45"))
_PAIR_DECIMALS_CACHE: dict[str, int] = {}
_PAIR_DECIMALS_FALLBACK = {"XBTEUR": 1, "ETHEUR": 2}


class KrakenExecutionError(RuntimeError):
    pass


class KrakenOrderStateUnknown(KrakenExecutionError):
    def __init__(self, order_id: str, message: str):
        super().__init__(message)
        self.order_id = order_id


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
    bid, ask = _get_bid_ask(asset)
    return (bid + ask) / 2.0


def _get_bid_ask(asset: str) -> tuple[float, float]:
    pair = _resolve_pair(asset)
    ticker = _public_query("Ticker", {"pair": pair})
    if not ticker:
        raise KrakenExecutionError(f"No ticker returned for {asset}")
    row = next(iter(ticker.values()))
    bid = float(row["b"][0])
    ask = float(row["a"][0])
    return bid, ask


def _pair_decimals(pair: str) -> int:
    """Price tick precision for `pair`, from Kraken's own AssetPairs metadata (cached). A
    post-only limit order needs its price rounded to the pair's real tick size -- the wrong
    precision gets the order rejected outright. Falls back to a known-correct hardcoded value
    for our two pairs if the lookup itself fails, rather than guessing for an unknown pair."""
    if pair in _PAIR_DECIMALS_CACHE:
        return _PAIR_DECIMALS_CACHE[pair]
    try:
        resp = _public_query("AssetPairs", {"pair": pair}, retries=1)
        row = resp.get(pair) or next(iter(resp.values()))
        decimals = int(row["pair_decimals"])
    except Exception:
        decimals = _PAIR_DECIMALS_FALLBACK.get(pair, 2)
    _PAIR_DECIMALS_CACHE[pair] = decimals
    return decimals


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


def _pre_trade_checks(asset: str, action: str, asset_amount: float, eur_amount: float) -> None:
    """Shared by every real-order path (market and post-only) so the safety gates can't drift
    apart between them -- this used to live only inside place_market_order; factored out rather
    than duplicated when post-only orders were added as a second real-order path."""
    if os.getenv("LIVE_TRADING_ENABLED", "").strip().lower() != "true":
        raise KrakenExecutionError("LIVE_TRADING_ENABLED is not true; refusing real order")
    if eur_amount > MAX_SINGLE_TRADE:
        raise KrakenExecutionError(f"Order {BASE_SYMBOL}{eur_amount:.2f} exceeds hard max single trade {BASE_SYMBOL}{MAX_SINGLE_TRADE:.2f}")
    if _latest_daily_loss_eur() > MAX_DAILY_LOSS:
        raise KrakenExecutionError(f"Daily loss guard exceeded {BASE_SYMBOL}{MAX_DAILY_LOSS:.2f}; refusing trade")
    _require_live_confirmation(asset, action, asset_amount, eur_amount)


def _find_order_by_userref(userref: int) -> str | None:
    """Look up whether an order tagged with this userref already exists (open or
    closed) on Kraken. Used to avoid double-submitting AddOrder when a retry follows
    a lost/timed-out response to a request Kraken may have already accepted."""
    for method, key in (("OpenOrders", "open"), ("ClosedOrders", "closed")):
        try:
            resp = _private_query(method, {"userref": userref}, retries=1)
        except Exception:
            continue
        orders = resp.get(key)
        if isinstance(orders, dict) and orders:
            return next(iter(orders.keys()))
    return None


def _place_order_idempotent(
    pair: str,
    action: str,
    volume: str,
    retries: int = 3,
    ordertype: str = "market",
    price: str | None = None,
    oflags: str | None = None,
) -> dict[str, Any]:
    """AddOrder with a userref-based idempotency check. The generic _private_query retry
    loop is safe to reuse for read-only calls (QueryOrders, balances, etc.), but blindly
    retrying AddOrder on a lost/timed-out response risks submitting a SECOND real market
    order if Kraken actually accepted the first one. Before each retry (and once more
    after the final attempt), check whether an order tagged with this call's userref
    already exists; only submit a fresh AddOrder if it doesn't."""
    userref = int(time.time() * 1000) % 2_000_000_000
    data: dict[str, Any] = {"pair": pair, "type": action, "ordertype": ordertype, "volume": volume, "userref": userref}
    if price is not None:
        data["price"] = price
    if oflags:
        data["oflags"] = oflags
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if attempt > 1:
            existing = _find_order_by_userref(userref)
            if existing:
                return {"txid": [existing]}
        try:
            return _private_query("AddOrder", data, retries=1)
        except Exception as exc:
            last_error = exc
            # A post-only order Kraken would have to fill immediately (price moved into the
            # spread) is rejected outright, not worth retrying at the same price -- let the
            # caller fall back to a market order right away instead of burning the retry budget.
            if "PostOnly" in str(exc) or "would execute immediately" in str(exc).lower():
                raise
            if attempt < retries:
                time.sleep(5.0)
    existing = _find_order_by_userref(userref)
    if existing:
        return {"txid": [existing]}
    raise KrakenExecutionError(f"Kraken AddOrder failed: {last_error}")


def _cancel_order(order_id: str) -> None:
    try:
        _private_query("CancelOrder", {"txid": order_id}, retries=1)
    except Exception:
        pass  # best-effort -- if it already filled or already closed, nothing to cancel


def _poll_post_only_fill(order_id: str, timeout_seconds: float) -> dict[str, float] | None:
    """Poll a resting post-only order for up to timeout_seconds. Returns fill info if it closed
    filled, or None if it's still open/unfilled (or canceled/expired on its own) when the timeout
    expires -- for a post-only order that is the NORMAL, expected outcome whenever the market
    doesn't come to the resting price in time, not an alarming state, so unlike
    confirm_order_filled (written for market orders, where a timeout is genuinely concerning)
    this does not raise for it. The caller cancels and falls back to a market order on None.
    Only raises if the order's fate is genuinely impossible to determine -- the query itself
    failing -- which needs a human, not an automatic retry."""
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        try:
            result = _private_query("QueryOrders", {"txid": order_id})
        except Exception as exc:
            msg = f"UNKNOWN_ORDER_STATE: {order_id} -- QueryOrders itself failed: {exc}. Check Kraken manually before any further trading."
            _send_discord(f"LIVE TRADE UNKNOWN STATE: {msg}")
            raise KrakenOrderStateUnknown(order_id, msg) from exc
        row = _extract_order_row(result, order_id)
        status = str(row.get("status", "unknown")) if isinstance(row, dict) else "not_found"
        if isinstance(row, dict) and status == "closed":
            vol = float(row.get("vol_exec", 0.0))
            cost = float(row.get("cost", 0.0))
            fee = float(row.get("fee", 0.0))
            price = cost / vol if vol > 0 else 0.0
            return {"filled_price": price, "filled_amount": vol, "fee_eur": fee}
        if isinstance(row, dict) and status in {"canceled", "expired"}:
            return None  # already gone on its own -- clean to fall back, nothing to cancel
        time.sleep(1.0)
    return None  # still open/pending at the deadline -- caller cancels and falls back


def place_market_order(
    asset: str,
    action: str,
    asset_amount: float,
    dry_run: bool = True,
    log_standalone: bool = True,
) -> dict[str, Any]:
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

    _pre_trade_checks(asset, action, float(asset_amount), eur_amount)

    pair = _resolve_pair(asset)
    resp = _place_order_idempotent(pair, action, f"{float(asset_amount):.10f}")
    txid = resp.get("txid", [])
    order_id = txid[0] if isinstance(txid, list) and txid else str(txid)
    fill = confirm_order_filled(order_id)
    actual_price = float(fill["filled_price"])
    if expected_price > 0 and abs(actual_price / expected_price - 1.0) > SLIPPAGE_WARN_PCT:
        _send_discord(f"SLIPPAGE WARNING: {asset} {action} moved from {BASE_SYMBOL}{expected_price:.2f} to {BASE_SYMBOL}{actual_price:.2f}")
    result = {
        "order_id": order_id,
        "status": "filled",
        "filled_price": actual_price,
        "filled_amount": float(fill["filled_amount"]),
        "fee_eur": float(fill.get("fee_eur", 0.0)),
        "timestamp": _now_iso(),
        "dry_run": False,
    }
    if log_standalone:
        slippage_bps = ((actual_price / expected_price) - 1.0) * 10000.0 if expected_price > 0 else 0.0
        _log_execution(
            {
                "timestamp": result["timestamp"],
                "base_currency": BASE_CURRENCY,
                "asset": asset,
                "action": action.upper(),
                "target_weight": "",
                "actual_weight": "",
                "eur_amount": float(result["filled_amount"]) * actual_price,
                "asset_amount": result["filled_amount"],
                "fill_price": actual_price,
                "expected_price": expected_price,
                "slippage_bps": slippage_bps,
                "fee_eur": result["fee_eur"],
                "order_id": order_id,
                "status": "filled",
                "dry_run": False,
                "error_message": "",
            }
        )
    return result


def place_post_only_order(
    asset: str,
    action: str,
    asset_amount: float,
    dry_run: bool = True,
    log_standalone: bool = True,
) -> dict[str, Any]:
    """Maker-only limit order, resting at the current best bid (BUY) / best ask (SELL) --
    halves the fee vs. a market order at this account's current Kraken tier (0.40% maker vs
    0.80% taker). Falls back to place_market_order (same guaranteed-fill behavior as before
    this existed) if the post-only order is rejected for crossing the spread, or if it simply
    hasn't filled within POST_ONLY_TIMEOUT_SECONDS -- a daily rebalance needs to complete same-day
    regardless, so this never risks leaving a position un-rebalanced waiting on a better price."""
    asset = asset.upper()
    action = action.lower()
    bid, ask = _get_bid_ask(asset)
    touch_price = bid if action == "buy" else ask
    expected_price = (bid + ask) / 2.0
    eur_amount = float(asset_amount) * expected_price

    if dry_run:
        result = {
            "order_id": f"DRYRUN-POSTONLY-{int(time.time())}",
            "status": "filled",
            "filled_price": touch_price,
            "filled_amount": float(asset_amount),
            "fee_eur": 0.0,
            "timestamp": _now_iso(),
            "dry_run": True,
        }
        print(f"DRY RUN (post-only): would {action.upper()} {asset_amount:.10f} {asset} at ~{BASE_SYMBOL}{touch_price:.2f} (resting), value {BASE_SYMBOL}{eur_amount:.2f}")
        return result

    _pre_trade_checks(asset, action, float(asset_amount), eur_amount)

    pair = _resolve_pair(asset)
    decimals = _pair_decimals(pair)
    price_str = f"{touch_price:.{decimals}f}"
    try:
        resp = _place_order_idempotent(pair, action, f"{float(asset_amount):.10f}", ordertype="limit", price=price_str, oflags="post")
    except Exception as exc:
        print(f"Post-only {asset} {action} rejected ({exc}); falling back to market order.")
        return place_market_order(asset, action, asset_amount, dry_run=False, log_standalone=log_standalone)

    txid = resp.get("txid", [])
    order_id = txid[0] if isinstance(txid, list) and txid else str(txid)
    # KrakenOrderStateUnknown (the query itself failing) is NOT caught here -- that's genuinely
    # ambiguous and must surface, the same way it does for a market order, rather than risk a
    # second real order on top of a fill whose outcome we can't actually confirm.
    fill = _poll_post_only_fill(order_id, POST_ONLY_TIMEOUT_SECONDS)
    if fill is None:
        _cancel_order(order_id)
        # Cancel can race a fill that completes in the instant before it lands -- Kraken simply
        # no-ops the cancel in that case, so check once more rather than assume "canceled" means
        # "unfilled" and risk placing a second order on top of one that actually went through.
        fill = _poll_post_only_fill(order_id, timeout_seconds=3.0)
    if fill is None:
        print(f"Post-only {asset} {action} did not fill within {POST_ONLY_TIMEOUT_SECONDS:.0f}s; canceled, falling back to market order.")
        return place_market_order(asset, action, asset_amount, dry_run=False, log_standalone=log_standalone)

    actual_price = float(fill["filled_price"])
    if expected_price > 0 and abs(actual_price / expected_price - 1.0) > SLIPPAGE_WARN_PCT:
        _send_discord(f"SLIPPAGE WARNING: {asset} {action} moved from {BASE_SYMBOL}{expected_price:.2f} to {BASE_SYMBOL}{actual_price:.2f}")
    result = {
        "order_id": order_id,
        "status": "filled",
        "filled_price": actual_price,
        "filled_amount": float(fill["filled_amount"]),
        "fee_eur": float(fill.get("fee_eur", 0.0)),
        "timestamp": _now_iso(),
        "dry_run": False,
    }
    if log_standalone:
        slippage_bps = ((actual_price / expected_price) - 1.0) * 10000.0 if expected_price > 0 else 0.0
        _log_execution(
            {
                "timestamp": result["timestamp"],
                "base_currency": BASE_CURRENCY,
                "asset": asset,
                "action": action.upper(),
                "target_weight": "",
                "actual_weight": "",
                "eur_amount": float(result["filled_amount"]) * actual_price,
                "asset_amount": result["filled_amount"],
                "fill_price": actual_price,
                "expected_price": expected_price,
                "slippage_bps": slippage_bps,
                "fee_eur": result["fee_eur"],
                "order_id": order_id,
                "status": "filled",
                "dry_run": False,
                "error_message": "",
            }
        )
    return result


def place_order(
    asset: str,
    action: str,
    asset_amount: float,
    dry_run: bool = True,
    log_standalone: bool = True,
) -> dict[str, Any]:
    """Dispatches to the post-only path when enabled, else the plain market-order path. Dry runs
    always use the plain market-order simulation regardless of USE_POST_ONLY_ORDERS -- there's no
    real order book or fee-tier difference to simulate, and this keeps existing dry-run behavior
    (and anything that depends on it) unchanged."""
    if dry_run or not USE_POST_ONLY_ORDERS:
        return place_market_order(asset, action, asset_amount, dry_run=dry_run, log_standalone=log_standalone)
    return place_post_only_order(asset, action, asset_amount, dry_run=False, log_standalone=log_standalone)


def _extract_order_row(result: dict[str, Any], order_id: str) -> dict[str, Any] | None:
    row = result.get(order_id)
    if isinstance(row, dict):
        return row

    # Kraken can occasionally return a single normalized order row that is not
    # keyed exactly as the txid we submitted. Treat one unambiguous row as ours.
    dict_rows = [v for v in result.values() if isinstance(v, dict)]
    if len(dict_rows) == 1:
        candidate = dict_rows[0]
        if "status" in candidate and "descr" in candidate:
            return candidate
    return None


def confirm_order_filled(order_id: str, timeout_seconds: int = 90) -> dict[str, float]:
    deadline = time.time() + int(timeout_seconds)
    attempt = 0
    last_status = "not_found"
    last_row: dict[str, Any] | None = None
    while time.time() < deadline:
        attempt += 1
        try:
            result = _private_query("QueryOrders", {"txid": order_id})
        except Exception as exc:
            # The order was already placed (we have a real order_id) -- if we can't even
            # query its state, that's an unknown-state situation needing manual review,
            # not a plain failure. Raising a bare exception here lost order_id entirely
            # once it reached execute_strategy_signal's generic except block (logged as
            # order_id=""), which broke reconciliation for exactly the orders that most
            # need it -- ones that were placed but whose outcome couldn't be confirmed.
            msg = f"UNKNOWN_ORDER_STATE: {order_id} -- QueryOrders itself failed: {exc}. Check Kraken manually before any further trading."
            _send_discord(f"LIVE TRADE UNKNOWN STATE: {msg}")
            raise KrakenOrderStateUnknown(order_id, msg) from exc
        row = _extract_order_row(result, order_id)
        if isinstance(row, dict):
            last_row = row
            last_status = str(row.get("status", "unknown"))
        else:
            last_status = "not_found"

        if isinstance(row, dict) and row.get("status") == "closed":
            vol = float(row.get("vol_exec", 0.0))
            cost = float(row.get("cost", 0.0))
            fee = float(row.get("fee", 0.0))
            price = cost / vol if vol > 0 else 0.0
            return {"filled_price": price, "filled_amount": vol, "fee_eur": fee}

        if isinstance(row, dict) and row.get("status") in {"canceled", "expired"}:
            raise KrakenExecutionError(f"ORDER {row.get('status')}: {order_id}")

        print(f"Order confirm attempt {attempt}: {order_id} status={last_status}", flush=True)
        time.sleep(1.0 if attempt <= 10 else 2.0)

    detail = ""
    if last_row:
        detail = f" last_status={last_status} vol_exec={last_row.get('vol_exec', '')}"
    msg = (
        f"UNKNOWN_ORDER_STATE: {order_id} not confirmed after {timeout_seconds}s."
        f"{detail} Check Kraken manually before any further trading."
    )
    _send_discord(f"LIVE TRADE UNKNOWN STATE: {msg}")
    raise KrakenOrderStateUnknown(order_id, msg)


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
            result = place_order(
                asset,
                str(plan["action"]).lower(),
                float(plan["asset_amount"]),
                dry_run=dry_run,
                log_standalone=False,
            )
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
        except KrakenOrderStateUnknown as exc:
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
                    "order_id": exc.order_id,
                    "status": "unknown",
                    "dry_run": dry_run,
                    "error_message": str(exc),
                }
            )
            _send_discord(f"LIVE TRADE UNKNOWN STATE: {exc}. Manual review required.")
            raise
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
        try:
            _log_daily_pnl(float(post.get("total_eur", 0.0)))
        except Exception:
            pass
    return report


def _log_daily_pnl(total_eur: float) -> None:
    """Append/replace today's real account snapshot in DAILY_PNL_LOG. Nothing wrote to this file
    before -- _latest_daily_loss_eur() (the daily-loss guard) always read an empty/missing file and
    returned 0.0, making MAX_DAILY_LOSS a permanent no-op. This makes that guard actually work, and
    doubles as the history a "live P&L" display reads from."""
    DAILY_PNL_LOG.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict[str, Any]] = []
    if DAILY_PNL_LOG.exists():
        try:
            rows = list(csv.DictReader(DAILY_PNL_LOG.open("r", encoding="utf-8")))
        except Exception:
            rows = []
    prev_total = None
    for r in reversed(rows):
        if str(r.get("date", "")) != today:
            try:
                prev_total = float(r.get("total_eur", ""))
            except Exception:
                prev_total = None
            break
    loss_eur = (prev_total - total_eur) if (prev_total is not None and total_eur < prev_total) else 0.0
    rows = [r for r in rows if str(r.get("date", "")) != today]
    rows.append({"date": today, "total_eur": f"{total_eur:.8f}", "realised_unrealised_loss_eur": f"{loss_eur:.8f}"})
    with DAILY_PNL_LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "total_eur", "realised_unrealised_loss_eur"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["date", "total_eur", "realised_unrealised_loss_eur"]})


def live_return_since_start(current_total_eur: float) -> float:
    """Return of the real account vs. the first-ever logged snapshot (i.e. since going live), as a
    FRACTION (0.0054 = 0.54%) -- matching the convention every other return field in this codebase
    uses with _fmt_pct (which multiplies by 100 itself). Caught by a pre-deploy test: returning a
    percent-point value here (5.4 meaning 5.4%) double-converts through _fmt_pct into "540%"."""
    if not DAILY_PNL_LOG.exists():
        return float("nan")
    try:
        rows = list(csv.DictReader(DAILY_PNL_LOG.open("r", encoding="utf-8")))
    except Exception:
        return float("nan")
    if not rows:
        return float("nan")
    try:
        start_total = float(rows[0]["total_eur"])
    except Exception:
        return float("nan")
    if start_total <= 0:
        return float("nan")
    return current_total_eur / start_total - 1.0


def compute_live_asset_pnl(asset: str, current_price: float) -> float:
    """Reconstruct a weighted-average cost basis for `asset` from real (non-dry-run) filled
    orders in EXECUTION_LOG, and return the current unrealized P&L as a FRACTION (same convention
    as live_return_since_start -- see its docstring). NaN if there's no real position (or no fill
    history) to compute one from."""
    if not EXECUTION_LOG.exists() or current_price <= 0:
        return float("nan")
    try:
        rows = list(csv.DictReader(EXECUTION_LOG.open("r", encoding="utf-8")))
    except Exception:
        return float("nan")
    qty = 0.0
    avg_cost = 0.0
    for r in rows:
        if str(r.get("asset", "")).upper() != asset.upper():
            continue
        if str(r.get("status", "")) != "filled":
            continue
        if str(r.get("dry_run", "")).strip().lower() == "true":
            continue
        action = str(r.get("action", "")).upper()
        try:
            amt = float(r.get("asset_amount", 0.0) or 0.0)
            price = float(r.get("fill_price", 0.0) or 0.0)
            fee = float(r.get("fee_eur", 0.0) or 0.0)
        except Exception:
            continue
        if action == "BUY" and amt > 0:
            new_qty = qty + amt
            if new_qty > 0:
                avg_cost = (avg_cost * qty + price * amt + fee) / new_qty
            qty = new_qty
        elif action == "SELL" and amt > 0:
            qty = max(0.0, qty - amt)
    if qty <= 1e-12 or avg_cost <= 0:
        return float("nan")
    return current_price / avg_cost - 1.0


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
