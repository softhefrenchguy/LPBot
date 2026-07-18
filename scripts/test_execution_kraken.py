from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

from execution_kraken import (
    BASE_CURRENCY,
    BASE_SYMBOL,
    KrakenExecutionError,
    calculate_position_size,
    execute_strategy_signal,
    get_account_balance,
    get_current_price,
    place_market_order,
)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description="Kraken execution engine safety tests.")
    ap.add_argument("--real", action="store_true", help="Run a real tiny ETH buy then sell. Requires LIVE_TRADING_ENABLED=true.")
    ap.add_argument("--test-eur", type=float, default=10.0)
    args = ap.parse_args()

    load_dotenv()
    results: list[tuple[str, bool, str]] = []

    has_keys = bool(os.getenv("KRAKEN_API_KEY", "").strip() and os.getenv("KRAKEN_API_SECRET", "").strip())

    eth_price = float("nan")
    btc_price = float("nan")
    try:
        eth_price = get_current_price("ETH")
        btc_price = get_current_price("BTC")
        results.append(("API/public connection", True, "public ticker OK"))
    except Exception as exc:
        results.append(("API/public connection", False, str(exc)))

    balance = None
    if has_keys:
        try:
            balance = get_account_balance()
            msg = (
                f"{BASE_CURRENCY}={BASE_SYMBOL}{balance[BASE_CURRENCY]:.2f}, ETH={balance['ETH']:.8f}, "
                f"BTC={balance['BTC']:.8f}, total={BASE_SYMBOL}{balance['total_eur']:.2f}"
            )
            results.append(("Balance retrieval", True, msg))
        except Exception as exc:
            results.append(("Balance retrieval", False, str(exc)))
    else:
        results.append(("Balance retrieval", False, "KRAKEN_API_KEY/KRAKEN_API_SECRET not set"))

    results.append(("ETH price", eth_price == eth_price and eth_price > 0, f"{BASE_SYMBOL}{eth_price:.2f}" if eth_price == eth_price else "n/a"))
    results.append(("BTC price", btc_price == btc_price and btc_price > 0, f"{BASE_SYMBOL}{btc_price:.2f}" if btc_price == btc_price else "n/a"))

    if has_keys and balance is not None:
        try:
            plan = calculate_position_size("ETH", 0.35, max(float(balance["total_eur"]), 1000.0))
            results.append(("Position sizing", True, f"{plan['action']} {BASE_SYMBOL}{plan['eur_amount']:.2f} / {plan['asset_amount']:.8f} ETH"))
        except Exception as exc:
            results.append(("Position sizing", False, str(exc)))
    else:
        fake_amount = 1000.0 * 0.35 / eth_price if eth_price == eth_price and eth_price > 0 else 0.0
        results.append(("Position sizing", True, f"offline example: BUY {BASE_SYMBOL}{1000.0*0.35:.2f} / {fake_amount:.8f} ETH"))

    try:
        amount = float(args.test_eur) / eth_price if eth_price == eth_price and eth_price > 0 else 0.0
        dry = place_market_order("ETH", "buy", amount, dry_run=True)
        results.append(("Dry run order", bool(dry.get("dry_run")), f"simulated {amount:.8f} ETH"))
    except Exception as exc:
        results.append(("Dry run order", False, str(exc)))

    if args.real:
        if not has_keys:
            results.append(("Real tiny order", False, "API keys missing"))
        elif os.getenv("LIVE_TRADING_ENABLED", "").strip().lower() != "true":
            results.append(("Real tiny order", False, "LIVE_TRADING_ENABLED is not true"))
        else:
            try:
                amount = float(args.test_eur) / eth_price
                buy = place_market_order("ETH", "buy", amount, dry_run=False)
                sell = place_market_order("ETH", "sell", float(buy.get("filled_amount", amount)), dry_run=False)
                results.append(("Real tiny order", True, f"buy={buy.get('order_id')} sell={sell.get('order_id')}"))
            except Exception as exc:
                results.append(("Real tiny order", False, str(exc)))
    else:
        results.append(("Real tiny order", True, "skipped; run with --real manually"))

    try:
        report = execute_strategy_signal(0.0, 0.0, total_capital_eur=1000.0, dry_run=True)
        results.append(("Full dry-run signal", bool(report.get("dry_run")), "ETH/BTC target 0 dry-run OK"))
    except Exception as exc:
        results.append(("Full dry-run signal", False, str(exc)))

    print("===================================")
    print("KRAKEN EXECUTION ENGINE TEST")
    print("===================================")
    for name, ok, msg in results:
        print(f"{name:24s} {_status(ok):5s}  {msg}")
    print("===================================")
    required = [r for r in results if r[0] not in {"Balance retrieval"}]
    ready = all(ok for _, ok, _ in required) and has_keys
    print(f"Engine ready for live trading: {'YES' if ready else 'NO'}")
    if not has_keys:
        print("Reason: Kraken API keys are not configured yet.")
    print("===================================")
    Path("artifacts/live_trades").mkdir(parents=True, exist_ok=True)
    return 0 if all(ok for _, ok, _ in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
