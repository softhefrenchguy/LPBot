from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import json
import sys
import time
from decimal import Decimal
from datetime import datetime, timezone
from web3 import Web3
from web3.exceptions import ContractLogicError

# ---------- RPC / ADDRS ----------
RPC_URL = "https://arb1.arbitrum.io/rpc"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

PM_ADDRESS = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")  # NonfungiblePositionManager
USDC = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
WETH = Web3.to_checksum_address("0x82af49447d8a07e3bd95bd0d56f35241523fbab1")

# Load ABIs (bundled JSONs next to this script)
def load_abi(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), "r", encoding="utf-8") as f:
        return json.load(f)

pm_abi = load_abi("position_manager_abi.json")
erc20_abi = load_abi("erc20_abi.json")


pm = w3.eth.contract(address=PM_ADDRESS, abi=pm_abi)
usdc = w3.eth.contract(address=USDC, abi=erc20_abi)
weth = w3.eth.contract(address=WETH, abi=erc20_abi)

# ---------- Helpers ----------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def load_last_cycle():
    try:
        with open(os.path.join(os.path.dirname(__file__), "last_cycle_data.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_last_cycle(data):
    with open(os.path.join(os.path.dirname(__file__), "last_cycle_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def print_status(msg):
    print(msg, flush=True)

def build_and_send(tx):
    tx = tx.build_transaction({
        "from": WALLET_ADDRESS,
        "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
        "maxFeePerGas": int(w3.eth.get_block("latest")["baseFeePerGas"] * 12 // 10),  # base * 1.2
        "maxPriorityFeePerGas": w3.to_wei("0.01", "gwei"),
        "chainId": 42161,
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash.hex(), receipt

def get_liquidity(token_id: int) -> int:
    # NonfungiblePositionManager.positions(tokenId) returns a struct; liquidity is index 7
    pos = pm.functions.positions(token_id).call()
    return int(pos[7])

def find_active_positions_owned() -> list[int]:
    # iterate owned NFTs by scanning events is heavy; instead rely on last created token if present
    # This fallback attempts to read a lot: we’ll keep it simple using last_cycle_data.json
    return []

def amount_to_usd(amount0, amount1, eth_price_usdc=3700.0) -> float:
    # USDC is 6 decimals, WETH is 18. Fees come in raw token units.
    a0 = Decimal(amount0) / Decimal(1e6)   # USDC
    a1 = Decimal(amount1) / Decimal(1e18)  # WETH
    return float(a0 + a1 * Decimal(eth_price_usdc))

# ---------- Core ----------
def main():
    print_status("✅ Connected to Arbitrum!")

    data = load_last_cycle()
    last_token = data.get("last_created_token_id")
    tokens_to_process = []

    if isinstance(last_token, int):
        tokens_to_process = [last_token]
    elif isinstance(last_token, str) and last_token.isdigit():
        tokens_to_process = [int(last_token)]
    else:
        # If we cannot find the last created token ID, we STOP to avoid touching wrong NFTs.
        print_status("⚠️ No last_created_token_id in last_cycle_data.json — refusing to proceed.")
        data["withdraw_ok"] = False
        data["failed_token_ids"] = []
        data["timestamp_withdraw"] = now_utc()
        save_last_cycle(data)
        sys.exit(1)

    print_status(f"🔍 Wallet owns (unknown count); targeting only last created token.")
    print_status(f"🎯 Positions to process: {tokens_to_process}")

    failed_tokens: list[int] = []
    total_fees0 = 0
    total_fees1 = 0

    for token_id in tokens_to_process:
        try:
            liq_before = get_liquidity(token_id)
            if liq_before == 0:
                print_status(f"ℹ️ Token {token_id} already has 0 liquidity — skipping decrease.")
            else:
                # 1) Pre-collect
                try:
                    params = (token_id, WALLET_ADDRESS, 2**128 - 1, 2**128 - 1)
                    txh, _ = build_and_send(pm.functions.collect(params))
                    print_status(f"💰 Fees collected (pre-withdraw). Tx: {txh}")
                except Exception as e:
                    print_status(f"⚠️ Pre-collect failed for {token_id}: {e}")

                # 2) Decrease liquidity FULL
                try:
                    dec_params = (token_id, liq_before, 0, 0, int(time.time()) + 600)
                    txh, _ = build_and_send(pm.functions.decreaseLiquidity(dec_params))
                    print_status(f"💧 Liquidity withdrawn for token {token_id}. Tx: {txh}")
                except ContractLogicError as e:
                    print_status(f"❌ decreaseLiquidity reverted for {token_id}: {e}")
                    failed_tokens.append(token_id)
                    continue
                except Exception as e:
                    print_status(f"❌ decreaseLiquidity error for {token_id}: {e}")
                    failed_tokens.append(token_id)
                    continue

            # 3) Post-collect all (to pull any owed tokens)
            try:
                params = (token_id, WALLET_ADDRESS, 2**128 - 1, 2**128 - 1)
                txh, receipt = build_and_send(pm.functions.collect(params))
                print_status(f"💸 Collected fees (post-withdraw). Tx: {txh}")

                # Try to parse logs best-effort (some ABIs mismatch; treat as best-effort)
                # We won't fail the run on log decode mismatches.
            except Exception as e:
                print_status(f"⚠️ Post-collect failed for {token_id}: {e}")

            # 4) Verify on-chain that liquidity is zero
            liq_after = get_liquidity(token_id)
            if liq_after != 0:
                print_status(f"❌ Verification failed: token {token_id} liquidity still {liq_after}")
                failed_tokens.append(token_id)
            else:
                print_status(f"✅ Verified: token {token_id} liquidity is now 0")

        except Exception as e:
            print_status(f"❌ Error with token {token_id}: {e}")
            failed_tokens.append(token_id)

    # Record summary
    data["timestamp_withdraw"] = now_utc()
    data["updated_at"] = data["timestamp_withdraw"]
    data["fees_collected_usd"] = data.get("fees_collected_usd", 0.0)  # leave as-is unless you compute precisely
    data["failed_token_ids"] = failed_tokens
    data["withdraw_ok"] = len(failed_tokens) == 0
    save_last_cycle(data)

    # Exit code drives main.py behavior
    if data["withdraw_ok"]:
        print_status("✅ Withdraw phase finished successfully.")
        sys.exit(0)
    else:
        print_status(f"🚨 Withdraw phase FAILED for tokens: {failed_tokens}. Aborting cycle.")
        sys.exit(1)

if __name__ == "__main__":
    # Hard safety: refuse to run without env
    if not WALLET_ADDRESS or not PRIVATE_KEY:
        print("❌ WALLET_ADDRESS / PRIVATE_KEY missing from .env")
        sys.exit(1)
    main()
