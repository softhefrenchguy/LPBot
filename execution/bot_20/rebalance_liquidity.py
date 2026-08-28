#!/usr/bin/env python3
"""
rebalance_liquidity.py — FORCE rebalance wallet to ETH-only.

Behavior:
- Reads wallet balances directly
- If USDC balance > $5 → swaps ALL USDC → WETH
- Stateless: does NOT rely on any JSON / cycle / LP data
- Safe to run anytime
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from web3 import Web3
import logging

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.getLogger("web3").setLevel(logging.ERROR)

# -------------------------------------------------
# ENV / CONFIG
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = "https://ethereum.publicnode.com"

WETH_ADDR = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC_ADDR = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")

# Uniswap v3 SwapRouter (v1) on Ethereum mainnet — matches the minimal router ABI below
SWAP_ROUTER_ADDR = Web3.to_checksum_address(
    "0xE592427A0AEce92De3Edee1F18E0157C05861564"
)

# Try common Uniswap v3 fee tiers
FEE_TIERS = [500, 3000, 10000]

MIN_USDC_TO_SWAP = 5.0  # don't bother swapping dust

# -------------------------------------------------
# WEB3 SETUP
# -------------------------------------------------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("❌ Failed to connect to Ethereum mainnet RPC")

print("✅ Connected to Ethereum mainnet RPC")

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise SystemExit("❌ Missing WALLET_ADDRESS or PRIVATE_KEY in .env")

WALLET_ADDRESS = Web3.to_checksum_address(WALLET_ADDRESS)

# Load ABIs
with open(os.path.join(script_dir, "erc20_abi.json"), "r") as f:
    erc20_abi = json.load(f)

with open(os.path.join(script_dir, "uniswap_v3_router_abi.json"), "r") as f:
    router_abi = json.load(f)

weth = w3.eth.contract(address=WETH_ADDR, abi=erc20_abi)
usdc = w3.eth.contract(address=USDC_ADDR, abi=erc20_abi)
router = w3.eth.contract(address=SWAP_ROUTER_ADDR, abi=router_abi)

WETH_DEC = weth.functions.decimals().call()
USDC_DEC = usdc.functions.decimals().call()

# -------------------------------------------------
# TX HELPERS
# -------------------------------------------------
def sign_and_send(tx, gas_limit=350_000):
    tx["from"] = WALLET_ADDRESS
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)

    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)

    tx["maxFeePerGas"] = int(base_fee * 2)
    try:
        priority = w3.eth.max_priority_fee
    except Exception:
        priority = w3.to_wei("0.05", "gwei")
    tx["maxPriorityFeePerGas"] = int(max(priority, w3.to_wei("0.05", "gwei")))
    tx["gas"] = gas_limit

    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"🔗 Tx {tx_hash.hex()} | ⛽ {receipt.gasUsed} | status={receipt.status}")
    return receipt

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
def main():
    weth_bal = weth.functions.balanceOf(WALLET_ADDRESS).call() / 10**WETH_DEC
    usdc_raw = usdc.functions.balanceOf(WALLET_ADDRESS).call()
    usdc_bal = usdc_raw / 10**USDC_DEC

    print(f"💼 Wallet BEFORE → {weth_bal:.6f} WETH | {usdc_bal:.2f} USDC")

    if usdc_bal < MIN_USDC_TO_SWAP:
        print("ℹ️ USDC balance too small — nothing to rebalance.")
        return

    print(f"🔄 Rebalancing {usdc_bal:.2f} USDC → WETH")

    # -------------------------------------------------
    # Approve router
    # -------------------------------------------------
    tx = usdc.functions.approve(
        SWAP_ROUTER_ADDR,
        usdc_raw
    ).build_transaction({
        "from": WALLET_ADDRESS
    })

    receipt = sign_and_send(tx, gas_limit=100_000)
    if receipt.status != 1:
        print("❌ USDC approval failed.")
        return

    print("✅ USDC approved for router.")

    # -------------------------------------------------
    # Try swap on different fee tiers
    # -------------------------------------------------
    swapped = False

    for fee in FEE_TIERS:
        try:
            params = {
                "tokenIn": USDC_ADDR,
                "tokenOut": WETH_ADDR,
                "fee": fee,
                "recipient": WALLET_ADDRESS,
                "deadline": int(time.time()) + 600,
                "amountIn": usdc_raw,
                "amountOutMinimum": 0,
                "sqrtPriceLimitX96": 0,
            }

            print(f"➡️ Trying Uniswap v3 swap (fee={fee})")

            tx = router.functions.exactInputSingle(
                params
            ).build_transaction({
                "from": WALLET_ADDRESS
            })

            receipt = sign_and_send(tx)

            if receipt.status == 1:
                print(f"✅ Swap successful (fee={fee})")
                swapped = True
                break
        except Exception as e:
            print(f"⚠️ Swap failed (fee={fee}): {e}")

    # -------------------------------------------------
    # Final balances
    # -------------------------------------------------
    weth_after = weth.functions.balanceOf(WALLET_ADDRESS).call() / 10**WETH_DEC
    usdc_after = usdc.functions.balanceOf(WALLET_ADDRESS).call() / 10**USDC_DEC

    print(f"🔍 Wallet AFTER → {weth_after:.6f} WETH | {usdc_after:.2f} USDC")

    if swapped and usdc_after < 1:
        print("✅ Rebalance SUCCESS — ETH-only achieved.")
    else:
        print("❌ Rebalance FAILED — still holding USDC.")

# -------------------------------------------------
if __name__ == "__main__":
    main()
