#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from web3 import Web3

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = os.getenv("RPC_URL", "https://ethereum.publicnode.com")
POOL_ADDRESS = Web3.to_checksum_address(
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # WETH/USDC 0.05% Ethereum mainnet
)

PRICE_FEED_FILE = os.getenv("PRICE_FEED_FILE")
if PRICE_FEED_FILE and not os.path.isabs(PRICE_FEED_FILE):
    PRICE_FEED_FILE = os.path.join(script_dir, PRICE_FEED_FILE)
if not PRICE_FEED_FILE:
    PRICE_FEED_FILE = os.path.join(script_dir, "price_feed.json")

CHECK_INTERVAL_SEC = float(os.getenv("PRICE_FEED_INTERVAL_SEC", "3"))

w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"proxies": {"http": None, "https": None}}))
if not w3.is_connected():
    raise SystemExit("Failed to connect to Ethereum mainnet RPC")

with open(os.path.join(script_dir, "pool_abi.json"), "r", encoding="utf-8") as f:
    pool_abi = json.load(f)

pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)


def get_eth_price():
    slot0 = pool.functions.slot0().call()
    sqrt_price_x96 = slot0[0]
    return float((sqrt_price_x96 / (2 ** 96)) ** 2 * 1e12)


def write_price(price):
    payload = {
        "ts": time.time(),
        "price": float(price),
        "iso": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    tmp_path = PRICE_FEED_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, PRICE_FEED_FILE)
    except PermissionError:
        try:
            with open(PRICE_FEED_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"[price_feed] write failed: {e}")


def main():
    print(f"[price_feed] writing to {PRICE_FEED_FILE} every {CHECK_INTERVAL_SEC:.1f}s")
    while True:
        try:
            price = get_eth_price()
            write_price(price)
        except Exception as e:
            print(f"[price_feed] price fetch failed: {e}")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
