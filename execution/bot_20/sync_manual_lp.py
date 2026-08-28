#!/usr/bin/env python3
"""
sync_manual_lp.py

Synchronises last_cycle_data.json with whatever LP NFT
you currently hold in the wallet.

Use this after manually calling create_position.py or
manually minting a position in any way.

Steps:
1. Detect the NFT the wallet owns.
2. Pull lowerTick, upperTick, liquidity, tokens owed.
3. Compute USD price bounds from ticks.
4. Update last_cycle_data.json so main_auto_v3 knows
   lp_active = True with correct ticks and band.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3

# -------------------------
# Load ENV
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS      = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")

LAST_CYCLE_FILE = os.path.join(script_dir, "last_cycle_data.json")

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
if not WALLET_ADDRESS:
    raise SystemExit("❌ WALLET_ADDRESS missing in .env")

WALLET_ADDRESS = Web3.to_checksum_address(WALLET_ADDRESS)

# -------------------------
# Web3
# -------------------------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("❌ Cannot connect to Ethereum mainnet RPC")

print("✅ Connected to Ethereum mainnet")

# -------------------------
# Load ABIs
# -------------------------
with open(os.path.join(script_dir, "NonfungiblePositionManager.json"), "r") as f:
    pm_abi = json.load(f)

with open(os.path.join(script_dir, "pool_abi.json"), "r") as f:
    pool_abi = json.load(f)

pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)

# -------------------------
# Helpers
# -------------------------
def sqrtPriceX96_to_price(sqrtPriceX96):
    return float((sqrtPriceX96 / 2**96)**2 * 1e12)  # USDC per WETH

def tick_to_price(tick):
    sqrtRatioX96 = (1.0001 ** (tick / 2)) * (2**96)
    return sqrtPriceX96_to_price(sqrtRatioX96)

# -------------------------
# Detect owned LP NFT
# -------------------------
balance = pm.functions.balanceOf(WALLET_ADDRESS).call()

if balance == 0:
    print("⚠️ Wallet owns NO LP NFT → Setting lp_active = False.")
    with open(LAST_CYCLE_FILE, "w") as f:
        json.dump({"lp_active": False}, f, indent=2)
    raise SystemExit()

# If multiple, pick the latest mint (highest tokenId)
token_ids = []
for i in range(balance):
    tid = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, i).call()
    token_ids.append(int(tid))

token_id = max(token_ids)
print(f"🔎 Found LP NFT tokenId = {token_id}")

# -------------------------
# Fetch NFT details
# -------------------------
pos = pm.functions.positions(token_id).call()

liquidity      = pos[7]
tick_lower     = pos[5]
tick_upper     = pos[6]

print(f"📌 lower_tick = {tick_lower}, upper_tick = {tick_upper}, liquidity = {liquidity}")

# -------------------------
# Current price
# -------------------------
slot0 = pool.functions.slot0().call()
sqrt_price = slot0[0]
spot_price = sqrtPriceX96_to_price(sqrt_price)

# Compute USD bounds
lower_price = tick_to_price(tick_lower)
upper_price = tick_to_price(tick_upper)

lower_price = float(lower_price)
upper_price = float(upper_price)

print(f"📐 Computed band: ${lower_price:.2f} → ${upper_price:.2f}")
print(f"📉 Spot price:   ${spot_price:.2f}")

# -------------------------
# Save into last_cycle_data.json
# -------------------------
new_data = {
    "timestamp_sync": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "lp_active": True,
    "tokenId": token_id,
    "lower_tick": tick_lower,
    "upper_tick": tick_upper,
    "lower_bound_usd": round(lower_price, 2),
    "upper_bound_usd": round(upper_price, 2),
    "spot_price": spot_price,
}

with open(LAST_CYCLE_FILE, "w") as f:
    json.dump(new_data, f, indent=2)

print("✅ last_cycle_data.json updated & synchronized.")
print("🎯 main_auto_v3 will now track this LP correctly.")
