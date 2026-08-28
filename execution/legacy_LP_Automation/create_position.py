from dotenv import load_dotenv
load_dotenv()

import json
from web3 import Web3
import os
import time
from datetime import datetime

# --------------------------
# Setup
# --------------------------
RPC_URL = "https://arb1.arbitrum.io/rpc"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("❌ Failed to connect to Arbitrum RPC")
print("✅ Connected to Arbitrum!")

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

POSITION_MANAGER_ADDRESS = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS = Web3.to_checksum_address("0xC6962004f452bE9203591991D15f6b388e09E8D0")  # WETH/USDC 0.05%

# --------------------------
# Load ABIs
# --------------------------
def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

pm_abi = load_json("NonfungiblePositionManager.json")
pool_abi = load_json("pool_abi.json")
erc20_abi = load_json("erc20_abi.json")

pm = w3.eth.contract(address=POSITION_MANAGER_ADDRESS, abi=pm_abi)
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)

# --------------------------
# Helpers
# --------------------------
def align_tick(tick, spacing, down=True):
    r = tick % spacing
    return tick - r if down else tick - r + spacing

def send(tx):
    tx.setdefault("from", WALLET_ADDRESS)
    tx.setdefault("gas", 600_000)
    tx.setdefault("maxFeePerGas", int(w3.eth.gas_price * 2))
    tx.setdefault("maxPriorityFeePerGas", int(w3.to_wei("0.01", "gwei")))
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(txh)
    if r.status != 1:
        raise Exception(f"❌ Transaction failed: {txh.hex()}")
    return r

def allow(token_addr, spender, amount):
    token = w3.eth.contract(address=token_addr, abi=erc20_abi)
    current = token.functions.allowance(WALLET_ADDRESS, spender).call()
    if current >= amount:
        return
    tx = token.functions.approve(spender, amount).build_transaction({
        "from": WALLET_ADDRESS,
        "gas": 150_000,
        "maxFeePerGas": int(w3.eth.gas_price * 2),
        "maxPriorityFeePerGas": int(w3.to_wei("0.01", "gwei")),
        "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS)
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(txh)
    if r.status != 1:
        raise Exception(f"❌ Approval failed: {txh.hex()}")

# --------------------------
# Adaptive Range Width (smooth scaling)
# --------------------------
def calculate_range_width():
    RANGE_MIN = 5   # Tightest possible
    RANGE_MAX = 25   # Widest possible
    RANGE_DEFAULT = 10
    TARGET_MINUTES = 10
    TARGET_MAX_MINUTES = 60

    try:
        if os.path.exists("last_cycle_data.json"):
            with open("last_cycle_data.json", "r") as f:
                last = json.load(f)
                start = datetime.strptime(last["timestamp_create"], "%Y-%m-%d %H:%M:%S UTC")
                end = datetime.strptime(last["timestamp_withdraw"], "%Y-%m-%d %H:%M:%S UTC")
                duration_min = (end - start).total_seconds() / 60

                # Linear interpolation: 10 min → 40 ticks, 60 min → 15 ticks
                if duration_min <= TARGET_MINUTES:
                    width = RANGE_MAX
                elif duration_min >= TARGET_MAX_MINUTES:
                    width = RANGE_MIN
                else:
                    # scale smoothly between RANGE_MAX and RANGE_MIN
                    ratio = (duration_min - TARGET_MINUTES) / (TARGET_MAX_MINUTES - TARGET_MINUTES)
                    width = RANGE_MAX - (RANGE_MAX - RANGE_MIN) * ratio

                width = round(width)
                print(f"⚙️] Adaptive range width: ±{width} ticks (last cycle {duration_min:.1f} min)")
                return width
        else:
            print("[⚙️] No previous data — default ±20 ticks.")
            return RANGE_DEFAULT
    except Exception as e:
        print(f"[⚠️] Adaptive width failed — defaulting to 20 ticks: {e}")
        return RANGE_DEFAULT


range_width = calculate_range_width()

# --------------------------
# Main logic
# --------------------------
def main():
    slot0 = pool.functions.slot0().call()
    current_tick = slot0[1]
    spacing = pool.functions.tickSpacing().call()

    tick_lower = align_tick(current_tick - range_width, spacing, True)
    tick_upper = align_tick(current_tick + range_width, spacing, False)
    print(f"📊 Using ticks → Lower: {tick_lower}, Upper: {tick_upper}")

    token0 = pool.functions.token0().call()
    token1 = pool.functions.token1().call()
    t0 = w3.eth.contract(address=token0, abi=erc20_abi)
    t1 = w3.eth.contract(address=token1, abi=erc20_abi)
    dec0 = t0.functions.decimals().call()
    dec1 = t1.functions.decimals().call()

    bal0 = t0.functions.balanceOf(WALLET_ADDRESS).call()
    bal1 = t1.functions.balanceOf(WALLET_ADDRESS).call()
    amt0 = int(bal0 * 0.95)
    amt1 = int(bal1 * 0.95)

    allow(token0, POSITION_MANAGER_ADDRESS, amt0)
    allow(token1, POSITION_MANAGER_ADDRESS, amt1)

    print(f"💰 Using {amt0/10**dec0:.6f} of token0 and {amt1/10**dec1:.6f} of token1 for liquidity...")

    params = {
        "token0": token0,
        "token1": token1,
        "fee": 500,
        "tickLower": tick_lower,
        "tickUpper": tick_upper,
        "amount0Desired": amt0,
        "amount1Desired": amt1,
        "amount0Min": 0,
        "amount1Min": 0,
        "recipient": WALLET_ADDRESS,
        "deadline": int(time.time()) + 600
    }

    tx = pm.functions.mint(params).build_transaction({
        "from": WALLET_ADDRESS
    })
    r = send(tx)
    print(f"✅ LP position created! Tx hash: {r.transactionHash.hex()}")

    # Extract tokenId
    token_id = None
    try:
        events = pm.events.Mint().process_receipt(r)
        if not events:
            events = pm.events.IncreaseLiquidity().process_receipt(r)
        if events:
            token_id = int(events[0]["args"]["tokenId"])
            print(f"🔢 New LP token ID: {token_id}")
    except Exception:
        try:
            new_balance = pm.functions.balanceOf(WALLET_ADDRESS).call()
            token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, new_balance - 1).call()
            print(f"🪙 New LP token ID detected (via fallback): {token_id}")
        except Exception as e2:
            print(f"Warning: Could not determine tokenId via fallback: {e2}")

    # Record metadata
    data = {}
    if os.path.exists("last_cycle_data.json"):
        with open("last_cycle_data.json", "r") as f:
            data = json.load(f)

    data["timestamp_create"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    data["range_width"] = range_width
    if token_id:
        data["last_created_token_id"] = token_id

    with open("last_cycle_data.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
