import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
from web3 import Web3
import os
import json
import time
from decimal import Decimal

# ---------------------------
# âš™ï¸ CONFIG
# ---------------------------
RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")  # USDC/WETH 0.05%
USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
assert w3.is_connected(), "âŒ RPC connection failed"

WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Load ABIs
with open("position_manager_abi.json") as f:
    pm_abi = json.load(f)
with open("pool_abi.json") as f:
    pool_abi = json.load(f)
with open("erc20_abi.json") as f:
    erc20_abi = json.load(f)


pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
usdc = w3.eth.contract(address=USDC, abi=erc20_abi)
weth = w3.eth.contract(address=WETH, abi=erc20_abi)

# ---------------------------
# ðŸ§® Price helpers
# ---------------------------
def get_tick():
    return pool.functions.slot0().call()[1]

def get_price_usd():
    sqrt_price_x96 = pool.functions.slot0().call()[0]
    return (sqrt_price_x96 / 2**96) ** 2 * 1e12

# ---------------------------
# ðŸ§  Adaptive range logic
# ---------------------------
def map_duration_to_range(duration_min):
    # Map 10â€“60 min â†’ Â±40 â†’ Â±15 ticks
    return int(40 - ((duration_min - 10) / 50) * (40 - 15))

# ---------------------------
# ðŸš€ Create new position
# ---------------------------
print("âœ… Connected to Ethereum mainnet!")

price = get_price_usd()
tick = get_tick()
print(f"ðŸ’° Wallet value before LP create = {price:.4f} USDC per WETH")

# Determine range width (default Â±50 if no previous data)
if os.path.exists("last_cycle_data.json"):
    
    nonce += 1
    print(f"ðŸ”— Approved token {token.address[-4:]}... | {tx_hash.hex()}")
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Mint LP
tx = pm.functions.mint(mint_params).build_transaction({
    "from": WALLET,
    "nonce": nonce,
    "gas": 800000,

})

    print(f"ðŸ”— Tx {tx_hash.hex()} | â›½ {receipt['gasUsed']:,} @ 0.02 gwei â†’ {receipt['gasUsed'] * 0.02e-9:.6f} ETH")
except Exception as e:
    print(f"âŒ Mint failed: {e}")
    raise SystemExit

# Extract new token ID
logs = pm.events.IncreaseLiquidity().process_receipt(receipt)
token_id = logs[0]["args"]["tokenId"] if logs else None
if not token_id:
    print("âš ï¸ Could not extract token ID â€” fallback to last ID")
    token_id = int(pm.functions.nextTokenId().call()) - 1

print(f"âœ… LP position created! NFT #{token_id}")

# Save data
with open("last_cycle_data.json", "w") as f:
    json.dump({
        "timestamp_create": time.time(),
        "lower_tick": lower_tick,
        "upper_tick": upper_tick,
        "lower_bound_usd": price * (1 - (range_width / 10000)),
        "upper_bound_usd": price * (1 + (range_width / 10000)),
        "tokenId": token_id,
    }, f, indent=2)

print(f"ðŸ§¾ Saved centered bounds â†’ ${price * (1 - (range_width / 10000)):.2f} â†’ ${price * (1 + (range_width / 10000)):.2f}")
print("âœ… create_position.py completed.")












# --- CLEAN TX PATCH FROM BOT_10 ---

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"ðŸ”— Tx {tx_hash.hex()} | â›½ {receipt['gasUsed']:,} @ 0.02 gwei â†’ {receipt['gasUsed'] * 0.02e-9:.6f} ETH")
except Exception as e:
    print(f"âŒ Mint failed: {e}")
    raise SystemExit

# Extract new token ID
logs = pm.events.IncreaseLiquidity().process_receipt(receipt)
token_id = logs[0]["args"]["tokenId"] if logs else None
if not token_id:
    print("âš ï¸ Could not extract token ID â€” fallback to last ID")
    token_id = int(pm.functions.nextTokenId().call()) - 1

print(f"âœ… LP position created! NFT #{token_id}")

# Save data
with open("last_cycle_data.json", "w") as f:
    json.dump({
        "timestamp_create": time.time(),
        "lower_tick": lower_tick,
        "upper_tick": upper_tick,
        "lower_bound_usd": price * (1 - (range_width / 10000)),
        "upper_bound_usd": price * (1 + (range_width / 10000)),
        "tokenId": token_id,
    }, f, indent=2)

print(f"ðŸ§¾ Saved centered bounds â†’ ${price * (1 - (range_width / 10000)):.2f} â†’ ${price * (1 + (range_width / 10000)):.2f}")
print("âœ… create_position.py completed.")













print(f"✅ Tx sent! Hash: {tx_hash.hex()}")
# --- END PATCH ---

# --- TX SIGN + SEND WITH NONCE RETRY ---
nonce = w3.eth.get_transaction_count(WALLET)
tx["nonce"] = nonce
signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)

try:
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
except Exception as e:
    if "nonce too low" in str(e):
        print("⚠️ Nonce too low — retrying with next nonce...")
        tx["nonce"] = w3.eth.get_transaction_count(WALLET)
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    else:
        raise

receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"✅ Tx sent! Hash: {tx_hash.hex()}")
# --- END PATCH ---
