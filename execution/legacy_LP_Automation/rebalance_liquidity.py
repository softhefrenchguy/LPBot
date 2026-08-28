from dotenv import load_dotenv
load_dotenv()

from web3 import Web3
import json
import os
import time
from datetime import datetime

# --------------------------
# ⚙️ Setup
# --------------------------
RPC_URL = "https://arb1.arbitrum.io/rpc"
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Arbitrum addresses
WETH = Web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")
USDC = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
FACTORY = Web3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984")
ROUTER  = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
FEE = 500  # 0.05%

# --------------------------
# 🔍 Load ABIs
# --------------------------
def load_abi(filename):
    with open(filename, "r") as f:
        return json.load(f)

erc20_abi = load_abi("erc20_abi.json")
router_abi = load_abi("swap_router_abi.json")
factory_abi = load_abi("uniswap_v3_factory_abi.json")
pool_abi = load_abi("pool_abi.json")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("❌ Failed to connect to Arbitrum RPC")
print("✅ Connected to Arbitrum!\n")

token_weth = w3.eth.contract(address=WETH, abi=erc20_abi)
token_usdc = w3.eth.contract(address=USDC, abi=erc20_abi)
router = w3.eth.contract(address=ROUTER, abi=router_abi)
factory = w3.eth.contract(address=FACTORY, abi=factory_abi)

# --------------------------
# 🧠 Helper utilities
# --------------------------
def sign_and_send(tx):
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    tx["maxFeePerGas"] = int(base_fee * 1.2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei("0.01", "gwei"))
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"🔗 Sent transaction: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        print("⚠️ Transaction reverted.")
        return None
    print(f"✅ Confirmed in block {receipt.blockNumber}")
    return receipt


def ensure_allowance(token_addr, amount):
    token = w3.eth.contract(address=token_addr, abi=erc20_abi)
    allowance = token.functions.allowance(WALLET_ADDRESS, ROUTER).call()
    if allowance >= amount:
        return
    print(f"🛂 Approving {amount} for router…")
    tx = token.functions.approve(ROUTER, amount).build_transaction({
        "from": WALLET_ADDRESS,
        "gas": 200000,
    })
    receipt = sign_and_send(tx)
    if receipt is None:
        raise Exception("❌ Approval failed")


def swap_exact_in(token_in, token_out, amount_in_wei):
    params = (
        token_in,
        token_out,
        FEE,
        WALLET_ADDRESS,
        int(time.time()) + 1200,
        int(amount_in_wei),
        0,
        0
    )
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": WALLET_ADDRESS,
        "gas": 600000,
    })
    print("🚀 Executing swap…")
    receipt = sign_and_send(tx)
    if receipt is None:
        raise Exception("❌ Swap failed")
    print("✅ Swap successful!\n")
    return receipt

# --------------------------
# 📈 Pool & price data
# --------------------------
pool_addr = factory.functions.getPool(USDC, WETH, FEE).call()
if int(pool_addr, 16) == 0:
    raise Exception("❌ USDC/WETH pool (0.05%) not found on Arbitrum")

pool = w3.eth.contract(address=pool_addr, abi=pool_abi)
token0_addr = pool.functions.token0().call()
token1_addr = pool.functions.token1().call()
token0 = w3.eth.contract(address=token0_addr, abi=erc20_abi)
token1 = w3.eth.contract(address=token1_addr, abi=erc20_abi)
dec0 = token0.functions.decimals().call()
dec1 = token1.functions.decimals().call()

sqrtP = pool.functions.slot0().call()[0]
price_1_per_0 = (sqrtP / (2 ** 96)) ** 2 * (10 ** (dec0 - dec1))

if token0_addr.lower() == USDC.lower() and token1_addr.lower() == WETH.lower():
    usdc_per_weth = 1.0 / price_1_per_0
elif token0_addr.lower() == WETH.lower() and token1_addr.lower() == USDC.lower():
    usdc_per_weth = price_1_per_0
else:
    raise Exception("❌ Pool tokens are not USDC/WETH")

print(f"📊 Price: 1 WETH = {usdc_per_weth:.2f} USDC\n")

# --------------------------
# 💰 Balances & targets
# --------------------------
bal_usdc = token_usdc.functions.balanceOf(WALLET_ADDRESS).call() / 1e6
bal_weth = token_weth.functions.balanceOf(WALLET_ADDRESS).call() / 1e18
print(f"💼 Balances: {bal_usdc:.2f} USDC | {bal_weth:.6f} WETH")

value_usdc_side = bal_usdc
value_weth_side = bal_weth * usdc_per_weth
total_value_before = value_usdc_side + value_weth_side
target_each = total_value_before / 2.0
THRESHOLD = 0.01  # 1% drift tolerance

# --------------------------
# ♻️ Rebalance logic
# --------------------------
drift = abs(value_usdc_side - value_weth_side) / total_value_before
if drift < THRESHOLD:
    print("✅ Portfolio already ~50/50. No action needed.")
    pnl_usd = 0.0
else:
    try:
        if value_usdc_side > target_each:
            delta_usdc = value_usdc_side - target_each
            amount_in = int(delta_usdc * 1e6)
            print(f"🔁 Swapping {delta_usdc:.2f} USDC → WETH (amount_in={amount_in})")
            ensure_allowance(USDC, amount_in)
            swap_exact_in(USDC, WETH, amount_in)
        else:
            delta_usdc_needed = target_each - value_usdc_side
            amount_weth = delta_usdc_needed / usdc_per_weth
            amount_in = int(amount_weth * 1e18)
            print(f"🔁 Swapping {amount_weth:.6f} WETH → USDC (amount_in={amount_in})")
            ensure_allowance(WETH, amount_in)
            swap_exact_in(WETH, USDC, amount_in)

        new_usdc = token_usdc.functions.balanceOf(WALLET_ADDRESS).call() / 1e6
        new_weth = token_weth.functions.balanceOf(WALLET_ADDRESS).call() / 1e18
        total_value_after = new_usdc + new_weth * usdc_per_weth
        pnl_usd = total_value_after - total_value_before

        print(f"\n📊 Total before = {total_value_before:.2f} | after = {total_value_after:.2f}")
        print(f"💹 PnL from rebalance = {pnl_usd:+.2f} USD\n")

    except Exception as e:
        print(f"❌ Rebalance failed: {e}")
        pnl_usd = 0.0

print("🎯 Rebalance complete.")

# --------------------------
# 💾 Record PnL
# --------------------------
try:
    with open("last_cycle_data.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    data = {}

data["pnl_usd"] = round(pnl_usd, 2)
data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

from logger_utils import save_cycle_data
save_cycle_data(data, append=True)
print(f"💾 Recorded PnL data → {data}")

# --------------------------
# 📏 Dynamic Range Width Logic
# --------------------------
RANGE_BASE = 20
RANGE_WIDE = 40
RANGE_TIGHT = 15

try:
    if os.path.exists("last_cycle_data.json"):
        with open("last_cycle_data.json", "r") as f:
            last_cycle = json.load(f)
            start = datetime.strptime(last_cycle["timestamp_create"], "%Y-%m-%d %H:%M:%S UTC")
            end = datetime.strptime(last_cycle["timestamp_withdraw"], "%Y-%m-%d %H:%M:%S UTC")
            duration_min = (end - start).total_seconds() / 60

            if duration_min < 10:
                range_width = RANGE_WIDE
            elif duration_min > 60:
                range_width = RANGE_TIGHT
            else:
                range_width = RANGE_BASE

            with open("next_range_width.txt", "w") as f:
                f.write(str(range_width))

            print(f"[📊] Cycle lasted {duration_min:.1f} min → next range width = {range_width} ticks")
    else:
        print("[⚠️] No last_cycle_data.json found — keeping default 20 ticks.")
except Exception as e:
    print(f"[⚠️] Could not determine next range width: {e}")


