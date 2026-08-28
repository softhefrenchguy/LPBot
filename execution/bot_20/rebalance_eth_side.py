import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
from web3 import Web3
import json
import os
import time
from datetime import datetime

# --------------------------
# âš™ï¸ Setup
# --------------------------
RPC_URL = "https://ethereum.publicnode.com"
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Ethereum mainnet addresses
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
FACTORY = Web3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984")
ROUTER  = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
FEE = 500  # 0.05%

# --------------------------
# ðŸ” Load ABIs
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
    raise Exception("âŒ Failed to connect to Ethereum mainnet RPC")
print("âœ… Connected to Ethereum mainnet!\n")

token_weth = w3.eth.contract(address=WETH, abi=erc20_abi)
token_usdc = w3.eth.contract(address=USDC, abi=erc20_abi)
router = w3.eth.contract(address=ROUTER, abi=router_abi)
factory = w3.eth.contract(address=FACTORY, abi=factory_abi)

# --------------------------
# ðŸ§  Helper utilities
# --------------------------
def sign_and_send(tx):
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    tx["maxFeePerGas"] = int(base_fee * 1.2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei("0.01", "gwei"))
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"ðŸ”— Sent transaction: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        print("âš ï¸ Transaction reverted.")
        return None
    print(f"âœ… Confirmed in block {receipt.blockNumber}")
    return receipt


def ensure_allowance(token_addr, amount):
    token = w3.eth.contract(address=token_addr, abi=erc20_abi)
    allowance = token.functions.allowance(WALLET_ADDRESS, ROUTER).call()
    if allowance >= amount:
        return
    print(f"ðŸ›‚ Approving {amount} for routerâ€¦")
    tx = token.functions.approve(ROUTER, amount).build_transaction({
        "from": WALLET_ADDRESS,
        "gas": 200000,
    })
    receipt = sign_and_send(tx)
    if receipt is None:
        raise Exception("âŒ Approval failed")


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
    print("ðŸš€ Executing swapâ€¦")
    receipt = sign_and_send(tx)
    if receipt is None:
        raise Exception("âŒ Swap failed")
    print("âœ… Swap successful!\n")
    return receipt

# --------------------------
# ðŸ“ˆ Pool & price data
# --------------------------
pool_addr = factory.functions.getPool(USDC, WETH, FEE).call()
if int(pool_addr, 16) == 0:
    raise Exception("âŒ USDC/WETH pool (0.05%) not found on Ethereum mainnet")

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
    raise Exception("âŒ Pool tokens are not USDC/WETH")

print(f"ðŸ“Š Price: 1 WETH = {usdc_per_weth:.2f} USDC\n")

# --------------------------
# ðŸ’° Balances & target (ETH-side)
# --------------------------
bal_usdc = token_usdc.functions.balanceOf(WALLET_ADDRESS).call() / 1e6
bal_weth = token_weth.functions.balanceOf(WALLET_ADDRESS).call() / 1e18
print(f"ðŸ’¼ Balances: {bal_usdc:.2f} USDC | {bal_weth:.6f} WETH")

# Target: 95% WETH, 5% USDC
value_usdc_side = bal_usdc
value_weth_side = bal_weth * usdc_per_weth
total_value = value_usdc_side + value_weth_side
target_usdc = total_value * 0.05
target_weth = total_value * 0.95 / usdc_per_weth

print(f"ðŸŽ¯ Target: {target_usdc:.2f} USDC | {target_weth:.6f} WETH equivalent")

# --------------------------
# â™»ï¸ Rebalance Logic
# --------------------------
try:
    pnl_usd = 0.0
    if bal_usdc > target_usdc * 1.05:
        delta_usdc = bal_usdc - target_usdc
        amount_in = int(delta_usdc * 1e6)
        print(f"ðŸ” Swapping {delta_usdc:.2f} USDC â†’ WETH (reducing stable side)")
        ensure_allowance(USDC, amount_in)
        swap_exact_in(USDC, WETH, amount_in)
    elif bal_usdc < target_usdc * 0.95:
        delta_usdc_needed = target_usdc - bal_usdc
        amount_weth = delta_usdc_needed / usdc_per_weth
        amount_in = int(amount_weth * 1e18)
        print(f"ðŸ” Swapping {amount_weth:.6f} WETH â†’ USDC (small top-up)")
        ensure_allowance(WETH, amount_in)
        swap_exact_in(WETH, USDC, amount_in)
    else:
        print("âœ… Within target range â€” no swap performed.")

    # Track PnL
    new_usdc = token_usdc.functions.balanceOf(WALLET_ADDRESS).call() / 1e6
    new_weth = token_weth.functions.balanceOf(WALLET_ADDRESS).call() / 1e18
    total_value_after = new_usdc + new_weth * usdc_per_weth
    pnl_usd = total_value_after - total_value

    print(f"\nðŸ“Š Total before = {total_value:.2f} | after = {total_value_after:.2f}")
    print(f"ðŸ’¹ PnL from rebalance = {pnl_usd:+.2f} USD\n")

except Exception as e:
    print(f"âŒ Rebalance failed: {e}")
    pnl_usd = 0.0

print("ðŸŽ¯ ETH-side rebalance complete.")

# --------------------------
# ðŸ’¾ Record PnL
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
print(f"ðŸ’¾ Recorded PnL data â†’ {data}")
