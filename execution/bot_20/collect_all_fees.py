import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import json
import os
import time
from datetime import datetime
from web3 import Web3

# --------------------------
# âš™ï¸ Setup
# --------------------------
RPC_URL = "https://ethereum.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("âŒ Could not connect to Ethereum mainnet RPC")
print("âœ… Connected to Ethereum mainnet!")

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
FACTORY = Web3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984")
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
POOL_FEE_BPS = 500

# --------------------------
# Load ABIs
# --------------------------
def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

pm_abi = load_json("NonfungiblePositionManager.json")
factory_abi = load_json("uniswap_v3_factory_abi.json")
pool_abi = load_json("pool_abi.json")

pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)
factory = w3.eth.contract(address=FACTORY, abi=factory_abi)

# --------------------------
# Helpers
# --------------------------
def sign_and_send(tx):
    tx.setdefault("from", WALLET_ADDRESS)
    tx.setdefault("nonce", w3.eth.get_transaction_count(WALLET_ADDRESS))
    tx.setdefault("gas", 500_000)
    tx.setdefault("maxFeePerGas", int(w3.eth.gas_price * 2))
    tx.setdefault("maxPriorityFeePerGas", int(w3.to_wei("0.01", "gwei")))
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(tx_hash)
    if r.status != 1:
        raise Exception(f"Transaction reverted: {tx_hash.hex()}")
    return r

def get_usdc_per_weth():
    pool_addr = factory.functions.getPool(USDC, WETH, POOL_FEE_BPS).call()
    pool = w3.eth.contract(address=pool_addr, abi=pool_abi)
    sqrtP = pool.functions.slot0().call()[0]
    token0 = pool.functions.token0().call()
    token1 = pool.functions.token1().call()
    price = (sqrtP / (2 ** 96)) ** 2
    return 1 / price if token0.lower() == USDC.lower() else price

# --------------------------
# ðŸ§¹ Collect All Fees
# --------------------------
def collect_all_fees():
    balance = pm.functions.balanceOf(WALLET_ADDRESS).call()
    if balance == 0:
        print("âš ï¸ No NFTs found in wallet.")
        return

    usdc_per_weth = get_usdc_per_weth()
    print(f"ðŸ’° Current ETH price: {usdc_per_weth:.2f} USDC")

    total_usd = 0.0
    print(f"ðŸ” Checking {balance} NFT positions...")

    for i in range(balance):
        token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, i).call()
        pos = pm.functions.positions(token_id).call()
        owed0, owed1 = pos[10], pos[11]

        if owed0 == 0 and owed1 == 0:
            continue  # skip if nothing owed

        print(f"\nðŸª™ Token {token_id}: owed0={owed0}, owed1={owed1} â†’ collecting...")

        tx = pm.functions.collect({
            "tokenId": token_id,
            "recipient": WALLET_ADDRESS,
            "amount0Max": 2**128 - 1,
            "amount1Max": 2**128 - 1
        }).build_transaction({"from": WALLET_ADDRESS})
        r = sign_and_send(tx)
        print(f"âœ… Collected from token {token_id}. Tx: {r.transactionHash.hex()}")

        # estimate USD value
        token0 = pos[2]
        token1 = pos[3]
        usd_val = 0.0
        if token0.lower() == USDC.lower():
            usd_val += owed0 / 1e6
        elif token0.lower() == WETH.lower():
            usd_val += (owed0 / 1e18) * usdc_per_weth
        if token1.lower() == USDC.lower():
            usd_val += owed1 / 1e6
        elif token1.lower() == WETH.lower():
            usd_val += (owed1 / 1e18) * usdc_per_weth
        total_usd += usd_val

        time.sleep(1)

    print("\n==================================================")
    print(f"âœ… All owed fees collected successfully.")
    print(f"ðŸ’µ Estimated total collected: â‰ˆ ${total_usd:.2f}")
    print(f"ðŸ•’ Completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("==================================================")

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    collect_all_fees()
