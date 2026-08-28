import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
from web3 import Web3
import json, time, os

RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
TOKEN_ID = 5053611  # ðŸ” replace with your NFT ID

w3 = Web3(Web3.HTTPProvider(RPC_URL))
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

pm_abi = json.load(open("NonfungiblePositionManager.json"))
pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)

# --- Get liquidity amount ---
pos = pm.functions.positions(TOKEN_ID).call()
liquidity = pos[7]
print(f"Liquidity = {liquidity}")

# --- Build decreaseLiquidity transaction ---
tx = pm.functions.decreaseLiquidity({
    "tokenId": TOKEN_ID,
    "liquidity": liquidity,
    "amount0Min": 0,
    "amount1Min": 0,
    "deadline": int(time.time()) + 600
}).build_transaction({
    "from": WALLET_ADDRESS,
    "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
    "maxFeePerGas": int(w3.eth.gas_price * 2),
    "maxPriorityFeePerGas": int(w3.to_wei("0.01", "gwei")),
    "gas": 700_000
})

signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
print("â³ Sent withdraw tx:", tx_hash.hex())
print("âœ… Waiting for confirmation...")
print(w3.eth.wait_for_transaction_receipt(tx_hash))
