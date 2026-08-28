from web3 import Web3
import json
import os

# --- RPC + wallet ---
RPC_URL = "https://arb1.arbitrum.io/rpc"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("❌ Failed to connect to Arbitrum RPC")
print("✅ Connected to Arbitrum!")

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

# --- Contract setup ---
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")

with open("NonfungiblePositionManager.json", "r") as f:
    pm_abi = json.load(f)

pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)

# --- Query wallet NFTs ---
balance = pm.functions.balanceOf(WALLET_ADDRESS).call()
print(f"🔍 Wallet owns {balance} LP NFT(s).")

if balance == 0:
    exit()

for i in range(balance):
    token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, i).call()
    pos = pm.functions.positions(token_id).call()
    liquidity = pos[7]
    tick_lower = pos[5]
    tick_upper = pos[6]
    print(f"🧩 Token {token_id} | Liquidity: {liquidity} | Range: [{tick_lower}, {tick_upper}]")
