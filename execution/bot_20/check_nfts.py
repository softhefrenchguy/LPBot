import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import json
from web3 import Web3

# --------------------------
# RPC + Wallet setup
# --------------------------
RPC_URL = "https://ethereum.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Replace this with your actual wallet address
WALLET_ADDRESS = Web3.to_checksum_address("0x4Ea56cbf82F3F93f7f5EC544de944F40Dc2D21C6")

# NonfungiblePositionManager on Ethereum mainnet
POSITION_MANAGER_ADDRESS = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")

# --------------------------
# Load ABI
# --------------------------
with open("abis/NonfungiblePositionManager.json", "r") as f:
    position_manager_abi = json.load(f)

position_manager = w3.eth.contract(address=POSITION_MANAGER_ADDRESS, abi=position_manager_abi)

# --------------------------
# Fetch owned positions
# --------------------------
balance = position_manager.functions.balanceOf(WALLET_ADDRESS).call()
print(f"Wallet owns {balance} LP NFT(s).")

if balance == 0:
    print("No LP positions found.")
else:
    for i in range(balance):
        token_id = position_manager.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, i).call()
        pos = position_manager.functions.positions(token_id).call()

        nonce = pos[0]
        token0 = pos[2]
        token1 = pos[3]
        fee = pos[4]
        tick_lower = pos[5]
        tick_upper = pos[6]
        liquidity = pos[7]
        tokens_owed0 = pos[10]
        tokens_owed1 = pos[11]

        print(f"\n🔹 Token ID: {token_id}")
        print(f"  • Token0: {token0}")
        print(f"  • Token1: {token1}")
        print(f"  • Fee Tier: {fee}")
        print(f"  • Tick Lower: {tick_lower}")
        print(f"  • Tick Upper: {tick_upper}")
        print(f"  • Liquidity: {liquidity}")
        print(f"  • Fees Owed (Token0): {tokens_owed0}")
        print(f"  • Fees Owed (Token1): {tokens_owed1}")
