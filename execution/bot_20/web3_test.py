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
from math import sqrt

# --- 1. Connect to the blockchain ---
w3 = Web3(Web3.HTTPProvider("https://ethereum.publicnode.com"))

if w3.is_connected():
    print("Connected to blockchain")
else:
    print("Failed to connect")
    exit()

# --- 2. Wallet ---
wallet_address = "0x4Ea56cbf82F3F93f7f5EC544de944F40Dc2D21C6"
print("Wallet address:", wallet_address)

# --- 3. Load ABIs ---
with open("uniswap_v3_position_manager_abi.json") as f:
    position_manager_abi = json.load(f)

with open("uniswap_v3_factory_abi.json") as f:
    factory_abi = json.load(f)

with open("uniswap_v3_pool_abi.json") as f:
    pool_abi = json.load(f)

with open("erc20_abi.json") as f:
    erc20_abi = json.load(f)

# --- 4. Load contracts ---
position_manager_address = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
position_manager = w3.eth.contract(
    address=w3.to_checksum_address(position_manager_address),
    abi=position_manager_abi
)
print("Position Manager contract loaded")

factory_address = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
factory_contract = w3.eth.contract(
    address=w3.to_checksum_address(factory_address),
    abi=factory_abi
)
print("Factory contract loaded")

# --- 5. List LP NFT positions ---
balance = position_manager.functions.balanceOf(wallet_address).call()
print(f"You have {balance} LP NFT(s)")

positions = []

for i in range(balance):
    token_id = position_manager.functions.tokenOfOwnerByIndex(wallet_address, i).call()
    pos = position_manager.functions.positions(token_id).call()
    liquidity = pos[7]
    token0_addr = pos[2]
    token1_addr = pos[3]

    # Load token contracts to get decimals & symbol
    token0_contract = w3.eth.contract(address=w3.to_checksum_address(token0_addr), abi=erc20_abi)
    token1_contract = w3.eth.contract(address=w3.to_checksum_address(token1_addr), abi=erc20_abi)
    token0_decimals = token0_contract.functions.decimals().call()
    token1_decimals = token1_contract.functions.decimals().call()
    token0_symbol = token0_contract.functions.symbol().call()
    token1_symbol = token1_contract.functions.symbol().call()

    # --- 6. Find pool address for 0.05% fee ---
    pool_address = factory_contract.functions.getPool(token0_addr, token1_addr, 500).call()
    pool_contract = w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=pool_abi)

    # --- 7. Read pool's current sqrtPriceX96 ---
    slot0 = pool_contract.functions.slot0().call()
    sqrtPriceX96 = slot0[0]  # first item is sqrtPriceX96

    # --- 8. Convert liquidity to token amounts ---
    tick_lower = pos[5]
    tick_upper = pos[6]

    sqrtPriceA = 1.0001 ** (tick_lower / 2)  # approximate conversion
    sqrtPriceB = 1.0001 ** (tick_upper / 2)
    sqrtPriceX = sqrtPriceX96 / (2 ** 96)

    def get_amounts(liquidity, sqrtPriceX, sqrtPriceA, sqrtPriceB, token0_decimals, token1_decimals):
        if sqrtPriceX <= sqrtPriceA:
            amount0 = liquidity * (sqrtPriceB - sqrtPriceA) / (sqrtPriceA * sqrtPriceB)
            amount1 = 0
        elif sqrtPriceX >= sqrtPriceB:
            amount0 = 0
            amount1 = liquidity * (sqrtPriceB - sqrtPriceA)
        else:
            amount0 = liquidity * (sqrtPriceB - sqrtPriceX) / (sqrtPriceX * sqrtPriceB)
            amount1 = liquidity * (sqrtPriceX - sqrtPriceA)
        amount0 /= 10 ** token0_decimals
        amount1 /= 10 ** token1_decimals
        return amount0, amount1

    amt0, amt1 = get_amounts(liquidity, sqrtPriceX, sqrtPriceA, sqrtPriceB, token0_decimals, token1_decimals)

    print(f"Position #{token_id}: {amt0:.6f} {token0_symbol}, {amt1:.6f} {token1_symbol}")

    positions.append({
        "tokenId": token_id,
        "liquidity": liquidity,
        "token0": token0_symbol,
        "token1": token1_symbol,
        "amount0": amt0,
        "amount1": amt1,
        "pool": pool_address
    })
