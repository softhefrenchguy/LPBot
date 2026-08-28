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

# ===========================
# CONFIGURE ETHEREUM MAINNET RPC
# ===========================
MAINNET_RPC = "https://ethereum.publicnode.com"  # Official Ethereum mainnet One RPC
w3 = Web3(Web3.HTTPProvider(MAINNET_RPC))

if not w3.is_connected():
    raise Exception("Failed to connect to Ethereum mainnet node")
print("âœ… Connected to Ethereum mainnet!")

# ===========================
# POOL ADDRESS (Change this!)
# ===========================
pool_address = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # Replace with the pool address you want to query

# ===========================
# UNISWAP V3 POOL ABI (Simplified to needed functions)
# ===========================
pool_abi = json.loads("""
[
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160","name": "sqrtPriceX96","type": "uint160"},
            {"internalType": "int24","name": "tick","type": "int24"},
            {"internalType": "uint16","name": "observationIndex","type": "uint16"},
            {"internalType": "uint16","name": "observationCardinality","type": "uint16"},
            {"internalType": "uint16","name": "observationCardinalityNext","type": "uint16"},
            {"internalType": "uint8","name": "feeProtocol","type": "uint8"},
            {"internalType": "bool","name": "unlocked","type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "tickSpacing",
        "outputs": [{"internalType": "int24","name": "","type": "int24"}],
        "stateMutability": "view",
        "type": "function"
    }
]
""")

# ===========================
# CONNECT TO POOL CONTRACT
# ===========================
pool_contract = w3.eth.contract(address=pool_address, abi=pool_abi)

# ===========================
# QUERY slot0 AND tickSpacing
# ===========================
try:
    slot0 = pool_contract.functions.slot0().call()
    tick_spacing = pool_contract.functions.tickSpacing().call()

    print("\nðŸ”¹ Pool Info:")
    print(f"SqrtPriceX96: {slot0[0]}")
    print(f"Current Tick: {slot0[1]}")
    print(f"Observation Index: {slot0[2]}")
    print(f"Observation Cardinality: {slot0[3]}")
    print(f"Next Observation Cardinality: {slot0[4]}")
    print(f"Fee Protocol: {slot0[5]}")
    print(f"Unlocked: {slot0[6]}")
    print(f"Tick Spacing: {tick_spacing}")

except Exception as e:
    print("âŒ Error querying pool:", e)
