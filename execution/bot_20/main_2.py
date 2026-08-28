import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
$code = @"
import json
import time
from web3 import Web3
import os

# --------------------------
# Web3 setup (Ethereum mainnet)
# --------------------------
RPC_URL = "https://ethereum.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("Failed to connect to Ethereum mainnet node")
print("ðŸš€ LP Automation Started (Ethereum mainnet)")

# --------------------------
# Wallet & contracts
# --------------------------
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

POSITION_MANAGER_ADDRESS = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
position_manager_abi_path = "abis/NonfungiblePositionManager.json"
with open(position_manager_abi_path, "r") as f:
    position_manager_abi = json.load(f)

position_manager = w3.eth.contract(address=POSITION_MANAGER_ADDRESS, abi=position_manager_abi)

# --------------------------
# History logging
# --------------------------
HISTORY_FILE = "history.log"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        pass

def log_action(action):
    timestamp = time.strftime("[%H:%M:%S]")
    with open(HISTORY_FILE, "a") as f:
        f.write(f"{timestamp} Logged action: {action}\n")
    print(f"{timestamp} Logged action: {action}")

# --------------------------
# Fetch active LP position
# --------------------------
def get_position(token_id):
    return position_manager.functions.positions(token_id).call()

# --------------------------
# Check if position is out of range
# --------------------------
def is_out_of_range(current_tick, tick_lower, tick_upper):
    return current_tick < tick_lower or current_tick > tick_upper

# --------------------------
# Placeholder functions for LP automation
# --------------------------
def withdraw_liquidity():
    print("Withdrawing liquidity...")
    log_action("withdraw_liquidity")
    time.sleep(1)

def rebalance_portfolio():
    print("Rebalancing portfolio to 50/50...")
    log_action("rebalance_portfolio")
    time.sleep(1)

def create_position():
    print("Creating new LP position...")
    log_action("create_position")
    time.sleep(1)

# --------------------------
# Main loop
# --------------------------
def main():
    # Replace this with your actual LP NFT token ID
    LP_TOKEN_ID = 1234

    # Get position tick bounds
    position = get_position(LP_TOKEN_ID)
    tick_lower = position[1]
    tick_upper = position[2]

    while True:
        try:
            slot0 = position_manager.functions.slot0().call()
            current_tick = slot0[1]

            if is_out_of_range(current_tick, tick_lower, tick_upper):
                print("âš ï¸  Position out of range detected! Executing maintenance cycle...")
                withdraw_liquidity()
                rebalance_portfolio()
                create_position()
                print("âœ… Cycle completed. Waiting for next trigger...\n")
            else:
                print("âœ… Position in range. Monitoring...")

            time.sleep(10)

        except Exception as e:
            print("âŒ Error during automation:", e)
            time.sleep(10)

if __name__ == "__main__":
    main()
"@

$code | Out-File -FilePath .\main.py -Encoding UTF8
Write-Host "main.py generated successfully!"
