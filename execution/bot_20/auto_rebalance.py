import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
# main.py
import json
from web3 import Web3
import os
import time
import pyautogui
import cv2
import numpy as np

# --------------------------
# Config
# --------------------------
RPC_URL = "https://ethereum.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

POSITION_MANAGER_ADDRESS = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
WETH_ADDRESS = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC_ADDRESS = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")

# --------------------------
# ABIs
# --------------------------
def load_abi(filename):
    with open(f"abis/{filename}.json", "r") as f:
        return json.load(f)

position_manager_abi = load_abi("NonfungiblePositionManager")
erc20_abi = load_abi("erc20")
position_manager = w3.eth.contract(address=POSITION_MANAGER_ADDRESS, abi=position_manager_abi)
weth = w3.eth.contract(address=WETH_ADDRESS, abi=erc20_abi)
usdc = w3.eth.contract(address=USDC_ADDRESS, abi=erc20_abi)

# --------------------------
# Color detection
# --------------------------
REGION = (440, 405, 120, 35)
LOWER_RED = np.array([110, 50, 70])
UPPER_RED = np.array([175, 255, 255])
TRIGGER_DELAY = 30

def detect_magenta():
    screenshot = pyautogui.screenshot(region=REGION)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    red_ratio = np.sum(mask > 0) / mask.size
    print(f"Magenta ratio: {red_ratio:.3f}")
    return red_ratio > 0.1

# --------------------------
# Helper to send tx
# --------------------------
def send_tx(tx):
    tx.update({"nonce": w3.eth.get_transaction_count(WALLET_ADDRESS)})
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

# --------------------------
# Withdraw + collect fees
# --------------------------
def withdraw_and_collect():
    balance = position_manager.functions.balanceOf(WALLET_ADDRESS).call()
    if balance == 0:
        print("[i] No LP positions found.")
        return
    for i in range(balance):
        token_id = position_manager.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, i).call()
        pos = position_manager.functions.positions(token_id).call()
        liquidity = pos[0]
        if liquidity == 0:
            print(f"[i] Token ID {token_id} has no liquidity.")
            continue
        print(f"Withdrawing liquidity from token ID {token_id}...")
        tx = position_manager.functions.decreaseLiquidity({
            "tokenId": token_id,
            "liquidity": liquidity,
            "amount0Min": 0,
            "amount1Min": 0,
            "deadline": w3.eth.get_block('latest')["timestamp"] + 300
        }).build_transaction({"from": WALLET_ADDRESS, "gas": 400_000,
                              "maxFeePerGas": w3.to_wei(10, "gwei"),
                              "maxPriorityFeePerGas": w3.to_wei(1, "gwei")})
        receipt = send_tx(tx)
        print(f"[âœ…] Liquidity withdrawn! Tx hash: {receipt.transactionHash.hex()}")
        print(f"Collecting fees for token ID {token_id}...")
        tx = position_manager.functions.collect({
            "tokenId": token_id,
            "recipient": WALLET_ADDRESS,
            "amount0Max": 2**128 - 1,
            "amount1Max": 2**128 - 1
        }).build_transaction({"from": WALLET_ADDRESS, "gas": 200_000,
                              "maxFeePerGas": w3.to_wei(10, "gwei"),
                              "maxPriorityFeePerGas": w3.to_wei(1, "gwei")})
        receipt = send_tx(tx)
        print(f"[âœ…] Fees collected! Tx hash: {receipt.transactionHash.hex()}")

# --------------------------
# Rebalance placeholder
# --------------------------
def rebalance_portfolio():
    print("[i] Rebalancing portfolio to 50/50... (placeholder)")
    # Here you can implement swapping WETH/USDC to 50/50 if needed
    time.sleep(1)

# --------------------------
# Create LP position placeholder
# --------------------------
def create_position():
    print("[i] Creating new LP position... (placeholder)")
    # Implement mint logic here as we did in your previous create_position.py
    time.sleep(1)

# --------------------------
# Main loop
# --------------------------
if __name__ == "__main__":
    print("ðŸš€ LP Automation Started (Ethereum mainnet)")
    while True:
        print("Refreshing page...")
        pyautogui.press('f5')
        time.sleep(5)
        print("Checking screen for magenta...")
        if detect_magenta():
            print("âš ï¸ Magenta detected â€” executing maintenance cycle...")
            withdraw_and_collect()
            rebalance_portfolio()
            create_position()
            print("âœ… Cycle completed. Waiting for next trigger...")
            time.sleep(TRIGGER_DELAY)
        else:
            time.sleep(2)


