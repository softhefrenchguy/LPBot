import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
from web3 import Web3
import json, os
from dotenv import load_dotenv
load_dotenv()

w3 = Web3(Web3.HTTPProvider("https://ethereum.publicnode.com"))
WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

with open("erc20_abi.json") as f:
    erc20 = json.load(f)

USDC = w3.eth.contract(address=Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), abi=erc20)
WETH = w3.eth.contract(address=Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), abi=erc20)

usdc = USDC.functions.balanceOf(WALLET).call() / 1e6
weth = WETH.functions.balanceOf(WALLET).call() / 1e18

print(f"Wallet: {WALLET}")
print(f"USDC balance: {usdc}")
print(f"WETH balance: {weth}")
