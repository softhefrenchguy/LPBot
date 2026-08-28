import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
from web3 import Web3
import os, json

RPC_URL = "https://ethereum.publicnode.com"
USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")

w3 = Web3(Web3.HTTPProvider(RPC_URL))

with open("erc20_abi.json") as f:
    erc20_abi = json.load(f)

t_usdc = w3.eth.contract(address=USDC, abi=erc20_abi)
t_weth = w3.eth.contract(address=WETH, abi=erc20_abi)
addr = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))

print("USDC balance:", t_usdc.functions.balanceOf(addr).call() / 1e6)
print("WETH balance:", t_weth.functions.balanceOf(addr).call() / 1e18)
