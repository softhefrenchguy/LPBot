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

RPC_URL = "https://ethereum.publicnode.com"
w3 = Web3(Web3.HTTPProvider(RPC_URL))
pool_addr = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")

with open("pool_abi.json", "r") as f:
    abi = json.load(f)

pool = w3.eth.contract(address=pool_addr, abi=abi)
print("token0:", pool.functions.token0().call())
print("token1:", pool.functions.token1().call())
