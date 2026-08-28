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
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

w3 = Web3(Web3.HTTPProvider(RPC_URL))
print("Connected:", w3.is_connected())

# load your real ABI
with open("pool_abi.json") as f:
    pool_abi = json.load(f)

pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)

try:
    slot0 = pool.functions.slot0().call()
    print("slot0():", slot0)
except Exception as e:
    print("slot0() call failed:", type(e).__name__, str(e)[:200])
