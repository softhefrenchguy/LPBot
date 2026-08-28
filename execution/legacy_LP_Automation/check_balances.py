from web3 import Web3
import json

w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
print("✅ Connected:", w3.is_connected())

WALLET = "0x4Ea56cbf82F3F93f7f5EC544de944F40Dc2D21C6"
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

TOKENS = {
    "USDC": "0xFF970A61A04b1Ca14834A43f5dE4533ebDdD67A6",
    "WETH": "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
}

for name, addr in TOKENS.items():
    try:
        token = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)
        symbol = token.functions.symbol().call()
        decimals = token.functions.decimals().call()
        balance = token.functions.balanceOf(WALLET).call() / (10 ** decimals)
        print(f"✅ {name} ({symbol}): {balance:.6f}")
    except Exception as e:
        print(f"❌ Error reading {name}: {e}")
