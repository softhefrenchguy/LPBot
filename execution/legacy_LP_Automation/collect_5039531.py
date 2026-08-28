from web3 import Web3
import json, os, time

w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

pm_addr = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
with open("NonfungiblePositionManager.json") as f:
    pm_abi = json.load(f)
pm = w3.eth.contract(address=pm_addr, abi=pm_abi)

token_id = 5040001  # your position

tx = pm.functions.collect({
    "tokenId": token_id,
    "recipient": WALLET_ADDRESS,
    "amount0Max": 2**128 - 1,
    "amount1Max": 2**128 - 1
}).build_transaction({
    "from": WALLET_ADDRESS,
    "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
    "gas": 300000,
    "maxFeePerGas": int(w3.eth.gas_price * 2),
    "maxPriorityFeePerGas": w3.to_wei("0.01", "gwei")
})

signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
print("✅ Final collect sent:", tx_hash.hex())
