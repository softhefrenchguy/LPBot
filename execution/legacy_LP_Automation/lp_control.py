from web3 import Web3
import json
import os

# --- 1. Connect to blockchain ---
w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
if not w3.is_connected():
    print("Failed to connect")
    exit()
print("Connected to blockchain")

# --- 2. Wallet ---
wallet_address = os.getenv("WALLET_ADDRESS")
private_key = os.getenv("PRIVATE_KEY")
if not wallet_address or not private_key:
    print("Please set WALLET_ADDRESS and PRIVATE_KEY as environment variables")
    exit()

# --- 3. Load ABIs ---
with open("uniswap_v3_position_manager_abi") as f:
    position_manager_abi = json.load(f)

# --- 4. Load Position Manager contract ---
position_manager_address = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
position_manager = w3.eth.contract(
    address=w3.to_checksum_address(position_manager_address),
    abi=position_manager_abi
)
print("Position Manager contract loaded")

# --- 5. Helper functions ---
def collect_fees(token_id):
    tx = position_manager.functions.collect({
        "tokenId": token_id,
        "recipient": wallet_address,
        "amount0Max": 2**128 - 1,
        "amount1Max": 2**128 - 1
    }).build_transaction({
        "from": wallet_address,
        "nonce": w3.eth.get_transaction_count(wallet_address, "pending"),
        "gas": 300000,
        "gasPrice": w3.to_wei("5", "gwei")
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Collected fees for position {token_id}: {w3.to_hex(tx_hash)}")
    return tx_hash

def remove_liquidity(token_id):
    position = position_manager.functions.positions(token_id).call()
    current_liquidity = position[7]  # liquidity
    print(f"Removing {current_liquidity} liquidity from position {token_id}")

    tx = position_manager.functions.decreaseLiquidity({
        "tokenId": token_id,
        "liquidity": current_liquidity,
        "amount0Min": 0,
        "amount1Min": 0,
        "deadline": w3.eth.get_block('latest')['timestamp'] + 600
    }).build_transaction({
        "from": wallet_address,
        "nonce": w3.eth.get_transaction_count(wallet_address, "pending"),
        "gas": 500000,
        "gasPrice": w3.to_wei("5", "gwei")
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Liquidity removal tx sent: {w3.to_hex(tx_hash)}")
    return tx_hash

# --- 6. Main execution ---
if __name__ == "__main__":
    balance = position_manager.functions.balanceOf(wallet_address).call()
    print(f"You have {balance} LP NFT(s)")

    for i in range(balance):
        token_id = position_manager.functions.tokenOfOwnerByIndex(wallet_address, i).call()
        collect_fees(token_id)
        remove_liquidity(token_id)
