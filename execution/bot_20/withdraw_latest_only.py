import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
# withdraw_liquidity.py
import json, os, time
from datetime import datetime
from web3 import Web3

# -------------------- CONFIG --------------------
RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("âŒ Could not connect to Ethereum mainnet RPC")
print("âœ… Connected to Ethereum mainnet!")

WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

pm_abi = json.load(open("NonfungiblePositionManager.json"))
pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)

STATE_FILE = "last_cycle_data.json"
HISTORY_FILE = "cycle_history.jsonl"

# -------------------- HELPERS --------------------
def send(tx):
    tx["from"] = WALLET_ADDRESS
    tx["gas"] = 700_000
    tx["maxFeePerGas"] = int(w3.eth.gas_price * 2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei("0.01", "gwei"))
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    return w3.eth.wait_for_transaction_receipt(txh)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)

def save_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -------------------- MAIN --------------------
def main():
    state = load_state()
    token_id = state.get("tokenId")

    # --- Fallback: latest token of wallet ---
    if not token_id:
        balance = pm.functions.balanceOf(WALLET_ADDRESS).call()
        if balance == 0:
            print("âš ï¸  No LP tokens found in wallet.")
            return
        token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, balance - 1).call()

    print(f"ðŸŽ¯ Latest NFT ID: {token_id}")

    # --- Check if token has liquidity ---
    position = pm.functions.positions(token_id).call()
    liquidity = position[7]

    if liquidity == 0:
        print(f"âš ï¸ Token {token_id} has zero liquidity â€” marking as closed and skipping withdraw.")
        state["status"] = "closed"
        state["tokenId"] = token_id
        save_state(state)
        return

    # --- Collect fees before withdrawing ---
    collect_params = {
        "tokenId": token_id,
        "recipient": WALLET_ADDRESS,
        "amount0Max": 2**128 - 1,
        "amount1Max": 2**128 - 1,
    }

    try:
        tx_collect = pm.functions.collect(collect_params).build_transaction({"from": WALLET_ADDRESS})
        receipt_collect = send(tx_collect)
        print(f"ðŸ’° Fees collected (pre-withdraw). Tx: {receipt_collect.transactionHash.hex()}")
    except Exception as e:
        print(f"âš ï¸ Collect failed or no fees: {e}")

    # --- Withdraw liquidity completely ---
    try:
        tx_withdraw = pm.functions.decreaseLiquidity(
            {"tokenId": token_id, "liquidity": liquidity, "amount0Min": 0, "amount1Min": 0, "deadline": int(time.time()) + 600}
        ).build_transaction({"from": WALLET_ADDRESS})
        receipt_withdraw = send(tx_withdraw)
        print(f"ðŸ’§ Liquidity withdrawn. Tx: {receipt_withdraw.transactionHash.hex()}")
    except Exception as e:
        print(f"âš ï¸ Withdraw failed: {e}")
        return

    # --- Final collect (residual tokens owed) ---
    try:
        tx_final = pm.functions.collect(collect_params).build_transaction({"from": WALLET_ADDRESS})
        receipt_final = send(tx_final)
        print(f"âœ… Final collect sent. Tx: {receipt_final.transactionHash.hex()}")
    except Exception as e:
        print(f"âš ï¸ Final collect failed: {e}")

    # --- Log the event ---
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    state["timestamp_withdraw"] = now
    state["logged_at"] = now
    state["updated_at"] = now
    save_state(state)

    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(state) + "\n")

    print("\n==================================================")
    print("ðŸ“˜  Cycle Summary")
    print("==================================================")
    print(f"ðŸ•’ Withdraw: {now}")
    print(f"ðŸ•“  Created:  {state.get('timestamp_create', 'N/A')}")
    print(f"ðŸŽ¯ Token ID:  {token_id}")
    print(f"ðŸŽ¯ Range:     {state.get('lower_bound_usd', '?')} â†’ {state.get('upper_bound_usd', '?')}")
    print("==================================================\n")
    print("âœ… withdraw_liquidity.py completed.\n")

# -------------------- EXECUTE --------------------
if __name__ == "__main__":
    main()
