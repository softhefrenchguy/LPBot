import os
import json
import time
import sys
from web3 import Web3
from dotenv import load_dotenv

# Load .env from folder
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit(" RPC connection failed")

WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PK     = os.getenv("PRIVATE_KEY")

if not WALLET or not PK:
    raise SystemExit(" WALLET_ADDRESS or PRIVATE_KEY missing in .env")

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, "NonfungiblePositionManager.json"), "r", encoding="utf-8") as f:
    pm_abi = json.load(f)

pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)

def sign(tx):
    # tx is built with {"from": WALLET}
    tx["nonce"] = w3.eth.get_transaction_count(WALLET)
    base = w3.eth.get_block("latest")["baseFeePerGas"]
    tx["maxFeePerGas"] = int(base * 2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei("0.01","gwei"))
    tx.setdefault("gas", 500000)

    signed = w3.eth.account.sign_transaction(tx, PK)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh)
    print(f" {txh.hex()} | {rcpt.gasUsed} gas")
    return rcpt

def main():
    balance = pm.functions.balanceOf(WALLET).call()
    print(f" Wallet {WALLET} owns {balance} LP NFTs")

    if balance == 0:
        print(" No LP positions found. Nothing to close.")
        return

    # Only process the most recent 40 NFTs
    start_index = max(balance - 40, 0)
    indices = range(start_index, balance)

    token_ids = []
    for i in indices:
        tid = pm.functions.tokenOfOwnerByIndex(WALLET, i).call()
        token_ids.append(int(tid))

    print(f" Targeting last {len(token_ids)} tokenIds:", token_ids)

    for tid in token_ids:
        print(f"\n===== Closing LP token {tid} =====")

        # Approve if needed
        approved = pm.functions.getApproved(tid).call()
        if approved.lower() != POSITION_MANAGER.lower():
            print(" Approving tokenId...")
            tx_approve = pm.functions.approve(POSITION_MANAGER, tid).build_transaction({"from": WALLET})
            sign(tx_approve)
        else:
            print(" Already approved")

        # Get liquidity
        pos = pm.functions.positions(tid).call()
        liquidity = pos[7]

        if liquidity == 0:
            print(" Liquidity already 0, skipping decrease")
        else:
            print(f" Removing liquidity = {liquidity}")
            tx_dec = pm.functions.decreaseLiquidity({
                "tokenId": tid,
                "liquidity": liquidity,
                "amount0Min": 0,
                "amount1Min": 0,
                "deadline": int(time.time()) + 300
            }).build_transaction({"from": WALLET})
            sign(tx_dec)

        # Collect everything
        print(" Collecting tokens + fees...")
        tx_col = pm.functions.collect({
            "tokenId": tid,
            "recipient": WALLET,
            "amount0Max": 2**128 - 1,
            "amount1Max": 2**128 - 1
        }).build_transaction({"from": WALLET})
        sign(tx_col)

        print(f" Finished token {tid}")

    print("\n Recent positions closed.")

if __name__ == "__main__":
    main()
