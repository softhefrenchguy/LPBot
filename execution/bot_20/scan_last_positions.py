from web3 import Web3
import json
from datetime import datetime

RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER_ADDR = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
WALLET = Web3.to_checksum_address("0xF3b76353a27B1E0dF5857F597a108fC103D5c48F")

# Minimal ABI: just what we need
POSITION_MANAGER_ABI = [
    {
        "inputs":[{"internalType":"address","name":"owner","type":"address"}],
        "name":"balanceOf",
        "outputs":[{"internalType":"uint256","name":"","type":"uint256"}],
        "stateMutability":"view",
        "type":"function"
    },
    {
        "inputs":[
            {"internalType":"address","name":"owner","type":"address"},
            {"internalType":"uint256","name":"index","type":"uint256"}
        ],
        "name":"tokenOfOwnerByIndex",
        "outputs":[{"internalType":"uint256","name":"","type":"uint256"}],
        "stateMutability":"view",
        "type":"function"
    },
    {
        "inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],
        "name":"positions",
        "outputs":[
            {"internalType":"uint96","name":"nonce","type":"uint96"},
            {"internalType":"address","name":"operator","type":"address"},
            {"internalType":"address","name":"token0","type":"address"},
            {"internalType":"address","name":"token1","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"int24","name":"tickLower","type":"int24"},
            {"internalType":"int24","name":"tickUpper","type":"int24"},
            {"internalType":"uint128","name":"liquidity","type":"uint128"},
            {"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},
            {"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},
            {"internalType":"uint128","name":"tokensOwed0","type":"uint128"},
            {"internalType":"uint128","name":"tokensOwed1","type":"uint128"}
        ],
        "stateMutability":"view",
        "type":"function"
    }
]

def main():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("❌ Could not connect to Ethereum mainnet RPC")
        return

    pm = w3.eth.contract(address=POSITION_MANAGER_ADDR, abi=POSITION_MANAGER_ABI)

    balance = pm.functions.balanceOf(WALLET).call()
    print(f"\n🔍 Wallet {WALLET} owns {balance} LP NFTs")

    if balance == 0:
        return

    # We want the *last* 20 (most recent by index)
    count = min(balance, 20)
    start_index = balance - count

    print(f"\n📋 Last {count} tokenIds (by ERC721 index):\n")

    for idx in range(balance - 1, start_index - 1, -1):
        token_id = pm.functions.tokenOfOwnerByIndex(WALLET, idx).call()
        try:
            pos = pm.functions.positions(token_id).call()
        except Exception as e:
            print(f"⚠ tokenId {token_id}: positions() reverted → {e}")
            continue

        (
            nonce,
            operator,
            token0,
            token1,
            fee,
            tickLower,
            tickUpper,
            liquidity,
            feeGrowth0,
            feeGrowth1,
            tokensOwed0,
            tokensOwed1,
        ) = pos

        has_liq = liquidity > 0
        status = "🟢 ACTIVE" if has_liq else "⚫️ no liquidity"

        print(
            f"  tokenId {token_id:<10} | pool fee {fee/10000:.2%} | "
            f"ticks [{tickLower}, {tickUpper}] | liquidity={liquidity} | {status}"
        )

    print("\n✅ Scan complete.\n")

if __name__ == "__main__":
    main()
