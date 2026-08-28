from web3 import Web3
import json
import os

# --- 1. Connect to blockchain ---
w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
if not w3.is_connected():
    print("Failed to connect")
    exit()

# --- 2. Load wallet info from environment ---
wallet_address = os.environ.get("WALLET_ADDRESS")
private_key = os.environ.get("PRIVATE_KEY")
if not wallet_address or not private_key:
    print("Please set WALLET_ADDRESS and PRIVATE_KEY as environment variables")
    exit()


# ABI Loader
# --------------------------
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
aggregator_abi = load_abi("aggregator_v3_abi.json")  # <-- Make sure this file exists in your abis/ folder


def load_abi(filename):
    path = os.path.join(BASE_DIR, "abis", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ABI file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# Example usage:
erc20_abi = load_abi("erc20_abi.json")
position_manager_abi = load_abi("uniswap_v3_position_manager_abi.json")
pool_abi = load_abi("uniswap_v3_pool_abi.json")


# --- 4. Load contracts ---
position_manager_address = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
position_manager = w3.eth.contract(address=w3.to_checksum_address(position_manager_address),
                                   abi=position_manager_abi)

WETH_ADDRESS = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
WETH = w3.eth.contract(address=WETH_ADDRESS, abi=erc20_abi)
USDC = w3.eth.contract(address=USDC_ADDRESS, abi=erc20_abi)

# Chainlink WETH/USD price feed on Arbitrum
PRICE_FEED = w3.eth.contract(address=w3.to_checksum_address("0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612"),
                             abi=aggregator_abi)

# Swap router
SWAP_ROUTER_ADDRESS = w3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")  # Uniswap v3 router
SWAP_ROUTER = w3.eth.contract(address=SWAP_ROUTER_ADDRESS, abi=swap_router_abi)

# --- Utility: send signed transaction with nonce handling ---
def send_tx(tx, nonce):
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

# --- Collect fees for a given LP NFT ---
def collect_fees(token_id, nonce):
    tx = position_manager.functions.collect({
        "tokenId": token_id,
        "recipient": wallet_address,
        "amount0Max": 2**128 - 1,
        "amount1Max": 2**128 - 1
    }).build_transaction({
        "from": wallet_address,
        "nonce": nonce,
        "gas": 500000,
        "maxFeePerGas": w3.to_wei("5", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei")
    })
    receipt = send_tx(tx, nonce)
    print(f"Collected fees for position {token_id}: {receipt.transactionHash.hex()}")
    return receipt

# --- Remove liquidity for a given LP NFT ---
def remove_liquidity(token_id, percentage, nonce):
    pos = position_manager.functions.positions(token_id).call()
    current_liquidity = pos[7]
    liquidity_to_remove = int(current_liquidity * percentage / 100)
    if liquidity_to_remove == 0:
        print(f"No liquidity to remove for position {token_id}")
        return

    tx = position_manager.functions.decreaseLiquidity({
        "tokenId": token_id,
        "liquidity": liquidity_to_remove,
        "amount0Min": 0,
        "amount1Min": 0,
        "deadline": w3.eth.get_block("latest")["timestamp"] + 600
    }).build_transaction({
        "from": wallet_address,
        "nonce": nonce,
        "gas": 500000,
        "maxFeePerGas": w3.to_wei("5", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei")
    })
    receipt = send_tx(tx, nonce)
    print(f"Removed {liquidity_to_remove} liquidity from position {token_id}: {receipt.transactionHash.hex()}")
    return receipt

# --- Read wallet balances ---
def get_balances():
    weth_balance = WETH.functions.balanceOf(wallet_address).call() / 10**18
    usdc_balance = USDC.functions.balanceOf(wallet_address).call() / 10**6
    return weth_balance, usdc_balance

# --- Fetch WETH price in USDC ---
def get_weth_price():
    latest_round = PRICE_FEED.functions.latestRoundData().call()
    price = latest_round[1] / 1e8
    return price

# --- Swap tokens via router ---
def swap_tokens(token_in, token_out, amount_in, nonce):
    if amount_in == 0:
        print(f"Swapping 0 {token_in.address} -> {token_out.address}")
        return
    allowance = token_in.functions.allowance(wallet_address, SWAP_ROUTER_ADDRESS).call()
    if allowance < amount_in:
        tx = token_in.functions.approve(SWAP_ROUTER_ADDRESS, 2**256 - 1).build_transaction({
            "from": wallet_address,
            "nonce": nonce,
            "gas": 100000,
            "maxFeePerGas": w3.to_wei("5", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("1", "gwei")
        })
        send_tx(tx, nonce)
        nonce += 1

    tx = SWAP_ROUTER.functions.exactInputSingle({
        "tokenIn": token_in.address,
        "tokenOut": token_out.address,
        "fee": 500,
        "recipient": wallet_address,
        "deadline": w3.eth.get_block("latest")["timestamp"] + 600,
        "amountIn": int(amount_in),
        "amountOutMinimum": 0,
        "sqrtPriceLimitX96": 0
    }).build_transaction({
        "from": wallet_address,
        "nonce": nonce,
        "gas": 500000,
        "maxFeePerGas": w3.to_wei("5", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei")
    })
    receipt = send_tx(tx, nonce)
    print(f"Swapped {amount_in} of {token_in.address} -> {token_out.address}: {receipt.transactionHash.hex()}")
    return receipt

# --- Auto-rebalance ---
def auto_rebalance(target_ratio=0.5):
    weth_balance, usdc_balance = get_balances()
    price = get_weth_price()
    total_value = weth_balance * price + usdc_balance
    target_weth_value = total_value * target_ratio
    target_usdc_value = total_value * (1 - target_ratio)

    nonce = w3.eth.get_transaction_count(wallet_address)
    if weth_balance * price < target_weth_value:
        delta_usdc = target_weth_value - weth_balance * price
        swap_tokens(USDC, WETH, int(delta_usdc * 10**6), nonce)
    elif weth_balance * price > target_weth_value:
        delta_weth = (weth_balance * price - target_weth_value) / price
        swap_tokens(WETH, USDC, int(delta_weth * 10**18), nonce)
    else:
        print("Already at target ratio")

# --- Create LP Position ---
def create_position(weth_amount, usdc_amount, pool_fee=500):
    if weth_amount < 0.001 or usdc_amount < 1:
        print("Balances too small to create LP position")
        return

    factory_address = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    with open("uniswap_v3_factory_abi.json") as f:
        factory_abi = json.load(f)
    factory = w3.eth.contract(address=w3.to_checksum_address(factory_address), abi=factory_abi)

    pool_address = factory.functions.getPool(WETH_ADDRESS, USDC_ADDRESS, pool_fee).call()
    if pool_address == "0x0000000000000000000000000000000000000000":
        print("Pool does not exist for this fee tier!")
        return

    with open("uniswap_v3_pool_abi.json") as f:
        pool_abi = json.load(f)
    pool = w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=pool_abi)

    slot0 = pool.functions.slot0().call()
    current_tick = slot0[1]
    tick_spacing = pool.functions.tickSpacing().call()
    tick_lower = (current_tick // tick_spacing - 1) * tick_spacing
    tick_upper = (current_tick // tick_spacing + 1) * tick_spacing

    token0 = WETH_ADDRESS
    token1 = USDC_ADDRESS
    if token0 > token1:
        token0, token1 = token1, token0
        weth_amount, usdc_amount = usdc_amount, weth_amount

    print(f"Creating LP position with ticks [{tick_lower}, {tick_upper}]")

    nonce = w3.eth.get_transaction_count(wallet_address)
    tx = position_manager.functions.mint({
        "token0": token0,
        "token1": token1,
        "fee": pool_fee,
        "tickLower": tick_lower,
        "tickUpper": tick_upper,
        "amount0Desired": int(weth_amount * 10**18),
        "amount1Desired": int(usdc_amount * 10**6),
        "amount0Min": 0,
        "amount1Min": 0,
        "recipient": wallet_address,
        "deadline": w3.eth.get_block("latest")["timestamp"] + 600
    }).build_transaction({
        "from": wallet_address,
        "nonce": nonce,
        "gas": 1000000,
        "maxFeePerGas": w3.to_wei("5", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei")
    })

    receipt = send_tx(tx, nonce)
    print(f"Created new LP position! Transaction: {receipt.transactionHash.hex()}")
    return receipt

# --- Main execution ---
if __name__ == "__main__":
    nonce = w3.eth.get_transaction_count(wallet_address)

    balance = position_manager.functions.balanceOf(wallet_address).call()
    positions = [position_manager.functions.tokenOfOwnerByIndex(wallet_address, i).call() for i in range(balance)]

    for token_id in positions:
        collect_fees(token_id, nonce)
        nonce += 1
        remove_liquidity(token_id, 100, nonce)
        nonce += 1

    weth_balance, usdc_balance = get_balances()
    print(f"Balances after LP removal: {weth_balance:.6f} WETH, {usdc_balance:.6f} USDC")

    auto_rebalance(target_ratio=0.5)

    weth_balance, usdc_balance = get_balances()
    print(f"Balances after rebalance: {weth_balance:.6f} WETH, {usdc_balance:.6f} USDC")

    create_position(weth_balance, usdc_balance, pool_fee=500)

