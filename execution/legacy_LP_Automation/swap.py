from web3 import Web3
import os, json

# --- 1. Connect to Arbitrum ---
w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
assert w3.is_connected(), "Failed to connect to Arbitrum"

# --- 2. Wallet + Private Key ---
wallet_address = "0x4Ea56cbf82F3F93f7f5EC544de944F40Dc2D21C6"
private_key = os.getenv("PRIVATE_KEY")
if not private_key:
    raise Exception("PRIVATE_KEY not set in environment variables!")

# --- 3. Load ABIs ---
with open("erc20_abi.json") as f:
    erc20_abi = json.load(f)
with open("uniswap_v3_router_abi.json") as f:
    router_abi = json.load(f)

# --- 4. Contract addresses ---
SWAP_ROUTER = w3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")  # Uniswap V3 SwapRouter on Arbitrum
WETH = w3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")
USDC = w3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")

router = w3.eth.contract(address=SWAP_ROUTER, abi=router_abi)

# --- 5. Helper: approve token if needed ---
def ensure_approval(token_addr, spender):
    token = w3.eth.contract(address=token_addr, abi=erc20_abi)
    allowance = token.functions.allowance(wallet_address, spender).call()
    balance = token.functions.balanceOf(wallet_address).call()
    if allowance < balance:
        print("Approving token for router...")
        tx = token.functions.approve(spender, 2**256 - 1).build_transaction({
            "from": wallet_address,
            "nonce": w3.eth.get_transaction_count(wallet_address),
            "gas": 100000,
            "gasPrice": w3.to_wei("1", "gwei")
        })
        signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"Approval tx sent: {w3.to_hex(tx_hash)}")
        w3.eth.wait_for_transaction_receipt(tx_hash)

# --- 6. Perform swap ---
def swap_tokens(token_in, token_out, amount_in_wei, fee=500):
    ensure_approval(token_in, SWAP_ROUTER)
    params = (
        token_in,
        token_out,
        fee,
        wallet_address,
        int(w3.eth.get_block('latest')['timestamp'] + 600),
        amount_in_wei,
        0,  # amountOutMinimum (slippage tolerance = 0 for now, adjust later)
        0  # sqrtPriceLimitX96 = 0 (no limit)
    )

    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": wallet_address,
        "nonce": w3.eth.get_transaction_count(wallet_address),
        "gas": 500000,
        "gasPrice": w3.to_wei("2", "gwei"),
        "value": amount_in_wei if token_in == WETH else 0
    })

    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Swap tx sent: {w3.to_hex(tx_hash)}")
    return tx_hash

# --- 7. Example usage ---
if __name__ == "__main__":
    # Choose swap direction
    direction = input("Enter swap direction (1 for WETH→USDC, 2 for USDC→WETH): ").strip()
    amount = float(input("Enter amount to swap: "))

    if direction == "1":
        amount_in_wei = w3.to_wei(amount, "ether")
        print(f"Swapping {amount} WETH → USDC...")
        swap_tokens(WETH, USDC, amount_in_wei)
    else:
        usdc_contract = w3.eth.contract(address=USDC, abi=erc20_abi)
        decimals = usdc_contract.functions.decimals().call()
        amount_in_wei = int(amount * (10 ** decimals))
        print(f"Swapping {amount} USDC → WETH...")
        swap_tokens(USDC, WETH, amount_in_wei)
