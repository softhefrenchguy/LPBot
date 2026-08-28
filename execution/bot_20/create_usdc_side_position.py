import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import json, os, time, math
from datetime import datetime
from web3 import Web3

# ---------- RPC + addresses ----------
RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS     = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")

WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY    = os.getenv("PRIVATE_KEY")

# ---------- Load ABIs ----------
def load_json(p): 
    with open(p,"r") as f: return json.load(f)
pm_abi   = load_json("NonfungiblePositionManager.json")
pool_abi = load_json("pool_abi.json")
erc20_abi= load_json("erc20_abi.json")

w3   = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected(): raise SystemExit("âŒ RPC connection failed.")
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
pm   = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)

# ---------- Helpers ----------
def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def send(tx):
    tx.setdefault("from", WALLET_ADDRESS)
    tx.setdefault("nonce", w3.eth.get_transaction_count(WALLET_ADDRESS))
    tx.setdefault("gas", 620_000)
    tx.setdefault("maxFeePerGas", int(w3.eth.gas_price * 2))
    tx.setdefault("maxPriorityFeePerGas", int(w3.to_wei("0.01", "gwei")))
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(txh)
    if r.status != 1:
        raise RuntimeError("Transaction failed")
    return r

def approve(token_addr, spender, amount):
    token = w3.eth.contract(address=token_addr, abi=erc20_abi)
    if token.functions.allowance(WALLET_ADDRESS, spender).call() >= amount:
        return
    tx = token.functions.approve(spender, amount).build_transaction({"from": WALLET_ADDRESS})
    send(tx)

def get_price_and_tick():
    slot0 = pool.functions.slot0().call()
    sqrtPriceX96, tick = slot0[0], slot0[1]
    price = (sqrtPriceX96 / (2 ** 96)) ** 2 * 1e12  # âœ… Correct scaling
    return float(price), tick

def ticks_for_usd_bounds(price_now, tick_now, lower_usd, upper_usd, spacing):
    if price_now <= 0 or lower_usd <= 0 or upper_usd <= 0:
        raise ValueError(f"Invalid inputs: price={price_now}, range=({lower_usd},{upper_usd})")
    def to_tick(target):
        dt = math.log(target / price_now) / math.log(1.0001)
        return tick_now + dt
    t_lower = int(math.ceil(to_tick(lower_usd) / spacing) * spacing)
    t_upper = int(math.floor(to_tick(upper_usd) / spacing) * spacing)
    if t_upper <= t_lower:
        t_upper = t_lower + spacing
    return t_lower, t_upper

# ---------- Main ----------
def main():
    print("âœ… Connected to Ethereum mainnet!")
    price, cur_tick = get_price_and_tick()
    spacing = pool.functions.tickSpacing().call()

    # Place range just BELOW price (-15 â†’ -2 USD)
    lower_usd = max(1.0, price - 15)
    upper_usd = price - 2
    if upper_usd <= 0:
        upper_usd = price * 0.99
    tL, tU = ticks_for_usd_bounds(price, cur_tick, lower_usd, upper_usd, spacing)

    print(f"ðŸŽ¯ USDC-side tick range (below price): {tL} â†’ {tU}")
    print(f"ðŸ’° ETH â‰ˆ {price:.2f} USDC")

    token0 = pool.functions.token0().call()
    token1 = pool.functions.token1().call()
    t0 = w3.eth.contract(address=token0, abi=erc20_abi)
    t1 = w3.eth.contract(address=token1, abi=erc20_abi)
    dec0 = t0.functions.decimals().call()
    dec1 = t1.functions.decimals().call()
    bal0 = t0.functions.balanceOf(WALLET_ADDRESS).call()
    bal1 = t1.functions.balanceOf(WALLET_ADDRESS).call()
    sym0 = t0.functions.symbol().call()
    sym1 = t1.functions.symbol().call()

    # Heavy USDC side
    if sym0.upper() == "USDC":
        amt0, amt1 = int(bal0*0.95), int(bal1*0.05)
    else:
        amt0, amt1 = int(bal0*0.05), int(bal1*0.95)

    approve(token0, POSITION_MANAGER, amt0)
    approve(token1, POSITION_MANAGER, amt1)
    print(f"ðŸ’¼ Supplying {amt0/10**dec0:.6f} {sym0} + {amt1/10**dec1:.6f} {sym1}")

    params = {
        "token0": token0,
        "token1": token1,
        "fee": 500,
        "tickLower": tL,
        "tickUpper": tU,
        "amount0Desired": amt0,
        "amount1Desired": amt1,
        "amount0Min": 0,
        "amount1Min": 0,
        "recipient": WALLET_ADDRESS,
        "deadline": int(time.time()) + 600
    }

    r = send(pm.functions.mint(params).build_transaction({"from": WALLET_ADDRESS}))
    print(f"âœ… LP created! Tx {r.transactionHash.hex()}")

    try:
        ev = pm.events.Mint().process_receipt(r)
        token_id = int(ev[0]["args"]["tokenId"]) if ev else None
    except Exception:
        token_id = None
    if token_id is None:
        count = pm.functions.balanceOf(WALLET_ADDRESS).call()
        token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, count-1).call()

    data = {
        "side": "USDC",
        "timestamp_create": now_utc(),
        "lower_bound_usd": round(lower_usd, 2),
        "upper_bound_usd": round(upper_usd, 2),
        "tokenId": int(token_id)
    }
    with open("last_cycle_data.json","w") as f: json.dump(data, f, indent=2)
    print("ðŸ§¾ Updated last_cycle_data.json (USDC-side).")

if __name__ == "__main__":
    main()





