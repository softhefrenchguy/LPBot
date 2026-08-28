# portfolio_tracker_plus.py
from web3 import Web3
import json, os, sys
from decimal import Decimal, getcontext
from datetime import datetime

getcontext().prec = 80
Q96 = 1 << 96
Q128 = 1 << 128

# --------------------------
# Setup
# --------------------------
RPC_URL = os.getenv("RPC_URL", "https://arb1.arbitrum.io/rpc")
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("❌ RPC connection failed")

WALLET = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
USDC  = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
WETH  = Web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")

POOL  = Web3.to_checksum_address("0xC6962004f452bE9203591991D15f6b388e09E8D0")   # WETH/USDC 0.05%
NFPM  = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")   # Position Manager
FEE_TIER = 500

with open("erc20_abi.json") as f:
    ERC20_ABI = json.load(f)
with open("pool_abi_min.json") as f:
    POOL_ABI = json.load(f)
with open("NonfungiblePositionManager.json") as f:
    NFPM_ABI = json.load(f)

t_usdc = w3.eth.contract(address=USDC, abi=ERC20_ABI)
t_weth = w3.eth.contract(address=WETH, abi=ERC20_ABI)
pool   = w3.eth.contract(address=POOL, abi=POOL_ABI)
nfpm   = w3.eth.contract(address=NFPM, abi=NFPM_ABI)

BASE_FILE    = "portfolio_baseline.json"
HISTORY_FILE = "portfolio_history.jsonl"

DEBUG = "--debug" in sys.argv
if "--reset" in sys.argv and os.path.exists(BASE_FILE):
    os.remove(BASE_FILE)
    print("♻️  Old baseline deleted — creating new one...")

# --------------------------
# TickMath.getSqrtRatioAtTick (exact)
# --------------------------
MIN_TICK = -887272
MAX_TICK =  887272

def get_sqrt_ratio_at_tick(tick: int) -> int:
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError("tick out of range")

    absTick = tick if tick >= 0 else -tick
    ratios = [
        0xfffcb933bd6fad37aa2d162d1a594001, 0xfff97272373d413259a46990580e213a,
        0xfff2e50f5f656932ef12357cf3c7fdcc, 0xffe5caca7e10e4e61c3624eaa0941cd0,
        0xffcb9843d60f6159c9db58835c926644, 0xff973b41fa98c081472e6896dfb254c0,
        0xff2ea16466c96a3843ec78b326b52861, 0xfe5dee046a99a2a811c461f1969c3053,
        0xfcbe86c7900a88aedcffc83b479aa3a4, 0xf987a7253ac413176f2b074cf7815e54,
        0xf3392b0822b70005940c7a398e4b70f3, 0xe7159475a2c29b7443b29c7fa6e889d9,
        0xd097f3bdfd2022b8845ad8f792aa5825, 0xa9f746462d870fdf8a65dc1f90e061e5,
        0x70d869a156d2a1b890bb3df62baf32f7, 0x31be135f97d08fd981231505542fcfa6,
        0x9aa508b5b7a84e1c677de54f3e99bc9,  0x5d6af8dedb81196699c329225ee604,
        0x2216e584f5fa1ea926041bedfe98,      0x48a170391f7dc42444e8fa2
    ]

    ratio = 0x100000000000000000000000000000000
    mask = 1
    for i in range(20):
        if absTick & mask:
            ratio = (ratio * ratios[i]) >> 128
        mask <<= 1

    if tick > 0:
        ratio = (1 << 256) // ratio
    sqrtPriceX96 = (ratio >> 32) + (1 if (ratio & ((1 << 32) - 1)) else 0)
    return sqrtPriceX96

# --------------------------
# Liquidity math (exact Uniswap v3 Q96)
# --------------------------
def get_amounts_from_liquidity(liquidity: int, sqrtPriceX96: int, sqrtA: int, sqrtB: int):
    if sqrtA > sqrtB:
        sqrtA, sqrtB = sqrtB, sqrtA
    if sqrtPriceX96 <= sqrtA:
        amount0 = ((liquidity * (sqrtB - sqrtA)) << 96) // (sqrtB * sqrtA)
        amount1 = 0
    elif sqrtPriceX96 >= sqrtB:
        amount0 = 0
        amount1 = (liquidity * (sqrtB - sqrtA)) // Q96
    else:
        amount0 = ((liquidity * (sqrtB - sqrtPriceX96)) << 96) // (sqrtB * sqrtPriceX96)
        amount1 = (liquidity * (sqrtPriceX96 - sqrtA)) // Q96
    return amount0, amount1

# --------------------------
# Price (USDC per WETH)
# --------------------------
def get_price_usdc_per_weth() -> Decimal:
    sqrtP = int(pool.functions.slot0().call()[0])
    sp = Decimal(sqrtP) / Decimal(Q96)
    price_1_per_0 = sp * sp * (Decimal(10) ** (18 - 6))
    return price_1_per_0.quantize(Decimal("0.0001"))  # higher precision

# --------------------------
# Latest tokenId (fast)
# --------------------------
def latest_token_id():
    try:
        if os.path.exists("last_cycle_data.json"):
            with open("last_cycle_data.json") as f:
                tid = json.load(f).get("last_created_token_id")
                if isinstance(tid, int) and tid > 0:
                    return tid
    except Exception:
        pass
    bal = nfpm.functions.balanceOf(WALLET).call()
    if bal == 0:
        return None
    return nfpm.functions.tokenOfOwnerByIndex(WALLET, bal - 1).call()

# --------------------------
# LP value (latest only + uncollected fees)
# --------------------------
def lp_amounts_latest(debug=False):
    tid = latest_token_id()
    if not tid:
        if debug: print("🔍 No LP NFT found.")
        return Decimal(0), Decimal(0)
    p = nfpm.functions.positions(tid).call()
    token0, token1, fee = p[2], p[3], p[4]
    tL, tU, L = int(p[5]), int(p[6]), int(p[7])
    owed0, owed1 = int(p[10]), int(p[11])  # uncollected fees

    if fee != FEE_TIER or {token0.lower(), token1.lower()} != {WETH.lower(), USDC.lower()}:
        if debug: print("⏭️ Not a WETH/USDC 0.05% position.")
        return Decimal(0), Decimal(0)

    sqrtP = int(pool.functions.slot0().call()[0])
    sqrtA = get_sqrt_ratio_at_tick(tL)
    sqrtB = get_sqrt_ratio_at_tick(tU)
    amt0, amt1 = get_amounts_from_liquidity(L, sqrtP, sqrtA, sqrtB)
    amt0 += owed0
    amt1 += owed1

    if token0.lower() == WETH.lower():
        weth = Decimal(amt0) / Decimal(10**18)
        usdc = Decimal(amt1) / Decimal(10**6)
    else:
        usdc = Decimal(amt0) / Decimal(10**6)
        weth = Decimal(amt1) / Decimal(10**18)

    if debug:
        print(f"🎫 tokenId={tid} tickL={tL} tickU={tU} L={L}")
        print(f"💎 LP amounts ≈ {float(weth):.6f} WETH + ${float(usdc):.2f} USDC (incl. owed fees)")
    return weth, usdc

# --------------------------
# Wallet balances + native ETH
# --------------------------
def wallet_balances():
    usdc = Decimal(int(t_usdc.functions.balanceOf(WALLET).call())) / Decimal(10**6)
    weth = Decimal(int(t_weth.functions.balanceOf(WALLET).call())) / Decimal(10**18)
    eth  = Decimal(w3.eth.get_balance(WALLET)) / Decimal(10**18)
    return weth, usdc, eth

# --------------------------
# Fees
# --------------------------
def last_fees_usd():
    try:
        if os.path.exists("last_cycle_data.json"):
            with open("last_cycle_data.json") as f:
                return Decimal(str(json.load(f).get("fees_collected_usd", 0.0)))
    except Exception:
        pass
    return Decimal(0)

# --------------------------
# Main
# --------------------------
def main():
    print(f"✅ Connected to Arbitrum (Chain ID {w3.eth.chain_id})")
    price = get_price_usdc_per_weth()
    print(f"💰 Live price: {float(price):.2f} USDC/WETH")

    free_weth, free_usdc, free_eth = wallet_balances()
    lp_weth, lp_usdc = lp_amounts_latest(debug=DEBUG)
    fees_usd = last_fees_usd()

    total_usd = (free_usdc + lp_usdc) + (free_weth + lp_weth + free_eth) * price + fees_usd

    if not os.path.exists(BASE_FILE):
        baseline = {
            "created": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "base_price": float(price),
            "total_value_usd": float(round(total_usd, 2))
        }
        with open(BASE_FILE, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"✅ Baseline recorded — total value ${float(total_usd):.2f}")
        return

    with open(BASE_FILE) as f:
        base = json.load(f)

    market_effect = total_usd - Decimal(str(base["total_value_usd"]))
    bot_effect = fees_usd

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "price_usdc_per_weth": float(price),
        "actual_value_usd": float(round(total_usd, 2)),
        "market_effect_usd": float(round(market_effect, 2)),
        "bot_effect_usd": float(round(bot_effect, 2)),
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print("\n📊 Portfolio Tracker Summary")
    print("=================================")
    print(f"💼 Free: {float(free_weth):.6f} WETH + {float(free_eth):.6f} ETH + ${float(free_usdc):.2f} USDC")
    print(f"💎 LP  : {float(lp_weth):.6f} WETH + ${float(lp_usdc):.2f} USDC")
    print(f"💰 Price: {float(price):.2f} USDC/WETH")
    print(f"🧮 Total portfolio: ${float(total_usd):.2f}")
    print(f"🌍 Market move:     {float(market_effect):+.2f} USD")
    print(f"🤖 Bot/fees effect: {float(bot_effect):+.2f} USD")
    print("=================================")
    print(f"📈 Change vs baseline: {(float(total_usd) - float(base['total_value_usd'])):+.2f} USD")

if __name__ == "__main__":
    main()


