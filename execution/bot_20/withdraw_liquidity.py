#!/usr/bin/env python3
"""
withdraw_liquidity.py — exit LP, burn NFT, and ALWAYS end ETH-only (USDC -> WETH).

FULL SCRIPT (not a patch).

Steps:
 0) Snapshot wallet
 1) If tokenId exists & owned: collect fees, decrease liquidity, collect principal, burn NFT.
 2) EVEN IF no tokenId: ALWAYS attempt USDC -> WETH rebalance.
 3) Swap strategy:
      - KyberSwap Aggregator API (approve Kyber routerAddress BEFORE build)
      - Uniswap V3 SwapRouter02 fallback (only if pool exists for fee tier)
 4) Update last_cycle_data.json and append history.

ETH-only invariant:
  - NEVER swap WETH -> USDC
  - If swap fails, set rebalance_failed=True so main bot can pause new LP creation if desired.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from web3 import Web3
import logging
import requests
import importlib.util

# Ensure local imports work even if launched from another cwd
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def _load_discord_webhook():
    try:
        from discord_webhook import send_discord_message  # type: ignore
        return send_discord_message
    except ModuleNotFoundError:
        mod_path = os.path.join(script_dir, "discord_webhook.py")
        if os.path.exists(mod_path):
            spec = importlib.util.spec_from_file_location("discord_webhook", mod_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, "send_discord_message", lambda content: None)
        def _noop(content: str):
            print("⚠️ discord_webhook missing; skipping webhook.")
        return _noop


send_discord_message = _load_discord_webhook()

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.getLogger("web3").setLevel(logging.ERROR)

# -------------------------------------------------
# ENV / CONFIG
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = "https://ethereum.publicnode.com"
CHAIN_NAME = "ethereum"

POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS      = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")

state_override = os.getenv("BOT_STATE_FILE")
history_override = os.getenv("BOT_HISTORY_FILE")
if state_override and not os.path.isabs(state_override):
    state_override = os.path.join(script_dir, state_override)
if history_override and not os.path.isabs(history_override):
    history_override = os.path.join(script_dir, history_override)

STATE_FILE = state_override or os.path.join(script_dir, "last_cycle_data.json")
HISTORY_FILE = history_override or os.path.join(script_dir, "cycle_history.jsonl")

WETH_ADDR = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC_ADDR = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")

# Uniswap V3 fallback (SwapRouter v1 address matches the minimal ABI below)
UNISWAP_ROUTER    = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
UNISWAP_FACTORY   = Web3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984")
UNISWAP_FEE_TIERS = [500, 3000, 10000]

# KyberSwap Aggregator
KYBER_API_BASE  = "https://aggregator-api.kyberswap.com"
KYBER_CLIENT_ID = os.getenv("KYBER_CLIENT_ID", "bot_20")

# Swap controls
USDC_TO_WETH_MIN_USD   = float(os.getenv("USDC_TO_WETH_MIN_USD", "5.0"))
SLIPPAGE_BPS           = float(os.getenv("SWAP_SLIPPAGE_BPS", "15"))  # 15 = 0.15%
REBALANCE_FAIL_USDC_USD = float(os.getenv("REBALANCE_FAIL_USDC_USD", "10.0"))
USDC_REBALANCE_PREMIUM_PCT = float(os.getenv("USDC_REBALANCE_PREMIUM_PCT", "0.0015"))  # 0.15% above upper band
USDC_REBALANCE_STEP_FRACTION = float(os.getenv("USDC_REBALANCE_STEP_FRACTION", "0.33"))  # % of USDC per tranche when premium breached
USDC_REBALANCE_STEP_SECONDS = float(os.getenv("USDC_REBALANCE_STEP_SECONDS", "60"))      # delay between tranches
USDC_REBALANCE_MAX_DURATION_SECONDS = float(os.getenv("USDC_REBALANCE_MAX_DURATION_SECONDS", "360"))  # safety cap

MAX_UINT256 = 2**256 - 1


# -------------------------------------------------
# WEB3 SETUP
# -------------------------------------------------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("❌ Failed to connect to Ethereum mainnet RPC")
print("✅ Connected to Ethereum mainnet RPC")

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise SystemExit("❌ Missing WALLET_ADDRESS or PRIVATE_KEY in .env")
WALLET_ADDRESS = Web3.to_checksum_address(WALLET_ADDRESS)

# Load ABIs
with open(os.path.join(script_dir, "NonfungiblePositionManager.json"), "r", encoding="utf-8") as f:
    pm_abi = json.load(f)
with open(os.path.join(script_dir, "pool_abi.json"), "r", encoding="utf-8") as f:
    pool_abi = json.load(f)
with open(os.path.join(script_dir, "erc20_abi.json"), "r", encoding="utf-8") as f:
    erc20_abi = json.load(f)

pm   = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)
pool = w3.eth.contract(address=POOL_ADDRESS,      abi=pool_abi)
weth = w3.eth.contract(address=WETH_ADDR,         abi=erc20_abi)
usdc = w3.eth.contract(address=USDC_ADDR,         abi=erc20_abi)

WETH_DEC = weth.functions.decimals().call()
USDC_DEC = usdc.functions.decimals().call()

# Optional Uniswap router ABI for exactInputSingle
uniswap_router = None
try:
    with open(os.path.join(script_dir, "uniswap_v3_router_abi.json"), "r", encoding="utf-8") as f:
        uniswap_router_abi = json.load(f)
    uniswap_router = w3.eth.contract(address=UNISWAP_ROUTER, abi=uniswap_router_abi)
    print("✅ Loaded Uniswap v3 SwapRouter ABI (uniswap_v3_router_abi.json).")
except FileNotFoundError:
    print("⚠️ uniswap_v3_router_abi.json not found — Uniswap fallback swap will be skipped.")
except Exception as e:
    print(f"⚠️ Failed to load Uniswap router ABI: {e}")

# Minimal Uniswap factory ABI
UNISWAP_FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "uint24",   "name": "fee",    "type": "uint24"},
        ],
        "name": "getPool",
        "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]
factory = w3.eth.contract(address=UNISWAP_FACTORY, abi=UNISWAP_FACTORY_ABI)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def sign_and_send(tx, gas_limit=400_000):
    tx["from"] = WALLET_ADDRESS
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)

    # Cap max fee to avoid overpaying during spikes, but keep at least 2x base
    max_fee_cap = w3.to_wei(float(os.getenv("MAX_FEE_GWEI", "10")), "gwei")
    tx["maxFeePerGas"] = int(min(max_fee_cap, base_fee * 2))
    # Use a small priority floor to avoid stuck txs if the network requires a tip
    try:
        priority = w3.eth.max_priority_fee
    except Exception:
        priority = w3.to_wei("0.05", "gwei")
    tx["maxPriorityFeePerGas"] = int(max(priority, w3.to_wei("0.05", "gwei")))
    tx["gas"] = gas_limit

    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"🔗 Tx {tx_hash.hex()} | ⛽ {receipt.gasUsed} gas | status={receipt.status}")
    return receipt


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return json.loads(raw) if raw else {}
    except Exception as e:
        print(f"⚠️ Could not read {STATE_FILE}: {e}")
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write state file: {e}")


def append_history(entry):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Could not write to history file: {e}")


def sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96: int) -> float:
    return float((sqrt_price_x96 / (2 ** 96)) ** 2 * 1e12)


def get_price_and_wallet():
    slot0 = pool.functions.slot0().call()
    sqrtp = slot0[0]
    price = sqrt_price_x96_to_price_usdc_per_weth(sqrtp)

    w = weth.functions.balanceOf(WALLET_ADDRESS).call() / 10**WETH_DEC
    u = usdc.functions.balanceOf(WALLET_ADDRESS).call() / 10**USDC_DEC
    total = u + w * price
    return price, w, u, total


def repair_state_invalid_token(extra=None):
    state = load_state()
    state["lp_active"] = False
    state["tokenId"] = None
    state["lower_bound_usd"] = None
    state["upper_bound_usd"] = None
    if isinstance(extra, dict):
        state.update(extra)
    save_state(state)
    print("🧹 Cleaned LP state (invalid token).")


def safe_approve(token_contract, spender, needed_raw, label="Token"):
    """
    Safer approve:
      - If allowance already >= needed_raw, do nothing
      - Else: approve(0) then approve(MAX_UINT256)
    This avoids USDC-style allowance quirks and avoids repeated approvals later.
    """
    try:
        current = token_contract.functions.allowance(WALLET_ADDRESS, spender).call()
    except Exception as e:
        print(f"❌ Could not read allowance for {label}: {e}")
        return False

    if current >= needed_raw:
        return True

    # 0 reset
    try:
        tx0 = token_contract.functions.approve(spender, 0).build_transaction({"from": WALLET_ADDRESS})
        r0 = sign_and_send(tx0, gas_limit=90_000)
        if r0.status != 1:
            print(f"⚠️ {label} approve(0) reverted.")
            return False
    except Exception as e:
        print(f"⚠️ {label} approve(0) failed: {e}")
        # some tokens don't require reset; continue to attempt max approve

    # max approve
    try:
        tx1 = token_contract.functions.approve(spender, MAX_UINT256).build_transaction({"from": WALLET_ADDRESS})
        r1 = sign_and_send(tx1, gas_limit=110_000)
        if r1.status != 1:
            print(f"⚠️ {label} approve(max) reverted.")
            return False
        print(f"✅ {label} approved (max).")
        return True
    except Exception as e:
        print(f"❌ {label} approve(max) failed: {e}")
        return False


# -------------------------------------------------
# KYBER SWAP (approve BEFORE build)
# -------------------------------------------------
def kyber_get_route(amount_in_raw: int):
    url = f"{KYBER_API_BASE}/{CHAIN_NAME}/api/v1/routes"
    params = {
        "tokenIn": USDC_ADDR,
        "tokenOut": WETH_ADDR,
        "amountIn": str(int(amount_in_raw)),
    }
    headers = {"x-client-id": KYBER_CLIENT_ID}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            try:
                j = r.json()
                print(f"⚠️ Kyber route error HTTP {r.status_code}: {j.get('message', j)}")
            except Exception:
                print(f"⚠️ Kyber route error HTTP {r.status_code}: {r.text[:200]}")
            return None

        j = r.json()
        if j.get("code") != 0:
            print(f"⚠️ Kyber route returned code={j.get('code')} msg={j.get('message')}")
            return None

        data = j.get("data") or {}
        if not data.get("routeSummary") or not data.get("routerAddress"):
            print("⚠️ Kyber route missing routeSummary/routerAddress.")
            return None

        return data
    except Exception as e:
        print(f"⚠️ Kyber route request failed: {e}")
        return None


def kyber_build_tx(route_data: dict):
    """
    IMPORTANT: enableGasEstimation=False to avoid build failing due to allowance estimation.
    """
    url = f"{KYBER_API_BASE}/{CHAIN_NAME}/api/v1/route/build"
    headers = {
        "x-client-id": KYBER_CLIENT_ID,
        "Content-Type": "application/json",
    }

    body = {
        "routeSummary": route_data["routeSummary"],
        "sender": WALLET_ADDRESS,
        "recipient": WALLET_ADDRESS,
        "origin": WALLET_ADDRESS,
        "slippageTolerance": SLIPPAGE_BPS,
        "enableGasEstimation": False,   # <<< key fix
        "source": KYBER_CLIENT_ID,
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if r.status_code != 200:
            try:
                j = r.json()
                print(f"⚠️ Kyber build error HTTP {r.status_code}: {j.get('message', j)}")
            except Exception:
                print(f"⚠️ Kyber build error HTTP {r.status_code}: {r.text[:200]}")
            return None

        j = r.json()
        if j.get("code") != 0:
            print(f"⚠️ Kyber build returned code={j.get('code')} msg={j.get('message')}")
            return None

        data = j.get("data") or {}
        if not data.get("data") or not data.get("to"):
            print("⚠️ Kyber build missing tx fields.")
            return None

        return data
    except Exception as e:
        print(f"⚠️ Kyber build request failed: {e}")
        return None


def try_swap_usdc_to_weth_kyber(usdc_raw: int) -> bool:
    route = kyber_get_route(usdc_raw)
    if not route:
        return False

    router_addr = Web3.to_checksum_address(route["routerAddress"])

    # ✅ FIX: approve Kyber router BEFORE build to avoid TRANSFER_FROM_FAILED in their estimator
    if not safe_approve(usdc, router_addr, usdc_raw, label="USDC (Kyber)"):
        return False

    built = kyber_build_tx(route)
    if not built:
        return False

    to_addr = Web3.to_checksum_address(built["to"])
    calldata = built["data"]
    value_wei = int(built.get("value", "0") or "0")

    try:
        tx = {"to": to_addr, "data": calldata, "value": value_wei}
        r = sign_and_send(tx, gas_limit=550_000)
        if r.status == 1:
            print("? Kyber USDC->WETH swap completed.")
            return True
        print(f"?? Kyber swap tx reverted (gasUsed={r.gasUsed}, status={r.status}).")
        return False
    except Exception as e:
        print(f"? Kyber swap failed: {e}")
        return False


# -------------------------------------------------
# UNISWAP FALLBACK (only if pool exists)
# -------------------------------------------------
def uniswap_pool_exists(fee: int) -> bool:
    try:
        p = factory.functions.getPool(USDC_ADDR, WETH_ADDR, int(fee)).call()
        if int(p, 16) == 0:
            return False
        return True
    except Exception:
        # token order might be opposite
        try:
            p = factory.functions.getPool(WETH_ADDR, USDC_ADDR, int(fee)).call()
            if int(p, 16) == 0:
                return False
            return True
        except Exception:
            return False


def try_swap_usdc_to_weth_uniswap(usdc_raw: int) -> bool:
    if uniswap_router is None:
        return False

    # approve router once (max)
    if not safe_approve(usdc, UNISWAP_ROUTER, usdc_raw, label="USDC (Uniswap)"):
        return False

    usdc_amt = usdc_raw / 10**USDC_DEC

    for fee in UNISWAP_FEE_TIERS:
        if not uniswap_pool_exists(fee):
            print(f"?? Uniswap pool missing for fee={fee} - skipping tier.")
            continue

        try:
            # Spot-price-based minimum (this is a fallback path; Kyber above already does real quoting/routing).
            # Not a true quote — ignores this trade's own price impact — so SLIPPAGE_BPS must cover that gap too.
            spot_price = sqrt_price_x96_to_price_usdc_per_weth(pool.functions.slot0().call()[0])
            expected_weth = (usdc_amt / spot_price) * (1 - fee / 1_000_000.0)
            min_weth_raw = int(expected_weth * (1 - SLIPPAGE_BPS / 10000.0) * 10**WETH_DEC)
            print(
                f"🛡️ Uniswap fallback slippage guard (fee={fee}): expected≈{expected_weth:.6f} WETH "
                f"-> amountOutMinimum={min_weth_raw} ({SLIPPAGE_BPS:.0f}bps tolerance)"
            )
            params = {
                "tokenIn": USDC_ADDR,
                "tokenOut": WETH_ADDR,
                "fee": int(fee),
                "recipient": WALLET_ADDRESS,
                "deadline": int(time.time()) + 600,
                "amountIn": int(usdc_raw),
                "amountOutMinimum": min_weth_raw,
                "sqrtPriceLimitX96": 0,
            }
            tx = uniswap_router.functions.exactInputSingle(params).build_transaction({"from": WALLET_ADDRESS})
            r = sign_and_send(tx, gas_limit=420_000)
            if r.status == 1:
                print(f"? Uniswap fallback USDC->WETH swap completed (fee={fee}).")
                return True
            print(f"?? Uniswap swap reverted (fee={fee}) - trying next tier...")
        except Exception as e:
            print(f"?? Uniswap swap attempt failed/reverted (fee={fee}) - trying next tier... ({e})")

    return False



# -------------------------------------------------
# REBALANCE WRAPPER
# -------------------------------------------------
def ensure_usdc_to_weth(price_now: float, band_upper: float | None):
    usdc_raw = usdc.functions.balanceOf(WALLET_ADDRESS).call()
    if usdc_raw == 0:
        print("?? No USDC to swap (balance=0).")
        return

    usdc_amt = usdc_raw / 10**USDC_DEC
    if usdc_amt < USDC_TO_WETH_MIN_USD:
        print(f"?? USDC balance {usdc_amt:.2f} (< ${USDC_TO_WETH_MIN_USD}) - not worth swapping.")
        return

    premium_guard = False
    price_ceiling = None
    if band_upper and USDC_REBALANCE_PREMIUM_PCT > 0:
        price_ceiling = band_upper * (1.0 + USDC_REBALANCE_PREMIUM_PCT)
        if price_now > price_ceiling:
            premium_guard = True
            pct_above = (price_now - band_upper) / band_upper * 100.0
            print(
                f"?? Price ${price_now:.2f} is {pct_above:.3f}% above band upper ${band_upper:.2f} "
                f"(ceiling {price_ceiling:.2f}) -> phased rebuy."
            )

    swapped_any = False
    start_ts = time.time()

    def do_swap(amount_raw: int):
        nonlocal swapped_any
        swapped = try_swap_usdc_to_weth_kyber(amount_raw)
        if not swapped:
            print("?? Kyber swap failed - trying Uniswap V3 fallback...")
            swapped = try_swap_usdc_to_weth_uniswap(amount_raw)
        if swapped:
            swapped_any = True
        return swapped

    state = load_state()

    while True:
        usdc_raw = usdc.functions.balanceOf(WALLET_ADDRESS).call()
        usdc_amt = usdc_raw / 10**USDC_DEC
        if usdc_amt < USDC_TO_WETH_MIN_USD:
            print(f"?? USDC balance {usdc_amt:.2f} below min ${USDC_TO_WETH_MIN_USD} - stopping.")
            break

        if premium_guard and (time.time() - start_ts) > USDC_REBALANCE_MAX_DURATION_SECONDS:
            print("?? Phased rebuy hit max duration cap; stopping further swaps.")
            break

        if premium_guard:
            step_amt = max(usdc_amt * USDC_REBALANCE_STEP_FRACTION, USDC_TO_WETH_MIN_USD)
            step_raw = int(step_amt * (10**USDC_DEC))
            print(f"?? Phased swap: {step_amt:.2f} USDC of {usdc_amt:.2f} (step {USDC_REBALANCE_STEP_FRACTION*100:.1f}%).")
            ok = do_swap(step_raw)
            if not ok:
                break

            price_now, _, _, _ = get_price_and_wallet()
            if price_ceiling and price_now <= price_ceiling:
                premium_guard = False
                print(f"? Price cooled to ${price_now:.2f} <= ceiling {price_ceiling:.2f}; finishing remaining USDC.")
            else:
                time.sleep(USDC_REBALANCE_STEP_SECONDS)
                continue
        else:
            print(f"?? Rebalancing USDC -> WETH: {usdc_amt:.6f} USDC (? ${usdc_amt:.2f})")
            do_swap(usdc_raw)
            break

    remaining_usdc = usdc.functions.balanceOf(WALLET_ADDRESS).call() / 10**USDC_DEC
    if not swapped_any:
        print("? Could not swap USDC->WETH via Kyber or Uniswap - staying in USDC.")
        state["rebalance_failed"] = True
        state["rebalance_failed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        save_state(state)
    else:
        state["rebalance_failed"] = False
        state["rebalance_failed_at"] = None
        save_state(state)
        if remaining_usdc >= REBALANCE_FAIL_USDC_USD:
            print(f"?? Remaining USDC {remaining_usdc:.2f} >= ${REBALANCE_FAIL_USDC_USD} after phased swaps.")
            state["rebalance_failed"] = True
            state["rebalance_failed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            save_state(state)


# -------------------------------------------------
# LP WITHDRAW
# -------------------------------------------------
def withdraw_if_possible():
    state = load_state()
    token_id = state.get("tokenId")

    if token_id is None:
        print("ℹ️ No tokenId in last_cycle_data.json — skipping LP withdraw steps.")
        repair_state_invalid_token()
        return {"ran": False, "gas_eth": 0.0, "token_id": None, "wallet_before": None}

    print(f"🏷 Withdrawing LP NFT #{token_id}")

    # Verify ownership
    try:
        owner = pm.functions.ownerOf(int(token_id)).call()
    except Exception as e:
        print(f"❌ NFT #{token_id} cannot be read ({e}) — repairing state.")
        repair_state_invalid_token()
        return {"ran": False, "gas_eth": 0.0, "token_id": token_id, "wallet_before": None}

    if owner.lower() != WALLET_ADDRESS.lower():
        print(f"❌ You do NOT own NFT #{token_id}! (Owned by {owner})")
        repair_state_invalid_token()
        return {"ran": False, "gas_eth": 0.0, "token_id": token_id, "wallet_before": None}

    price_before, weth_before, usdc_before, total_before = get_price_and_wallet()
    print(f"📉 Price before = ${price_before:.2f}")
    print(f"🔍 BEFORE → {weth_before:.6f} WETH + {usdc_before:.2f} USDC (${total_before:.2f})")

    pos = pm.functions.positions(int(token_id)).call()
    liquidity = int(pos[7])
    print(f"💧 Liquidity = {liquidity}")

    total_gas_eth = 0.0

    fee_collect0 = 0.0  # fees in token0 (WETH)
    fee_collect1 = 0.0  # fees in token1 (USDC)
    principal_collect0 = 0.0  # principal in token0 from second collect
    principal_collect1 = 0.0  # principal in token1 from second collect

    def _sum_collect(receipt, as_fees: bool):
        nonlocal fee_collect0, fee_collect1, principal_collect0, principal_collect1
        try:
            evs = pm.events.Collect().process_receipt(receipt)
            for ev in evs:
                a0 = ev["args"].get("amount0", 0)
                a1 = ev["args"].get("amount1", 0)
                if as_fees:
                    fee_collect0 += a0 / 10**WETH_DEC
                    fee_collect1 += a1 / 10**USDC_DEC
                else:
                    principal_collect0 += a0 / 10**WETH_DEC
                    principal_collect1 += a1 / 10**USDC_DEC
        except Exception:
            pass

    # Step 1: Collect fees
    try:
        tx = pm.functions.collect({
            "tokenId": int(token_id),
            "recipient": WALLET_ADDRESS,
            "amount0Max": 2**128 - 1,
            "amount1Max": 2**128 - 1,
        }).build_transaction({"from": WALLET_ADDRESS})
        r = sign_and_send(tx, gas_limit=220_000)
        total_gas_eth += (r.gasUsed * r.effectiveGasPrice) / 1e18
        print("🧾 Collected fees.")
        _sum_collect(r, as_fees=True)
    except Exception as e:
        print(f"⚠️ Fee collection failed: {e}")

    # Step 2: Decrease liquidity
    if liquidity > 0:
        try:
            decrease_params = {
                "tokenId": int(token_id),
                "liquidity": liquidity,
                "amount0Min": 0,
                "amount1Min": 0,
                "deadline": int(time.time()) + 600,
            }
            # Simulate first (read-only static call, no tx sent) to get expected amounts,
            # then apply real slippage protection instead of accepting any amount.
            try:
                expected0, expected1 = pm.functions.decreaseLiquidity(decrease_params).call({"from": WALLET_ADDRESS})
                min0 = int(expected0 * (1 - SLIPPAGE_BPS / 10000.0))
                min1 = int(expected1 * (1 - SLIPPAGE_BPS / 10000.0))
                decrease_params["amount0Min"] = min0
                decrease_params["amount1Min"] = min1
                print(
                    f"🛡️ decreaseLiquidity slippage guard: expected0={expected0}, expected1={expected1} "
                    f"-> min0={min0}, min1={min1} ({SLIPPAGE_BPS:.0f}bps tolerance)"
                )
            except Exception as sim_e:
                print(f"⚠️ Could not simulate decreaseLiquidity for slippage guard ({sim_e}) — proceeding with amount0Min=amount1Min=0.")

            tx = pm.functions.decreaseLiquidity(decrease_params).build_transaction({"from": WALLET_ADDRESS})
            r = sign_and_send(tx, gas_limit=350_000)
            total_gas_eth += (r.gasUsed * r.effectiveGasPrice) / 1e18
            print("💧 Liquidity withdrawn.")
        except Exception as e:
            print(f"❌ decreaseLiquidity failed: {e}")
    else:
        print("ℹ️ Position already has zero liquidity on-chain.")

    # Step 3: Collect principal
    try:
        tx = pm.functions.collect({
            "tokenId": int(token_id),
            "recipient": WALLET_ADDRESS,
            "amount0Max": 2**128 - 1,
            "amount1Max": 2**128 - 1,
        }).build_transaction({"from": WALLET_ADDRESS})
        r = sign_and_send(tx, gas_limit=220_000)
        total_gas_eth += (r.gasUsed * r.effectiveGasPrice) / 1e18
        print("🧾 Final collection complete.")
        _sum_collect(r, as_fees=False)
    except Exception as e:
        print(f"⚠️ Final collect failed: {e}")

    # Step 4: Burn NFT
    try:
        tx = pm.functions.burn(int(token_id)).build_transaction({"from": WALLET_ADDRESS})
        r = sign_and_send(tx, gas_limit=170_000)
        total_gas_eth += (r.gasUsed * r.effectiveGasPrice) / 1e18
        print(f"🔥 Burned NFT #{token_id}")
    except Exception as e:
        print(f"⚠️ Could not burn NFT: {e}")

    # Update state: LP closed
    state = load_state()
    state["lp_active"] = False
    state["tokenId"] = None
    state["lower_bound_usd"] = None
    state["upper_bound_usd"] = None
    state["timestamp_withdraw"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    save_state(state)

    return {
        "ran": True,
        "gas_eth": float(total_gas_eth),
        "token_id": token_id,
        "wallet_before": {"weth": float(weth_before), "usdc": float(usdc_before), "total": float(total_before)},
        "price_before": float(price_before),
        "fees_weth": float(fee_collect0),
        "fees_usdc": float(fee_collect1),
        "principal_collect0": float(principal_collect0),
        "principal_collect1": float(principal_collect1),
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    state_initial = load_state()
    band_upper = state_initial.get("upper_bound_usd")
    # snapshot at start
    price0, weth0, usdc0, total0 = get_price_and_wallet()

    # withdraw if possible (but NEVER return early)
    withdraw_stats = withdraw_if_possible()

    # wallet mid
    price_mid, weth_mid, usdc_mid, total_mid = get_price_and_wallet()
    print(f"🔍 AFTER withdraw → {weth_mid:.6f} WETH + {usdc_mid:.2f} USDC (${total_mid:.2f})")

    # rebalance always
    ensure_usdc_to_weth(price_mid, band_upper)

    # final wallet
    price_after, weth_after, usdc_after, total_after = get_price_and_wallet()
    print(f"🔍 FINAL (after USDC→WETH) → {weth_after:.6f} WETH + {usdc_after:.2f} USDC (${total_after:.2f})")

    # Implied rebalance price from mid -> final (USDC -> WETH conversion)
    conv_weth = weth_after - weth_mid
    implied_rebalance_price = None
    if usdc_mid > 0 and conv_weth > 0:
        implied_rebalance_price = usdc_mid / conv_weth
        print(
            f"?? Implied rebalance price: ${implied_rebalance_price:.2f} (converted {usdc_mid:.2f} USDC -> {conv_weth:.6f} WETH)"
        )

    # pnl (wallet view) - baseline MUST be the portfolio value before create, not the small wallet stub
    entry_value = float(load_state().get("value_before_create", total0))
    before_total = withdraw_stats["wallet_before"]["total"] if withdraw_stats.get("ran") and withdraw_stats.get("wallet_before") else total0
    gas_eth = float(withdraw_stats.get("gas_eth", 0.0))
    gas_usd = gas_eth * price_after
    pnl = total_after - entry_value
    net = pnl - gas_usd

    # mark failure if still meaningful USDC
    remaining_usdc = usdc.functions.balanceOf(WALLET_ADDRESS).call() / 10**USDC_DEC
    rebalance_failed = remaining_usdc >= REBALANCE_FAIL_USDC_USD

    state = load_state()
    state["wallet_after_withdraw"] = {"weth": float(weth_after), "usdc": float(usdc_after), "total_usd": float(total_after)}
    fee_weth = float(withdraw_stats.get("fees_weth", 0.0))
    fee_usdc = float(withdraw_stats.get("fees_usdc", 0.0))
    fee_usd = fee_weth * price_after + fee_usdc
    principal_weth = float(withdraw_stats.get("principal_collect0", 0.0))
    principal_usdc = float(withdraw_stats.get("principal_collect1", 0.0))
    principal_usd = principal_weth * price_after + principal_usdc

    # Track post-rebalance WETH balance explicitly (to monitor ownership drift)
    state["weth_after_rebalance"] = float(weth_after)

    state["withdraw_stats"] = {
        "value_in_usd": float(round(entry_value, 2)),
        "value_out_usd": float(round(total_after, 2)),
        "raw_pnl_usd": float(round(pnl, 2)),
        "gas_eth": float(gas_eth),
        "gas_usd": float(round(gas_usd, 4)),
        "net_pnl_usd": float(round(net, 4)),
        "price_start": float(price0),
        "price_exit": float(price_after),
        "weth_start": float(weth0),
        "usdc_start": float(usdc0),
        "weth_final": float(weth_after),
        "usdc_final": float(usdc_after),
        "fees_weth": fee_weth,
        "fees_usdc": fee_usdc,
        "fees_usd": float(round(fee_usd, 4)),
        "principal_weth": principal_weth,
        "principal_usdc": principal_usdc,
        "principal_usd": float(round(principal_usd, 4)),
    }
    state["rebalance_failed"] = bool(rebalance_failed)
    state["rebalance_failed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if rebalance_failed else None
    state["rebalance_failed_usdc"] = float(round(remaining_usdc, 6))
    save_state(state)

    # --- Summary logging: wallet PnL, position PnL, APR-ish signal ---
    position_start = float(state.get("value_before_create", entry_value))
    position_pnl = total_after - float(position_start or entry_value)

    dur_minutes = None
    try:
        ts_create = state.get("timestamp_create")
        if ts_create:
            dur_minutes = max(
                0.0,
                (datetime.now(timezone.utc) - datetime.fromisoformat(ts_create.replace(" UTC", "+00:00"))).total_seconds() / 60.0,
            )
    except Exception:
        pass

    apr_pct = None
    if dur_minutes and dur_minutes > 0 and position_start:
        apr_pct = (position_pnl / float(position_start)) * (525600.0 / dur_minutes) * 100.0

    print(
        f"📊 Wallet PnL Δ=${pnl:.2f} (net=${net:.2f} after gas ${gas_usd:.4f}); "
        f"Position PnL Δ=${position_pnl:.2f}; "
        f"Fees=${fee_usd:.2f} ({fee_weth:.6f} WETH + {fee_usdc:.2f} USDC); "
        f"WETH after rebalance={weth_after:.6f}"
    )
    if implied_rebalance_price is not None:
        print(f"?? Rebalance price (USDC->WETH): ${implied_rebalance_price:.2f}")
    if apr_pct is not None:
        print(f"📈 Implied APR vs entry ${position_start:.2f} over {dur_minutes:.1f}m: {apr_pct:.2f}%")
    print(f"⛽ Gas cost: {gas_eth:.6f} ETH (${gas_usd:.4f}) | Rebalance_failed={rebalance_failed}")

    append_history({
        "type": "withdraw_liquidity",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "withdraw_ran": bool(withdraw_stats.get("ran")),
        "token_id": withdraw_stats.get("token_id"),
        "price_exit": float(price_after),
        "wallet_start": {"weth": float(weth0), "usdc": float(usdc0), "total": float(total0)},
        "wallet_after_withdraw": {"weth": float(weth_mid), "usdc": float(usdc_mid), "total": float(total_mid)},
        "wallet_final": {"weth": float(weth_after), "usdc": float(usdc_after), "total": float(total_after)},
        "stats": state.get("withdraw_stats", {}),
        "rebalance_failed": bool(state.get("rebalance_failed")),
        "rebalance_failed_usdc": float(state.get("rebalance_failed_usdc", 0.0)),
        "exit_reason": state.get("exit_reason"),
        "exit_breakout_side": state.get("exit_breakout_side"),
        "gas_eth": float(gas_eth),
        "wallet_pnl_usd": float(round(pnl, 4)),
        "wallet_net_pnl_usd": float(round(net, 4)),
        "position_pnl_usd": float(round(position_pnl, 4)),
        "apr_pct": float(round(apr_pct, 4)) if apr_pct is not None else None,
        "duration_minutes": float(round(dur_minutes, 2)) if dur_minutes is not None else None,
        "fees_weth": fee_weth,
        "fees_usdc": fee_usdc,
        "fees_usd": float(round(fee_usd, 4)),
        "principal_weth": principal_weth,
        "principal_usdc": principal_usdc,
        "principal_usd": float(round(principal_usd, 4)),
        "weth_after_rebalance": float(weth_after),
        "implied_rebalance_price": float(round(implied_rebalance_price, 6)) if implied_rebalance_price is not None else None,
    })

    if rebalance_failed:
        print(f"❌ Rebalance FAILED: still holding ~{remaining_usdc:.2f} USDC. (State flag set: rebalance_failed=True)")
    print("✅ withdraw_liquidity.py completed (ETH-only invariance enforced).\n")

    # Discord notification with PnL/fees summary
    try:
        msg = (
            f"?? Withdraw #{withdraw_stats.get('token_id')} | price=${price_after:.2f}\n"
            f"Wallet PnL: ${pnl:.2f} (net ${net:.2f} after gas ${gas_usd:.4f})\n"
            f"Position PnL: ${position_pnl:.2f} | APR: {apr_pct:.2f}% over {dur_minutes:.1f}m\n"
            f"Fees: ${fee_usd:.2f} ({fee_weth:.6f} WETH + {fee_usdc:.2f} USDC)\n"
            f"WETH after rebalance={weth_after:.6f}, gas={gas_eth:.6f} ETH\n"
            f"Principal: ${principal_usd:.2f} ({principal_weth:.6f} WETH + {principal_usdc:.2f} USDC)"
        )
        if implied_rebalance_price is not None:
            msg += f"\nRebalance price (USDC->WETH): ${implied_rebalance_price:.2f}"
        send_discord_message(msg)
    except Exception:
        pass


if __name__ == "__main__":
    main()
