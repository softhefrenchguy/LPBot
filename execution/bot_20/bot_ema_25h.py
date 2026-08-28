#!/usr/bin/env python3
import os
import sys
import json
import time
import math
import subprocess
from datetime import datetime, timezone

from dotenv import load_dotenv
from web3 import Web3

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BOT_NAME = "ema_25h"
EMA_WINDOW_MIN = 1500
SCHEDULE_MIN = 1500
BAND_WIDTH_USD = 50.0
CAP_USD = 1000.0
CHECK_INTERVAL_SEC = 3.0
EMA_SAMPLE_SECONDS = 3.0
EMA_WARMUP_MINUTES = EMA_WINDOW_MIN
FEED_MAX_AGE_SEC = 20.0

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = os.getenv("RPC_URL", "https://ethereum.publicnode.com")
POOL_ADDRESS = Web3.to_checksum_address(
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # WETH/USDC 0.05% Ethereum mainnet
)
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")

STATE_FILE = os.path.join(script_dir, "ema_25h_state.json")
HISTORY_FILE = os.path.join(script_dir, "ema_25h_history.jsonl")
WITHDRAW_SCRIPT = os.path.join(script_dir, "withdraw_liquidity.py")

PRICE_FEED_FILE = os.getenv("PRICE_FEED_FILE")
if PRICE_FEED_FILE and not os.path.isabs(PRICE_FEED_FILE):
    PRICE_FEED_FILE = os.path.join(script_dir, PRICE_FEED_FILE)
if not PRICE_FEED_FILE:
    PRICE_FEED_FILE = os.path.join(script_dir, "price_feed.json")

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise SystemExit("WALLET_ADDRESS or PRIVATE_KEY missing in .env")
WALLET_ADDRESS = Web3.to_checksum_address(WALLET_ADDRESS)

# Web3
w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"proxies": {"http": None, "https": None}}))
if not w3.is_connected():
    raise SystemExit("Failed to connect to Ethereum mainnet RPC")

with open(os.path.join(script_dir, "pool_abi.json"), "r", encoding="utf-8") as f:
    pool_abi = json.load(f)
with open(os.path.join(script_dir, "NonfungiblePositionManager.json"), "r", encoding="utf-8") as f:
    pm_abi = json.load(f)
with open(os.path.join(script_dir, "erc20_abi.json"), "r", encoding="utf-8") as f:
    erc20_abi = json.load(f)

pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default or {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return json.loads(raw) if raw else (default or {})
    except Exception:
        return default or {}


def load_state():
    return _load_json(STATE_FILE, {})


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[{BOT_NAME}] state write failed: {e}")


def append_history(entry):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[{BOT_NAME}] history append failed: {e}")

def get_last_withdraw_entry():
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None
    for raw in reversed(lines):
        try:
            entry = json.loads(raw)
        except Exception:
            continue
        if entry.get("type") == "withdraw_liquidity":
            return entry
    return None


def get_last_create_entry():
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None
    for raw in reversed(lines):
        try:
            entry = json.loads(raw)
        except Exception:
            continue
        if entry.get("type") == "create_position":
            return entry
    return None


def sync_state_with_history():
    state = load_state()
    token_id = state.get("tokenId")
    if token_id:
        return state
    last_create = get_last_create_entry()
    if not last_create:
        return state
    token_id = last_create.get("token_id")
    if not token_id:
        return state
    try:
        owner = pm.functions.ownerOf(int(token_id)).call()
    except Exception:
        return state
    if owner and owner.lower() == WALLET_ADDRESS.lower():
        state["tokenId"] = int(token_id)
        state["lp_active"] = True
        save_state(state)
    return state


def estimate_weth_out(withdraw_entry):
    if not withdraw_entry:
        return None
    stats = withdraw_entry.get("stats") or {}
    price_exit = withdraw_entry.get("price_exit") or stats.get("price_exit")
    try:
        price_exit = float(price_exit)
    except Exception:
        return None
    if price_exit <= 0:
        return None
    def _f(val):
        try:
            return float(val)
        except Exception:
            return 0.0
    principal_weth = _f(withdraw_entry.get("principal_weth", stats.get("principal_weth")))
    principal_usdc = _f(withdraw_entry.get("principal_usdc", stats.get("principal_usdc")))
    fees_weth = _f(withdraw_entry.get("fees_weth", stats.get("fees_weth")))
    fees_usdc = _f(withdraw_entry.get("fees_usdc", stats.get("fees_usdc")))
    return principal_weth + fees_weth + (principal_usdc + fees_usdc) / price_exit


def read_price_from_feed():
    if not os.path.exists(PRICE_FEED_FILE):
        return None, None
    try:
        with open(PRICE_FEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        price = float(data.get("price"))
        ts = float(data.get("ts"))
    except Exception:
        return None, None
    age = time.time() - ts
    if age > FEED_MAX_AGE_SEC:
        return None, age
    return price, age


def ema_update(prev, value, alpha):
    if prev is None:
        return value
    return alpha * value + (1.0 - alpha) * prev


def compute_alpha(window_minutes, sample_seconds):
    if window_minutes <= 0:
        return 1.0
    n = (window_minutes * 60.0) / sample_seconds
    if n <= 1:
        return 1.0
    return 2.0 / (n + 1.0)


ALPHA = compute_alpha(EMA_WINDOW_MIN, EMA_SAMPLE_SECONDS)


def price_to_tick(price, dec0, dec1):
    # price is token1 per token0 (USDC per WETH). Adjust by decimals.
    scale = 10 ** (dec1 - dec0)
    return int(math.floor(math.log(price * scale) / math.log(1.0001)))


def _get_pending_nonce():
    try:
        return w3.eth.get_transaction_count(WALLET_ADDRESS, "pending")
    except Exception:
        return w3.eth.get_transaction_count(WALLET_ADDRESS)


def sign_and_send(tx, gas_limit=400_000):
    tx["from"] = WALLET_ADDRESS
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", w3.eth.gas_price)
    max_fee_cap = w3.to_wei(float(os.getenv("MAX_FEE_GWEI", "10")), "gwei")
    tx["maxFeePerGas"] = int(min(max_fee_cap, base_fee * 2))
    try:
        priority = w3.eth.max_priority_fee
    except Exception:
        priority = w3.to_wei("0.05", "gwei")
    tx["maxPriorityFeePerGas"] = int(max(priority, w3.to_wei("0.05", "gwei")))
    tx["gas"] = gas_limit

    for attempt in range(2):
        tx["nonce"] = _get_pending_nonce()
        signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        try:
            txh = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(txh)
            print(f"[{BOT_NAME}] tx {txh.hex()} | {receipt.gasUsed} gas | status={receipt.status}")
            return receipt
        except Exception as e:
            msg = str(e)
            if attempt == 0 and "nonce too low" in msg.lower():
                time.sleep(0.5)
                continue
            raise


def safe_approve(token_contract, spender, needed_raw, label):
    try:
        current = token_contract.functions.allowance(WALLET_ADDRESS, spender).call()
    except Exception as e:
        print(f"[{BOT_NAME}] allowance read failed for {label}: {e}")
        return False
    if current >= needed_raw:
        return True
    try:
        tx0 = token_contract.functions.approve(spender, 0).build_transaction({"from": WALLET_ADDRESS})
        sign_and_send(tx0, gas_limit=90_000)
    except Exception:
        pass
    try:
        tx1 = token_contract.functions.approve(spender, 2**256 - 1).build_transaction({"from": WALLET_ADDRESS})
        r1 = sign_and_send(tx1, gas_limit=110_000)
        return r1.status == 1
    except Exception as e:
        print(f"[{BOT_NAME}] approve failed for {label}: {e}")
        return False


def run_withdraw():
    env = os.environ.copy()
    env["BOT_STATE_FILE"] = STATE_FILE
    env["BOT_HISTORY_FILE"] = HISTORY_FILE
    print(f"[{BOT_NAME}] running withdraw_liquidity.py")
    res = subprocess.run(
        f'python "{WITHDRAW_SCRIPT}"',
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    return res.returncode


def create_position(lower_price, upper_price, price_now):
    if price_now > upper_price:
        print(f"[{BOT_NAME}] price {price_now:.2f} above upper {upper_price:.2f}; skip mint")
        return None

    token0 = pool.functions.token0().call()
    token1 = pool.functions.token1().call()
    t0 = w3.eth.contract(address=token0, abi=erc20_abi)
    t1 = w3.eth.contract(address=token1, abi=erc20_abi)
    dec0 = t0.functions.decimals().call()
    dec1 = t1.functions.decimals().call()
    sym0 = t0.functions.symbol().call()
    sym1 = t1.functions.symbol().call()
    spacing = pool.functions.tickSpacing().call()
    if sym0.upper() == "WETH" and sym1.upper() == "USDC":
        lower_for_tick = lower_price
        upper_for_tick = upper_price
    elif sym0.upper() == "USDC" and sym1.upper() == "WETH":
        lower_for_tick = 1.0 / upper_price
        upper_for_tick = 1.0 / lower_price
    else:
        print(f"[{BOT_NAME}] unsupported token order: {sym0}/{sym1}")
        return None

    raw_lower = price_to_tick(lower_for_tick, dec0, dec1)
    raw_upper = price_to_tick(upper_for_tick, dec0, dec1)
    if raw_upper <= raw_lower:
        raw_upper = raw_lower + spacing

    lower_tick = (raw_lower // spacing) * spacing
    upper_tick = (raw_upper // spacing) * spacing
    if upper_tick <= lower_tick:
        upper_tick = lower_tick + spacing

    bal0 = t0.functions.balanceOf(WALLET_ADDRESS).call()
    bal1 = t1.functions.balanceOf(WALLET_ADDRESS).call()

    if sym0.upper() == "WETH":
        weth_bal = bal0 / 10**dec0
    elif sym1.upper() == "WETH":
        weth_bal = bal1 / 10**dec1
    else:
        print(f"[{BOT_NAME}] WETH not found in pool tokens")
        return None

    cap_weth = CAP_USD / price_now
    offer_weth = min(weth_bal, cap_weth)
    if offer_weth <= 0.0001:
        print(f"[{BOT_NAME}] insufficient WETH to mint (have {weth_bal:.6f})")
        return None

    offer_weth *= 0.999

    if sym0.upper() == "WETH":
        amt0_offer = int(offer_weth * (10**dec0))
        amt1_offer = 0
        token_weth = t0
        label = "WETH token0"
    else:
        amt0_offer = 0
        amt1_offer = int(offer_weth * (10**dec1))
        token_weth = t1
        label = "WETH token1"

    if not safe_approve(token_weth, POSITION_MANAGER, max(amt0_offer, amt1_offer), label=label):
        print(f"[{BOT_NAME}] approve failed; abort mint")
        return None

    params = {
        "token0": token0,
        "token1": token1,
        "fee": 500,
        "tickLower": int(lower_tick),
        "tickUpper": int(upper_tick),
        "amount0Desired": int(amt0_offer),
        "amount1Desired": int(amt1_offer),
        "amount0Min": 0,
        "amount1Min": 0,
        "recipient": WALLET_ADDRESS,
        "deadline": int(time.time()) + 600,
    }

    try:
        mint_tx = pm.functions.mint(params).build_transaction({"from": WALLET_ADDRESS})
        receipt = sign_and_send(mint_tx, gas_limit=600_000)
    except Exception as e:
        print(f"[{BOT_NAME}] mint failed: {e}")
        return None

    if receipt.status != 1:
        print(f"[{BOT_NAME}] mint reverted")
        return None

    token_id = None
    try:
        events = pm.events.Mint().process_receipt(receipt)
        if events:
            token_id = int(events[0]["args"]["tokenId"])
    except Exception:
        pass

    if token_id is None:
        bal_nfts = pm.functions.balanceOf(WALLET_ADDRESS).call()
        if bal_nfts > 0:
            token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, bal_nfts - 1).call()

    if token_id is None:
        print(f"[{BOT_NAME}] mint succeeded but tokenId missing")
        return None

    print(
        f"[{BOT_NAME}] minted token {token_id} | band ${lower_price:.2f}-${upper_price:.2f} | "
        f"ticks {lower_tick} -> {upper_tick}"
    )

    portfolio_value = offer_weth * price_now
    state = load_state()
    state.update({
        "lp_active": True,
        "tokenId": int(token_id),
        "timestamp_create": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "lower_bound_usd": float(lower_price),
        "upper_bound_usd": float(upper_price),
        "range_width_usd": float(BAND_WIDTH_USD),
        "price_usd": float(price_now),
        "value_before_create": float(round(portfolio_value, 2)),
    })
    save_state(state)

    append_history({
        "type": "create_position",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "token_id": int(token_id),
        "lower_bound_usd": float(lower_price),
        "upper_bound_usd": float(upper_price),
        "price_usd": float(price_now),
        "weth_used": float(round(offer_weth, 6)),
        "value_usd": float(round(portfolio_value, 2)),
    })

    return {
        "token_id": token_id,
        "weth_used": float(offer_weth),
    }


def main():
    state = sync_state_with_history()
    ema = state.get("ema")
    ema_age_min = float(state.get("ema_age_min", 0.0))
    last_price = state.get("last_price")
    last_recenter_ts = float(state.get("last_recenter_ts", 0.0))
    pending_recenter = bool(state.get("pending_recenter", False))
    lp_active = bool(state.get("lp_active", False))
    token_id = state.get("tokenId")

    schedule_sec = SCHEDULE_MIN * 60.0

    print(f"[{BOT_NAME}] starting | EMA={EMA_WINDOW_MIN}m | schedule={SCHEDULE_MIN}m | width=${BAND_WIDTH_USD:.2f}")

    last_ts = time.time()

    while True:
        now = time.time()
        dt = now - last_ts
        if dt <= 0:
            dt = CHECK_INTERVAL_SEC
        last_ts = now

        price, age = read_price_from_feed()
        if price is None:
            if age is None:
                print(f"[{BOT_NAME}] price feed missing; waiting")
            else:
                print(f"[{BOT_NAME}] price feed stale ({age:.1f}s); waiting")
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        ema = ema_update(ema, price, ALPHA)
        ema_age_min += dt / 60.0
        last_price = price

        due = (now - last_recenter_ts) >= schedule_sec
        if due and ema is not None and ema_age_min >= EMA_WARMUP_MINUTES:
            pending_recenter = True
            state = load_state()
            state["pending_recenter"] = True
            pending_ts = state.get("pending_started_ts", now)
            if pending_ts is None:
                pending_ts = now
            state["pending_started_ts"] = float(pending_ts)
            save_state(state)
            print(f"[{BOT_NAME}] schedule due; waiting for price <= EMA (EMA={ema:.2f}, price={price:.2f})")

        if pending_recenter and ema is not None and ema_age_min >= EMA_WARMUP_MINUTES:
            if price > ema:
                print(f"[{BOT_NAME}] pending recenter; price {price:.2f} > EMA {ema:.2f} -> waiting")
            else:
                print(f"[{BOT_NAME}] pending recenter; price {price:.2f} <= EMA {ema:.2f} -> proceed")
                if lp_active and token_id:
                    state = load_state()
                    state["exit_reason"] = "scheduled_recenter"
                    state["exit_breakout_side"] = None
                    save_state(state)
                    rc = run_withdraw()
                    state = load_state()
                    lp_active = bool(state.get("lp_active", False))
                    token_id = state.get("tokenId")
                    if rc != 0:
                        print(f"[{BOT_NAME}] withdraw failed; will retry when price <= EMA")
                        time.sleep(CHECK_INTERVAL_SEC)
                        continue
                    withdraw_entry = get_last_withdraw_entry()
                    last_withdraw_ts = state.get("virtual_last_withdraw_ts")
                    if withdraw_entry and withdraw_entry.get("timestamp") != last_withdraw_ts:
                        weth_out = estimate_weth_out(withdraw_entry)
                        if weth_out is not None:
                            state["virtual_weth"] = float(state.get("virtual_weth", 0.0)) + float(weth_out)
                            state["virtual_last_withdraw_ts"] = withdraw_entry.get("timestamp")
                            append_history({
                                "type": "virtual_update",
                                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "virtual_weth": float(state.get("virtual_weth", 0.0)),
                                "weth_out": float(weth_out),
                                "price_exit": float(withdraw_entry.get("price_exit", 0.0) or 0.0),
                                "source": "withdraw",
                            })
                        save_state(state)

                if not lp_active:
                    lower = float(ema)
                    upper = float(ema + BAND_WIDTH_USD)
                    created = create_position(lower, upper, price)
                    if created is None:
                        print(f"[{BOT_NAME}] create failed; will retry when price <= EMA")
                        time.sleep(CHECK_INTERVAL_SEC)
                        continue
                    state = load_state()
                    state["virtual_weth"] = float(state.get("virtual_weth", 0.0)) - float(created.get("weth_used", 0.0))
                    append_history({
                        "type": "virtual_update",
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "virtual_weth": float(state.get("virtual_weth", 0.0)),
                        "weth_in": float(created.get("weth_used", 0.0)),
                        "price_create": float(price),
                        "source": "create",
                    })
                    save_state(state)

                last_recenter_ts = now
                pending_recenter = False
                state = load_state()
                state["last_recenter_ts"] = last_recenter_ts
                state["pending_recenter"] = False
                state["pending_started_ts"] = None
                save_state(state)

        state = load_state()
        state["ema"] = float(ema) if ema is not None else None
        state["ema_age_min"] = float(ema_age_min)
        state["last_price"] = float(last_price)
        state["last_recenter_ts"] = float(last_recenter_ts)
        state["pending_recenter"] = bool(pending_recenter)
        save_state(state)

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
