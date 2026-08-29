import os
import json
import time
import math
from datetime import datetime
import sys

# Ensure local imports work even if launched from another cwd
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import importlib.util
from web3 import Web3
from dotenv import load_dotenv

# Robust local import of discord_webhook even if cwd is different
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
        # Fallback no-op to avoid crashing
        def _noop(content: str):
            print("⚠️ discord_webhook missing; skipping webhook.")
        return _noop

send_discord_message = _load_discord_webhook()

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------
# ⚙️ CONFIG
# ----------------------------
RPC_URL = "https://ethereum.publicnode.com"
POSITION_MANAGER = Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88")
POOL_ADDRESS = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")  # WETH/USDC 0.05%

STATE_FILE = "last_cycle_data.json"
HISTORY_FILE = "cycle_history.jsonl"

# Adaptive width config (ticks)
MIN_WIDTH_TICKS = 10
MAX_WIDTH_TICKS = 120
MAX_WIDTH_STEP = 30  # max change per cycle
USDC_DUST_THRESHOLD = 1_000  # 0.001 USDC (6 decimals) treated as zero for single-sided

# Position mode: "full_range" (default) spans the whole curve, both tokens deposited, no rebalance
# churn from price moving within the range. "tight_range" is the original adaptive ±5-25 tick,
# WETH-only, fast-cycling design -- kept available but opt-in, since mainnet gas economics make
# that design's frequent close/reopen cycling far more expensive than it was on Arbitrum.
POSITION_MODE = os.getenv("POSITION_MODE", "full_range").strip().lower()
UNISWAP_MIN_TICK = -887272
UNISWAP_MAX_TICK = 887272


def full_range_ticks(spacing: int) -> tuple[int, int]:
    """Widest tick range allowed for this pool's spacing: ceil(MIN_TICK/spacing) to floor(MAX_TICK/spacing)."""
    lower = -((-UNISWAP_MIN_TICK) // spacing) * spacing  # ceil division, stays >= MIN_TICK
    upper = (UNISWAP_MAX_TICK // spacing) * spacing        # floor division, stays <= MAX_TICK
    return lower, upper

# Vol/slope-based widening
VOL_WIDEN_THRESHOLD = float(os.getenv("VOL_WIDEN_THRESHOLD", "0.00018"))       # vol_ewma (abs returns)
SLOPE_WIDEN_THRESHOLD = float(os.getenv("SLOPE_WIDEN_THRESHOLD", "0.18"))      # %/min (slope5)
VOL_WIDEN_MULT = float(os.getenv("VOL_WIDEN_MULT", "1.5"))
SLOPE_WIDEN_MULT = float(os.getenv("SLOPE_WIDEN_MULT", "1.35"))

# Target duration window (mins)
TARGET_DUR_LOW = 12.0
TARGET_DUR_HIGH = 17.0

# USD band: normal vs panic re-entry
NORMAL_BAND_FRAC = 0.004  # ±0.4% ≈ ±$11 at $2800
PANIC_BAND_FRAC = 0.008   # ±0.8% ≈ ±$22 at $2800
TARGET_BAND_USD = float(os.getenv("TARGET_BAND_USD", "2.0"))

# Safety: slippage protection on mint + hard cap on position size (checked before minting)
SLIPPAGE_BPS = float(os.getenv("SWAP_SLIPPAGE_BPS", "15"))       # 15 = 0.15% (same standard as withdraw_liquidity.py's swap leg)
MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD", "500"))   # hard ceiling on USD value ever deposited into a single mint

# ----------------------------
# 🔗 Connect & ENV
# ----------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("❌ Failed to connect to Ethereum mainnet RPC")
print("✅ Connected to Ethereum mainnet!")

WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise SystemExit("❌ WALLET_ADDRESS or PRIVATE_KEY missing in .env")

# ----------------------------
# 📄 ABIs & Contracts
# ----------------------------
with open("NonfungiblePositionManager.json", "r", encoding="utf-8") as f:
    pm_abi = json.load(f)
with open("pool_abi.json", "r", encoding="utf-8") as f:
    pool_abi = json.load(f)
with open("erc20_abi.json", "r", encoding="utf-8") as f:
    erc20_abi = json.load(f)

pm = w3.eth.contract(address=POSITION_MANAGER, abi=pm_abi)
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)


# ----------------------------
# Helpers
# ----------------------------
def send(tx):
    tx["from"] = WALLET_ADDRESS
    tx["gas"] = 600_000
    tx["maxFeePerGas"] = int(w3.eth.gas_price * 2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei("0.01", "gwei"))
    tx["nonce"] = w3.eth.get_transaction_count(WALLET_ADDRESS)
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    return w3.eth.wait_for_transaction_receipt(txh)


def get_eth_price():
    sqrt_price_x96 = pool.functions.slot0().call()[0]
    return float((sqrt_price_x96 / (2 ** 96)) ** 2 * 1e12)  # USDC per WETH


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def append_history(entry: dict):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def load_last_duration():
    """Return last valid cycle duration_minutes from history, if any."""
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None
    if not lines:
        return None

    for raw in reversed(lines):
        try:
            entry = json.loads(raw)
        except Exception:
            continue
        dur = entry.get("duration_minutes")
        if dur is not None:
            try:
                return float(dur)
            except Exception:
                continue
    return None


def apply_vol_slope_widen(width_ticks: int, vol_ewma, slope5_pct_per_min: float):
    """Widen band when vol or slope are elevated."""
    widened = width_ticks
    if vol_ewma is not None and vol_ewma >= VOL_WIDEN_THRESHOLD:
        widened = int(round(widened * VOL_WIDEN_MULT))
        print(f"⚠ vol_ewma={vol_ewma:.6f} ≥ {VOL_WIDEN_THRESHOLD} → widen ×{VOL_WIDEN_MULT:.2f}")

    if slope5_pct_per_min is not None and abs(slope5_pct_per_min) >= SLOPE_WIDEN_THRESHOLD:
        widened = int(round(widened * SLOPE_WIDEN_MULT))
        print(f"⚠ |slope5|={slope5_pct_per_min:.3f}%/min ≥ {SLOPE_WIDEN_THRESHOLD}% → widen ×{SLOPE_WIDEN_MULT:.2f}")

    widened = max(MIN_WIDTH_TICKS, min(MAX_WIDTH_TICKS, widened))
    return widened


def compute_adaptive_width(prev_width, panic_reentry=False):
    """
    Adaptive width based on last duration:
      - Target duration window: 12–17 minutes.
      - If last duration < 12 → widen by up to MAX_WIDTH_STEP (capped at MAX_WIDTH_TICKS).
      - If last duration > 17 → tighten by up to MAX_WIDTH_STEP (down to MIN_WIDTH_TICKS).
      - Otherwise keep width.
    Panic re-entry: override and use maximum width.
    """
    if panic_reentry:
        print(f"📏 Panic re-entry → forcing width to ±{MAX_WIDTH_TICKS} ticks.")
        return MAX_WIDTH_TICKS, None

    last_dur = load_last_duration()
    if last_dur is None:
        print(f"📏 No previous duration → keeping previous width ±{prev_width} ticks.")
        return prev_width, None

    # Too fast (< TARGET_DUR_LOW) → widen
    if last_dur < TARGET_DUR_LOW:
        step = MAX_WIDTH_STEP
        new_w = min(prev_width + step, MAX_WIDTH_TICKS)
        print(
            f"📏 Last cycle was fast (~{last_dur:.2f} min < {TARGET_DUR_LOW} min) "
            f"→ widening from ±{prev_width} to ±{new_w} ticks"
        )
        return new_w, last_dur

    # Too slow (> TARGET_DUR_HIGH) → tighten
    if last_dur > TARGET_DUR_HIGH:
        step = MAX_WIDTH_STEP
        new_w = max(prev_width - step, MIN_WIDTH_TICKS)
        print(
            f"📏 Last cycle was slow (~{last_dur:.2f} min > {TARGET_DUR_HIGH} min) "
            f"→ tightening from ±{prev_width} to ±{new_w} ticks"
        )
        return new_w, last_dur

    # Within target window → keep width
    print(
        f"📏 Last cycle ~{last_dur:.2f} min (within {TARGET_DUR_LOW}-{TARGET_DUR_HIGH} min) "
        f"→ keeping width at ±{prev_width} ticks"
    )
    return prev_width, last_dur


# ----------------------------
# MAIN
# ----------------------------
def main():
    state = load_state()
    panic_reentry = bool(state.get("panic_reentry", False))
    vol_ewma = state.get("vol_ewma")
    slope5_pct_per_min = state.get("slope5")

    slot0 = pool.functions.slot0().call()
    cur_tick = slot0[1]
    spacing = pool.functions.tickSpacing().call()
    tick_base = (cur_tick // spacing) * spacing
    price_now = get_eth_price()
    if not (price_now > 0):
        # A zero/negative price (bad oracle read, broken slot0) silently zeroes out the
        # WETH side of _usd_value() below, which would let the MAX_POSITION_USD cap be
        # bypassed entirely (the whole point of the cap is to bound USD exposure -- it
        # can't do that if the price feed it depends on is broken). Hard-stop instead.
        raise RuntimeError(f"get_eth_price() returned a non-positive price ({price_now}); refusing to mint with a broken price feed.")
    print(f"🔢 Pool tick={cur_tick}, spacing={spacing}, tick_base={tick_base}")

    print(f"📍 Position mode: {POSITION_MODE}")

    if POSITION_MODE == "full_range":
        lower_tick, upper_tick = full_range_ticks(spacing)
        width_ticks = upper_tick - lower_tick
        print(f"🌐 Full-range mint: ticks {lower_tick} -> {upper_tick} (spans the entire curve — no exit/re-open needed while price stays on-chain)")
    else:
        # Previous width (or default)
        prev_width = int(state.get("range_width", 20))
        if prev_width < MIN_WIDTH_TICKS or prev_width > MAX_WIDTH_TICKS:
            prev_width = max(MIN_WIDTH_TICKS, min(MAX_WIDTH_TICKS, prev_width))

        width_ticks, last_dur = compute_adaptive_width(prev_width, panic_reentry=panic_reentry)
        width_ticks = apply_vol_slope_widen(width_ticks, vol_ewma, slope5_pct_per_min)

        if price_now > 0 and TARGET_BAND_USD > 0:
            target_pct = TARGET_BAND_USD / price_now
            target_ticks = int(round(math.log(1.0 + target_pct) / math.log(1.0001)))
            width_ticks = max(MIN_WIDTH_TICKS, min(MAX_WIDTH_TICKS, target_ticks))
            print(f"Target USD width ${TARGET_BAND_USD:.2f} -> {width_ticks} ticks")

        # Snap width to a multiple of spacing, and also center ticks on spacing grid
        if width_ticks % spacing != 0:
            snapped = (width_ticks // spacing) * spacing
            if snapped == 0:
                snapped = spacing
            print(
                f"📐 Snapping width from ±{width_ticks} to ±{snapped} "
                f"(multiple of spacing={spacing})"
            )
            width_ticks = snapped

        # Place range ABOVE current price so position starts 100% WETH (price below band -> token0 only)
        lower_tick = tick_base + spacing * 2  # buffer above current tick
        upper_tick = lower_tick + width_ticks

    # --- Token info ---
    token0 = pool.functions.token0().call()
    token1 = pool.functions.token1().call()
    t0 = w3.eth.contract(address=token0, abi=erc20_abi)
    t1 = w3.eth.contract(address=token1, abi=erc20_abi)
    dec0 = t0.functions.decimals().call()
    dec1 = t1.functions.decimals().call()
    sym0 = t0.functions.symbol().call()
    sym1 = t1.functions.symbol().call()
    bal0 = t0.functions.balanceOf(WALLET_ADDRESS).call()
    bal1 = t1.functions.balanceOf(WALLET_ADDRESS).call()
    print(f"🔎 token0={sym0} bal0={bal0} | token1={sym1} bal1={bal1}")

    if POSITION_MODE == "full_range":
        # Full-range spans the current price, so a real position needs both tokens roughly
        # in the price's ratio -- Uniswap's mint uses whichever side is the binding constraint
        # and leaves the rest of the "longer" side untouched in the wallet.
        amt0_offer = int(bal0 * 0.99)
        amt1_offer = int(bal1 * 0.99)
    else:
        # Offer 99% of WETH balance; force WETH-only mint (amount1Desired=0)
        amt0_offer = int(bal0 * 0.99)
        amt1_offer = 0

    # --- Position-size safety cap (checked before minting) ---
    def _usd_value(amt_raw, dec, sym):
        return (amt_raw / 10**dec) if sym.upper() == "USDC" else (amt_raw / 10**dec) * price_now

    offer_value_usd = _usd_value(amt0_offer, dec0, sym0) + _usd_value(amt1_offer, dec1, sym1)
    if offer_value_usd > MAX_POSITION_USD:
        scale = MAX_POSITION_USD / offer_value_usd
        capped_amt0 = int(amt0_offer * scale)
        capped_amt1 = int(amt1_offer * scale)
        print(
            f"🛑 Position size capped: offer was ${offer_value_usd:,.2f} (> MAX_POSITION_USD=${MAX_POSITION_USD:,.2f}) "
            f"— scaling {sym0} deposit from {amt0_offer/10**dec0:.6f} to {capped_amt0/10**dec0:.6f}"
            f"{f', {sym1} deposit from {amt1_offer/10**dec1:.6f} to {capped_amt1/10**dec1:.6f}' if amt1_offer > 0 else ''}."
        )
        amt0_offer = capped_amt0
        amt1_offer = capped_amt1
        offer_value_usd = MAX_POSITION_USD

    print(f"⚙️ Mint range ({POSITION_MODE}): {lower_tick} → {upper_tick}")
    print(f"💼 Offering {amt0_offer/10**dec0:.6f} {sym0} + {amt1_offer/10**dec1:.6f} {sym1}")

    # --- Portfolio analytics before LP ---
    if sym0.upper() == "WETH":
        weth_units = bal0 / 10**dec0
        usdc_units = bal1 / 10**dec1
    else:
        weth_units = bal1 / 10**dec1
        usdc_units = bal0 / 10**dec0

    portfolio_value_usd = round(usdc_units + weth_units * price_now, 2)
    value_before_create = portfolio_value_usd

    print(
        f"🔎 Wallet snapshot → {weth_units:.6f} WETH + {usdc_units:.2f} USDC  |  ETH≈${price_now:.2f}"
    )
    print(f"📊 Starting portfolio value (WETH+USDC): ${portfolio_value_usd}")

    # --- Approvals (only if needed) ---
    for token, amt, sym, dec in [
        (token0, amt0_offer, sym0, dec0),
        (token1, amt1_offer, sym1, dec1),
    ]:
        contract = w3.eth.contract(address=token, abi=erc20_abi)
        current_allow = contract.functions.allowance(WALLET_ADDRESS, POSITION_MANAGER).call()
        if current_allow < amt:
            tx = contract.functions.approve(POSITION_MANAGER, amt).build_transaction({
                "from": WALLET_ADDRESS,
                "gas": 120_000,
                "maxFeePerGas": int(w3.eth.gas_price * 2),
                "maxPriorityFeePerGas": int(w3.to_wei("0.01", "gwei")),
                "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
            })
            receipt = send(tx)
            print(f"✅ Approved {amt/10**dec:.4f} {sym} for {token[:6]}… | ⛽ {receipt.gasUsed} gas")

    # --- Mint position ---
    params = {
        "token0": token0,
        "token1": token1,
        "fee": 500,
        "tickLower": lower_tick,
        "tickUpper": upper_tick,
        "amount0Desired": amt0_offer,
        "amount1Desired": amt1_offer,
        "amount0Min": int(amt0_offer * (1 - SLIPPAGE_BPS / 10000.0)),
        "amount1Min": int(amt1_offer * (1 - SLIPPAGE_BPS / 10000.0)),  # correctly 0 when amt1_offer=0 (tight_range mode)
        "recipient": WALLET_ADDRESS,
        "deadline": int(time.time()) + 600,
    }

    mint_tx = pm.functions.mint(params).build_transaction({"from": WALLET_ADDRESS})
    receipt = send(mint_tx)
    print(
        f"🔗 mint: {receipt.transactionHash.hex()} | ⛽ {receipt.gasUsed} gas | status={receipt.status}"
    )
    if receipt.status != 1:
        print("❌ Mint reverted on-chain, aborting.")
        return

    print("✅ LP position created!")

    # --- Extract tokenId + actual amounts used ---
    token_id = None
    used0 = used1 = None

    try:
        events = pm.events.Mint().process_receipt(receipt)
        if events:
            ev = events[0]["args"]
            token_id = int(ev["tokenId"])
            used0 = int(ev.get("amount0", 0))
            used1 = int(ev.get("amount1", 0))
    except Exception:
        pass

    if token_id is None:
        bal_nfts = pm.functions.balanceOf(WALLET_ADDRESS).call()
        if bal_nfts == 0:
            raise RuntimeError("Mint succeeded but wallet has 0 NFTs?")
        token_id = pm.functions.tokenOfOwnerByIndex(WALLET_ADDRESS, bal_nfts - 1).call()

    if (used0 is None or used1 is None) or (used0 == 0 and used1 == 0):
        try:
            inc_events = pm.events.IncreaseLiquidity().process_receipt(receipt)
            if inc_events:
                a = inc_events[0]["args"]
                used0 = int(a.get("amount0", 0))
                used1 = int(a.get("amount1", 0))
        except Exception:
            pass

    # --- Compute ACTUAL position value from used amounts ---
    if sym0.upper() == "USDC":
        used_usdc = (used0 or 0) / (10**dec0)
        used_weth = (used1 or 0) / (10**dec1)
    elif sym1.upper() == "USDC":
        used_usdc = (used1 or 0) / (10**dec1)
        used_weth = (used0 or 0) / (10**dec0)
    else:
        used_usdc = (used1 or 0) / (10**dec1)
        used_weth = (used0 or 0) / (10**dec0)

    position_value_usd = round(used_usdc + used_weth * price_now, 2)
    print(
        f"💧 Position value actually deposited: ${position_value_usd} "
        f"({used_usdc:.2f} USDC + {used_weth:.6f} WETH)"
    )

    # --- Wallet snapshot after mint ---
    bal0_after = t0.functions.balanceOf(WALLET_ADDRESS).call()
    bal1_after = t1.functions.balanceOf(WALLET_ADDRESS).call()
    if sym0.upper() == "WETH":
        weth_after = bal0_after / 10**dec0
        usdc_after = bal1_after / 10**dec1
    else:
        weth_after = bal1_after / 10**dec1
        usdc_after = bal0_after / 10**dec0

    delta_weth = round(weth_after - weth_units, 6)
    delta_usdc = round(usdc_after - usdc_units, 2)

    # --- Centered USD bounds (derived from ticks to reflect shifted bands) ---
    if panic_reentry:
        band = PANIC_BAND_FRAC
        print(f"🧯 Panic re-entry: using wide USD band ±{band*100:.2f}%")
    else:
        band = NORMAL_BAND_FRAC

    # price scales by 1.0001 per tick; anchor to current tick
    def price_at_tick(target_tick: int) -> float:
        return price_now * (1.0001 ** (target_tick - cur_tick))

    lower_usd = round(price_at_tick(lower_tick), 2)
    upper_usd = round(price_at_tick(upper_tick), 2)

    # --- Determine next cycle number ---
    cycle_number = 1
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for raw in reversed(lines):
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if "cycle_number" in obj:
                    cycle_number = int(obj.get("cycle_number", 0)) + 1
                    break
        except Exception:
            pass

    # --- Save cycle state ---
    data = {
        "cycle_number": cycle_number,
        "timestamp_create": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "range_width": int(width_ticks),
        "lower_bound_usd": lower_usd,
        "upper_bound_usd": upper_usd,
        "lower_tick": int(lower_tick),
        "upper_tick": int(upper_tick),
        "tokenId": int(token_id),
        "portfolio_value_usd": float(portfolio_value_usd),
        "value_before_create": float(value_before_create),
        "position_value_usd": float(position_value_usd),
        "price_usd": float(price_now),
        "weth_before": float(weth_units),
        "usdc_before": float(usdc_units),
        "weth_after": float(weth_after),
        "usdc_after": float(usdc_after),
        "used_weth": float(used_weth),
        "used_usdc": float(used_usdc),
        "delta_weth": float(delta_weth),
        "delta_usdc": float(delta_usdc),
        "fees_usd": float(state.get("fees_usd", 0.0)),
        "pnl_usd": float(state.get("pnl_usd", 0.0)),
        # Creation succeeded
        "lp_active": True,
        # Panic flags reset on new LP
        "panic_mode": False,
        "panic_reentry": False,
    }

    save_state(data)

    append_history({
        "type": "create_position",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "cycle_number": cycle_number,
        "token_id": int(token_id),
        "range_width": int(width_ticks),
        "price_usd": float(price_now),
        "ticks": {"lower": int(lower_tick), "upper": int(upper_tick)},
        "weth_before": float(weth_units),
        "usdc_before": float(usdc_units),
        "weth_after": float(weth_after),
        "usdc_after": float(usdc_after),
        "used_weth": float(used_weth),
        "used_usdc": float(used_usdc),
        "delta_weth": float(delta_weth),
        "delta_usdc": float(delta_usdc),
        "portfolio_value_usd": float(portfolio_value_usd),
        "position_value_usd": float(position_value_usd),
    })

    # Discord notification
    try:
        msg = (
            f"🟢 LP created #{token_id} | price=${price_now:.2f}\n"
            f"Ticks: {lower_tick} → {upper_tick}\n"
            f"WETH used={used_weth:.6f}, USDC used={used_usdc:.2f}\n"
            f"WETH before/after={weth_units:.6f}/{weth_after:.6f}, ΔWETH={delta_weth:.6f}\n"
            f"Position value=${position_value_usd:.2f} (portfolio=${portfolio_value_usd:.2f})"
        )
        send_discord_message(msg)
    except Exception:
        pass

    print(f"🧾 Saved centered USD range → ${lower_usd} → ${upper_usd}")
    print(f"📏 Final width used: ±{width_ticks} ticks")
    print(f"🌀 Cycle #{cycle_number} initiated.")
    print("✅ create_position.py completed.\n")


if __name__ == "__main__":
    main()
