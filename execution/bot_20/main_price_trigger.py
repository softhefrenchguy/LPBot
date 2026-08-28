import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import json, os, time, subprocess
from web3 import Web3

RPC_URL = "https://ethereum.publicnode.com"
POOL_ADDRESS = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")
TRIGGER_GAP   = 5.0    # $ buffer beyond your bounds
POLL_SECONDS  = 3      # monitor cadence

# ---------- web3 / pool ----------
pool_abi = json.load(open("pool_abi.json"))
w3   = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("âŒ Could not connect to Ethereum mainnet RPC")
pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
print("âœ… Connected to Ethereum mainnet â€” Ethereum mainnet One")

def get_price():
    # Correct USDC per WETH from sqrtPriceX96 (6 vs 18 decimals -> 1e12)
    sqrtPriceX96 = pool.functions.slot0().call()[0]
    return float((sqrtPriceX96 / (2 ** 96)) ** 2 * 1e12)

def run(pyfile):
    print(f"âš™ï¸ Running {pyfile} ...")
    subprocess.run(["python", pyfile], check=True)
    print(f"âœ… {pyfile} completed.\n")

def load_state():
    if os.path.exists("last_cycle_data.json"):
        with open("last_cycle_data.json") as f:
            return json.load(f)
    return {}

def short_range(state):
    lo = state.get("lower_bound_usd")
    hi = state.get("upper_bound_usd")
    side = state.get("side")
    if lo is None or hi is None or side not in ("ETH", "USDC"):
        raise RuntimeError("last_cycle_data.json missing side/lower/upper")
    return side, float(lo), float(hi)

def log_in_range(price, lower, upper):
    # Distances to boundaries (positive = inside the band)
    d_lower = price - lower     # >0 means above the lower bound
    d_upper = upper - price     # >0 means below the upper bound
    print(f"âœ… In range ${lower:.2f} â†’ ${upper:.2f} | dLower {d_lower:+.2f} | dUpper {d_upper:+.2f}")

def main():
    print("ðŸš€ LP Automation (Bidirectional Auto-Flip v2)")
    print("ðŸ’¡ Automatically switches between ETH and USDC sides.\n")

    state = load_state()
    if not state:
        print("âš ï¸ No saved state â€” creating initial ETH-side position...")
        run("create_eth_side_position.py")
        state = load_state()

    side, lower, upper = short_range(state)
    print(f"âœ… Loaded last position â†’ Range ${lower:.2f} â†’ ${upper:.2f}\n")

    while True:
        price = get_price()
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] ðŸ’° Price = ${price:.2f}")

        if side == "ETH":
            # If price goes significantly BELOW the lower bound, stay ETH and re-place the band lower.
            if price < lower - TRIGGER_GAP:
                diff = (lower - TRIGGER_GAP) - price
                print(f"âš ï¸ Price {price:.2f} < lower-{TRIGGER_GAP} by ${diff:.2f} â†’ refreshing ETH-side lower...")
                run("withdraw_latest_only.py")
                run("create_eth_side_position.py")
                state = load_state()
                side, lower, upper = short_range(state)
                print(f"âœ… New ETH-side range: ${lower:.2f} â†’ ${upper:.2f}\n")
                continue

            # If price goes significantly ABOVE the upper bound, flip to USDC.
            if price > upper + TRIGGER_GAP:
                diff = price - (upper + TRIGGER_GAP)
                print(f"âš ï¸ Price {price:.2f} > upper+{TRIGGER_GAP} by ${diff:.2f} â†’ flipping to USDC-side...")
                run("withdraw_latest_only.py")
                run("create_usdc_side_position.py")
                state = load_state()
                side, lower, upper = short_range(state)
                print(f"âœ… New USDC-side range: ${lower:.2f} â†’ ${upper:.2f}\n")
                continue

        else:  # USDC side
            # If price goes significantly ABOVE the upper bound (we sit below), stay USDC and re-place higher.
            if price > upper + TRIGGER_GAP:
                diff = price - (upper + TRIGGER_GAP)
                print(f"âš ï¸ Price {price:.2f} > upper+{TRIGGER_GAP} by ${diff:.2f} â†’ refreshing USDC-side higher...")
                run("withdraw_latest_only.py")
                run("create_usdc_side_position.py")
                state = load_state()
                side, lower, upper = short_range(state)
                print(f"âœ… New USDC-side range: ${lower:.2f} â†’ ${upper:.2f}\n")
                continue

            # If price goes significantly BELOW the lower bound, flip back to ETH.
            if price < lower - TRIGGER_GAP:
                diff = (lower - TRIGGER_GAP) - price
                print(f"âš ï¸ Price {price:.2f} < lower-{TRIGGER_GAP} by ${diff:.2f} â†’ flipping to ETH-side...")
                run("withdraw_latest_only.py")
                run("create_eth_side_position.py")
                state = load_state()
                side, lower, upper = short_range(state)
                print(f"âœ… New ETH-side range: ${lower:.2f} â†’ ${upper:.2f}\n")
                continue

        # Still inside the band (+/- gap not breached)
        log_in_range(price, lower, upper)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nðŸ›‘ Stopped by user.")







