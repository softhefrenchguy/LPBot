import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os, json, time, statistics
from datetime import datetime
from web3 import Web3

RPC_URL = "https://ethereum.publicnode.com"

RANGE_TIGHT = 25
RANGE_BASE  = 50
RANGE_WIDE  = 80
LOOKBACK_CYCLES = 5
SHORT_LIMIT_MIN = 10
LONG_LIMIT_MIN  = 60

GAS_THRESHOLD_GWEI  = 0.05
PAUSE_MINUTES_WHEN_EXPENSIVE = 10
LOW_ETH_BALANCE     = 0.002

HISTORY_FILE = "cycle_history.jsonl"
RANGE_FILE   = "next_range_width.txt"

def _w3(): return Web3(Web3.HTTPProvider(RPC_URL))
def _now(): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def current_gas_gwei():
    try: return _w3().eth.get_block("latest")["baseFeePerGas"] / 1e9
    except Exception: return None

def eth_balance_eth(address):
    try: return _w3().eth.get_balance(address) / 1e18
    except Exception: return 0.0

def recent_durations():
    out = []
    if not os.path.exists(HISTORY_FILE): return out
    with open(HISTORY_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines[-LOOKBACK_CYCLES:]:
        try:
            e = json.loads(ln)
            d = e.get("cycle_duration_min")
            if isinstance(d, (int, float)): out.append(float(d))
        except Exception: pass
    return out

def decide_range_from_volatility(durations):
    if not durations: return RANGE_BASE, "no history â†’ base"
    short = sum(1 for d in durations if d < SHORT_LIMIT_MIN)
    avg = statistics.mean(durations)
    if short >= 3: return RANGE_WIDE, f"volatile ({short}/{len(durations)} short)"
    if avg > LONG_LIMIT_MIN: return RANGE_TIGHT, f"stable (avg {avg:.1f} min)"
    return RANGE_BASE, f"normal (avg {avg:.1f} min)"

def write_next_range(width):
    try:
        with open(RANGE_FILE, "w") as f: f.write(str(int(width)))
        return True
    except Exception: return False

def gas_and_range_controller(wallet, weth_price_now=None, weth_price_prev=None):
    ts = _now()
    gas = current_gas_gwei()
    eth_bal = eth_balance_eth(wallet)
    if gas is None: gas = 0.03
    print(f"[{ts}] â›½ Gas: {gas:.3f} gwei | ETH balance: {eth_bal:.5f}")
    durations = recent_durations()
    width, reason = decide_range_from_volatility(durations)
    if eth_bal < LOW_ETH_BALANCE:
        width = RANGE_WIDE
        reason += "; low ETH â†’ widen"
    if gas > GAS_THRESHOLD_GWEI:
        width = RANGE_WIDE
        reason += f"; gas high ({gas:.3f}) â†’ widen & pause"
    write_next_range(width)
    print(f"ðŸ“ Next range width â†’ Â±{width} ticks ({reason})")
    if gas > GAS_THRESHOLD_GWEI:
        print(f"â¸ï¸ Pausing {PAUSE_MINUTES_WHEN_EXPENSIVE} min to save gas...")
        time.sleep(PAUSE_MINUTES_WHEN_EXPENSIVE * 60)
    return gas

