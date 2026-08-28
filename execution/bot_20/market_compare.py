import json
import math
import os
from collections import defaultdict
from datetime import datetime

from web3 import Web3
from dotenv import load_dotenv

# ---------------------------
# CONFIG
# ---------------------------
SCRIPT_DIR = os.path.dirname(__file__)
env_path = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = os.getenv("ARB_RPC_URL", "https://ethereum.publicnode.com")
POOL_ADDRESS = Web3.to_checksum_address("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")

HISTORY_FILE = os.path.join(SCRIPT_DIR, "cycle_history.jsonl")

# How many recent cycles to focus on when comparing vs market
RECENT_CYCLES = 20

# Half-life for recency weighting (in cycles)
HALF_LIFE_CYCLES = 10.0
LAMBDA = math.log(2) / HALF_LIFE_CYCLES


# ---------------------------
# Helpers
# ---------------------------

def f_float(d, key, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def f_int(d, key, default=0):
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------
# Load cycle history
# ---------------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ {HISTORY_FILE} not found.")
        return []

    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def split_cycles_rebalances(records):
    cycles = [r for r in records if "cycle_number" in r]
    rebalances = [r for r in records if "timestamp_rebalance" in r]
    return cycles, rebalances


# ---------------------------
# Bot performance analysis
# ---------------------------

def analyse_recent_cycles(cycles, rebalances, recent_n=20):
    """
    Look at the last `recent_n` cycles and their associated rebalances.
    Return a dict with stats we can compare against market.
    """
    if not cycles:
        return None

    recent_cycles = cycles[-recent_n:]
    N = len(recent_cycles)

    # Map cycle_number -> rebalances that occur after its withdraw (approx)
    # (This is heuristic; we just slice the last few rebalances.)
    recent_rebalances = rebalances[-(2 * recent_n):] if rebalances else []

    stats = {
        "n_cycles": N,
        "recent_cycles": recent_cycles,
        "recent_rebalances": recent_rebalances,
        "cycle_pnls": [],
        "cycle_durs": [],
        "cycle_times": [],
        "raw_total_pnl": 0.0,
        "raw_avg_pnl": 0.0,
        "weighted_avg_pnl": 0.0,
        "short_avg": None,
        "long_avg": None,
        "panic_avg": None,
        "rebalance_avg": None,
    }

    # --- cycles: raw + weighted pnls ---
    total_raw = 0.0
    total_w = 0.0
    total_w_pnl = 0.0
    durations = []
    times = []

    for idx, c in enumerate(recent_cycles):
        pnl = f_float(c, "net_pnl_usd", 0.0)
        dur = f_float(c, "duration_minutes", 0.0)
        ts_str = c.get("timestamp_withdraw") or c.get("timestamp_create")

        total_raw += pnl
        durations.append(dur)
        stats["cycle_pnls"].append(pnl)
        stats["cycle_durs"].append(dur)
        times.append(ts_str)

        age = (N - 1) - idx  # newest cycle has age=0
        w = math.exp(-LAMBDA * age)
        total_w += w
        total_w_pnl += w * pnl

    stats["cycle_times"] = times
    stats["raw_total_pnl"] = total_raw
    stats["raw_avg_pnl"] = total_raw / N if N > 0 else 0.0
    stats["weighted_avg_pnl"] = total_w_pnl / total_w if total_w > 0 else 0.0

    # short vs long
    short_pnls = []
    long_pnls = []
    for c in recent_cycles:
        pnl = f_float(c, "net_pnl_usd", 0.0)
        dur = f_float(c, "duration_minutes", 0.0)
        if dur < 10.0:
            short_pnls.append(pnl)
        if dur >= 30.0:
            long_pnls.append(pnl)

    if short_pnls:
        stats["short_avg"] = sum(short_pnls) / len(short_pnls)
    if long_pnls:
        stats["long_avg"] = sum(long_pnls) / len(long_pnls)

    # --- rebalances ---
    by_type = defaultdict(list)
    for r in recent_rebalances:
        t = r.get("type", "unknown")
        pn = f_float(r, "net_usd", 0.0)
        by_type[t].append(pn)

    if by_type.get("panic_100_usdc"):
        pvals = by_type["panic_100_usdc"]
        stats["panic_avg"] = sum(pvals) / len(pvals)

    if by_type.get("rebalance_50_50"):
        rvals = by_type["rebalance_50_50"]
        stats["rebalance_avg"] = sum(rvals) / len(rvals)

    return stats


# ---------------------------
# Market analysis (Uniswap pool)
# ---------------------------

def load_pool():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise RuntimeError("Failed to connect to Ethereum mainnet RPC")

    with open(os.path.join(SCRIPT_DIR, "pool_abi.json"), "r", encoding="utf-8") as f:
        pool_abi = json.load(f)

    pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
    return w3, pool


def tick_to_price(tick):
    """
    Approx price from tick: 1.0001^tick * 1e12 (USDC per WETH).
    """
    return (1.0001 ** tick) * 1e12


def get_current_price_and_tick(pool):
    slot0 = pool.functions.slot0().call()
    sqrt_price_x96, tick = slot0[0], slot0[1]
    # Use direct sqrtPriceX96 formula (same as your other scripts)
    price = float((sqrt_price_x96 / (2**96)) ** 2 * 1e12)
    return price, tick


def get_market_regime(pool):
    """
    Use Uniswap V3 observe() to probe average ticks over different time windows.
    This won't reconstruct full OHLC, but it gives us trend hints.
    """
    # secondsAgo windows: 5m, 15m, 1h, 4h
    horizons = [300, 900, 3600, 14400]
    seconds_agos = horizons + [0]  # last element must be 0 per Uniswap docs

    try:
        (tick_cumulatives, _) = pool.functions.observe(seconds_agos).call()
    except Exception as e:
        print(f"⚠️ Could not call observe() on pool: {e}")
        return None

    # current price/tick from slot0
    current_price, current_tick = get_current_price_and_tick(pool)

    # For each horizon H, avg tick over last H seconds:
    # avg_tick(H) = (tickCum(now) - tickCum(now-H)) / H
    regime = {
        "current_price": current_price,
        "current_tick": current_tick,
        "horizons": [],
    }

    for i, H in enumerate(horizons):
        tick_now = tick_cumulatives[-1]
        tick_past = tick_cumulatives[i]
        dt = H
        avg_tick = (tick_now - tick_past) / dt
        avg_price = tick_to_price(avg_tick)

        # relative to current price
        rel_move = (current_price - avg_price) / avg_price if avg_price > 0 else 0.0

        regime["horizons"].append(
            {
                "seconds": H,
                "minutes": H / 60,
                "avg_tick": avg_tick,
                "avg_price": avg_price,
                "rel_move": rel_move,
            }
        )

    return regime


def classify_regime(regime):
    """
    Turn the raw regime info into a human label: flat / choppy / trending up/down.
    """
    if regime is None:
        return "unknown", {}

    # Look primarily at 1h window (3600s) and 15m window (900s)
    h1 = next((h for h in regime["horizons"] if h["seconds"] == 3600), None)
    h15 = next((h for h in regime["horizons"] if h["seconds"] == 900), None)

    if not h1 or not h15:
        return "unknown", {}

    move_1h = h1["rel_move"]  # current vs 1h avg
    move_15 = h15["rel_move"]

    # thresholds (tuneable)
    TREND_STRONG = 0.03   # 3%
    TREND_WEAK = 0.01     # 1%

    info = {
        "move_1h": move_1h,
        "move_15m": move_15,
        "current_price": regime["current_price"],
    }

    # Decide trend
    if abs(move_1h) < TREND_WEAK and abs(move_15) < TREND_WEAK:
        label = "flat/mean-reverting"
    elif move_1h > TREND_STRONG:
        label = "strong uptrend"
    elif move_1h < -TREND_STRONG:
        label = "strong downtrend"
    elif move_15 > TREND_STRONG:
        label = "short-term pump"
    elif move_15 < -TREND_STRONG:
        label = "short-term dump"
    else:
        label = "choppy/sideways with noise"

    return label, info


# ---------------------------
# Main compare logic
# ---------------------------

def coach_comment(perf, regime_label, regime_info):
    """
    Print detailed, regime-aware advice.
    """
    print("🧠 Market-aware Coach — Performance vs Market")
    print("--------------------------------------------")

    if perf is None:
        print("• No cycles found in history yet.")
        return

    n_cycles = perf["n_cycles"]
    raw_avg = perf["raw_avg_pnl"]
    w_avg = perf["weighted_avg_pnl"]
    short_avg = perf["short_avg"]
    long_avg = perf["long_avg"]
    panic_avg = perf["panic_avg"]
    rebalance_avg = perf["rebalance_avg"]

    print(f"• Analysing your last {n_cycles} cycles (recent behaviour).")
    print(f"  Raw avg PnL/cycle:       {raw_avg:+.2f} USDC")
    print(f"  Recency-weighted PnL:    {w_avg:+.2f} USDC\n")

    print(f"• Current market regime: {regime_label}")
    if regime_label != "unknown":
        print(
            f"  (Price now ≈ {regime_info['current_price']:.2f} USDC/WETH, "
            f"1h rel move {regime_info['move_1h']*100:.2f}%, "
            f"15m rel move {regime_info['move_15m']*100:.2f}%)\n"
        )
    else:
        print("  (Could not classify market regime from pool oracle.)\n")

    # --- Regime-specific reasoning ---

    if regime_label in ("strong uptrend", "strong downtrend"):
        # Trending environment
        if w_avg < 0:
            print(
                "🔎 The market is trending strongly, but your recent cycles are losing.\n"
                "   → Interpretation: your ranges are likely too tight or your exits too sensitive,\n"
                "     so you get knocked out as the trend continues instead of riding it.\n"
            )
            print("   Suggested adjustments for a trending regime:")
            print("   • Widen your base tick range (e.g. +40–60 ticks wider than current).")
            print("   • Increase MIN_CYCLE_MIN so you don't exit after tiny counter-moves.")
            print("   • Make panic harder to trigger (larger USD band + tick buffer).")
        else:
            print(
                "✅ Market is trending and your recent cycles are positive.\n"
                "   → Your current range seems to cope reasonably with trend.\n"
            )
            print("   Still, to reduce risk of whipsaw in strong trends:")
            print("   • Avoid very tight ranges right after large moves.")
            print("   • Keep panics rare and reserved for true cascades.\n")

    elif regime_label in ("short-term pump", "short-term dump", "choppy/sideways with noise"):
        # Noisy / short-term bursts regime
        if short_avg is not None and short_avg > 0 and (long_avg is None or long_avg <= 0):
            print(
                "📈 Recent short cycles (<10 min) are doing well in a noisy regime.\n"
                "   → Fast rotations during local bursts are profitable for you.\n"
            )
            print("   To lean into this:")
            print("   • Allow slightly tighter ranges immediately after confirming chop.")
            print("   • Keep panics VERY strict so you don't overreact to mini-wicks.\n")
        elif w_avg < 0:
            print(
                "📉 Market looks choppy/noisy, but your net recent PnL is negative.\n"
                "   → Likely over-reacting to noise: too many exits and rebalances.\n"
            )
            print("   Suggested tweaks for a choppy regime:")
            print("   • Increase USD_TOLERANCE before exiting (let price breathe).")
            print("   • Widen range modestly so small swings don't kick you out.")
            print("   • Ensure MIN_CYCLE_MIN is high enough to avoid micro-cycles.\n")
        else:
            print(
                "➖ Market is choppy/short-bursty and your PnL is roughly flat or slightly positive.\n"
                "   → You're close. Small reductions in churn (fewer rebalances/panics)\n"
                "     could turn this clearly positive.\n"
            )

    elif regime_label == "flat/mean-reverting":
        # Flat mean-reverting regime
        if w_avg < 0:
            print(
                "📉 Market looks relatively flat/mean-reverting, but you're still losing.\n"
                "   → This usually means gas + rebalance + panic costs are eating the fees.\n"
            )
            print("   For flatter regimes:")
            print("   • You can safely narrow the range a bit BUT reduce panic to almost zero.")
            print("   • Consider raising drift threshold even more to rebalance less often.")
            print("   • Make sure LP stays in range long enough to earn fees (min duration).\n")
        else:
            print(
                "✅ Flat/mean-reverting market and you're at least not bleeding badly.\n"
                "   → This is where tight ranges and low panic sensitivity can shine.\n"
            )

    # --- Rebalance & panic behaviour ---

    if panic_avg is not None:
        print(
            f"• Recent panic events average {panic_avg:+.2f} USDC.\n"
            "  → If this is significantly negative (e.g. < -1), your panic logic is too aggressive.\n"
            "    Consider:\n"
            "    - Requiring BOTH a deeper tick break AND a larger USD move.\n"
            "    - Longer cooldowns or str str stricter EMA confirmation before re-entry.\n"
        )

    if rebalance_avg is not None:
        print(
            f"• Recent normal 50/50 rebalances average {rebalance_avg:+.2f} USDC.\n"
            "  → If this is negative, treat rebalancing as a cost, not a benefit. To reduce it:\n"
            "    - Increase the drift threshold (e.g. 3–5% instead of 1%).\n"
            "    - Rebalance only as part of a full exit/remint cycle (no free-floating rebalances).\n"
        )

    # --- Final summary line ---

    if w_avg < 0:
        print(
            f"\n⚠️ Bottom line: in the current {regime_label} regime, your recent "
            f"recency-weighted PnL per cycle is {w_avg:+.2f} USDC.\n"
            "   Focus on: wider ranges, fewer panics, and stricter rules to avoid churning "
            "on small moves."
        )
    else:
        print(
            f"\n✅ Bottom line: in the current {regime_label} regime, your recent "
            f"recency-weighted PnL per cycle is {w_avg:+.2f} USDC.\n"
            "   Focus on nudging down costs (rebalances, panics) without breaking what's working."
        )

    print()


def main():
    # 1) Load history
    records = load_history()
    if not records:
        print("No data in cycle_history.jsonl.")
        return

    cycles, rebalances = split_cycles_rebalances(records)
    perf = analyse_recent_cycles(cycles, rebalances, RECENT_CYCLES)

    # 2) Load market / pool state
    try:
        w3, pool = load_pool()
    except Exception as e:
        print(f"⚠️ Could not connect to Ethereum mainnet / pool: {e}")
        regime = None
        regime_label = "unknown"
        regime_info = {}
    else:
        regime = get_market_regime(pool)
        regime_label, regime_info = classify_regime(regime)

    # 3) Print comparison
    coach_comment(perf, regime_label, regime_info)


if __name__ == "__main__":
    main()
