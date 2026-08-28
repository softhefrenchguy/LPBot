#!/usr/bin/env python3
"""
main_auto_v3.py - ETH-only LP controller (Balanced B1 + Oscillation + Forced Top Exit)

Modes:
  - "LP"       - active Uniswap position, farming fees
  - "FLAT_ETH" - 100% ETH, no LP

While LP is active:
  - Allow normal oscillations in and out of the band.
  - EXIT LP ONLY on:
      1) A *forced top* move:
           - price > upper band
           - distance >= 0.25% above band
           - exit immediately (we don't want to ride too far above our band)
      2) A *real breakout* (Balanced B1):
           - price outside band
           - stays there long enough (persistence)
           - AND distance from band big enough (0.60%)
           - AND trend/volatility confirm (EMA gap, slope, vol_ewma)

While FLAT_ETH:
  - Detect *consolidation* in TWO WAYS:
      1) EMA-based calm: small gap/slope/vol for some time.
      2) Oscillation-based: price crossing a center level multiple times
         in a compact window (you don't care if it leaves the range as
         long as it keeps coming back).
  - When EMAs are warm and (EMA-calm OR oscillation-pattern) is true,
    call create_position.py to build a new ETH-only LP band.

No USDC hedge mode in this version: we always exit back to ETH-only
(handled via withdraw_liquidity.py which keeps everything in your wallet).
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime, timezone
from collections import deque

from dotenv import load_dotenv
from web3 import Web3

# Ensure stdout can emit UTF-8 symbols on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -------------------------------------------------
# ENV / CONFIG
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

RPC_URL = "https://ethereum.publicnode.com"
POOL_ADDRESS = Web3.to_checksum_address(
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # WETH/USDC 0.05% Ethereum mainnet
)

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
if not WALLET_ADDRESS:
    raise SystemExit("? WALLET_ADDRESS missing in .env")
WALLET_ADDRESS = Web3.to_checksum_address(WALLET_ADDRESS)

STATE_FILE = os.path.join(script_dir, "bot_state.json")
LAST_CYCLE_FILE = os.path.join(script_dir, "last_cycle_data.json")

CHECK_INTERVAL_SEC = 3.0

# ===== EMA config =====
EMA_FAST_WINDOW = 5        # EMA5 (minutes-equivalent)
EMA_SLOW_WINDOW = 30       # EMA30
EMA_SAMPLE_SECONDS = 3.0   # approx sampling interval
SLOPE5M_WINDOW_SEC = 5 * 60

# Volatility EWMA on absolute returns
VOL_EWMA_ALPHA = 0.1

# ===== Breakout detection (exit LP ? FLAT_ETH) =====
# Balanced B1: fairly conservative breakout detection
OUT_OF_RANGE_PCT_MIN = 0.20 / 100.0   # 0.20% beyond band edge (normal breakout)
OUT_OF_RANGE_TICKS_MIN = 15           # ~45s if 3s / tick

# Trend/vol thresholds for confirming breakout
GAP_BREAKOUT_PCT = 0.12               # |EMA5 - EMA30| / EMA30 >= 0.12%
SLOPE_BREAKOUT_PCT_PER_MIN = 0.10     # |slope5| >= 0.10%/min
VOL_BREAKOUT_MIN = 0.00025            # vol_ewma >= 0.025% abs returns

# ===== Forced top exit =====
# Allow small buffer above the upper band to reduce churn.
TOP_FORCED_EXIT_PCT = 0.0             # forced top exit disabled

# ===== EMA-based consolidation detection (re-enter LP) =====
CONSOLIDATION_GAP_MAX = 0.15          # |gap| <= 0.15%
CONSOLIDATION_SLOPE_MAX = 0.08        # |slope5| <= 0.08%/min
CONSOLIDATION_VOL_MAX = 0.00012       # vol_ewma <= 0.012% abs returns
EMA_CONSOLIDATION_TICKS_MIN = 60      # ~3 minutes if 3s / tick

EMA_WARMUP_MINUTES = 10.0             # wait until EMAs "mature"

# ===== Oscillation-based consolidation (re-enter LP) =====
# We treat price oscillating around a local "center" as consolidation zone.
# We treat price oscillating around a local "center" as consolidation zone.
OSC_WINDOW_PCT = 0.003                # +/-0.3% around osc_center
OSC_MIN_OSCILLATIONS = 5              # minimum crossings for pattern
OSC_PATTERN_TICKS_MIN = 50            # minimum ticks since pattern started

# Logging control
# Throttle noisy per-tick logs (100 ticks ~ 5 min at 3s/tick)
LOG_EVERY_N_TICKS = 100
LOG_MIN_SECONDS = 3.0  # do not log more frequently than this even if thresholds hit
REENTRY_MAX_ABS_SLOPE = 0.25          # block re-entry if |slope5| above this (%/min)
REENTRY_MAX_VOL = CONSOLIDATION_VOL_MAX * 1.2  # block re-entry if vol too high
# Regime policy (goal: hold ETH in uptrends, LP in downtrends)
# Exit LP to go 100% WETH when an uptrend is detected (prevents the LP from converting WETH -> USDC).
UPTREND_HODL_EXIT_SLOPE = 0.02        # %/min; if 5m slope > this (and persists), exit LP and hold ETH
UPTREND_HODL_PERSIST_TICKS = 40       # require slope > threshold for this many ticks (~2 min @3s) before exit
MIN_HOLD_UPTREND_EXIT_SECONDS = 120   # do not fire uptrend exit until position held at least this long
# Only (re-)enter LP when not trending up.
DOWNTREND_REENTRY_SLOPE_MAX = 0.0     # %/min; require slope5 <= this to attempt LP entry


# -------------------------------------------------
# Web3 / ABIs
# -------------------------------------------------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise SystemExit("? Failed to connect to Ethereum mainnet RPC")

print("? Connected to Ethereum mainnet RPC")
print("?? LP Automation main_auto_v3 (ETH-only, Balanced B1 + Oscillation + Forced Top Exit)")

with open(os.path.join(script_dir, "pool_abi.json"), "r", encoding="utf-8") as f:
    pool_abi = json.load(f)

pool = w3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)
with open(os.path.join(script_dir, "NonfungiblePositionManager.json"), "r", encoding="utf-8") as f:
    pm_abi = json.load(f)
pm = w3.eth.contract(address=Web3.to_checksum_address("0xC36442b4a4522E871399CD717aBDD847Ab11FE88"), abi=pm_abi)


# -------------------------------------------------
# Helpers: State & Files
# -------------------------------------------------
def _load_json_file(path: str, default: dict | None = None):
    """Load JSON with a forgiving fallback if the file is empty or corrupted."""
    default = default or {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = raw.replace("\x00", "").strip()
        if not cleaned:
            return default
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"?? {path} contains invalid JSON ({e}); resetting to defaults.")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
        except Exception as write_err:
            print(f"?? Could not reset {path}: {write_err}")
        return default
    except Exception as e:
        print(f"?? Could not read {path}: {e}")
        return default


def load_state():
    return _load_json_file(STATE_FILE, {})


def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"?? Could not write {STATE_FILE}: {e}")


def load_last_cycle():
    return _load_json_file(LAST_CYCLE_FILE, {})


def save_last_cycle(data: dict):
    try:
        with open(LAST_CYCLE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"?? Could not write {LAST_CYCLE_FILE}: {e}")


def parse_timestamp_utc(ts_str: str):
    """Parse 'YYYY-MM-DD HH:MM:SS UTC' into epoch seconds; return None on failure."""
    if not ts_str:
        return None
    try:
        ts_clean = ts_str.replace("UTC", "").strip()
        dt = datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


# -------------------------------------------------
# Helpers: Math / EMA / Price
# -------------------------------------------------
def get_eth_price():
    slot0 = pool.functions.slot0().call()
    sqrt_price_x96 = slot0[0]
    # price token1/token0 (USDC per WETH), scale factor 1e12 (USDC 6 decimals, WETH 18)
    return float((sqrt_price_x96 / (2 ** 96)) ** 2 * 1e12)


def ema_update(prev, value, alpha):
    if prev is None:
        return value
    return alpha * value + (1.0 - alpha) * prev


def compute_alpha(window_minutes: float, sample_seconds: float):
    if window_minutes <= 0:
        return 1.0
    n = (window_minutes * 60.0) / sample_seconds
    if n <= 1:
        return 1.0
    return 2.0 / (n + 1.0)


ALPHA_FAST = compute_alpha(EMA_FAST_WINDOW, EMA_SAMPLE_SECONDS)
ALPHA_SLOW = compute_alpha(EMA_SLOW_WINDOW, EMA_SAMPLE_SECONDS)


# -------------------------------------------------
# Helpers: Actions (hook to other scripts)
# -------------------------------------------------
def run_cmd(cmd: str, backoff_sec: float = 5.0):
    print(f"[run] {cmd}")
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode != 0:
            print(f"[run] failed code={res.returncode}: {cmd}")
            if backoff_sec > 0:
                time.sleep(backoff_sec)
        return res.returncode
    except Exception as e:
        print(f"[run] exception while running {cmd}: {e}")
        if backoff_sec > 0:
            time.sleep(backoff_sec)
        return 1


def trigger_exit_to_flat_eth(
    exit_reason: str,
    breakout_side: str,
    price: float,
    band_lower: float,
    band_upper: float,
    gap_pct: float,
    slope5_pct_per_min: float,
    vol_ewma
):
    """
    Exit LP because of confirmed breakout or forced top.
    -> withdraw_liquidity.py (which burns NFT and leaves you ETH+USDC in wallet).
    NOTE: we do NOT clear tokenId here; withdraw_liquidity.py needs it.
    """
    last = load_last_cycle()
    last["exit_reason"] = exit_reason
    last["exit_breakout_side"] = breakout_side  # "UP" or "DOWN"
    last["exit_price"] = float(price)
    last["exit_band_lower"] = float(band_lower) if band_lower is not None else None
    last["exit_band_upper"] = float(band_upper) if band_upper is not None else None
    last["exit_gap_pct"] = float(gap_pct)
    last["exit_slope5_pct_per_min"] = float(slope5_pct_per_min)
    last["exit_vol_ewma"] = float(vol_ewma) if vol_ewma is not None else None
    save_last_cycle(last)

    # Exit LP (full withdraw + fees back to wallet)
    run_cmd(f'python "{os.path.join(script_dir, "withdraw_liquidity.py")}"')


def trigger_enter_lp_from_flat_eth(vol_ewma, slope5_pct_per_min):
    """
    From FLAT_ETH, after consolidation detection: re-enter LP.
    Store vol_ewma + slope5 into last_cycle so create_position.py
    can choose width / lower bound using AI/live_policy.
    """
    last = load_last_cycle()
    last["vol_ewma"] = float(vol_ewma) if vol_ewma is not None else None
    last["slope5"] = float(slope5_pct_per_min)
    # ensure clean state before creation; create_position.py will set lp_active=True on success
    last["lp_active"] = False
    # Reset oscillation/consolidation counters after an exit
    last["ema_consolidation_ticks"] = 0
    last["osc_center"] = None
    last["oscillations"] = 0
    last["osc_pattern_ticks"] = 0
    last["last_osc_side"] = 0
    save_last_cycle(last)

    run_cmd(f'python "{os.path.join(script_dir, "create_position.py")}"')

    # After create_position, check whether LP was actually created
    last_after = load_last_cycle()
    lp_active = bool(last_after.get("lp_active", False))
    return lp_active, last_after


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
def main():
    state = load_state()
    tick_count = state.get("tick_count", 0)

    ema_fast = state.get("ema_fast")
    ema_slow = state.get("ema_slow")
    vol_ewma = state.get("vol_ewma")
    last_price = state.get("last_price")
    ema_age_min = state.get("ema_age_min", 0.0)
    slope_up_ticks = state.get("slope_up_ticks", 0)

    # Out-of-range persistence counters (for breakout)
    oor_up_count = state.get("oor_up_count", 0)
    oor_down_count = state.get("oor_down_count", 0)

    # EMA-based consolidation ticks
    ema_consolidation_ticks = state.get("ema_consolidation_ticks", 0)

    # Oscillation tracking
    osc_center = state.get("osc_center")          # float or None
    oscillations = state.get("oscillations", 0)   # number of crossings
    osc_pattern_ticks = state.get("osc_pattern_ticks", 0)
    last_osc_side = state.get("last_osc_side")    # -1, 0, +1, or None

    # Mode: "LP" or "FLAT_ETH"
    mode = state.get("mode")
    last_log_ts = state.get("last_log_ts", 0.0)

    # -------- Initial sync with last_cycle to avoid ghost states --------
    last_cycle_initial = load_last_cycle()
    lp_active_initial = bool(last_cycle_initial.get("lp_active", False))
    band_lower_init = last_cycle_initial.get("lower_bound_usd")
    band_upper_init = last_cycle_initial.get("upper_bound_usd")
    have_band_init = (
        isinstance(band_lower_init, (int, float)) and
        isinstance(band_upper_init, (int, float)) and
        band_lower_init < band_upper_init
    )

    # If mode not set yet, infer from lp_active
    if mode not in ("LP", "FLAT_ETH"):
        mode = "LP" if lp_active_initial and have_band_init else "FLAT_ETH"

    # If we think we're in LP but JSON says no LP ? correct to FLAT_ETH
    if mode == "LP" and not (lp_active_initial and have_band_init):
        print("?? Startup sync: mode=LP but no active LP in last_cycle ? switching to FLAT_ETH.")
        mode = "FLAT_ETH"

    # If we think we're FLAT_ETH but JSON says LP is active ? correct to LP
    if mode == "FLAT_ETH" and lp_active_initial and have_band_init:
        print("?? Startup sync: mode=FLAT_ETH but lp_active=True & band present ? switching to LP.")
        mode = "LP"

    last_ts = time.time()
    price_window = deque()

    print(f"?? Starting mode: {mode}")
    print(f"?? Checks every {CHECK_INTERVAL_SEC:.1f}s\n")

    # --------------------------
    # Main loop
    # --------------------------
    while True:
        loop_ts = time.time()
        dt = loop_ts - last_ts
        if dt <= 0:
            dt = CHECK_INTERVAL_SEC
        last_ts = loop_ts
        mode_prev = mode

        # ------------------------------------------
        # 1) Fetch price
        # ------------------------------------------
        try:
            price = get_eth_price()
        except Exception as e:
            print(f"?? Failed to fetch price: {e}")

        price_window.append((loop_ts, price))
        while price_window and (loop_ts - price_window[0][0]) > SLOPE5M_WINDOW_SEC:
            price_window.popleft()

        # Update EMAs
        ema_fast = ema_update(ema_fast, price, ALPHA_FAST)
        ema_slow = ema_update(ema_slow, price, ALPHA_SLOW)
        ema_age_min += dt / 60.0

        # Vol EWMA on returns
        if last_price:
            ret = (price - last_price) / last_price
            vol_ewma = ema_update(vol_ewma, abs(ret), VOL_EWMA_ALPHA)
        last_price = price

        # Gap & "slope"
        if ema_slow:
            gap_frac = (ema_fast - ema_slow) / ema_slow
        else:
            gap_frac = 0.0

        gap_pct = gap_frac * 100.0
        slope5_pct_per_min = gap_pct  # keep same semantic as before

        # 5-minute slope (percent per minute)
        slope5m_pct_per_min = 0.0
        if len(price_window) >= 2:
            oldest_ts, oldest_price = price_window[0]
            dt_min = (loop_ts - oldest_ts) / 60.0
            if dt_min > 0 and oldest_price:
                slope5m_pct_per_min = ((price - oldest_price) / oldest_price) * 100.0 / dt_min

        # Track persistence of sustained positive slope for uptrend exit (LP mode only)
        if mode == "LP" and slope5m_pct_per_min > UPTREND_HODL_EXIT_SLOPE:
            slope_up_ticks += 1
        else:
            slope_up_ticks = 0

        # ------------------------------------------
        # 2) Band from last_cycle (active LP)
        # ------------------------------------------
        last_cycle = load_last_cycle()
        lp_active = bool(last_cycle.get("lp_active", False))
        band_lower = last_cycle.get("lower_bound_usd")
        band_upper = last_cycle.get("upper_bound_usd")
        token_id = last_cycle.get("tokenId")
        timestamp_create = last_cycle.get("timestamp_create")
        position_age_sec = None
        if timestamp_create:
            ts_epoch = parse_timestamp_utc(timestamp_create)
            if ts_epoch:
                position_age_sec = time.time() - ts_epoch

        have_band = (
            isinstance(band_lower, (int, float)) and
            isinstance(band_upper, (int, float)) and
            band_lower < band_upper
        )

        # On-chain sanity: if JSON says lp_active=False but tokenId exists and wallet still owns it, flip back to LP.
        if token_id and not lp_active:
            try:
                owner = pm.functions.ownerOf(int(token_id)).call()
                if owner and owner.lower() == WALLET_ADDRESS.lower():
                    lp_active = True
                    last_cycle["lp_active"] = True
                    save_last_cycle(last_cycle)
                    if mode == "FLAT_ETH" and have_band:
                        print("?? Detected active LP NFT on-chain; switching back to LP.")
                        mode = "LP"
            except Exception as e:
                print(f"?? Could not verify token ownership for {token_id}: {e}")

        # Distances to band (for logging & breakout detection)
        distance_above_pct = 0.0
        distance_below_pct = 0.0
        price_above = price_below = False

        if have_band:
            if price > band_upper:
                price_above = True
                distance_above_pct = (price - band_upper) / band_upper * 100.0
            elif price < band_lower:
                price_below = True
                distance_below_pct = (band_lower - price) / band_lower * 100.0

        # ------------------------------------------
        # 3) EMA-based consolidation detection
        # ------------------------------------------
        ema_consolidating = False
        if (
            abs(gap_pct) <= CONSOLIDATION_GAP_MAX and
            abs(slope5_pct_per_min) <= CONSOLIDATION_SLOPE_MAX and
            vol_ewma is not None and vol_ewma <= CONSOLIDATION_VOL_MAX and
            ema_age_min >= EMA_WARMUP_MINUTES
        ):
            ema_consolidating = True
            ema_consolidation_ticks += 1
        else:
            ema_consolidation_ticks = 0

        # ------------------------------------------
        # 4) Oscillation-based consolidation detection
        # ------------------------------------------
        osc_consolidating = False

        if ema_age_min >= EMA_WARMUP_MINUTES:
            # initialise center if needed
            if osc_center is None:
                osc_center = price
                oscillations = 0
                osc_pattern_ticks = 0
                last_osc_side = 0
            else:
                # if we drift too far from the current center, softly re-anchor
                drift = (price - osc_center) / osc_center
                if abs(drift) > OSC_WINDOW_PCT * 2:
                    # reset pattern around new center
                    osc_center = price
                    oscillations = 0
                    osc_pattern_ticks = 0
                    last_osc_side = 0
                else:
                    # update center slowly to follow local mean
                    osc_center = osc_center * 0.98 + price * 0.02

            # track pattern
            diff = price - osc_center
            side = 0
            if diff > 0:
                side = 1
            elif diff < 0:
                side = -1

            if last_osc_side is None:
                last_osc_side = side

            if side != 0 and last_osc_side is not None and last_osc_side != 0 and side != last_osc_side:
                oscillations += 1
            if side != 0:
                last_osc_side = side

            osc_pattern_ticks += 1

            if (
                oscillations >= OSC_MIN_OSCILLATIONS and
                osc_pattern_ticks >= OSC_PATTERN_TICKS_MIN
            ):
                osc_consolidating = True
        else:
            # EMAs not warmed yet -> reset oscillation pattern
            osc_center = None
            oscillations = 0
            osc_pattern_ticks = 0
            last_osc_side = 0

        tick_count += 1

        # ------------------------------------------
        # 5) Out-of-range persistence counters (for breakout)
        # ------------------------------------------
        if mode == "LP" and lp_active and have_band:
            if price_above:
                oor_up_count += 1
                oor_down_count = 0
            elif price_below:
                oor_down_count += 1
                oor_up_count = 0
            else:
                oor_up_count = 0
                oor_down_count = 0
        else:
            oor_up_count = 0
            oor_down_count = 0

        # ------------------------------------------
        # 6) Logging snapshot (throttled)
        # ------------------------------------------
        vol_str = "init" if vol_ewma is None else f"{vol_ewma:.6f}"
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        band_state_change = (have_band != state.get("have_band_last")) or (lp_active != state.get("lp_active_last"))
        mode_change = mode != mode_prev
        should_log = (
            ((tick_count % LOG_EVERY_N_TICKS) == 0) or band_state_change or mode_change
        ) and (loop_ts - last_log_ts >= LOG_MIN_SECONDS)

        if should_log:
            last_log_ts = loop_ts
            show_band = have_band and lp_active and mode == "LP"
            if show_band:
                print(
                    f"[{now_str}] price ${price:,.2f} | band ${band_lower:,.2f}-${band_upper:,.2f} (src=LP_JSON)"
                )
            else:
                print(
                    f"[{now_str}] price ${price:,.2f} | band N/A (no active LP JSON)"
                )

            print(
                f"    EMA5={ema_fast:.2f} | EMA30={ema_slow:.2f} | "
                f"gap={gap_pct:+.3f}% | slope5={slope5_pct_per_min:+.3f}%/min | "
                f"vol_EWMA={vol_str} | age={ema_age_min:.2f}m | "
                f"mode={mode} | lp_active={lp_active} | "
                f"oor_up={oor_up_count} | oor_down={oor_down_count} | "
                f"ema_consolid_ticks={ema_consolidation_ticks} | "
                f"osc_center={osc_center if osc_center is not None else 'None'} | "
                f"oscillations={oscillations} | pattern_ticks={osc_pattern_ticks}"
            )

            if mode == "FLAT_ETH":
                print(
                    f"FLAT_ETH: consolidation/oscillation building "
                    f"(EMA_ticks={ema_consolidation_ticks}/{EMA_CONSOLIDATION_TICKS_MIN}, "
                    f"osc_pattern_ticks={osc_pattern_ticks}/{OSC_PATTERN_TICKS_MIN}, "
                    f"oscillations={oscillations})."
                )

            if show_band:
                if price_above:
                    print(
                        f"price above band: dist_above={distance_above_pct:.3f}% "
                        f"(min breakout {OUT_OF_RANGE_PCT_MIN*100:.3f}%)"
                    )
                elif price_below:
                    print(
                        f"price below band: dist_below={distance_below_pct:.3f}% "
                        f"(min breakout {OUT_OF_RANGE_PCT_MIN*100:.3f}%)"
                    )

        # ------------------------------------------
        # 7) Decision logic
        # ------------------------------------------

        if mode == "LP" and (not lp_active or not have_band):
            print("?? Runtime sync: mode=LP but no active LP band ? switching to FLAT_ETH.")
            mode = "FLAT_ETH"
            ema_consolidation_ticks = 0

        # ---------- Mode: LP ----------
        if mode == "LP":
            if not lp_active or not have_band:
                print("?? mode=LP but no active LP band ? switching to FLAT_ETH.")
                mode = "FLAT_ETH"
                ema_consolidation_ticks = 0
            else:
                # Regime switch: in an uptrend, exit LP and hold 100% WETH.
                # This avoids the LP slowly (or fully) converting WETH into USDC as price rises.
                if (
                    ema_age_min >= EMA_WARMUP_MINUTES and
                    slope5m_pct_per_min > UPTREND_HODL_EXIT_SLOPE and
                    slope_up_ticks >= UPTREND_HODL_PERSIST_TICKS and
                    (position_age_sec is None or position_age_sec >= MIN_HOLD_UPTREND_EXIT_SECONDS)
                ):
                    print(
                        f"? Uptrend detected (slope5m={slope5m_pct_per_min:+.3f}%/min > {UPTREND_HODL_EXIT_SLOPE:.3f}, "
                        f"persist_ticks={slope_up_ticks}/{UPTREND_HODL_PERSIST_TICKS}, "
                        f"age={(position_age_sec or 0)/60:.2f}m). Exiting LP to hold ETH."
                    )
                    trigger_exit_to_flat_eth(
                        exit_reason="uptrend_hodl",
                        breakout_side="UP",
                        price=price,
                        band_lower=band_lower,
                        band_upper=band_upper,
                        gap_pct=gap_pct,
                        slope5_pct_per_min=slope5m_pct_per_min,
                        vol_ewma=vol_ewma,
                    )
                    mode = "FLAT_ETH"
                    oor_up_count = 0
                    oor_down_count = 0
                    ema_consolidation_ticks = 0
                    slope_up_ticks = 0
                    # reset oscillation pattern for new regime
                    osc_center = None
                    oscillations = 0
                    osc_pattern_ticks = 0
                    last_osc_side = 0
                    continue

                # --- 7.a Forced TOP exit (0.25% above upper band) ---
                if price_above and distance_above_pct >= TOP_FORCED_EXIT_PCT * 100.0:
                    print(
                        f"?? FORCED TOP EXIT: price {distance_above_pct:.3f}% above upper band "
                        f"(limit {TOP_FORCED_EXIT_PCT*100:.3f}%) ? exiting LP immediately."
                    )

                    trigger_exit_to_flat_eth(
                        exit_reason="forced_top",
                        breakout_side="UP",
                        price=price,
                        band_lower=band_lower,
                        band_upper=band_upper,
                        gap_pct=gap_pct,
                        slope5_pct_per_min=slope5_pct_per_min,
                        vol_ewma=vol_ewma,
                    )
                    mode = "FLAT_ETH"
                    oor_up_count = 0
                    oor_down_count = 0
                    ema_consolidation_ticks = 0
                    slope_up_ticks = 0

                else:
                    # --- 7.b Normal breakout logic (Balanced B1) ---
                    def trend_confirmed():
                        trend_flags = 0
                        if abs(gap_pct) >= GAP_BREAKOUT_PCT:
                            trend_flags += 1
                        if abs(slope5_pct_per_min) >= SLOPE_BREAKOUT_PCT_PER_MIN:
                            trend_flags += 1
                        if vol_ewma is not None and vol_ewma >= VOL_BREAKOUT_MIN:
                            trend_flags += 1
                        return trend_flags >= 2

                    breakout_up = False
                    breakout_down = False

                    if price_above and oor_up_count >= OUT_OF_RANGE_TICKS_MIN:
                        if distance_above_pct >= OUT_OF_RANGE_PCT_MIN * 100.0 and trend_confirmed():
                            breakout_up = True

                    if price_below and oor_down_count >= OUT_OF_RANGE_TICKS_MIN:
                        if distance_below_pct >= OUT_OF_RANGE_PCT_MIN * 100.0 and trend_confirmed():
                            breakout_down = True

                    if breakout_up or breakout_down:
                        side = "UP" if breakout_up else "DOWN"
                        dist_used = distance_above_pct if breakout_up else distance_below_pct
                        print(
                            f"?? Confirmed breakout {side} ? exiting LP and going FLAT_ETH "
                            f"(distance={dist_used:.3f}%, gap={gap_pct:.3f}%, "
                            f"slope5={slope5_pct_per_min:.3f}%/min, vol={vol_str})."
                        )

                        trigger_exit_to_flat_eth(
                            exit_reason="breakout",
                            breakout_side=side,
                            price=price,
                            band_lower=band_lower,
                            band_upper=band_upper,
                            gap_pct=gap_pct,
                            slope5_pct_per_min=slope5_pct_per_min,
                            vol_ewma=vol_ewma,
                        )

                        mode = "FLAT_ETH"
                        oor_up_count = 0
                        oor_down_count = 0
                        ema_consolidation_ticks = 0
                        slope_up_ticks = 0

        # ---------- Mode: FLAT_ETH ----------
        elif mode == "FLAT_ETH":
            # Re-enter LP only when 5m slope is negative (skip consolidation checks)
            if slope5m_pct_per_min >= 0:
                if tick_count % LOG_EVERY_N_TICKS == 0:
                    print(
                        f"? Trend guard: slope5m={slope5m_pct_per_min:+.3f}%/min >= 0.000 "
                        "-> staying FLAT_ETH (hold ETH), not entering LP."
                    )
                ema_consolidation_ticks = 0
                osc_center = None
                oscillations = 0
                osc_pattern_ticks = 0
                last_osc_side = 0
                continue

            print("? Negative 5m slope ? re-entering LP from FLAT_ETH.")

            lp_active_after, last_after = trigger_enter_lp_from_flat_eth(
                vol_ewma=vol_ewma,
                slope5_pct_per_min=slope5m_pct_per_min,
            )

            if lp_active_after:
                print("? LP created successfully ? switching mode=LP.")
                mode = "LP"
                oor_up_count = 0
                oor_down_count = 0
                ema_consolidation_ticks = 0
                # reset oscillation pattern for new regime
                osc_center = None
                oscillations = 0
                osc_pattern_ticks = 0
                last_osc_side = 0
            else:
                print(
                    "?? LP creation failed or lp_active=False after create_position "
                    "? staying in FLAT_ETH."
                )
                ema_consolidation_ticks = 0
                osc_center = None
                oscillations = 0
                osc_pattern_ticks = 0
                last_osc_side = 0

        # 8) Save state
        # ------------------------------------------
        state = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "vol_ewma": vol_ewma,
            "last_price": last_price,
            "ema_age_min": ema_age_min,
            "oor_up_count": oor_up_count,
            "oor_down_count": oor_down_count,
            "ema_consolidation_ticks": ema_consolidation_ticks,
            "osc_center": osc_center,
            "oscillations": oscillations,
            "osc_pattern_ticks": osc_pattern_ticks,
            "last_osc_side": last_osc_side,
            "mode": mode,
            "tick_count": tick_count,
            "have_band_last": have_band,
            "lp_active_last": lp_active,
            "last_log_ts": last_log_ts,
            "slope_up_ticks": slope_up_ticks,
        }
        save_state(state)

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
