# ai_core/live_policy.py
# Central AI logic for real-time LP decisions (trend-up, reposition-lower, dynamic ranges)

import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Optional ML model (you already trained panic_model; we can extend later)
try:
    from joblib import load
    PANIC_MODEL = load(os.path.join(MODEL_DIR, "panic_model.pkl"))
except Exception:
    PANIC_MODEL = None


# ---------------------------------------------------------
# 1. Trend-up decision (EMA + optional ML override)
# ---------------------------------------------------------
def decide_trend_up(ema_fast, ema_slow, ema_gap_rel, ema_fast_slope):
    """
    Returns True if conditions indicate a strong upward trend.
    Rule-based fallback:
      - EMA5 > EMA30
      - gap > 0.4%
      - slope > 0.2% per minute

    If PANIC_MODEL is available, we can override with model probability.
    """

    # Simple rule-based decision
    rule_flag = (
        ema_fast is not None
        and ema_slow is not None
        and ema_fast > ema_slow
        and ema_gap_rel is not None
        and ema_gap_rel > 0.004          # 0.4%
        and ema_fast_slope is not None
        and ema_fast_slope > 0.002       # 0.2% per minute
    )

    # Optional ML override (e.g. probability of continuation)
    if PANIC_MODEL and ema_gap_rel is not None and ema_fast_slope is not None:
        X = np.array([[ema_gap_rel, ema_fast_slope]])
        try:
            prob = PANIC_MODEL.predict_proba(X)[0][1]
            return prob > 0.70
        except Exception:
            # If anything fails, fall back to rules
            return rule_flag

    return rule_flag


# ---------------------------------------------------------
# 2. Reposition-lower decision
# ---------------------------------------------------------
def should_reposition_lower(price, lower_usd, ema_fast, ema_slow, ema_gap_rel, volatility):
    """
    Decide if we should withdraw and recreate LP lower (ETH-heavy re-entry).
    Conditions:
        - price below band (under lower_usd)
        - below by at least 0.15%
        - EMA5 < EMA30 (local downtrend)
        - EMA gap sufficiently negative
        - volatility not ultra-low (avoid micro-noise)
    """

    if price is None or lower_usd is None:
        return False

    # positive if price is below band
    dist = (lower_usd - price) / lower_usd

    # still inside band
    if dist <= 0:
        return False

    # Require at least 0.15% below band
    if dist < 0.0015:
        return False

    # EMA downtrend: fast below slow
    if ema_fast is None or ema_slow is None or not (ema_fast < ema_slow):
        return False

    # Gap should show meaningful downside separation
    if ema_gap_rel is None or ema_gap_rel > -0.001:   # require at least -0.1%
        return False

    # Volatility filter (avoid ultra-flat noise)
    if volatility is not None and volatility < 0.001:
        return False

    return True


# ---------------------------------------------------------
# 3. Range parameter suggestion
# ---------------------------------------------------------
# ai_core/live_policy.py

def suggest_range_parameters(vol_ewma, ema_gap_rel=None):
    """
    Returns:
        width_pct   -- interpreted by create_position.py as pct * 10_000 ticks
        offset_pct  -- interpreted by create_position.py as pct * 10_000 ticks

    So: return width_pct = 0.006 → 60 ticks
         return width_pct = 0.012 → 120 ticks
    """

    # ========= SAFE BOUNDS (hard enforced) =========
    MIN_WIDTH_TICKS  = 60
    MAX_WIDTH_TICKS  = 120
    MIN_OFFSET_TICKS = 20
    MAX_OFFSET_TICKS = 80

    # convert ticks back to % (since create_position does pct * 10,000)
    # 1 tick ≈ 0.01% → 100 ticks ≈ 1%
    tick_to_pct = 1.0 / 10_000.0

    # ========= Volatility logic (simple version) =========
    # If no vol data exists (first cycle), use safe defaults
    if vol_ewma is None:
        target_width = 90    # neutral safe width (in ticks)
        target_offset = 40
    else:
        # vol_ewma expected range: 0.3 → 2.0-like behavior
        # You can refine this over time
        if vol_ewma < 0.5:
            target_width = 60
            target_offset = 25
        elif vol_ewma < 1.0:
            target_width = 80
            target_offset = 35
        elif vol_ewma < 1.5:
            target_width = 100
            target_offset = 45
        else:
            target_width = 120
            target_offset = 55

    # ========= Trend logic (ema_gap_rel) =========
    if ema_gap_rel is not None:
        # If strong uptrend → raise offset slightly
        if ema_gap_rel > 0.004:      # ~0.4%
            target_offset += 10

        # If downtrend → pull offset down, but not below minimum
        elif ema_gap_rel < -0.004:
            target_offset -= 10

    # ========= Clamp to safe bounds =========
    target_width  = int(max(MIN_WIDTH_TICKS,  min(target_width,  MAX_WIDTH_TICKS)))
    target_offset = int(max(MIN_OFFSET_TICKS, min(target_offset, MAX_OFFSET_TICKS)))

    # ========= Convert ticks → pct (so create_position can convert back) =========
    width_pct  = target_width  * tick_to_pct
    offset_pct = target_offset * tick_to_pct

    # print(f"[AI] width_ticks={target_width}  offset_ticks={target_offset}")
    # print(f"[AI] width_pct={width_pct*100:.3f}% offset_pct={offset_pct*100:.3f}%")

    return width_pct, offset_pct
