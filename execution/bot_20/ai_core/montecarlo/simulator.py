import numpy as np
from math import sqrt

from ai_core.live_policy import decide_trend_up, should_reposition_lower, suggest_range_parameters
from ai_core.montecarlo.lp_sim_state import LPSimulatorState

def amounts_from_liquidity(L, P, Pa, Pb):
    sqrtP  = sqrt(P)
    sqrtPa = sqrt(Pa)
    sqrtPb = sqrt(Pb)

    if P <= Pa:
        amount0 = L * (1.0 / sqrtPa - 1.0 / sqrtPb)
        amount1 = 0.0
    elif P >= Pb:
        amount0 = 0.0
        amount1 = L * (sqrtPb - sqrtPa)
    else:
        amount0 = L * (1.0 / sqrtP - 1.0 / sqrtPb)
        amount1 = L * (sqrtP - sqrtPa)
    return amount0, amount1

def liquidity_from_amount0_below_range(amount0, Pa, Pb):
    sqrtPa = sqrt(Pa)
    sqrtPb = sqrt(Pb)
    denom = (1.0 / sqrtPa - 1.0 / sqrtPb)
    if denom <= 0:
        return 0.0
    return amount0 / denom

def run_strategy_on_path(
    prices,
    volume_path,
    width_ticks: int,
    offset_ticks: int,
    fee_tier: float = 0.0005,
    pool_share: float = 0.00003,
):
    state = LPSimulatorState(start_eth=1.0, start_usdc=0.0, price0=prices[0])

    in_lp = False
    L = 0.0
    Pa = Pb = None
    fees_usd = 0.0

    equity_curve = []

    def band_from_ticks(price, width_ticks, offset_ticks):
        lower = price * (1.0 + offset_ticks / 10_000.0)
        upper = lower * (1.0 + width_ticks / 10_000.0)
        return lower, upper

    for i in range(1, len(prices)):
        price = float(prices[i])
        prev_price = float(prices[i - 1])
        state.price = price

        # compute simple trend variables (or pass more elaborate ones)
        if i > 30:
            window = prices[max(0, i - 60):i]
            ema_fast = np.mean(window[-5:])
            ema_slow = np.mean(window)
            ema_gap_rel = (ema_fast - ema_slow) / ema_slow
            ema_fast_slope = (prices[i] - prices[i - 5]) / prices[i - 5] / 5
        else:
            ema_fast = ema_slow = ema_gap_rel = ema_fast_slope = None

        if not in_lp:
            # rebalance: convert USDC → ETH at current price before re-enter
            if state.usdc > 0:
                state.eth += state.usdc / price
                state.usdc = 0.0

            lower_price, upper_price = band_from_ticks(price, width_ticks, offset_ticks)
            Pa, Pb = lower_price, upper_price

            if state.eth <= 0:
                equity_curve.append(state.eth + state.usdc / price)
                continue

            L = liquidity_from_amount0_below_range(state.eth, Pa, Pb)
            state.eth = 0.0
            in_lp = True

            amt0, amt1 = amounts_from_liquidity(L, price, Pa, Pb)
            tvl_eth = amt0 + amt1 / price
            equity_curve.append(tvl_eth + state.usdc / price)
            continue

        # If in LP
        if in_lp:
            amt0, amt1 = amounts_from_liquidity(L, price, Pa, Pb)

            # fees: use real historical volume for this step
            volume_usd = float(volume_path[i])
            tvl_usd = amt1 + amt0 * price
            if tvl_usd > 0:
                step_fees_usd = fee_tier * volume_usd * pool_share
                fees_usd += step_fees_usd

            # exit conditions
            if price >= Pb or decide_trend_up(ema_fast, ema_slow, ema_gap_rel, ema_fast_slope):
                state.usdc += amt1 + fees_usd
                state.eth += amt0
                L = 0.0
                fees_usd = 0.0
                in_lp = False

        # Mark-to-market
        if in_lp:
            amt0, amt1 = amounts_from_liquidity(L, price, Pa, Pb)
            tvl_eth = amt0 + (amt1 + fees_usd) / price
        else:
            tvl_eth = 0.0
        total_eth = state.eth + state.usdc / price + tvl_eth
        equity_curve.append(total_eth)

    # Final liquidation if still in LP
    final_price = float(prices[-1])
    if in_lp and L > 0.0:
        amt0, amt1 = amounts_from_liquidity(L, final_price, Pa, Pb)
        state.usdc += amt1 + fees_usd
        state.eth += amt0
        in_lp = False

    final_value_eth = state.eth + state.usdc / final_price
    fees_earned_eth = (fees_usd / final_price) if final_price > 0 else 0.0

    eq = np.array(equity_curve, dtype=float)
    if eq.size > 0:
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd))
    else:
        max_dd = 0.0

    return final_value_eth, fees_earned_eth, max_dd

