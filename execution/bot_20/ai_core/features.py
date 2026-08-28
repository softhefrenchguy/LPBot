import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureConfig:
    ema_fast_window: int = 5          # in candles (assume 1m candles)
    ema_slow_window: int = 30
    trend_slope_lookback: int = 5     # minutes
    vol_lookback_short: int = 60      # minutes
    vol_lookback_long: int = 240      # minutes
    fvg_lookback: int = 240           # how far back to track FVGs (candles)
    overlap_lookback: int = 60
    autocorr_lookback: int = 60
    drift_lookback: int = 60
    wick_stats_lookback: int = 60
    sweep_lookback: int = 60
    vol_shock_window: int = 5         # minutes
    vol_shock_lookback: int = 30      # minutes
    vol_shock_threshold: float = 0.03 # 3% price move
    volume_1h_window: int = 60
    volume_24h_window: int = 1440     # 24h if 1m bars


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def _compute_trend_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"]

    # EMAs
    ema_fast = _ema(close, cfg.ema_fast_window)
    ema_slow = _ema(close, cfg.ema_slow_window)

    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["ema_gap_pct"] = (ema_fast - ema_slow) / ema_slow

    # slope of fast EMA over last k minutes
    k = cfg.trend_slope_lookback
    ema_fast_shifted = ema_fast.shift(k)
    out["ema_fast_slope"] = (ema_fast - ema_fast_shifted) / (ema_fast_shifted.replace(0, np.nan) * k)

    # 1h and 4h range percentages (if we have enough data)
    # Assume candles are 1m; you can adjust if different
    high = df["high"]
    low = df["low"]

    range_1h_high = high.rolling(window=60, min_periods=10).max()
    range_1h_low = low.rolling(window=60, min_periods=10).min()
    range_1h = (range_1h_high - range_1h_low) / ((range_1h_high + range_1h_low) / 2.0)

    range_4h_high = high.rolling(window=240, min_periods=30).max()
    range_4h_low = low.rolling(window=240, min_periods=30).min()
    range_4h = (range_4h_high - range_4h_low) / ((range_4h_high + range_4h_low) / 2.0)

    out["range_1h_pct"] = range_1h
    out["range_4h_pct"] = range_4h
    out["range_ratio"] = range_1h / range_4h.replace(0, np.nan)

    return out


def _compute_volatility_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # log returns
    r = np.log(close / close.shift(1))

    rv_short = _rolling_std(r, cfg.vol_lookback_short)
    rv_long = _rolling_std(r, cfg.vol_lookback_long)

    out["rv_short"] = rv_short
    out["rv_long"] = rv_long
    out["rv_ratio"] = rv_short / rv_long.replace(0, np.nan)

    # ATR-style volatility
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr_norm"] = true_range.rolling(window=cfg.vol_lookback_short, min_periods=10).mean() / close

    # Volatility shock: max move over window
    window = cfg.vol_shock_window
    lookback = cfg.vol_shock_lookback

    # rolling max move over vol_shock_window
    pct_change_window = close.pct_change(window).abs()
    max_move = pct_change_window.rolling(lookback, min_periods=1).max()

    out["max_move_recent"] = max_move
    out["vol_shock_flag"] = (max_move > cfg.vol_shock_threshold).astype(int)

    return out


def _compute_fvg_and_microstructure_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Detects fair value gaps + some microstructure stats.
    """
    out = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    # --- Fair Value Gaps (3-candle pattern) ---
    # bullish FVG: low_t > high_{t-2}
    # bearish FVG: high_t < low_{t-2}
    high_shift2 = high.shift(2)
    low_shift2 = low.shift(2)

    bullish_fvg = (low > high_shift2) & high_shift2.notna()
    bearish_fvg = (high < low_shift2) & low_shift2.notna()

    # Track active FVGs with a simple approach:
    # We will maintain lists of open FVGs in local arrays and compute counts/distances.
    num_bull_active = []
    num_bear_active = []
    dist_to_nearest_bull = []
    dist_to_nearest_bear = []
    fvg_density = []

    # each FVG: dict with fields: "low", "high", "direction"
    active_bull = []
    active_bear = []

    prices = close.values
    highs = high.values
    lows = low.values

    # We only keep FVGs within fvg_lookback candles
    max_len = cfg.fvg_lookback

    for i in range(len(df)):
        # drop very old ones if list too long
        if len(active_bull) > max_len:
            active_bull = active_bull[-max_len:]
        if len(active_bear) > max_len:
            active_bear = active_bear[-max_len:]

        # add new FVG if forms here
        if bullish_fvg.iloc[i]:
            # bullish gap between high_{t-2} and low_t
            gap_low = high_shift2.iloc[i]
            gap_high = low.iloc[i]
            if pd.notna(gap_low) and pd.notna(gap_high):
                active_bull.append({"low": gap_low, "high": gap_high})

        if bearish_fvg.iloc[i]:
            # bearish gap between high_t and low_{t-2}
            gap_low = high.iloc[i]
            gap_high = low_shift2.iloc[i]
            if pd.notna(gap_low) and pd.notna(gap_high):
                active_bear.append({"low": gap_high, "high": gap_low})

        price_i = prices[i]

        # update "active" status: mark FVGs as filled if price trades through gap
        def filter_active(fvgs, price_now):
            still_active = []
            for f in fvgs:
                # consider gap filled if price has fully crossed the zone
                if not (f["low"] <= price_now <= f["high"]):
                    still_active.append(f)
            return still_active

        active_bull = filter_active(active_bull, price_i)
        active_bear = filter_active(active_bear, price_i)

        num_bull_active.append(len(active_bull))
        num_bear_active.append(len(active_bear))

        # distances to nearest FVG
        def nearest_distance(fvgs, price_now):
            if not fvgs:
                return np.nan
            dists = []
            for f in fvgs:
                # distance to center of the gap
                center = (f["low"] + f["high"]) / 2.0
                dists.append(abs(price_now - center) / price_now)
            return min(dists) if dists else np.nan

        dist_to_nearest_bull.append(nearest_distance(active_bull, price_i))
        dist_to_nearest_bear.append(nearest_distance(active_bear, price_i))

        # density = total number of FVGs that formed in last fvg_lookback bars (approx using lists size)
        fvg_density.append(len(active_bull) + len(active_bear))

    out["num_bull_fvg_active"] = num_bull_active
    out["num_bear_fvg_active"] = num_bear_active
    out["dist_to_nearest_bull_fvg_pct"] = dist_to_nearest_bull
    out["dist_to_nearest_bear_fvg_pct"] = dist_to_nearest_bear
    out["fvg_density"] = fvg_density

    # --- Mean-reversion vs trend: overlap & autocorrelation ---

    # Overlap ratio between consecutive candles
    prev_low = low.shift(1)
    prev_high = high.shift(1)

    # intersection of [low, high] and [prev_low, prev_high]
    overlap_low = np.maximum(low, prev_low)
    overlap_high = np.minimum(high, prev_high)
    overlap_len = (overlap_high - overlap_low).clip(lower=0)
    candle_range = (high - low).replace(0, np.nan)
    overlap_ratio = overlap_len / candle_range

    out["overlap_ratio"] = overlap_ratio

    out["avg_overlap_recent"] = (
        overlap_ratio.rolling(cfg.overlap_lookback, min_periods=10).mean()
    )

    # return autocorrelation
    returns = np.log(close / close.shift(1))
    # rolling autocorrelation of r_t with r_{t-1}
    def rolling_autocorr(x, lag=1):
        x1 = x
        x2 = x.shift(lag)
        return (
            x1.rolling(cfg.autocorr_lookback, min_periods=10)
            .corr(x2)
        )

    out["ret_autocorr"] = rolling_autocorr(returns, lag=1)

    # --- Wick / liquidity-hunt style patterns ---
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    upper_wick = (high - body_high).clip(lower=0)
    lower_wick = (body_low - low).clip(lower=0)
    body = (body_high - body_low).replace(0, np.nan)

    # wick/body ratios
    upper_wick_ratio = upper_wick / body
    lower_wick_ratio = lower_wick / body

    out["upper_wick_ratio"] = upper_wick_ratio
    out["lower_wick_ratio"] = lower_wick_ratio

    large_wick_up = (upper_wick_ratio > 2.0).astype(int)
    large_wick_down = (lower_wick_ratio > 2.0).astype(int)

    out["large_wick_up_count"] = large_wick_up.rolling(cfg.wick_stats_lookback, min_periods=1).sum()
    out["large_wick_down_count"] = large_wick_down.rolling(cfg.wick_stats_lookback, min_periods=1).sum()

    # sweeps: new high but close back into previous range
    # For simplicity: new 1h high but close < previous 1h high
    window = cfg.sweep_lookback
    rolling_high = high.rolling(window, min_periods=10).max()
    rolling_low = low.rolling(window, min_periods=10).min()

    prev_rolling_high = rolling_high.shift(1)
    prev_rolling_low = rolling_low.shift(1)

    sweep_high = (high > prev_rolling_high) & (close < prev_rolling_high)
    sweep_low = (low < prev_rolling_low) & (close > prev_rolling_low)

    out["sweep_high_count"] = sweep_high.rolling(window, min_periods=1).sum()
    out["sweep_low_count"] = sweep_low.rolling(window, min_periods=1).sum()

    return out


def _compute_speed_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # Rolling regression on log price
    log_price = np.log(close)

    drift_values = []
    noise_values = []

    window = cfg.drift_lookback
    times = np.arange(len(df))

    for i in range(len(df)):
        if i < window:
            drift_values.append(np.nan)
            noise_values.append(np.nan)
            continue

        idx_slice = slice(i - window, i)
        y = log_price.iloc[idx_slice].values
        x = times[idx_slice]

        if np.any(np.isnan(y)):
            drift_values.append(np.nan)
            noise_values.append(np.nan)
            continue

        # fit y = a*x + b
        a, b = np.polyfit(x, y, 1)
        # drift per minute in log space -> approx pct per minute
        drift_per_min = a
        # predicted
        y_hat = a * x + b
        residuals = y - y_hat
        noise = np.std(residuals)

        drift_values.append(drift_per_min)
        noise_values.append(noise)

    drift_per_min = pd.Series(drift_values, index=df.index)
    noise = pd.Series(noise_values, index=df.index)

    # convert drift to approx % per hour
    out["drift_pct_per_hour"] = drift_per_min * 60.0
    out["drift_noise_ratio"] = noise / (drift_per_min.abs() + 1e-9)

    return out


def _compute_volume_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    df_pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    If df_pool is provided, it should at least have:
      - index or column 'timestamp' aligned (or mergeable) with df
      - 'volume_usd' (or 'volume')
      - optionally 'fees_usd' and 'liquidity_usd'
    """
    out = pd.DataFrame(index=df.index)

    if df_pool is None:
        # basic on-chain candle volume only
        vol = df.get("volume", pd.Series(index=df.index, dtype=float))
        out["volume_1h"] = vol.rolling(cfg.volume_1h_window, min_periods=1).sum()
        out["volume_24h"] = vol.rolling(cfg.volume_24h_window, min_periods=1).sum()
        out["volume_ratio"] = out["volume_1h"] / (out["volume_24h"] + 1e-9)
        out["fee_apr_est"] = np.nan
        return out

    # If df_pool provided, align on index or timestamp
    pool = df_pool.copy()
    if "timestamp" in pool.columns:
        pool = pool.set_index("timestamp")

    # reindex pool to candles
    pool = pool.reindex(df.index, method="ffill")

    if "volume_usd" in pool.columns:
        vol = pool["volume_usd"]
    elif "volume" in pool.columns:
        vol = pool["volume"]
    else:
        vol = pd.Series(0.0, index=df.index)

    out["volume_1h"] = vol.rolling(cfg.volume_1h_window, min_periods=1).sum()
    out["volume_24h"] = vol.rolling(cfg.volume_24h_window, min_periods=1).sum()
    out["volume_ratio"] = out["volume_1h"] / (out["volume_24h"] + 1e-9)

    # approximate fee APR if fees and liquidity known
    if "fees_usd_24h" in pool.columns and "liquidity_usd" in pool.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            fee_apr = (pool["fees_usd_24h"] / pool["liquidity_usd"]).replace([np.inf, -np.inf], np.nan)
        out["fee_apr_est"] = fee_apr * 365.0
    else:
        out["fee_apr_est"] = np.nan

    return out


def build_market_features(
    df_candles: pd.DataFrame,
    df_pool: Optional[pd.DataFrame] = None,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Main entry point.

    Parameters
    ----------
    df_candles : DataFrame
        Must have columns: ['open', 'high', 'low', 'close'].
        'volume' is optional but recommended.
        Index should be a DateTimeIndex or something time-like.

    df_pool : DataFrame, optional
        Pool-level metrics (volume, fees, liquidity) indexed by timestamp
        or with a 'timestamp' column.

    cfg : FeatureConfig, optional
        Configuration for windows and thresholds.

    Returns
    -------
    features : DataFrame
        DataFrame indexed like df_candles with many feature columns.
    """
    if cfg is None:
        cfg = FeatureConfig()

    # Ensure index aligned and sorted
    df = df_candles.sort_index().copy()

    trend = _compute_trend_features(df, cfg)
    vol = _compute_volatility_features(df, cfg)
    micro = _compute_fvg_and_microstructure_features(df, cfg)
    speed = _compute_speed_features(df, cfg)
    volume = _compute_volume_features(df, cfg, df_pool)

    # merge all feature blocks
    features = pd.concat(
        [trend, vol, micro, speed, volume],
        axis=1,
    )

    return features
