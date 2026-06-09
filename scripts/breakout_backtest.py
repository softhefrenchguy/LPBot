from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must include timestamp and close columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def _compute_sigma_ann(r: pd.Series, bar_minutes: int, vol_window: int) -> pd.Series:
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    sigma = r.rolling(vol_window, min_periods=vol_window).std()
    return sigma * np.sqrt(bars_per_year)


def main() -> None:
    p = argparse.ArgumentParser(description="Breakout-after-compression backtest.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--lookback", type=int, default=288, help="Range lookback in bars")
    p.add_argument("--compress-window", type=int, default=288, help="Compression window in bars")
    p.add_argument("--compress-quantile", type=float, default=0.25)
    p.add_argument("--hold-bars", type=int, default=0, help="Hold N bars after breakout")
    p.add_argument("--trend-ema", type=int, default=0, help="Optional trend EMA on higher timeframe (bars)")
    p.add_argument(
        "--trend-timeframe",
        choices=["same", "1h", "4h", "1d"],
        default="same",
        help="Timeframe for trend EMA (default: same)",
    )
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--out", default="artifacts/backtest/breakout_backtest.csv")
    args = p.parse_args()

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    # rolling range + compression
    rolling_high = close.rolling(args.lookback, min_periods=args.lookback).max()
    rolling_low = close.rolling(args.lookback, min_periods=args.lookback).min()
    rolling_range = (rolling_high - rolling_low) / rolling_low

    range_pct = rolling_range.rolling(args.compress_window, min_periods=args.compress_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    compress = range_pct <= args.compress_quantile

    # breakout signal: close above prior high after compression
    breakout = (close > rolling_high.shift(1)) & compress.shift(1)
    breakdown = (close < rolling_low.shift(1)) & compress.shift(1)

    # optional higher-timeframe trend filter
    if args.trend_ema and args.trend_ema > 0:
        tf_map = {"same": f"{args.bar_minutes}min", "1h": "1h", "4h": "4h", "1d": "1d"}
        tf = tf_map[args.trend_timeframe]
        # Use raw values to avoid label alignment against RangeIndex, which would create NaNs.
        tmp = pd.DataFrame({"close": close.to_numpy()}, index=df["timestamp"])
        if tf == f"{args.bar_minutes}min":
            trend_close = tmp["close"]
            trend_ema = trend_close.ewm(span=args.trend_ema, adjust=False).mean()
            trend_on = (trend_close > trend_ema).values
        else:
            # Use only completed higher-timeframe bars to avoid lookahead bias.
            trend_close = tmp["close"].resample(tf).last().shift(1).ffill()
            trend_ema = trend_close.ewm(span=args.trend_ema, adjust=False).mean()
            trend_on = (trend_close > trend_ema).reindex(tmp.index, method="ffill").to_numpy()
    else:
        trend_on = np.ones(len(df), dtype=bool)

    sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan)
    weight_raw = weight_raw.fillna(0.0).clip(lower=0.0, upper=args.w_max)

    # gate: long on breakout, optional hold, flat on breakdown
    gate_on = np.zeros(len(df), dtype=bool)
    hold = 0
    for i in range(len(df)):
        if breakdown.iat[i]:
            hold = 0
            gate_on[i] = False
            continue
        if breakout.iat[i]:
            hold = max(int(args.hold_bars), 0)
        if hold > 0:
            gate_on[i] = True
            hold -= 1
        else:
            gate_on[i] = breakout.iat[i]

        if not trend_on[i]:
            gate_on[i] = False
    weight = weight_raw * gate_on.astype(float)

    core_r = weight.shift(1).fillna(0.0) * r
    eq = np.exp(core_r.cumsum())

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "close": close,
            "r": r,
            "rolling_high": rolling_high,
            "rolling_low": rolling_low,
            "rolling_range": rolling_range,
            "range_pct": range_pct,
            "compress": compress,
            "breakout": breakout,
            "breakdown": breakdown,
            "trend_on": trend_on,
            "gate_on": gate_on,
            "sigma_ann": sigma_ann,
            "weight_raw": weight_raw,
            "weight": weight,
            "core_r": core_r,
            "eq": eq,
        }
    )
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
