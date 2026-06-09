from __future__ import annotations

import argparse
import time
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


def _compute_frame(
    price_path: Path,
    bar_minutes: int,
    lookback: int,
    compress_window: int,
    compress_quantile: float,
    hold_bars: int,
    target_vol: float,
    w_max: float,
    vol_window: int,
) -> pd.DataFrame:
    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    rolling_high = close.rolling(lookback, min_periods=lookback).max()
    rolling_low = close.rolling(lookback, min_periods=lookback).min()
    rolling_range = (rolling_high - rolling_low) / rolling_low

    range_pct = rolling_range.rolling(compress_window, min_periods=compress_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    compress = range_pct <= compress_quantile

    breakout = (close > rolling_high.shift(1)) & compress.shift(1)
    breakdown = (close < rolling_low.shift(1)) & compress.shift(1)

    sigma_ann = _compute_sigma_ann(r, bar_minutes, vol_window)
    weight_raw = (target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan)
    weight_raw = weight_raw.fillna(0.0).clip(lower=0.0, upper=w_max)

    gate_on = np.zeros(len(df), dtype=bool)
    hold = 0
    for i in range(len(df)):
        if breakdown.iat[i]:
            hold = 0
            gate_on[i] = False
            continue
        if breakout.iat[i]:
            hold = max(int(hold_bars), 0)
        if hold > 0:
            gate_on[i] = True
            hold -= 1
        else:
            gate_on[i] = breakout.iat[i]
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
            "gate_on": gate_on,
            "sigma_ann": sigma_ann,
            "weight_raw": weight_raw,
            "weight": weight,
            "core_r": core_r,
            "eq": eq,
        }
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Breakout paper service (append-only).")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--compress-window", type=int, default=96)
    p.add_argument("--compress-quantile", type=float, default=0.50)
    p.add_argument("--hold-bars", type=int, default=24)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--out", default="artifacts/paper/breakout_paper.csv")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        frame = _compute_frame(
            price_path=price_path,
            bar_minutes=args.bar_minutes,
            lookback=args.lookback,
            compress_window=args.compress_window,
            compress_quantile=args.compress_quantile,
            hold_bars=args.hold_bars,
            target_vol=args.target_vol,
            w_max=args.w_max,
            vol_window=args.vol_window,
        )

        if out_path.exists():
            try:
                last_ts = pd.read_csv(out_path, usecols=["timestamp"]).iloc[-1, 0]
                last_ts = pd.to_datetime(last_ts, utc=True, errors="coerce")
            except Exception:
                last_ts = None
        else:
            last_ts = None

        if last_ts is not None:
            new_rows = frame[frame["timestamp"] > last_ts].copy()
        else:
            new_rows = frame.copy()

        if not new_rows.empty:
            new_rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            print(f"appended_rows={len(new_rows)} last_ts={new_rows['timestamp'].iloc[-1]}")
        else:
            print("no new rows")

        if args.once:
            break
        time.sleep(max(int(args.interval_sec), 1))


if __name__ == "__main__":
    main()
