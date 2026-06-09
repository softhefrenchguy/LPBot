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
    p = argparse.ArgumentParser(description="Vol-regime + soft trend filter backtest.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--trend-span", type=int, default=200, help="EMA span in bars")
    p.add_argument("--vol-window", type=int, default=288, help="Vol window in bars")
    p.add_argument("--vol-low", type=float, default=0.12, help="Annualized vol lower band")
    p.add_argument("--vol-high", type=float, default=0.60, help="Annualized vol upper band")
    p.add_argument("--vol-scale-outside", type=float, default=0.0)
    p.add_argument("--trend-scale-off", type=float, default=0.35)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--out", default="artifacts/backtest/vol_regime_backtest.csv")
    args = p.parse_args()

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    logp = np.log(close)
    ema = logp.ewm(span=args.trend_span, adjust=False).mean()
    trend_on = logp > ema

    sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan)
    weight_raw = weight_raw.fillna(0.0).clip(lower=0.0, upper=args.w_max)

    vol_in_band = (sigma_ann >= args.vol_low) & (sigma_ann <= args.vol_high)
    vol_scale = np.where(vol_in_band, 1.0, float(args.vol_scale_outside))
    trend_scale = np.where(trend_on, 1.0, float(args.trend_scale_off))

    weight = weight_raw * vol_scale * trend_scale
    weight = pd.Series(weight, index=df.index).clip(lower=0.0, upper=args.w_max)

    core_r = weight.shift(1).fillna(0.0) * r
    eq = np.exp(core_r.cumsum())

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "close": close,
            "r": r,
            "ema_logp": ema,
            "trend_on": trend_on,
            "sigma_ann": sigma_ann,
            "vol_in_band": vol_in_band,
            "vol_scale": vol_scale,
            "trend_scale": trend_scale,
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
