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


def _tiered_exposure(score: pd.Series, hi: float, mid: float) -> pd.Series:
    base = pd.Series(0.0, index=score.index, dtype=float)
    base = base.mask(score >= mid, 0.5)
    base = base.mask(score >= hi, 1.0)
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="Velocity+acceleration gate backtest.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-span", type=int, default=50, help="EMA span in bars")
    p.add_argument("--norm-window", type=int, default=200, help="Std window for velocity z-score")
    p.add_argument("--accel-weight", type=float, default=0.5)
    p.add_argument("--score-hi", type=float, default=1.0)
    p.add_argument("--score-mid", type=float, default=0.3)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36, help="Realized vol window in bars")
    p.add_argument("--out", default="artifacts/backtest/va_backtest.csv")
    args = p.parse_args()

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    logp = np.log(close)
    ema = logp.ewm(span=args.ema_span, adjust=False).mean()
    v = ema.diff()
    a = v.diff()

    v_std = v.rolling(args.norm_window, min_periods=args.norm_window).std()
    z_v = v / (v_std + 1e-12)
    z_a = a / (v_std + 1e-12)
    score = z_v + args.accel_weight * z_a

    base_exposure = _tiered_exposure(score, args.score_hi, args.score_mid)
    sigma_ann = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma_ann).replace([np.inf, -np.inf], np.nan)
    weight_raw = weight_raw.fillna(0.0).clip(lower=0.0, upper=args.w_max)
    weight = (weight_raw * base_exposure).clip(lower=0.0, upper=args.w_max)

    core_r = weight.shift(1).fillna(0.0) * r
    eq = np.exp(core_r.cumsum())

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "close": close,
            "r": r,
            "ema_logp": ema,
            "v": v,
            "a": a,
            "z_v": z_v,
            "z_a": z_a,
            "score": score,
            "base_exposure": base_exposure,
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
