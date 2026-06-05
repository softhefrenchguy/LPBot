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


def _parse_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_weights(raw: str, n: int) -> list[float]:
    if not raw:
        return [1.0 / n] * n
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError("weights length must match horizons length")
    total = sum(vals)
    if total <= 0:
        raise ValueError("weights must sum to > 0")
    return [v / total for v in vals]


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-horizon momentum backtest.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument(
        "--horizons",
        default="12,48,288",
        help="Comma-separated horizons in bars (default: 1h,4h,1d on 5m data)",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Comma-separated weights for horizons (normalized). Leave empty for equal.",
    )
    p.add_argument("--norm-window", type=int, default=288, help="Score z-score window (bars)")
    p.add_argument("--score-hi", type=float, default=1.0)
    p.add_argument("--score-mid", type=float, default=0.3)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36, help="Realized vol window in bars")
    p.add_argument("--out", default="artifacts/backtest/multi_mom_backtest.csv")
    args = p.parse_args()

    horizons = _parse_list(args.horizons)
    weights = _parse_weights(args.weights, len(horizons))

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    logp = np.log(close)

    mom_components = []
    for h in horizons:
        mom_components.append(logp - logp.shift(h))

    score_raw = pd.Series(0.0, index=df.index, dtype=float)
    for w, comp in zip(weights, mom_components):
        score_raw = score_raw + w * comp

    score_std = score_raw.rolling(args.norm_window, min_periods=args.norm_window).std()
    score_z = score_raw / (score_std + 1e-12)

    base_exposure = _tiered_exposure(score_z, args.score_hi, args.score_mid)
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
            "score_raw": score_raw,
            "score_z": score_z,
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
