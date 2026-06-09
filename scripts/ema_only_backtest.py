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


def _apply_rebalance_limits(target: pd.Series, min_delta: float, max_dw: float) -> pd.Series:
    w = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, t in enumerate(target):
        if abs(t - prev) < min_delta:
            w[i] = prev
            continue
        if max_dw > 0:
            step = np.clip(t - prev, -max_dw, max_dw)
            prev = prev + step
        else:
            prev = t
        w[i] = prev
    return pd.Series(w, index=target.index)


def main() -> None:
    p = argparse.ArgumentParser(description="EMA-only trend backtest with costs.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-span", type=int, default=200)
    p.add_argument("--hyst-on-pct", type=float, default=0.001)
    p.add_argument("--hyst-off-pct", type=float, default=0.001)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--min-rebalance-delta", type=float, default=0.05)
    p.add_argument("--max-dw-per-bar", type=float, default=0.10)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--last-days", type=int, default=365)
    p.add_argument("--out", default="artifacts/backtest/ema_only_backtest.csv")
    args = p.parse_args()

    price_path = Path(args.price_5m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_price(price_path)
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    ema = close.ewm(span=args.ema_span, adjust=False).mean()

    state = np.zeros(len(df), dtype=float)
    cur = 0.0
    for i in range(len(df)):
        if cur > 0 and close.iat[i] < ema.iat[i] * (1 - args.hyst_off_pct):
            cur = 0.0
        elif cur == 0 and close.iat[i] > ema.iat[i] * (1 + args.hyst_on_pct):
            cur = 1.0
        state[i] = cur

    sigma = _compute_sigma_ann(r, args.bar_minutes, args.vol_window)
    weight_raw = (args.target_vol / sigma).replace([np.inf, -np.inf], np.nan)
    weight_raw = weight_raw.fillna(0.0).clip(lower=0.0, upper=args.w_max)
    target_weight = weight_raw * state
    weight = _apply_rebalance_limits(
        target_weight,
        min_delta=args.min_rebalance_delta,
        max_dw=args.max_dw_per_bar,
    )

    w_prev = weight.shift(1).fillna(0.0)
    trade_cost_r = -abs(weight - w_prev) * (args.trade_cost_bps / 10000.0)
    core_r = w_prev * r + trade_cost_r
    eq = np.exp(core_r.cumsum())

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "close": close,
            "r": r,
            "ema": ema,
            "state": state,
            "sigma_ann": sigma,
            "weight_raw": weight_raw,
            "target_weight": target_weight,
            "weight": weight,
            "trade_cost_r": trade_cost_r,
            "core_r": core_r,
            "eq": eq,
        }
    )

    if args.last_days and args.last_days > 0:
        end = out["timestamp"].iloc[-1]
        start = end - pd.Timedelta(days=args.last_days)
        out = out[out["timestamp"] >= start].copy()

    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
