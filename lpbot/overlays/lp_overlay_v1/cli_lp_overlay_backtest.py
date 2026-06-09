from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns


def _bars_per_year(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    return 365 * 24 * (60 / bar_minutes)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _stats(returns: pd.Series, bar_minutes: int) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    bars_per_year = _bars_per_year(bar_minutes)
    equity = np.exp(r.cumsum())
    years = len(r) / bars_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    std = float(r.std())
    ann_vol = std * np.sqrt(bars_per_year)
    sharpe = float(r.mean() / std) * np.sqrt(bars_per_year) if std > 0 else float("nan")
    mdd = _max_drawdown(equity)
    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--fee-rate-ann", type=float, required=True)
    p.add_argument("--il-k", type=float, required=True)
    p.add_argument("--lp-vol-on", type=float, required=True)
    p.add_argument("--lp-scale", type=float, required=True)
    p.add_argument("--lp-weight-max", type=float, required=True)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)
    p.add_argument("--out", default=None)
    p.add_argument("--print-rows", action="store_true", help="Print full output rows.")
    args = p.parse_args()

    exposure = pd.read_csv(Path(args.exposure_csv))
    prices = pd.read_csv(Path(args.price_csv))

    required_exp = {"timestamp", "weight", "sigma_ann_smooth", "gate"}
    missing_exp = required_exp - set(exposure.columns)
    if missing_exp:
        raise SystemExit(f"exposure-csv missing columns: {sorted(missing_exp)}")
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        raise SystemExit("price-csv must contain columns: timestamp, close")

    exposure["timestamp"] = pd.to_datetime(exposure["timestamp"], utc=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    exposure = exposure.sort_values("timestamp")
    prices = prices.sort_values("timestamp")

    merged = pd.merge(prices, exposure, on="timestamp", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping timestamps between exposure and price data.")
    if "close" not in merged.columns:
        if "close_x" in merged.columns:
            merged = merged.rename(columns={"close_x": "close"})
        elif "close_y" in merged.columns:
            merged = merged.rename(columns={"close_y": "close"})
    if "close_x" in merged.columns:
        merged = merged.drop(columns=["close_x"])
    if "close_y" in merged.columns:
        merged = merged.drop(columns=["close_y"])

    out_df = compute_lp_overlay_returns(
        merged,
        bar_minutes=args.bar_minutes,
        fee_rate_ann=args.fee_rate_ann,
        il_k=args.il_k,
        lp_vol_on=args.lp_vol_on,
        lp_scale=args.lp_scale,
        lp_weight_max=args.lp_weight_max,
        lp_min_on_bars=args.lp_min_on_bars,
        lp_cooldown_bars=args.lp_cooldown_bars,
    )

    core_equity = np.exp(out_df["core_r"].fillna(0.0).cumsum())
    combined_equity = np.exp(out_df["combined_r"].fillna(0.0).cumsum())
    out_df["core_equity"] = core_equity
    out_df["combined_equity"] = combined_equity

    core_stats = _stats(out_df["core_r"], args.bar_minutes)
    combined_stats = _stats(out_df["combined_r"], args.bar_minutes)

    print("Core strategy:")
    print(
        f"CAGR={core_stats['cagr']:.4f} | ann_vol={core_stats['ann_vol']:.4f} | "
        f"Sharpe={core_stats['sharpe']:.4f} | max_drawdown={core_stats['max_drawdown']:.4f}"
    )
    print("Core + LP overlay:")
    print(
        f"CAGR={combined_stats['cagr']:.4f} | ann_vol={combined_stats['ann_vol']:.4f} | "
        f"Sharpe={combined_stats['sharpe']:.4f} | max_drawdown={combined_stats['max_drawdown']:.4f}"
    )

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    if args.print_rows:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
