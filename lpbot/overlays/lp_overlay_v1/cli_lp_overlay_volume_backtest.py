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
        return {"cagr": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    bars_per_year = _bars_per_year(bar_minutes)
    n_bars = len(r)
    equity = np.exp(r.cumsum())
    cagr = float(equity.iloc[-1] ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else float("nan")
    ann_vol = float(r.std()) * np.sqrt(bars_per_year)
    sharpe = cagr / ann_vol if ann_vol > 0 else float("nan")
    mdd = _max_drawdown(equity)
    return {"cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--volume-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--fee-tier", type=float, required=True)
    p.add_argument("--in-range-frac", type=float, required=True)
    p.add_argument("--range-sigma-ref", type=float, default=2.0)
    p.add_argument("--min-in-range-frac", type=float, default=0.05)
    p.add_argument("--max-in-range-frac", type=float, default=0.25)
    p.add_argument("--pool-tvl-usd", type=float, default=None)
    p.add_argument("--tvl-csv", default=None)
    p.add_argument("--il-k", type=float, required=True)
    p.add_argument("--churn-k", type=float, default=0.0)
    p.add_argument("--tightness-exp", type=float, default=1.0)
    p.add_argument("--range-sigma", type=float, default=2.0)
    p.add_argument("--lp-vol-on", type=float, required=True)
    p.add_argument("--lp-scale", type=float, required=True)
    p.add_argument("--lp-weight-max", type=float, required=True)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exposure = pd.read_csv(Path(args.exposure_csv))
    prices = pd.read_csv(Path(args.price_csv))
    volume = pd.read_csv(Path(args.volume_csv))

    required_exp = {"timestamp", "weight", "sigma_ann_smooth", "gate"}
    missing_exp = required_exp - set(exposure.columns)
    if missing_exp:
        raise SystemExit(f"exposure-csv missing columns: {sorted(missing_exp)}")
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        raise SystemExit("price-csv must contain columns: timestamp, close")
    if "timestamp" not in volume.columns or "volume_usd" not in volume.columns:
        raise SystemExit("volume-csv must contain columns: timestamp, volume_usd")

    if args.pool_tvl_usd is None and args.tvl_csv is None:
        raise SystemExit("Provide --pool-tvl-usd or --tvl-csv.")

    exposure["timestamp"] = pd.to_datetime(exposure["timestamp"], utc=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    volume["timestamp"] = pd.to_datetime(volume["timestamp"], utc=True)

    exposure = exposure.sort_values("timestamp")
    prices = prices.sort_values("timestamp")
    volume = volume.sort_values("timestamp")

    merged = pd.merge(prices, exposure, on="timestamp", how="inner")
    merged = pd.merge(merged, volume[["timestamp", "volume_usd"]], on="timestamp", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping timestamps between exposure, price, and volume data.")

    if "close" not in merged.columns:
        if "close_x" in merged.columns:
            merged = merged.rename(columns={"close_x": "close"})
        elif "close_y" in merged.columns:
            merged = merged.rename(columns={"close_y": "close"})
    if "close_x" in merged.columns:
        merged = merged.drop(columns=["close_x"])
    if "close_y" in merged.columns:
        merged = merged.drop(columns=["close_y"])

    if args.tvl_csv is not None:
        tvl = pd.read_csv(Path(args.tvl_csv))
        if "timestamp" not in tvl.columns or "tvl_usd" not in tvl.columns:
            raise SystemExit("tvl-csv must contain columns: timestamp, tvl_usd")
        tvl["timestamp"] = pd.to_datetime(tvl["timestamp"], utc=True)
        tvl = tvl.sort_values("timestamp")
        merged = pd.merge(merged, tvl[["timestamp", "tvl_usd"]], on="timestamp", how="inner")
        if merged.empty:
            raise SystemExit("No overlapping timestamps after merging tvl data.")
        pool_tvl = merged["tvl_usd"]
    else:
        pool_tvl = float(args.pool_tvl_usd)

    merged = merged.sort_values("timestamp")

    base_df = compute_lp_overlay_returns(
        merged,
        bar_minutes=args.bar_minutes,
        fee_rate_ann=0.0,
        il_k=args.il_k,
        lp_vol_on=args.lp_vol_on,
        lp_scale=args.lp_scale,
        lp_weight_max=args.lp_weight_max,
        lp_min_on_bars=args.lp_min_on_bars,
        lp_cooldown_bars=args.lp_cooldown_bars,
    )

    base_df = base_df.merge(
        merged[["timestamp", "close", "volume_usd"]], on="timestamp", how="left"
    )
    if args.tvl_csv is not None:
        base_df = base_df.merge(merged[["timestamp", "tvl_usd"]], on="timestamp", how="left")

    lp_weight_lag = base_df["lp_weight"].shift(1).fillna(0.0)
    volume_usd = pd.to_numeric(base_df["volume_usd"], errors="coerce").fillna(0.0)
    if args.tvl_csv is not None:
        pool_tvl_usd = pd.to_numeric(base_df["tvl_usd"], errors="coerce")
    else:
        pool_tvl_usd = float(pool_tvl)

    in_range_frac_eff = float(args.in_range_frac) * (
        float(args.range_sigma_ref) / float(args.range_sigma)
    )
    in_range_frac_eff = float(
        np.clip(
            in_range_frac_eff,
            float(args.min_in_range_frac),
            float(args.max_in_range_frac),
        )
    )
    fee_per_bar = args.fee_tier * volume_usd * in_range_frac_eff / pool_tvl_usd
    fee_per_bar = pd.to_numeric(fee_per_bar, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fee_r = lp_weight_lag * fee_per_bar

    il_r = pd.to_numeric(base_df["il_r"], errors="coerce").fillna(0.0)
    bars_per_year = 365 * 24 * (60 / args.bar_minutes)
    sigma_bar = pd.to_numeric(base_df["sigma_ann_smooth"], errors="coerce").fillna(0.0)
    sigma_bar = sigma_bar / np.sqrt(bars_per_year)
    move = base_df["r"].abs().fillna(0.0)
    band = float(args.range_sigma) * sigma_bar
    churn_intensity = (move / (band + 1e-12)) - 1.0
    churn_intensity = churn_intensity.clip(lower=0.0)
    churn_r = -lp_weight_lag * float(args.churn_k) * churn_intensity
    core_r = pd.to_numeric(base_df["core_r"], errors="coerce").fillna(0.0)
    overlay_r = fee_r + il_r + churn_r
    combined_r = core_r + overlay_r

    base_df["fee_r"] = fee_r
    base_df["in_range_frac_eff"] = in_range_frac_eff
    base_df["churn_intensity"] = churn_intensity
    base_df["churn_r"] = churn_r
    base_df["overlay_r"] = overlay_r
    base_df["combined_r"] = combined_r

    base_df = base_df.dropna(subset=["r"])
    base_df["core_equity"] = np.exp(base_df["core_r"].cumsum())
    base_df["combined_equity"] = np.exp(base_df["combined_r"].cumsum())

    core_stats = _stats(base_df["core_r"], args.bar_minutes)
    combined_stats = _stats(base_df["combined_r"], args.bar_minutes)

    lp_on = base_df["lp_on"].astype(bool)
    lp_on_pct = float(lp_on.mean() * 100.0)
    if lp_on.any():
        mean_fee_r_on = float(base_df.loc[lp_on, "fee_r"].mean())
        mean_il_r_on = float(base_df.loc[lp_on, "il_r"].mean())
        mean_churn_r_on = float(base_df.loc[lp_on, "churn_r"].mean())
        mean_churn_intensity_on = float(base_df.loc[lp_on, "churn_intensity"].mean())
        mean_in_range_eff_on = float(base_df.loc[lp_on, "in_range_frac_eff"].mean())
        mean_overlay_r_on = float(base_df.loc[lp_on, "overlay_r"].mean())
    else:
        mean_fee_r_on = float("nan")
        mean_il_r_on = float("nan")
        mean_churn_r_on = float("nan")
        mean_churn_intensity_on = float("nan")
        mean_in_range_eff_on = float("nan")
        mean_overlay_r_on = float("nan")

    print("Core strategy:")
    print(
        f"CAGR={core_stats['cagr']:.4f} | ann_vol={core_stats['ann_vol']:.4f} | "
        f"Sharpe={core_stats['sharpe']:.4f} | max_drawdown={core_stats['max_drawdown']:.4f}"
    )
    print("Core + LP overlay (volume fees):")
    print(
        f"CAGR={combined_stats['cagr']:.4f} | ann_vol={combined_stats['ann_vol']:.4f} | "
        f"Sharpe={combined_stats['sharpe']:.4f} | max_drawdown={combined_stats['max_drawdown']:.4f}"
    )
    print(
        f"lp_on_pct={lp_on_pct:.2f}% | fee_on={mean_fee_r_on:.6f} | "
        f"il_on={mean_il_r_on:.6f} | churn_intensity_on={mean_churn_intensity_on:.6f} | "
        f"in_range_eff_on={mean_in_range_eff_on:.6f} | churn_on={mean_churn_r_on:.6f} | "
        f"overlay_on={mean_overlay_r_on:.6f}"
    )

    out_cols = [
        "timestamp",
        "close",
        "volume_usd",
        "r",
        "weight",
        "sigma_ann_smooth",
        "gate",
        "lp_on",
        "lp_weight",
        "core_r",
        "fee_r",
        "in_range_frac_eff",
        "churn_r",
        "il_r",
        "overlay_r",
        "combined_r",
        "core_equity",
        "combined_equity",
    ]
    out_df = base_df[out_cols]

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")


if __name__ == "__main__":
    main()
