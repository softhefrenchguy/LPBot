from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def _sanitize_float(val: float) -> str:
    s = f"{val:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--lp-vol-on", type=float, required=True)
    p.add_argument("--lp-scale", type=float, required=True)
    p.add_argument("--lp-weight-max", type=float, required=True)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)
    p.add_argument("--il-k-list", required=True)
    p.add_argument("--fee-rate-ann-list", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    il_k_list = _parse_float_list(args.il_k_list)
    fee_list = _parse_float_list(args.fee_rate_ann_list)
    if not il_k_list or not fee_list:
        raise SystemExit("Provide non-empty --il-k-list and --fee-rate-ann-list.")

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
    merged = merged.sort_values("timestamp")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fee_rate_ann in fee_list:
        for il_k in il_k_list:
            out_df = compute_lp_overlay_returns(
                merged,
                bar_minutes=args.bar_minutes,
                fee_rate_ann=fee_rate_ann,
                il_k=il_k,
                lp_vol_on=args.lp_vol_on,
                lp_scale=args.lp_scale,
                lp_weight_max=args.lp_weight_max,
                lp_min_on_bars=args.lp_min_on_bars,
                lp_cooldown_bars=args.lp_cooldown_bars,
            )

            out_df = out_df.sort_values("timestamp")
            out_df = out_df.dropna(subset=["r"])
            for col in ["core_r", "combined_r", "fee_r", "il_r", "overlay_r"]:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0.0)

            core_equity = np.exp(out_df["core_r"].cumsum())
            combined_equity = np.exp(out_df["combined_r"].cumsum())
            out_df["core_equity"] = core_equity
            out_df["combined_equity"] = combined_equity

            core_stats = _stats(out_df["core_r"], args.bar_minutes)
            combined_stats = _stats(out_df["combined_r"], args.bar_minutes)

            lp_on = out_df["lp_on"].astype(bool)
            lp_on_pct = float(lp_on.mean() * 100.0)
            if lp_on.any():
                mean_fee_r_on = float(out_df.loc[lp_on, "fee_r"].mean())
                mean_il_r_on = float(out_df.loc[lp_on, "il_r"].mean())
                mean_overlay_r_on = float(out_df.loc[lp_on, "overlay_r"].mean())
            else:
                mean_fee_r_on = float("nan")
                mean_il_r_on = float("nan")
                mean_overlay_r_on = float("nan")

            print(
                f"fee={fee_rate_ann:.4f} il_k={il_k:.4f} | "
                f"core: CAGR={core_stats['cagr']:.4f} ann_vol={core_stats['ann_vol']:.4f} "
                f"Sharpe={core_stats['sharpe']:.4f} mdd={core_stats['max_drawdown']:.4f} | "
                f"combined: CAGR={combined_stats['cagr']:.4f} ann_vol={combined_stats['ann_vol']:.4f} "
                f"Sharpe={combined_stats['sharpe']:.4f} mdd={combined_stats['max_drawdown']:.4f} | "
                f"lp_on_pct={lp_on_pct:.2f}% fee_on={mean_fee_r_on:.6f} "
                f"il_on={mean_il_r_on:.6f} overlay_on={mean_overlay_r_on:.6f}"
            )

            fee_tag = _sanitize_float(fee_rate_ann)
            il_tag = _sanitize_float(il_k)
            out_path = out_dir / f"lp_overlay_fee_{fee_tag}_ilk_{il_tag}.csv"
            out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
