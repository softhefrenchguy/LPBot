from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _annualization_factor(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return np.sqrt(bars_per_year)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _cagr_approx(equity: pd.Series, bar_minutes: int) -> float:
    if equity.empty:
        return float("nan")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    years = len(equity) / bars_per_year
    if years <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exposure = pd.read_csv(Path(args.exposure_csv))
    prices = pd.read_csv(Path(args.price_csv))

    if "timestamp" not in exposure.columns or "weight" not in exposure.columns:
        raise SystemExit("exposure-csv must contain columns: timestamp, weight")
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        raise SystemExit("price-csv must contain columns: timestamp, close")

    exposure["timestamp"] = pd.to_datetime(exposure["timestamp"], utc=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    exposure = exposure.sort_values("timestamp")
    prices = prices.sort_values("timestamp")

    df = pd.merge(prices, exposure[["timestamp", "weight"]], on="timestamp", how="left")
    df = df.sort_values("timestamp")
    if df.empty:
        raise SystemExit("No price data available after merge.")

    df["r"] = np.log(df["close"] / df["close"].shift(1))
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["strat_r"] = df["weight"].shift(1).fillna(0.0) * df["r"]
    df = df.dropna(subset=["r", "strat_r"])

    df["equity"] = np.exp(df["strat_r"].cumsum())

    peak = df["equity"].cummax()
    drawdown = df["equity"] / peak - 1.0

    ann_factor = _annualization_factor(args.bar_minutes)
    strat_r_std = float(df["strat_r"].std())
    ann_vol = strat_r_std * ann_factor
    sharpe = float(df["strat_r"].mean() / strat_r_std) * ann_factor if strat_r_std > 0 else float("nan")
    mdd = _max_drawdown(df["equity"])
    cagr = _cagr_approx(df["equity"], args.bar_minutes)
    avg_weight = float(df["weight"].mean())
    dd_mask = drawdown < -0.3
    avg_weight_dd = float(df.loc[dd_mask, "weight"].mean()) if dd_mask.any() else float("nan")

    trough_idx = drawdown.idxmin()
    if pd.isna(trough_idx):
        mdd_start = None
        mdd_end = None
    else:
        mdd_end = df.loc[trough_idx, "timestamp"]
        peak_idx = df.loc[:trough_idx, "equity"].idxmax()
        mdd_start = df.loc[peak_idx, "timestamp"] if not pd.isna(peak_idx) else None

    print(f"ann_vol: {ann_vol:.6f}")
    print(f"sharpe: {sharpe:.4f}")
    print(f"max_drawdown: {mdd:.4f}")
    print(f"cagr: {cagr:.4f}")
    print(f"avg_weight: {avg_weight:.6f}")
    print(f"avg_weight_dd_lt_30pct: {avg_weight_dd:.6f}")
    if mdd_start is not None and mdd_end is not None:
        print(f"mdd_range: {mdd_start} -> {mdd_end}")

    bars_per_year = 365 * 24 * (60 / args.bar_minutes)
    df["year"] = df["timestamp"].dt.year
    yearly_rows = []
    for year, g in df.groupby("year"):
        if g.empty:
            continue
        strat_r = g["strat_r"]
        eq_year = np.exp(strat_r.cumsum())
        years = len(g) / bars_per_year
        cagr_y = float(eq_year.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
        std_y = float(strat_r.std())
        ann_vol_y = std_y * np.sqrt(bars_per_year)
        sharpe_y = float(strat_r.mean() / std_y) * np.sqrt(bars_per_year) if std_y > 0 else float("nan")
        mdd_y = _max_drawdown(eq_year)
        avg_w_y = float(g["weight"].mean())
        pct_zero_y = float((g["weight"] == 0).mean() * 100.0)
        yearly_rows.append(
            {
                "year": int(year),
                "cagr": cagr_y,
                "ann_vol": ann_vol_y,
                "sharpe": sharpe_y,
                "max_drawdown": mdd_y,
                "avg_weight": avg_w_y,
                "pct_weight_zero": pct_zero_y,
            }
        )

    if yearly_rows:
        yearly_df = pd.DataFrame(yearly_rows).set_index("year")
        print("\nYearly performance:")
        print(yearly_df.to_string())
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        yearly_df.to_csv(out_dir / "yearly_stats.csv")

    out_df = df[["timestamp", "weight", "r", "strat_r", "equity"]]
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
