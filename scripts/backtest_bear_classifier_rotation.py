from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_forex_optimised import _stats


"""
Backtest BEAR-type defensive rotation against current gold fallback.

This is a research script. It does not alter production config.

Method:
- Baseline return = existing LPBot daily return with current gold sleeve logic.
- Classifier-gated return = baseline minus existing gold sleeve return, plus:
  - PANIC_BEAR major periods: VIXY long-vol ETF proxy.
  - INFLATION_BEAR major periods: USO energy/oil ETF proxy.
  - UNCLASSIFIED, GEOPOLITICAL_BEAR, and minor_dip periods: unchanged current
    gold sleeve return.

Classifier decisions are read from bear_period_classifications.csv and therefore
use only data known at bear_start. The chosen defensive asset is locked for the
entire BEAR period and is not reclassified mid-period.
"""


ACTIONABLE_ASSETS = {
    "PANIC_BEAR": {"proxy": "VIXY", "label": "vixy_long_vol"},
    "INFLATION_BEAR": {"proxy": "USO", "label": "uso_energy"},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest BEAR classifier defensive rotation.")
    p.add_argument("--daily", default="artifacts/backtest/extended_overlay_daily.csv")
    p.add_argument("--bear-input", default="artifacts/bear_classifier/bear_period_classifier_inputs.csv")
    p.add_argument("--classifications", default="artifacts/bear_classifier/bear_period_classifications.csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--defensive-cap", type=float, default=0.30)
    p.add_argument("--gross-cap", type=float, default=0.80)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="artifacts/bear_classifier/cache")
    p.add_argument("--out-summary", default="artifacts/bear_classifier/bear_classifier_rotation_summary.csv")
    p.add_argument("--out-daily", default="artifacts/bear_classifier/bear_classifier_rotation_daily.csv")
    p.add_argument("--out-periods", default="artifacts/bear_classifier/bear_classifier_rotation_periods.csv")
    return p.parse_args()


def _load_daily(path: Path, start: str, end: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    date_col = next((c for c in ["day", "timestamp", "date"] if c in d.columns), None)
    if date_col is None:
        raise SystemExit(f"{path} has no day/timestamp/date column.")
    d["day"] = pd.to_datetime(d[date_col], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").drop_duplicates("day", keep="last")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    d = d[(d["day"] >= start_ts) & (d["day"] <= end_ts)].copy()
    for col in [
        "combined_return",
        "return_mean_reversion",
        "gold_strategy_return",
        "gold_weight_exec",
        "alloc_eth",
        "alloc_btc",
        "eth_weight_exec",
        "btc_weight_exec",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    if "return_mean_reversion" in d.columns:
        d["baseline_current_return"] = d["return_mean_reversion"]
    else:
        d["baseline_current_return"] = d["combined_return"]
    if "gold_strategy_return" not in d.columns:
        d["gold_strategy_return"] = 0.0
    if "gold_weight_exec" not in d.columns:
        d["gold_weight_exec"] = 0.0
    if "alloc_eth" not in d.columns:
        d["alloc_eth"] = d.get("eth_weight_exec", 0.0)
    if "alloc_btc" not in d.columns:
        d["alloc_btc"] = d.get("btc_weight_exec", 0.0)
    return d.reset_index(drop=True)


def _download_prices(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache: Path, refresh: bool) -> pd.DataFrame:
    if cache.exists() and not refresh:
        out = pd.read_csv(cache)
        out["day"] = pd.to_datetime(out["day"], utc=True, errors="coerce").dt.floor("D")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out.dropna(subset=["day", "close"])

    import yfinance as yf

    raw = yf.download(
        symbol,
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No yfinance data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else "index"
    out = pd.DataFrame(
        {
            "day": pd.to_datetime(raw[date_col], utc=True, errors="coerce").dt.floor("D"),
            "close": pd.to_numeric(raw["Close"], errors="coerce"),
        }
    ).dropna(subset=["day", "close"]).sort_values("day").drop_duplicates("day", keep="last")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _asset_returns(daily: pd.DataFrame, prices: pd.DataFrame, name: str) -> pd.Series:
    px = prices.rename(columns={"close": f"{name}_close"})
    out = daily[["day"]].merge(px, on="day", how="left")
    out[f"{name}_close"] = out[f"{name}_close"].ffill()
    return out[f"{name}_close"].pct_change().fillna(0.0)


def _period_mask(days: pd.Series, start: str, end: str) -> pd.Series:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    return (days >= s) & (days <= e)


def _build_period_assignments(bear_input: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    periods = bear_input.copy()
    periods["assigned_bear_type"] = "UNCLASSIFIED"
    periods["classifier_proxy"] = "gold_fallback"
    periods["classifier_actionable"] = False

    cls = classifications[["bear_start", "assigned_bear_type"]].copy()
    periods = periods.drop(columns=["assigned_bear_type"], errors="ignore").merge(cls, on="bear_start", how="left")
    periods["assigned_bear_type"] = periods["assigned_bear_type"].fillna("UNCLASSIFIED")
    periods.loc[periods["bear_scale"].ne("major_bear"), "assigned_bear_type"] = "UNCLASSIFIED"

    for bear_type, meta in ACTIONABLE_ASSETS.items():
        mask = periods["assigned_bear_type"].eq(bear_type)
        periods.loc[mask, "classifier_proxy"] = meta["label"]
        periods.loc[mask, "classifier_actionable"] = True
    return periods


def _summarise_periods(daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in periods.iterrows():
        mask = _period_mask(daily["day"], p["bear_start"], p["bear_end"])
        g = daily[mask]
        if g.empty:
            continue
        rows.append(
            {
                "bear_start": p["bear_start"],
                "bear_end": p["bear_end"],
                "duration_days": int(p["duration_days"]),
                "bear_scale": p["bear_scale"],
                "assigned_bear_type": p["assigned_bear_type"],
                "classifier_proxy": p["classifier_proxy"],
                "baseline_period_return": float((1.0 + g["baseline_current_return"]).prod() - 1.0),
                "classifier_period_return": float((1.0 + g["classifier_return"]).prod() - 1.0),
                "classifier_minus_baseline": float((1.0 + g["classifier_return"]).prod() - (1.0 + g["baseline_current_return"]).prod()),
                "defensive_rotation_return": float((1.0 + g["classifier_defensive_return"]).prod() - 1.0),
                "baseline_gold_return": float((1.0 + g["gold_strategy_return"]).prod() - 1.0),
                "n_defensive_days": int((g["classifier_defensive_weight"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = _parse_args()
    daily = _load_daily(Path(args.daily), args.start, args.end)
    if daily.empty:
        raise SystemExit("No daily rows in requested backtest window.")

    bear_input = pd.read_csv(args.bear_input)
    classifications = pd.read_csv(args.classifications)
    periods = _build_period_assignments(bear_input, classifications)

    start_ts = daily["day"].min()
    end_ts = daily["day"].max()
    cache_dir = Path(args.cache_dir)
    vixy_ret = _asset_returns(daily, _download_prices("VIXY", start_ts, end_ts, cache_dir / "VIXY.csv", args.refresh), "vixy")
    uso_ret = _asset_returns(daily, _download_prices("USO", start_ts, end_ts, cache_dir / "USO.csv", args.refresh), "uso")

    daily["classifier_asset"] = "gold_fallback"
    daily["classifier_target_weight"] = 0.0
    daily["classifier_asset_return"] = 0.0

    used_cap = (daily["alloc_eth"].abs() + daily["alloc_btc"].abs()).clip(lower=0.0)
    eligible_weight = (args.gross_cap - used_cap).clip(lower=0.0, upper=args.defensive_cap)

    for _, p in periods.iterrows():
        mask = _period_mask(daily["day"], p["bear_start"], p["bear_end"])
        if not bool(p["classifier_actionable"]):
            continue
        if p["assigned_bear_type"] == "PANIC_BEAR":
            daily.loc[mask, "classifier_asset"] = "vixy_long_vol"
            daily.loc[mask, "classifier_asset_return"] = vixy_ret.loc[mask].to_numpy()
        elif p["assigned_bear_type"] == "INFLATION_BEAR":
            daily.loc[mask, "classifier_asset"] = "uso_energy"
            daily.loc[mask, "classifier_asset_return"] = uso_ret.loc[mask].to_numpy()
        daily.loc[mask, "classifier_target_weight"] = eligible_weight.loc[mask].to_numpy()

    daily["classifier_defensive_weight"] = daily["classifier_target_weight"].shift(1).fillna(0.0)
    prev_w = daily["classifier_defensive_weight"].shift(1).fillna(0.0)
    daily["classifier_turnover"] = (daily["classifier_defensive_weight"] - prev_w).abs()
    daily["classifier_cost"] = daily["classifier_turnover"] * (args.cost_bps / 10000.0)
    replacement = daily["classifier_defensive_weight"] * daily["classifier_asset_return"] - daily["classifier_cost"]

    fallback = daily["gold_strategy_return"]
    actionable_mask = daily["classifier_asset"].isin(["vixy_long_vol", "uso_energy"])
    daily["classifier_defensive_return"] = np.where(actionable_mask, replacement, fallback)
    daily["base_ex_gold_return"] = daily["baseline_current_return"] - daily["gold_strategy_return"]
    daily["classifier_return"] = daily["base_ex_gold_return"] + daily["classifier_defensive_return"]

    baseline_stats = _stats(daily["baseline_current_return"])
    classifier_stats = _stats(daily["classifier_return"])
    diff = classifier_stats["sharpe"] - baseline_stats["sharpe"]

    period_summary = _summarise_periods(daily, periods)
    actionable = period_summary[period_summary["classifier_proxy"].isin(["vixy_long_vol", "uso_energy"])]
    if actionable.empty:
        concentration = pd.DataFrame(columns=["bear_start", "classifier_minus_baseline", "share_of_total_improvement"])
    else:
        total_improvement = float(period_summary["classifier_minus_baseline"].sum())
        concentration = actionable[["bear_start", "classifier_minus_baseline"]].copy()
        concentration["share_of_total_improvement"] = (
            concentration["classifier_minus_baseline"] / total_improvement if abs(total_improvement) > 1e-12 else np.nan
        )

    summary = pd.DataFrame(
        [
            {"config": "baseline_gold_fallback", **baseline_stats},
            {"config": "classifier_panic_inflation", **classifier_stats},
            {
                "config": "delta_classifier_minus_baseline",
                "return": classifier_stats["return"] - baseline_stats["return"],
                "cagr": classifier_stats["cagr"] - baseline_stats["cagr"],
                "sharpe": diff,
                "raw_sharpe": classifier_stats["raw_sharpe"] - baseline_stats["raw_sharpe"],
                "maxdd": classifier_stats["maxdd"] - baseline_stats["maxdd"],
                "ann_vol": classifier_stats["ann_vol"] - baseline_stats["ann_vol"],
            },
        ]
    )

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    daily.to_csv(args.out_daily, index=False)
    period_summary.to_csv(args.out_periods, index=False)

    print("=" * 100)
    print("BEAR CLASSIFIER ROTATION BACKTEST")
    print("=" * 100)
    print(f"Window: {daily['day'].min().date()} to {daily['day'].max().date()}")
    print("Panic proxy: VIXY long-vol ETF. Inflation proxy: USO oil/energy ETF.")
    print("Minor dips and unclassified major bears: unchanged current gold fallback.")
    print()
    for _, row in summary.iterrows():
        print(
            f"{row['config']:<34} "
            f"Sharpe {row['sharpe']:.3f}  CAGR {row['cagr'] * 100:.2f}%  "
            f"MaxDD {row['maxdd'] * 100:.2f}%  Return {row['return'] * 100:.2f}%"
        )
    print()
    print("Actionable classifier periods:")
    if actionable.empty:
        print("  none")
    else:
        print(
            actionable[
                [
                    "bear_start",
                    "bear_end",
                    "assigned_bear_type",
                    "classifier_proxy",
                    "baseline_period_return",
                    "classifier_period_return",
                    "classifier_minus_baseline",
                    "n_defensive_days",
                ]
            ].to_string(index=False)
        )
    print()
    print("Improvement concentration:")
    if concentration.empty:
        print("  none")
    else:
        print(concentration.to_string(index=False))
    print()
    print(f"Saved summary: {args.out_summary}")
    print(f"Saved daily:   {args.out_daily}")
    print(f"Saved periods: {args.out_periods}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
