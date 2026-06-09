from __future__ import annotations

import argparse

import pandas as pd

from lpbot.paper.paper_trade_v1.paper_log import append_rows, read_last_timestamp
from lpbot.paper.paper_trade_v1.paper_tick import compute_paper_rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append paper-trade diagnostics log.")
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--volume-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--fee-tier", type=float, required=True)
    p.add_argument("--trade-cost-bps", type=float, default=0.0)
    p.add_argument("--pool-tvl-usd", type=float, required=True)
    p.add_argument("--in-range-frac", type=float, required=True)
    p.add_argument("--range-sigma", type=float, required=True)
    p.add_argument("--range-sigma-ref", type=float, required=True)
    p.add_argument("--min-in-range-frac", type=float, required=True)
    p.add_argument("--max-in-range-frac", type=float, required=True)
    p.add_argument("--churn-k", type=float, required=True)
    p.add_argument("--il-k", type=float, required=True)
    p.add_argument("--lp-vol-on", type=float, required=True)
    p.add_argument("--lp-scale", type=float, required=True)
    p.add_argument("--lp-weight-max", type=float, required=True)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)
    p.add_argument("--log-csv", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exposure = pd.read_csv(args.exposure_csv)
    price = pd.read_csv(args.price_csv)
    volume = pd.read_csv(args.volume_csv)

    rows = compute_paper_rows(
        exposure,
        price,
        volume,
        bar_minutes=args.bar_minutes,
        fee_tier=args.fee_tier,
        trade_cost_bps=args.trade_cost_bps,
        pool_tvl_usd=args.pool_tvl_usd,
        in_range_frac=args.in_range_frac,
        range_sigma=args.range_sigma,
        range_sigma_ref=args.range_sigma_ref,
        min_in_range_frac=args.min_in_range_frac,
        max_in_range_frac=args.max_in_range_frac,
        churn_k=args.churn_k,
        il_k=args.il_k,
        lp_vol_on=args.lp_vol_on,
        lp_scale=args.lp_scale,
        lp_weight_max=args.lp_weight_max,
        lp_min_on_bars=args.lp_min_on_bars,
        lp_cooldown_bars=args.lp_cooldown_bars,
    )

    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")

    last_ts = read_last_timestamp(args.log_csv)
    if last_ts is not None:
        rows = rows[rows["timestamp"] > last_ts]

    appended_rows = len(rows)
    last_timestamp = None
    lp_on_pct = float("nan")
    churn_mean = float("nan")
    if appended_rows > 0:
        last_timestamp = rows["timestamp"].iloc[-1]
        lp_on_pct = float(rows["lp_on"].mean()) * 100
        churn_mean = float(rows["churn_intensity"].mean())
        append_rows(args.log_csv, rows)
    else:
        last_timestamp = last_ts

    print(f"appended_rows={appended_rows}")
    print(f"last_timestamp={last_timestamp}")
    print(f"lp_on_pct_appended={lp_on_pct}")
    print(f"mean_churn_intensity_appended={churn_mean}")


if __name__ == "__main__":
    main()
