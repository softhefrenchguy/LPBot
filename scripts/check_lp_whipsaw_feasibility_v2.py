from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNISWAP_V3_LAUNCH = pd.Timestamp("2021-05-05", tz="UTC")


def _exact_v2_il(price_ratio: float) -> float:
    if price_ratio <= 0 or not np.isfinite(price_ratio):
        return np.nan
    return 2.0 * np.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="LP fee-vs-IL for whipsaw stretches, v2: drops pre-Uniswap-v3 stretches entirely (does not silently include them under a v3 fee assumption), and reports breakeven fee APY per stretch instead of asserting a specific historical rate (real historical volume data was not accessible -- Graph API needs a paid/GRT-funded key, Dune has no free API tier, and DeFiLlama's per-pool historical chart endpoint was unreachable after ~10 attempts across two sessions).")
    ap.add_argument("--shape-csv", default="artifacts/backtest/flat_stretch_price_shape.csv")
    ap.add_argument("--out", default="artifacts/backtest/lp_whipsaw_feasibility_v2.csv")
    args = ap.parse_args()

    shapes = pd.read_csv(args.shape_csv)
    whipsaw = shapes[shapes["category"] == "failed_trend_whipsaw"].copy()
    whipsaw["start"] = pd.to_datetime(whipsaw["start"], utc=True)
    whipsaw["end"] = pd.to_datetime(whipsaw["end"], utc=True)
    whipsaw["pre_v3"] = whipsaw["start"] < UNISWAP_V3_LAUNCH

    dropped = whipsaw[whipsaw["pre_v3"]]
    kept = whipsaw[~whipsaw["pre_v3"]].copy()

    print("=" * 120)
    print("LP FEE vs IL, v2 -- pre-Uniswap-v3 stretches DROPPED (not silently included)")
    print("=" * 120)
    print(f"Dropped ({len(dropped)}, all predate the 2021-05-05 Uniswap v3 launch -- correction: this is 3, not the 2 in the original ask):")
    for _, r in dropped.iterrows():
        print(f"  {r['start'].date()} -> {r['end'].date()}")
    print(f"\nRemaining stretches evaluated: {len(kept)}")
    print()
    print("Data note: real historical Uniswap v3 volume was NOT obtained. Graph gateway requires a funded/GRT-based key")
    print("under the post-hosted-service Graph Network model (100k free queries/mo, then ~$2/100k -- but needs account +")
    print("billing setup beyond a simple API key); Dune's free tier has no API access at all; DeFiLlama's live /pools")
    print("snapshot worked once (apyBase ~11%, apyBase7d ~28% for the largest ETH/USDC v3 pool) but its per-pool historical")
    print("chart endpoint was unreachable (connection reset) across ~10 retries. Reporting BREAKEVEN fee APY per stretch")
    print("instead of asserting a specific historical rate -- this is the defensible framing given the data gap.")
    print()

    rows = []
    for _, r in kept.iterrows():
        length_days = int(r["length_days"])
        years = length_days / 365.25
        net_pct = float(r["eth_net_pct"])
        price_ratio = 1.0 + net_pct / 100.0
        il_full = _exact_v2_il(price_ratio)
        il_conc = il_full * 4.0  # ~4x amplification, moderate concentration (il_k 0.5 vs 0.125 full-range)

        # breakeven APY: fee_apy * years + il = 0  =>  fee_apy = -il / years
        breakeven_full = -il_full / years if years > 0 else np.nan
        breakeven_conc = -il_conc / years if years > 0 else np.nan

        print(f"{r['start'].date()} -> {r['end'].date()} ({length_days}d, ETH net {net_pct:+.1f}%)")
        print(f"  IL: full-range {il_full*100:.2f}%   moderately concentrated {il_conc*100:.2f}%")
        print(f"  Breakeven fee APY needed: full-range {breakeven_full*100:.1f}%   concentrated {breakeven_conc*100:.1f}%")
        rows.append({
            "start": str(r["start"].date()), "end": str(r["end"].date()), "length_days": length_days, "eth_net_pct": net_pct,
            "il_full_range_pct": il_full * 100, "il_concentrated_pct": il_conc * 100,
            "breakeven_apy_full_range_pct": breakeven_full * 100, "breakeven_apy_concentrated_pct": breakeven_conc * 100,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Breakeven APY range (full-range): {out_df['breakeven_apy_full_range_pct'].min():.1f}% to {out_df['breakeven_apy_full_range_pct'].max():.1f}%, median {out_df['breakeven_apy_full_range_pct'].median():.1f}%")
    print(f"Breakeven APY range (concentrated): {out_df['breakeven_apy_concentrated_pct'].min():.1f}% to {out_df['breakeven_apy_concentrated_pct'].max():.1f}%, median {out_df['breakeven_apy_concentrated_pct'].median():.1f}%")
    n_low_bar = int((out_df["breakeven_apy_full_range_pct"] < 10).sum())
    n_total = len(out_df)
    print(f"\n{n_low_bar}/{n_total} stretches need less than 10% fee APY (full-range) to breakeven -- a bar that's been cleared by")
    print("the real live-checked DeFiLlama snapshot (11-28%) and is broadly consistent with widely-reported historical")
    print("Uniswap v3 ETH/USDC fee APY ranges, but I could not verify this stretch-by-stretch against real historical data.")
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
