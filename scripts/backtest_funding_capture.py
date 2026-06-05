from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_btc_full_stack import _download_binance_daily, classify_regime_v2_on_btc, perf


def load_trend_returns(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "day" in d.columns:
        d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    elif "timestamp" in d.columns:
        d["day"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.floor("D")
    else:
        raise ValueError(f"{path} must contain day or timestamp")
    ret_col = "combined_return" if "combined_return" in d.columns else "strategy_return"
    d["trend_return"] = pd.to_numeric(d[ret_col], errors="coerce").fillna(0.0)
    return d[["day", "trend_return"]].dropna(subset=["day"]).copy()


def load_daily_funding(path: Path, start: str, end: str) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["timestamp", "funding_rate"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["funding_rate"] = pd.to_numeric(d["funding_rate"], errors="coerce")
    d = d.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp")
    d = d[(d["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (d["timestamp"] <= pd.Timestamp(end, tz="UTC"))].copy()

    # The feature file is 5-minute bars with the current funding value repeated.
    # Use one value per 8-hour bucket to approximate actual funding payments.
    d["funding_bucket"] = d["timestamp"].dt.floor("8h")
    buckets = d.groupby("funding_bucket", as_index=False)["funding_rate"].last()
    buckets["day"] = buckets["funding_bucket"].dt.floor("D")
    daily = buckets.groupby("day", as_index=False)["funding_rate"].sum().rename(columns={"funding_rate": "funding_daily"})
    return daily


def load_eth_daily(start: str, end: str) -> pd.DataFrame:
    d = _download_binance_daily("ETHUSDC", start, end)
    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH market-neutral funding capture backtest")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_6y.csv")
    ap.add_argument("--trend-daily-csv", default="artifacts/backtest/eth_btc_portfolio_daily_20bps.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/funding_capture_summary.csv")
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/funding_capture_daily.csv")
    args = ap.parse_args()

    eth = load_eth_daily(args.start, args.end)
    funding = load_daily_funding(Path(args.perp_csv), args.start, args.end)
    d = eth.merge(funding, on="day", how="left").sort_values("day")
    d["funding_daily"] = pd.to_numeric(d["funding_daily"], errors="coerce").fillna(0.0)

    active = 0
    target = []
    for f in d["funding_daily"]:
        if active == 0 and f > 0.0003:
            active = 1
        elif active == 1 and f < 0.0001:
            active = 0
        target.append(active)
    d["position_target"] = target
    d["position_exec"] = d["position_target"].shift(1).fillna(0.0)
    prev = d["position_exec"].shift(1).fillna(0.0)
    d["turnover"] = (d["position_exec"] - prev).abs()

    # Market-neutral spot/perp hedge: price return mostly nets out. Model keeps
    # funding earned while active, entry/exit trading cost, and daily basis drag.
    d["entry_exit_cost"] = d["turnover"] * 0.002
    d["basis_drag"] = d["position_exec"] * 0.0002
    d["strategy_return"] = d["position_exec"] * d["funding_daily"] - d["entry_exit_cost"] - d["basis_drag"]
    d["equity"] = (1.0 + d["strategy_return"]).cumprod()

    m = perf(d["strategy_return"])
    tim = float((d["position_exec"] > 0).mean())
    avg_funding = float(d.loc[d["position_exec"] > 0, "funding_daily"].mean()) if (d["position_exec"] > 0).any() else np.nan
    regime_funding = d.groupby("regime_v2")["funding_daily"].mean().to_dict()

    trend = load_trend_returns(Path(args.trend_daily_csv))
    merged = d[["day", "strategy_return"]].merge(trend, on="day", how="inner")
    corr = float(merged["strategy_return"].corr(merged["trend_return"])) if len(merged) > 2 else np.nan
    trend_m = perf(merged["trend_return"])
    merged["combined_70_30"] = 0.7 * merged["trend_return"] + 0.3 * merged["strategy_return"]
    comb_m = perf(merged["combined_70_30"])

    verdict = "PASS" if m["sharpe"] >= 0.7 and corr < 0.3 and comb_m["sharpe"] > trend_m["sharpe"] else "MARGINAL" if m["sharpe"] >= 0.4 else "FAIL"
    worth = "YES" if comb_m["sharpe"] > trend_m["sharpe"] and corr < 0.5 else "NO"

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_daily, index=False)

    summary = pd.DataFrame(
        [
            {
                "strategy": "funding_capture",
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "time_in_market": tim,
                "avg_daily_funding_when_active": avg_funding,
                "bull_avg_funding": regime_funding.get("BULL", np.nan),
                "chop_avg_funding": regime_funding.get("CHOP", np.nan),
                "bear_avg_funding": regime_funding.get("BEAR", np.nan),
                "corr_vs_trend": corr,
                "trend_sharpe": trend_m["sharpe"],
                "trend_max_dd": trend_m["max_dd"],
                "combined_70_30_sharpe": comb_m["sharpe"],
                "combined_70_30_max_dd": comb_m["max_dd"],
                "verdict": verdict,
                "worth_combining": worth,
            }
        ]
    )
    summary.to_csv(args.out_summary_csv, index=False)

    print("=" * 64)
    print("ETH FUNDING RATE CAPTURE BACKTEST")
    print("2019-2024 | Market Neutral")
    print("=" * 64)
    print(f"CAGR:              {m['cagr']*100:.2f}%")
    print(f"Sharpe:            {m['sharpe']:.3f}")
    print(f"MaxDD:             {m['max_dd']*100:.2f}%")
    print(f"Time in market:    {tim*100:.2f}%")
    print(f"Avg daily funding: {avg_funding*100:.4f}%" if np.isfinite(avg_funding) else "Avg daily funding: n/a")
    print("")
    print("Funding by regime:")
    print(f"BULL avg: {regime_funding.get('BULL', np.nan)*100:.4f}%/day")
    print(f"CHOP avg: {regime_funding.get('CHOP', np.nan)*100:.4f}%/day")
    print(f"BEAR avg: {regime_funding.get('BEAR', np.nan)*100:.4f}%/day")
    print("")
    print(f"Correlation vs trend strategy: {corr:.3f}")
    print("")
    print("Combined 70/30 portfolio:")
    print(f"Sharpe: {comb_m['sharpe']:.3f}  MaxDD: {comb_m['max_dd']*100:.2f}%")
    print(f"vs trend only: {trend_m['sharpe']:.3f}  {trend_m['max_dd']*100:.2f}%")
    print("")
    print(f"Verdict: {verdict}")
    print(f"Worth combining: {worth}")
    print("=" * 64)
    print(f"Saved: {args.out_summary_csv}")
    print(f"Saved: {args.out_daily_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
