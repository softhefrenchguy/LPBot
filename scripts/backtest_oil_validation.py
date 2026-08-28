from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_commodities_trend import _load_commodity, _load_or_download, run_backtest, _priority_allocation_sweep
from backtest_forex_optimised import _crypto_main, _stats


OIL_SPANS = (10, 25, 75)
OIL_CONFIRM = 3
OIL_KWARGS = dict(long_only=True, vol_filter=True, asym_sizing=True, transition=True, carry=True, oil_seasonal=False)


def _trade_list(frame: pd.DataFrame) -> pd.DataFrame:
    w = frame["weight_exec"].to_numpy()
    ret = frame["strategy_return"].to_numpy()
    close = frame["close"].to_numpy()
    ts = frame["timestamp"].to_numpy()
    active = np.abs(w) > 1e-9
    if not active.any():
        return pd.DataFrame(columns=["entry_date", "exit_date", "days_held", "entry_price", "exit_price", "return_pct"])
    n = len(active)
    diff = np.diff(active.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if active[-1] else end_candidates
    log_ret = np.log1p(np.clip(ret, -0.999999, None))
    cum0 = np.concatenate(([0.0], np.cumsum(log_ret)))
    trade_returns = np.expm1(cum0[ends + 1] - cum0[starts])
    rows = []
    for s, e, r in zip(starts, ends, trade_returns):
        rows.append({
            "entry_date": pd.Timestamp(ts[s]).date().isoformat(),
            "exit_date": pd.Timestamp(ts[e]).date().isoformat(),
            "days_held": int(e - s + 1),
            "entry_price": float(close[s]),
            "exit_price": float(close[e]),
            "return_pct": float(r * 100.0),
        })
    return pd.DataFrame(rows)


def _yearly_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["year"] = pd.to_datetime(d["day"]).dt.year
    rows = []
    for yr, g in d.groupby("year"):
        strat_ret = float((1.0 + g["strategy_return"]).prod() - 1.0)
        spot_ret = float((1.0 + g["exec_return"].fillna(0.0)).prod() - 1.0)
        rows.append({"year": int(yr), "strategy_return_pct": strat_ret * 100.0, "oil_spot_return_pct": spot_ret * 100.0})
    return pd.DataFrame(rows)


def _load_oil_period(data_dir: Path, start: str, end: str, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"{start}_{end}".replace("-", "")
    fut = _load_or_download("CL=F", start, end, data_dir / f"_raw_OIL_futures_{tag}.csv", refresh)
    etf = _load_or_download("USO", start, end, data_dir / f"_raw_OIL_etf_{tag}.csv", refresh)
    return fut, etf


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the Oil long-only trend config: trade-level detail, yearly breakdown, and a 2010-2018 out-of-sample test.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--oos-start", default="2010-01-01")
    ap.add_argument("--oos-end", default="2018-12-31")
    ap.add_argument("--data-dir", default="data/commodities")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--out-validation", default="artifacts/backtest/oil_validation.csv")
    ap.add_argument("--out-trades", default="artifacts/backtest/oil_trades.csv")
    ap.add_argument("--out-oos", default="artifacts/backtest/oil_oos_summary.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--crypto-cost-bps", type=float, default=20.0)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--refresh-crypto", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)

    # ---- STEP 1 + 2: trade list and yearly breakdown on the 2019-2024 config ----
    d, etf, source = _load_commodity("OIL", data_dir, args.start, args.end, bool(args.refresh_data))
    frame = run_backtest(d, etf, "OIL", OIL_SPANS, OIL_CONFIRM, cost_bps=args.cost_bps, data_dir=data_dir, start=args.start, end=args.end, refresh=bool(args.refresh_data), **OIL_KWARGS)

    trades = _trade_list(frame)
    trades.to_csv(args.out_trades, index=False)

    n_trades = len(trades)
    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    win_rate = len(wins) / n_trades if n_trades else float("nan")
    avg_winner = float(wins["return_pct"].mean()) if len(wins) else 0.0
    avg_loser = float(losses["return_pct"].mean()) if len(losses) else 0.0
    largest = trades.loc[trades["return_pct"].abs().idxmax()] if n_trades else None
    total_ret_sum = trades["return_pct"].sum() if n_trades else 0.0
    top3_sum = trades.nlargest(3, "return_pct")["return_pct"].sum() if n_trades >= 1 else 0.0
    top3_share = (top3_sum / total_ret_sum * 100.0) if total_ret_sum != 0 else float("nan")

    yearly = _yearly_breakdown(frame)

    print("=" * 90); print("STEP 1: TRADE-BY-TRADE ANALYSIS (2019-2024)"); print("=" * 90)
    print(trades.to_string(index=False))
    print("-" * 90)
    print(f"Total trades: {n_trades}")
    print(f"Win rate: {win_rate*100:.1f}%")
    print(f"Avg winner: {avg_winner:.2f}%  Avg loser: {avg_loser:.2f}%")
    if largest is not None:
        print(f"Largest single trade: {largest['return_pct']:.2f}% ({largest['entry_date']} -> {largest['exit_date']})")
    print(f"Top-3 trades share of summed trade return: {top3_share:.1f}%")
    small_sample_flag = top3_share > 80.0
    print(f"Small-sample concern (top 3 > 80% of return): {'YES -- fragile' if small_sample_flag else 'NO -- spread across many trades'}")
    print("=" * 90)

    print("STEP 2: YEARLY BREAKDOWN"); print("=" * 90)
    print(yearly.to_string(index=False))
    max_year = yearly.loc[yearly["strategy_return_pct"].idxmax()]
    total_positive_years_return = yearly.loc[yearly["strategy_return_pct"] > 0, "strategy_return_pct"].sum()
    dominant_year_share = (max_year["strategy_return_pct"] / total_positive_years_return * 100.0) if total_positive_years_return != 0 else float("nan")
    print(f"Largest single year: {int(max_year['year'])} ({max_year['strategy_return_pct']:.1f}%) -- {dominant_year_share:.1f}% of summed positive-year returns")
    print(f"Years with positive strategy return: {int((yearly['strategy_return_pct']>0).sum())} / {len(yearly)}")
    print("=" * 90)

    validation_df = pd.DataFrame([{
        "config": "OIL long-only EMA10/25/75 confirm3 volfilterON carryON",
        "period": f"{args.start} to {args.end}",
        "trades": n_trades, "win_rate": win_rate, "avg_winner_pct": avg_winner, "avg_loser_pct": avg_loser,
        "largest_trade_pct": float(largest["return_pct"]) if largest is not None else np.nan,
        "top3_share_pct": top3_share, "small_sample_flag": small_sample_flag,
        "max_year": int(max_year["year"]), "max_year_return_pct": max_year["strategy_return_pct"],
        "dominant_year_share_pct": dominant_year_share,
    }])

    # ---- STEP 3: out-of-sample 2010-2018 ----
    oos_d, oos_etf = _load_oil_period(data_dir, args.oos_start, args.oos_end, bool(args.refresh_data))
    oos_frame = run_backtest(oos_d, oos_etf, "OIL", OIL_SPANS, OIL_CONFIRM, cost_bps=args.cost_bps, data_dir=data_dir, start=args.oos_start, end=args.oos_end, refresh=bool(args.refresh_data), **OIL_KWARGS)
    oos_daily = oos_frame.groupby("day", as_index=False)["strategy_return"].apply(lambda s: (1.0 + s).prod() - 1.0)
    oos_stats = _stats(oos_daily["strategy_return"])
    oos_trades = _trade_list(oos_frame)
    oos_yearly = _yearly_breakdown(oos_frame)

    print("STEP 3: OUT-OF-SAMPLE 2010-2018 (same config, no parameter changes)"); print("=" * 90)
    print(f"Sharpe (raw): {oos_stats['raw_sharpe']:.3f}")
    print(f"CAGR: {oos_stats['cagr']*100:.1f}%")
    print(f"MaxDD: {oos_stats['maxdd']*100:.1f}%")
    print(f"Trades: {len(oos_trades)}  ({len(oos_trades)/9:.1f}/year over 9 years)")
    print(oos_yearly.to_string(index=False))
    print("=" * 90)

    oos_verdict = "ROBUST" if oos_stats["raw_sharpe"] > 0.5 else "PERIOD-SPECIFIC" if oos_stats["raw_sharpe"] < 0.0 else "MIXED"
    print(f"Verdict: {oos_verdict}")
    print("=" * 90)

    oos_df = pd.DataFrame([{
        "period": f"{args.oos_start} to {args.oos_end}", "raw_sharpe": oos_stats["raw_sharpe"], "sharpe": oos_stats["sharpe"],
        "cagr": oos_stats["cagr"], "maxdd": oos_stats["maxdd"], "ann_vol": oos_stats["ann_vol"], "trades": len(oos_trades), "verdict": oos_verdict,
    }])
    oos_df.to_csv(args.out_oos, index=False)

    # ---- STEP 4: conditional combined-portfolio rerun ----
    if oos_stats["raw_sharpe"] > 0.5:
        main_df = _crypto_main(args.start, args.end, Path(args.work_dir), float(args.crypto_cost_bps), bool(args.refresh_crypto))
        main_stats = _stats(main_df["main_return"])
        sweep = _priority_allocation_sweep(frame, main_df, Path(args.work_dir), [0.10, 0.20], args.cost_bps)
        print("STEP 4: COMBINED PORTFOLIO (out-of-sample passed)"); print("=" * 90)
        for _, r in sweep.iterrows():
            print(f"  Main + Oil {r['allocation']*100:.0f}%: Sharpe {r['sharpe']:.3f}  CAGR {r['cagr']*100:.1f}%  MaxDD {r['maxdd']*100:.1f}%")
        beats_main = bool((sweep["sharpe"] > main_stats["sharpe"]).any())
        print(f"Beats main only ({main_stats['sharpe']:.3f}): {beats_main}")
        final_decision = "IMPLEMENT" if beats_main else "RESEARCH"
        print(f"Decision: {final_decision}")
        sweep.insert(0, "configuration", "main_plus_oil")
        validation_df = pd.concat([validation_df, sweep], ignore_index=True)
    else:
        print("STEP 4: SKIPPED -- out-of-sample Sharpe did not clear 0.5")
        print("Decision: SKIP  (2019-2024 result likely period-specific, not a repeatable edge)")
        final_decision = "SKIP"

    validation_df["final_decision"] = final_decision
    validation_df.to_csv(args.out_validation, index=False)

    print(f"\nSaved: {args.out_validation}")
    print(f"Saved: {args.out_trades}")
    print(f"Saved: {args.out_oos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
