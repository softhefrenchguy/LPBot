"""Decompose the resize-deadband's improvement into (a) mechanical cost savings from fewer
trades, vs (b) any genuine benefit independent of cost -- re-run the identical sweep with
cost_bps=0. If the Sharpe-improves-with-wider-deadband pattern disappears at zero cost, the whole
effect is "fewer trades cost less," true regardless of signal quality, not evidence the deadband
catches real noise. If it survives at zero cost, that's a genuine effect beyond cost avoidance.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import pandas as pd
from backtest_forex_optimised import _mean_reversion_overlay, _stats

WORK = Path("artifacts/backtest/deadband_sweep")
WORK.mkdir(parents=True, exist_ok=True)
DEADBANDS = [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


def run_config(start: str, end: str, deadband: float, cost_bps: float, tag: str) -> pd.DataFrame:
    daily = WORK / f"daily_{tag}_db{deadband}_c{cost_bps}.csv"
    summary = WORK / f"summary_{tag}_db{deadband}_c{cost_bps}.csv"
    cmd = [
        sys.executable, "scripts/backtest_eth_btc_portfolio.py",
        "--start", start, "--end", end,
        "--vol-filter", "--transition-momentum", "--asymmetric-sizing",
        "--allocation-mode", "signal_weighted",
        "--cost-mode", "weight_change",
        "--cost-bps", str(cost_bps),
        "--gross-cap", "0.8",
        "--eth-confirm-days", "3", "--btc-confirm-days", "5",
        "--eth-ema", "50,120,300", "--btc-ema", "15,40,120",
        "--include-gold", "--gold-symbol", "PAXG-USD",
        "--gold-ema", "25,65,180", "--gold-cap", "0.3",
        "--gold-cost-bps", str(cost_bps),
        "--eth-defensive-csv", "artifacts/backtest/direction_event_model_v1_flat_defensive_6y.csv",
        "--resize-deadband", str(deadband),
        "--out-summary-csv", str(summary),
        "--out-daily-csv", str(daily),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    base = pd.read_csv(daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    mr = _mean_reversion_overlay(base, gross_cap=0.8, cost_bps=cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr["main_return"] = pd.to_numeric(mr["combined_return"], errors="coerce").fillna(0.0) + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    return mr[["day", "main_return"]].dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


for start, end, tag, label in [
    ("2019-01-01", "2024-12-31", "validated", "Validated reference window (2019-2024)"),
    ("2025-01-01", "2026-02-07", "oos", "Genuinely out-of-sample window (2025-Feb'26)"),
]:
    print(f"\n{'='*86}\n{label}\n{'='*86}")
    print(f"{'deadband':>9} | {'Sharpe@60bps':>13} {'CAGR@60bps':>11} | {'Sharpe@0bps':>12} {'CAGR@0bps':>10}")
    for db in DEADBANDS:
        ret_cost = run_config(start, end, db, 60.0, tag)
        s_cost = _stats(ret_cost["main_return"])
        ret_free = run_config(start, end, db, 0.0, tag)
        s_free = _stats(ret_free["main_return"])
        print(f"{db:>9.2f} | {s_cost['sharpe']:>13.3f} {s_cost['cagr']*100:>10.1f}% | {s_free['sharpe']:>12.3f} {s_free['cagr']*100:>9.1f}%")
