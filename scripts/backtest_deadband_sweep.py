"""Sweep --resize-deadband on the EXACT validated production config (verified from
backtest_walkforward_production.py's own comment + _crypto_main's real CLI call), including the
mean-reversion CHOP overlay that's layered on top in production. Checks both the original
2019-2024 validated window and the full history through the most recent available data, since a
single week of real evidence (the €4.78/week fee finding) is nowhere near enough to trust alone.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import pandas as pd
from backtest_forex_optimised import _mean_reversion_overlay, _stats

WORK = Path("artifacts/backtest/deadband_sweep")
WORK.mkdir(parents=True, exist_ok=True)
COST_BPS = 60.0
DEADBANDS = [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


def run_config(start: str, end: str, deadband: float, tag: str) -> pd.DataFrame:
    daily = WORK / f"daily_{tag}_db{deadband}.csv"
    summary = WORK / f"summary_{tag}_db{deadband}.csv"
    cmd = [
        sys.executable, "scripts/backtest_eth_btc_portfolio.py",
        "--start", start, "--end", end,
        "--vol-filter", "--transition-momentum", "--asymmetric-sizing",
        "--allocation-mode", "signal_weighted",
        "--cost-mode", "weight_change",
        "--cost-bps", str(COST_BPS),
        "--gross-cap", "0.8",
        "--eth-confirm-days", "3", "--btc-confirm-days", "5",
        "--eth-ema", "50,120,300", "--btc-ema", "15,40,120",
        "--include-gold", "--gold-symbol", "PAXG-USD",
        "--gold-ema", "25,65,180", "--gold-cap", "0.3",
        "--gold-cost-bps", str(COST_BPS),
        "--eth-defensive-csv", "artifacts/backtest/direction_event_model_v1_flat_defensive_6y.csv",
        "--resize-deadband", str(deadband),
        "--out-summary-csv", str(summary),
        "--out-daily-csv", str(daily),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    base = pd.read_csv(daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    mr = _mean_reversion_overlay(base, gross_cap=0.8, cost_bps=COST_BPS, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr["main_return"] = pd.to_numeric(mr["combined_return"], errors="coerce").fillna(0.0) + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    return mr[["day", "main_return"]].dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


def count_trades(daily_path: Path) -> float:
    d = pd.read_csv(daily_path, low_memory=False)
    turn = pd.to_numeric(d.get("alloc_turnover_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return float((turn > 0).sum())


for start, end, tag, label in [
    ("2019-01-01", "2024-12-31", "validated", "Validated reference window (2019-2024)"),
]:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"{'deadband':>9} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'resize_events':>14}")
    for db in DEADBANDS:
        ret = run_config(start, end, db, tag)
        s = _stats(ret["main_return"])
        n_resizes = count_trades(WORK / f"daily_{tag}_db{db}.csv")
        print(f"{db:>9.2f} {s['sharpe']:>8.3f} {s['cagr']*100:>7.1f}% {s['maxdd']*100:>7.1f}% {n_resizes:>14.0f}")

# genuinely out-of-sample window: 2025-01-01 through the defensive-proxy file's real coverage limit
print(f"\n{'='*70}\nGenuinely out-of-sample window (2025-01-01 to 2026-02-07)\n{'='*70}")
print(f"{'deadband':>9} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'resize_events':>14}")
for db in DEADBANDS:
    ret = run_config("2025-01-01", "2026-02-07", db, "oos")
    s = _stats(ret["main_return"])
    n_resizes = count_trades(WORK / f"daily_oos_db{db}.csv")
    print(f"{db:>9.2f} {s['sharpe']:>8.3f} {s['cagr']*100:>7.1f}% {s['maxdd']*100:>7.1f}% {n_resizes:>14.0f}")
