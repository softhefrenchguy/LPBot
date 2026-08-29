from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from backtest_overlay_strategies import _mean_reversion_overlay
from backtest_forex_optimised import _stats


EMA_CANDIDATES = {
    "10/30/90": "10,30,90", "15/40/120": "15,40,120", "21/55/144": "21,55,144",
    "25/65/150": "25,65,150", "30/80/200": "30,80,200", "50/120/300": "50,120,300", "20/50/100": "20,50,100",
}

WINDOWS = [
    {"n": 1, "train_start": "2019-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"n": 2, "train_start": "2019-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"n": 3, "train_start": "2019-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"n": 4, "train_start": "2019-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRUE_START = "2019-01-01"  # earliest available data; used to give OOS runs proper indicator warm-up


def _apply_meanrev_overlay_series(base: pd.DataFrame, gross_cap: float, cost_bps: float) -> pd.Series:
    """There is no --meanrev-overlay flag on backtest_eth_btc_portfolio.py -- the mean-
    reversion CHOP overlay only exists as post-hoc logic inside _crypto_main (backtest_
    forex_optimised.py), applied AFTER the sleeve backtest runs. Reconstruct it the same
    way here. Without this step "full stack" would silently exclude the overlay while
    still being labelled full stack."""
    base = base.copy()
    if "gold_strategy_return" not in base.columns:
        base["gold_strategy_return"] = 0.0
    mr = _mean_reversion_overlay(base, gross_cap=gross_cap, cost_bps=cost_bps, z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    return base["combined_return"] + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)


def _run_subprocess_with_retry(cmd: list[str], desc: str, retries: int) -> None:
    last_err = None
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        last_err = f"backtest failed ({desc}), attempt {attempt}/{retries}:\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        if attempt < retries:
            time.sleep(5 * attempt)  # transient network errors (Binance API resets) -- back off and retry
    raise RuntimeError(last_err)


def _build_cmd(start: str, end: str, eth_ema: str, btc_ema: str, summary: Path, daily: Path, full_stack: bool) -> list[str]:
    cmd = [
        sys.executable, "scripts/backtest_eth_btc_portfolio.py",
        "--start", start, "--end", end,
        "--eth-ema", eth_ema, "--btc-ema", btc_ema,
        "--eth-confirm-days", "3", "--btc-confirm-days", "5",
        "--vol-filter", "--transition-momentum",
        "--allocation-mode", "signal_weighted", "--gross-cap", "0.8",
        "--cost-bps", "60", "--cost-mode", "weight_change",  # was 20; corrected to match real Kraken taker fees at ~$1k-10k/month volume
        "--out-summary-csv", str(summary), "--out-daily-csv", str(daily),
    ]
    if full_stack:
        cmd += ["--asymmetric-sizing", "--include-gold", "--gold-symbol", "PAXG-USD", "--gold-ema", "25,65,180", "--gold-cap", "0.3", "--gold-cost-bps", "60"]
    return cmd


def run_backtest(start: str, end: str, eth_ema: str, btc_ema: str, work_dir: Path, tag: str, full_stack: bool = False, resume: bool = True, retries: int = 3) -> dict:
    summary = work_dir / f"{tag}_summary.csv"
    daily = work_dir / f"{tag}_daily.csv"
    if not (resume and summary.exists() and daily.exists()):
        _run_subprocess_with_retry(_build_cmd(start, end, eth_ema, btc_ema, summary, daily, full_stack), f"{start} to {end}, eth={eth_ema} btc={btc_ema}", retries)
    row = pd.read_csv(summary).iloc[0].to_dict()
    if full_stack:
        base = pd.read_csv(daily, low_memory=False)
        base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
        main_return = _apply_meanrev_overlay_series(base, gross_cap=0.8, cost_bps=60.0)  # was 20.0
        overlay_stats = _stats(main_return)
        row["combined_sharpe_no_overlay"] = row["combined_sharpe"]
        row["combined_sharpe"] = overlay_stats["sharpe"]
        row["combined_cagr"] = overlay_stats["cagr"]
        row["combined_max_dd"] = overlay_stats["maxdd"]
    return row


def run_oos_warm(test_start: str, test_end: str, eth_ema: str, btc_ema: str, work_dir: Path, tag: str, full_stack: bool = False, resume: bool = True, retries: int = 3) -> dict:
    """OOS evaluation with proper indicator warm-up: run continuously from TRUE_START
    through test_end (so 120/300-day EMAs, the vol-filter's rolling(252) percentile, and
    the gold EMA(180) all have multi-year history exactly as live production would),
    then slice out just [test_start, test_end] before computing stats. A cold restart on
    test_start would starve the slow indicators of history and understate performance --
    this is the same reasoning the production walk-forward's calendar-year-slice approach
    already relies on, applied here to the walk-forward's per-window best params."""
    summary = work_dir / f"{tag}_summary.csv"
    daily = work_dir / f"{tag}_daily.csv"
    if not (resume and summary.exists() and daily.exists()):
        _run_subprocess_with_retry(_build_cmd(TRUE_START, test_end, eth_ema, btc_ema, summary, daily, full_stack), f"{TRUE_START} to {test_end} (oos-warm), eth={eth_ema} btc={btc_ema}", retries)
    base = pd.read_csv(daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    main_return = _apply_meanrev_overlay_series(base, gross_cap=0.8, cost_bps=60.0) if full_stack else base["combined_return"]  # cost_bps was 20.0
    sliced = main_return[(base["day"] >= test_start) & (base["day"] <= test_end)]
    return _stats(sliced)


def optimise_window(train_start: str, train_end: str, work_dir: Path, window_n: int, full_stack: bool = False, resume: bool = True) -> tuple[str, str, float, pd.DataFrame]:
    """Coordinate descent: optimise ETH EMA holding BTC fixed at a neutral starting
    point, then optimise BTC EMA holding ETH fixed at the just-found best. A full joint
    7x7 grid is not needed to answer the walk-forward question (would the same params
    keep getting picked, and would they hold up OOS) and this halves the run count."""
    results: list[dict] = []
    start_btc = "21/55/144"

    for label, spans in EMA_CANDIDATES.items():
        tag = f"w{window_n}_eth_{label.replace('/', '_')}"
        r = run_backtest(train_start, train_end, spans, EMA_CANDIDATES[start_btc], work_dir, tag, full_stack=full_stack, resume=resume)
        r.update({"search_stage": "eth", "eth_label": label, "btc_label": start_btc})
        results.append(r)
    best_eth = max((r for r in results if r["search_stage"] == "eth"), key=lambda r: r["combined_sharpe"])["eth_label"]

    for label, spans in EMA_CANDIDATES.items():
        tag = f"w{window_n}_btc_{label.replace('/', '_')}"
        r = run_backtest(train_start, train_end, EMA_CANDIDATES[best_eth], spans, work_dir, tag, full_stack=full_stack, resume=resume)
        r.update({"search_stage": "btc", "eth_label": best_eth, "btc_label": label})
        results.append(r)
    best_btc = max((r for r in results if r["search_stage"] == "btc"), key=lambda r: r["combined_sharpe"])["btc_label"]

    best_row = next(r for r in results if r["search_stage"] == "btc" and r["btc_label"] == best_btc)
    return best_eth, best_btc, float(best_row["combined_sharpe"]), pd.DataFrame(results)
