from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAXG_GRID = [
    ("21/55/144", "21,55,144"),
    ("15/40/120", "15,40,120"),
    ("25/65/180", "25,65,180"),
    ("10/30/90", "10,30,90"),
    ("34/89/233", "34,89,233"),
    ("50/120/300", "50,120,300"),
]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _slug(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").lower()


def _run_portfolio(
    work_dir: Path,
    stem: str,
    cost_bps: float,
    include_paxg: bool,
    paxg_ema: str = "21,55,144",
) -> tuple[Path, Path]:
    daily = work_dir / f"{stem}_daily.csv"
    summary = work_dir / f"{stem}_summary.csv"
    cmd = [
        sys.executable,
        "scripts/backtest_eth_btc_portfolio.py",
        "--start",
        "2019-01-01",
        "--end",
        "2024-12-31",
        "--vol-filter",
        "--transition-momentum",
        "--asymmetric-sizing",
        "--allocation-mode",
        "signal_weighted",
        "--cost-mode",
        "weight_change",
        "--cost-bps",
        str(float(cost_bps)),
        "--eth-confirm-days",
        "3",
        "--btc-confirm-days",
        "5",
        "--eth-ema",
        "50,120,300",
        "--btc-ema",
        "15,40,120",
        "--out-summary-csv",
        str(summary),
        "--out-daily-csv",
        str(daily),
    ]
    if include_paxg:
        cmd.extend(
            [
                "--include-gold",
                "--gold-symbol",
                "PAXG-USD",
                "--gold-ema",
                paxg_ema,
                "--gold-cap",
                "0.3",
                "--gold-cost-bps",
                str(float(cost_bps)),
            ]
        )
    _run(cmd)
    return daily, summary


def _run_meanrev(work_dir: Path, stem: str, daily: Path, cost_bps: float) -> tuple[Path, Path]:
    mr_daily = work_dir / f"{stem}_mr_daily.csv"
    mr_summary = work_dir / f"{stem}_mr_summary.csv"
    _run(
        [
            sys.executable,
            "scripts/backtest_overlay_strategies.py",
            "--baseline-daily",
            str(daily),
            "--cost-bps",
            str(float(cost_bps)),
            "--out-summary",
            str(mr_summary),
            "--out-daily",
            str(mr_daily),
        ]
    )
    return mr_daily, mr_summary


def _meanrev_row(summary_path: Path) -> dict[str, float]:
    s = pd.read_csv(summary_path)
    row = s[s["configuration"].eq("+ Mean-rev in CHOP")].iloc[0]
    return {
        "sharpe": float(row["sharpe"]),
        "maxdd": float(row["maxdd"]),
        "cagr": float(row["cagr"]),
    }


def _paxg_trade_diagnostics(daily_path: Path) -> dict[str, object]:
    d = pd.read_csv(daily_path, low_memory=False)
    if "gold_weight_exec" not in d.columns:
        return {"fires": 0, "avg_paxg_position_return": np.nan}

    d["day"] = pd.to_datetime(d["day"], errors="coerce")
    d["gold_weight_exec"] = pd.to_numeric(d["gold_weight_exec"], errors="coerce").fillna(0.0)
    d["gold_strategy_return"] = pd.to_numeric(d.get("gold_strategy_return", 0.0), errors="coerce").fillna(0.0)
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    active = d["gold_weight_exec"] > 0.0
    entries = active & ~active.shift(1, fill_value=False)
    fires = int(entries.sum())

    returns: list[float] = []
    if fires:
        run_id = (active != active.shift(1, fill_value=False)).cumsum()
        for _, w in d[active].groupby(run_id[active]):
            r = float((1.0 + pd.to_numeric(w["gold_strategy_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
            returns.append(r)

    return {
        "fires": fires,
        "avg_paxg_position_return": float(np.mean(returns)) if returns else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid-test PAXG reserve EMA gate speeds on validated ETH/BTC stack.")
    ap.add_argument("--out", default="artifacts/backtest/paxg_ema_grid.csv")
    ap.add_argument("--work-dir", default="artifacts/backtest/paxg_ema_grid")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--baseline-sharpe", type=float, default=1.752)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    base_daily, _ = _run_portfolio(
        work_dir=work_dir,
        stem="baseline_no_paxg",
        cost_bps=float(args.cost_bps),
        include_paxg=False,
    )
    _, base_mr_summary = _run_meanrev(work_dir, "baseline_no_paxg", base_daily, float(args.cost_bps))
    base_stats = _meanrev_row(base_mr_summary)
    rows.append(
        {
            "config": "No PAXG (baseline)",
            "paxg_ema": "",
            **base_stats,
            "fires": np.nan,
            "avg_paxg_position_return": np.nan,
            "daily_csv": str(base_daily),
        }
    )

    for label, spans in PAXG_GRID:
        stem = f"paxg_{_slug(label)}"
        daily, _ = _run_portfolio(
            work_dir=work_dir,
            stem=stem,
            cost_bps=float(args.cost_bps),
            include_paxg=True,
            paxg_ema=spans,
        )
        _, mr_summary = _run_meanrev(work_dir, stem, daily, float(args.cost_bps))
        stats = _meanrev_row(mr_summary)
        diag = _paxg_trade_diagnostics(daily)
        rows.append(
            {
                "config": f"PAXG {label}",
                "paxg_ema": label,
                **stats,
                **diag,
                "daily_csv": str(daily),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    baseline = df.iloc[0]
    candidates = df.iloc[1:].copy()
    best = candidates.sort_values(["sharpe", "maxdd", "cagr"], ascending=[False, False, False]).iloc[0]
    decision = "KEEP PAXG" if float(best["sharpe"]) > float(baseline["sharpe"]) else "REMOVE PAXG"

    print("")
    print("=" * 54)
    print("PAXG EMA GRID RESULTS")
    print("=" * 54)
    print("Config              Sharpe  MaxDD    CAGR   Fires")
    print("-" * 54)
    for _, r in df.iterrows():
        fires = "-" if pd.isna(r["fires"]) else f"{int(r['fires'])}"
        print(
            f"{str(r['config'])[:19]:<19}"
            f"{float(r['sharpe']):>7.3f} "
            f"{float(r['maxdd']) * 100:>7.2f}% "
            f"{float(r['cagr']) * 100:>6.2f}% "
            f"{fires:>6}"
        )
    print("-" * 54)
    print(
        f"Best PAXG: {best['config']} | Sharpe {float(best['sharpe']):.3f} "
        f"| avg PAXG return {float(best['avg_paxg_position_return']) * 100:.2f}%"
    )
    print(f"Baseline actual: {float(baseline['sharpe']):.3f} (reference requested: {float(args.baseline_sharpe):.3f})")
    print(f"Decision: {decision}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
