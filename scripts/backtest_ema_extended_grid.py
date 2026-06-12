from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ETH_GRID = [
    ("21/55/144", "21,55,144"),
    ("25/65/180", "25,65,180"),
    ("30/80/200", "30,80,200"),
    ("34/89/233", "34,89,233"),
    ("40/100/250", "40,100,250"),
    ("50/120/300", "50,120,300"),
]

BTC_GRID = [
    ("15/40/120", "15,40,120"),
    ("20/50/150", "20,50,150"),
    ("25/65/180", "25,65,180"),
    ("30/80/200", "30,80,200"),
    ("34/89/233", "34,89,233"),
]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _slug(combo: str) -> str:
    return combo.replace("/", "_")


def _run_config(
    eth_label: str,
    eth_ema: str,
    btc_label: str,
    btc_ema: str,
    work_dir: Path,
    cost_bps: float,
    include_paxg: bool,
) -> dict[str, object]:
    stem = f"eth_{_slug(eth_label)}__btc_{_slug(btc_label)}"
    daily = work_dir / f"{stem}_daily.csv"
    summary = work_dir / f"{stem}_summary.csv"
    overlay_daily = work_dir / f"{stem}_mr_daily.csv"
    overlay_summary = work_dir / f"{stem}_mr_summary.csv"

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
        eth_ema,
        "--btc-ema",
        btc_ema,
        "--out-summary-csv",
        str(summary),
        "--out-daily-csv",
        str(daily),
    ]
    if include_paxg:
        cmd.extend(["--include-gold", "--gold-symbol", "PAXG-USD", "--gold-cap", "0.3", "--gold-cost-bps", str(float(cost_bps))])
    _run(cmd)

    _run(
        [
            sys.executable,
            "scripts/backtest_overlay_strategies.py",
            "--baseline-daily",
            str(daily),
            "--cost-bps",
            str(float(cost_bps)),
            "--out-summary",
            str(overlay_summary),
            "--out-daily",
            str(overlay_daily),
        ]
    )

    base = pd.read_csv(summary).iloc[0].to_dict()
    ov = pd.read_csv(overlay_summary)
    mr = ov[ov["configuration"].eq("+ Mean-rev in CHOP")].iloc[0].to_dict()
    return {
        "eth_combo": eth_label,
        "btc_combo": btc_label,
        "include_paxg": bool(include_paxg),
        "pre_meanrev_sharpe": float(base["combined_sharpe"]),
        "pre_meanrev_maxdd": float(base["combined_max_dd"]),
        "pre_meanrev_cagr": float(base["combined_cagr"]),
        "sharpe": float(mr["sharpe"]),
        "maxdd": float(mr["maxdd"]),
        "cagr": float(mr["cagr"]),
        "daily_csv": str(daily),
        "overlay_daily_csv": str(overlay_daily),
    }


def _print_table(title: str, rows: list[dict[str, object]], combo_key: str, current_combo: str) -> None:
    print("")
    print("=" * 44)
    print(title)
    print("=" * 44)
    print("Combo            Sharpe   MaxDD    CAGR")
    for r in rows:
        combo = str(r[combo_key])
        suffix = " (cur)" if combo == current_combo else ""
        label = f"{combo}{suffix}"[:16]
        print(f"{label:<16} {float(r['sharpe']):>6.3f}  {float(r['maxdd']) * 100:>7.2f}% {float(r['cagr']) * 100:>6.2f}%")
    best = max(rows, key=lambda x: (float(x["sharpe"]), float(x["maxdd"]), float(x["cagr"])))
    print(f"Best: {best[combo_key]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extended independent ETH/BTC EMA grid for validated stack.")
    ap.add_argument("--out", default="artifacts/backtest/ema_extended_grid.csv")
    ap.add_argument("--work-dir", default="artifacts/backtest/ema_extended_grid")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--include-paxg", action="store_true", help="Include PAXG reserve in every run.")
    ap.add_argument("--switch-threshold", type=float, default=0.03)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    eth_rows: list[dict[str, object]] = []
    for eth_label, eth_ema in ETH_GRID:
        row = _run_config(
            eth_label=eth_label,
            eth_ema=eth_ema,
            btc_label="15/40/120",
            btc_ema="15,40,120",
            work_dir=work_dir,
            cost_bps=float(args.cost_bps),
            include_paxg=bool(args.include_paxg),
        )
        row["stage"] = "eth_grid"
        rows.append(row)
        eth_rows.append(row)

    best_eth = max(eth_rows, key=lambda x: (float(x["sharpe"]), float(x["maxdd"]), float(x["cagr"])))
    best_eth_label = str(best_eth["eth_combo"])
    best_eth_ema = dict(ETH_GRID)[best_eth_label]

    btc_rows: list[dict[str, object]] = []
    for btc_label, btc_ema in BTC_GRID:
        existing = next((r for r in eth_rows if str(r["eth_combo"]) == best_eth_label and str(r["btc_combo"]) == btc_label), None)
        if existing is not None:
            row = dict(existing)
        else:
            row = _run_config(
                eth_label=best_eth_label,
                eth_ema=best_eth_ema,
                btc_label=btc_label,
                btc_ema=btc_ema,
                work_dir=work_dir,
                cost_bps=float(args.cost_bps),
                include_paxg=bool(args.include_paxg),
            )
        row["stage"] = "btc_grid"
        rows.append(row)
        btc_rows.append(row)

    best_final = max(btc_rows, key=lambda x: (float(x["sharpe"]), float(x["maxdd"]), float(x["cagr"])))
    current = next(r for r in eth_rows if str(r["eth_combo"]) == "21/55/144" and str(r["btc_combo"]) == "15/40/120")
    improvement = float(best_final["sharpe"]) - float(current["sharpe"])
    decision = "SWITCH" if improvement > float(args.switch_threshold) else "KEEP_CURRENT"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    _print_table("ETH EMA GRID (BTC fixed 15/40/120)", eth_rows, "eth_combo", "21/55/144")
    _print_table(f"BTC EMA GRID (ETH fixed at {best_eth_label})", btc_rows, "btc_combo", "15/40/120")
    print("")
    print("=" * 44)
    print("FINAL")
    print("=" * 44)
    print(f"Current (21/55/144 + 15/40/120): {float(current['sharpe']):.3f}")
    print(f"Best found ({best_final['eth_combo']} + {best_final['btc_combo']}): {float(best_final['sharpe']):.3f}")
    print(f"Improvement: {improvement:+.3f}")
    print(f"Decision: {decision}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
