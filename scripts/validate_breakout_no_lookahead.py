from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _run_backtest(cmd_base: list[str], price_csv: Path, out_csv: Path) -> None:
    cmd = cmd_base + ["--price-5m", str(price_csv), "--out", str(out_csv)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build_cmd_base(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "scripts/breakout_backtest.py",
        "--bar-minutes",
        str(args.bar_minutes),
        "--lookback",
        str(args.lookback),
        "--compress-window",
        str(args.compress_window),
        "--compress-quantile",
        str(args.compress_quantile),
        "--hold-bars",
        str(args.hold_bars),
        "--trend-ema",
        str(args.trend_ema),
        "--trend-timeframe",
        str(args.trend_timeframe),
        "--target-vol",
        str(args.target_vol),
        "--w-max",
        str(args.w_max),
        "--vol-window",
        str(args.vol_window),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Validate breakout script for lookahead leakage.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--compress-window", type=int, default=96)
    p.add_argument("--compress-quantile", type=float, default=0.50)
    p.add_argument("--hold-bars", type=int, default=24)
    p.add_argument("--trend-ema", type=int, default=50)
    p.add_argument("--trend-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--checkpoints", type=int, default=25)
    p.add_argument("--warmup-bars", type=int, default=1500)
    p.add_argument("--weight-tol", type=float, default=1e-12)
    args = p.parse_args()

    price_path = Path(args.price_5m)
    df_price = pd.read_csv(price_path)
    if "timestamp" not in df_price.columns:
        raise ValueError("price csv needs timestamp column")
    n = len(df_price)
    if n <= args.warmup_bars + 10:
        raise ValueError("not enough rows for validation")

    cmd_base = _build_cmd_base(args)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        full_out = tmp / "full.csv"
        _run_backtest(cmd_base, price_path, full_out)
        full = pd.read_csv(full_out)
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True, errors="coerce")
        full = full.dropna(subset=["timestamp"]).set_index("timestamp")

        idxs = np.linspace(args.warmup_bars, n - 1, args.checkpoints, dtype=int)
        mismatches: list[dict[str, object]] = []

        for i in idxs:
            prefix_csv = tmp / f"prefix_{i}.csv"
            prefix_out = tmp / f"prefix_{i}_out.csv"
            df_price.iloc[: i + 1].to_csv(prefix_csv, index=False)

            _run_backtest(cmd_base, prefix_csv, prefix_out)
            pref = pd.read_csv(prefix_out)
            pref["timestamp"] = pd.to_datetime(pref["timestamp"], utc=True, errors="coerce")
            pref = pref.dropna(subset=["timestamp"]).set_index("timestamp")

            ts = pref.index[-1]
            if ts not in full.index:
                mismatches.append({"timestamp": str(ts), "reason": "timestamp missing in full"})
                continue

            a = pref.loc[ts]
            b = full.loc[ts]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[-1]
            if isinstance(b, pd.DataFrame):
                b = b.iloc[-1]

            diffs = {
                "gate_on": bool(a["gate_on"]) != bool(b["gate_on"]),
                "trend_on": bool(a["trend_on"]) != bool(b["trend_on"]),
                "breakout": bool(a["breakout"]) != bool(b["breakout"]),
                "breakdown": bool(a["breakdown"]) != bool(b["breakdown"]),
                "weight_diff": abs(float(a["weight"]) - float(b["weight"])),
            }
            if (
                diffs["gate_on"]
                or diffs["trend_on"]
                or diffs["breakout"]
                or diffs["breakdown"]
                or diffs["weight_diff"] > args.weight_tol
            ):
                mismatches.append(
                    {
                        "timestamp": str(ts),
                        "gate_on_prefix": bool(a["gate_on"]),
                        "gate_on_full": bool(b["gate_on"]),
                        "trend_on_prefix": bool(a["trend_on"]),
                        "trend_on_full": bool(b["trend_on"]),
                        "weight_prefix": float(a["weight"]),
                        "weight_full": float(b["weight"]),
                        "weight_diff": diffs["weight_diff"],
                    }
                )

        print(f"checkpoints={len(idxs)}")
        print(f"mismatches={len(mismatches)}")
        if mismatches:
            print("first_mismatches:")
            for row in mismatches[:10]:
                print(row)
            raise SystemExit(2)
        print("PASS: no lookahead mismatch detected across checkpoints.")


if __name__ == "__main__":
    main()
