#!/usr/bin/env python3
"""Summarize data health reports across regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes-dir", default="data/regimes")
    args = p.parse_args()

    base = Path(args.regimes_dir)
    rows = []

    for tf_dir in base.iterdir():
        if not tf_dir.is_dir():
            continue
        timeframe = tf_dir.name
        for profile_dir in tf_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            profile = profile_dir.name
            for sym_dir in profile_dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                health = sym_dir / f"data_health_{timeframe}.csv"
                if not health.exists():
                    continue
                df = pd.read_csv(health)
                metrics = (
                    df[df["type"] == "metric"].set_index("name")["value"].to_dict()
                )
                rows.append(
                    {
                        "timeframe": timeframe,
                        "profile": profile,
                        "symbol": sym_dir.name,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False) if not out.empty else "No data health files found.")


if __name__ == "__main__":
    main()