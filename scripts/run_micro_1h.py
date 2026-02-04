#!/usr/bin/env python3
"""Run 1h micro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-3)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "1h",
            *sym_args,
        ]
    )

    _run(
        [
            sys.executable,
            "src/obs/build_obs.py",
            "--data-dir",
            str(data_dir),
            "--timeframe",
            "1h",
            *sym_args,
        ]
    )

    _run(
        [
            sys.executable,
            "src/hmm/fit_hmm.py",
            "--obs-dir",
            str(obs_dir),
            "--out-dir",
            str(args.out_dir),
            "--timeframe",
            "1h",
            "--profile",
            "micro",
            "--K",
            *[str(k) for k in args.K],
            "--covariance-type",
            args.covariance_type,
            "--min-covar",
            str(args.min_covar),
            *sym_args,
        ]
    )


if __name__ == "__main__":
    main()