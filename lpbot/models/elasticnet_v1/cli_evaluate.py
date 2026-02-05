from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    args = p.parse_args()

    metrics_path = Path(args.artifacts_dir) / "metrics_windows.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise SystemExit("No metrics rows.")

    summary = df[["mse", "mae", "corr", "directional_acc", "nonzero"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
