from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.dataset import build_dataset, save_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")

    args = p.parse_args()

    dataset = build_dataset(
        path_1m=args.input_1m,
        horizon_min=args.horizon_min,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        regime_path=args.regime_path,
    )

    save_dataset(dataset, Path(args.out))
    print(f"rows={len(dataset)} cols={len(dataset.columns)} out={args.out}")


if __name__ == "__main__":
    main()
