from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.walk_forward import walk_forward_train


def _parse_float_list(val: str) -> list[float]:
    return [float(x.strip()) for x in val.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--val-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--alpha-grid", default="1e-4,1e-3,1e-2,1e-1,1.0")
    p.add_argument("--l1-grid", default="0.1,0.5,0.9")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument(
        "--target-type",
        choices=["return", "vol"],
        default="return",
        help="Training target type (default: return).",
    )

    args = p.parse_args()

    if args.dataset is None and args.input_1m is None:
        raise SystemExit("Provide --dataset or --input-1m")

    if args.dataset is not None:
        dataset = pd.read_csv(Path(args.dataset))
    else:
        dataset = build_supervised_dataset(
            path_1m=args.input_1m,
            horizon_min=args.horizon_min,
            include_time_features=not args.no_time_features,
            atr_window=args.atr_window,
            regime_path=args.regime_path,
            target_type=args.target_type,
        )

    alphas = _parse_float_list(args.alpha_grid)
    l1_ratios = _parse_float_list(args.l1_grid)

    results, _, _, _ = walk_forward_train(
        dataset=dataset,
        artifacts_dir=args.artifacts_dir,
        horizon_min=args.horizon_min,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        alphas=alphas,
        l1_ratios=l1_ratios,
        cv_splits=args.cv_splits,
        target_type=args.target_type,
    )

    for r in results:
        print(
            f"{r.val_start} -> {r.val_end} | "
            f"mse={r.mse:.6f} mae={r.mae:.6f} corr={r.corr:.4f} "
            f"dir_acc={r.directional_acc:.4f} nz={r.nonzero} "
            f"alpha={r.best_alpha} l1={r.best_l1_ratio}"
        )

    print(f"artifacts_dir={args.artifacts_dir}")


if __name__ == "__main__":
    main()
