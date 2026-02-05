from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.inference import predict_latest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--n-latest", type=int, default=1)
    p.add_argument("--out", default=None)

    args = p.parse_args()

    out_df = predict_latest(
        path_1m=args.input_1m,
        artifacts_dir=args.artifacts_dir,
        regime_path=args.regime_path,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        n_latest=args.n_latest,
    )

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
