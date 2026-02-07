from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.features import load_ohlcv_1m
from lpbot.models.elasticnet_v1.walk_forward import _time_windows


def _series_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return {"mse": float("nan"), "mae": float("nan"), "corr": float("nan"), "spearman": float("nan")}
    y = aligned.iloc[:, 0].to_numpy()
    yhat = aligned.iloc[:, 1].to_numpy()
    mse = float(np.mean((y - yhat) ** 2))
    mae = float(np.mean(np.abs(y - yhat)))
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    spearman = float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))
    return {"mse": mse, "mae": mae, "corr": corr, "spearman": spearman}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--input-1m", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    args = p.parse_args()

    meta_path = Path(args.artifacts_dir) / "metadata.json"
    target_type = "return"
    horizon_min = None
    train_days = 60
    val_days = 7
    step_days = 7
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target_type = meta.get("target_type", "return")
        horizon_min = meta.get("horizon_min", None)
        train_days = meta.get("train_days", train_days)
        val_days = meta.get("val_days", val_days)
        step_days = meta.get("step_days", step_days)

    metrics_path = Path(args.artifacts_dir) / "metrics_windows.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise SystemExit("No metrics rows.")

    if target_type == "vol":
        required = ["mse", "mae", "corr", "spearman"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing metrics columns: {missing}")
        if horizon_min is None:
            raise SystemExit("Missing horizon_min in metadata.json.")
        if args.input_1m is None:
            raise SystemExit("Provide --input-1m to compute volatility baselines.")

        dataset = build_supervised_dataset(
            path_1m=args.input_1m,
            horizon_min=int(horizon_min),
            include_time_features=not args.no_time_features,
            atr_window=args.atr_window,
            regime_path=args.regime_path,
            target_type="vol",
        )
        y_series = (
            dataset[["timestamp", "target"]]
            .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True))
            .set_index("timestamp")["target"]
        )
        if y_series.empty:
            raise SystemExit("No target rows available for baselines.")

        df_1m = load_ohlcv_1m(args.input_1m)
        r1 = np.log(df_1m["close"] / df_1m["close"].shift(1))
        sigma_past = r1.rolling(window=int(horizon_min), min_periods=int(horizon_min)).std()
        yhat_persist = np.log1p(sigma_past)
        persist_series = pd.Series(yhat_persist.values, index=df_1m["timestamp"])
        const_value = float(y_series.mean())
        const_series = pd.Series(const_value, index=y_series.index)

        windows = list(_time_windows(y_series.index.to_series(), train_days, val_days, step_days))
        if not windows:
            raise SystemExit("No evaluation windows available for baselines.")

        persist_metrics = []
        const_metrics = []
        for _, _, val_start, val_end in windows:
            val_mask = (y_series.index >= val_start) & (y_series.index < val_end)
            y_val = y_series.loc[val_mask]
            if y_val.empty:
                continue
            persist_metrics.append(_series_metrics(y_val, persist_series.reindex(y_val.index)))
            const_metrics.append(_series_metrics(y_val, const_series.reindex(y_val.index)))

        if not persist_metrics or not const_metrics:
            raise SystemExit("No baseline metrics computed; check input data alignment.")

        model_summary = df[required].mean()
        persist_summary = pd.DataFrame(persist_metrics).mean()
        const_summary = pd.DataFrame(const_metrics).mean()

        summary_df = pd.DataFrame(
            {
                "mse": [model_summary["mse"], persist_summary["mse"], const_summary["mse"]],
                "mae": [model_summary["mae"], persist_summary["mae"], const_summary["mae"]],
                "corr": [model_summary["corr"], persist_summary["corr"], const_summary["corr"]],
                "spearman": [
                    model_summary["spearman"],
                    persist_summary["spearman"],
                    const_summary["spearman"],
                ],
            },
            index=["model", "persistence", "constant"],
        )
        print(summary_df.to_string())
        print("directional_acc: N/A")
    else:
        summary = df[["mse", "mae", "corr", "directional_acc", "nonzero"]].mean()
        print(summary.to_string())


if __name__ == "__main__":
    main()
