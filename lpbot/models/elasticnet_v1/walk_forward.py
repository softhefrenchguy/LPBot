from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowResult:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    best_alpha: float
    best_l1_ratio: float
    mse: float
    mae: float
    corr: float
    directional_acc: float
    nonzero: int


def _time_windows(
    timestamps: pd.Series, train_days: int, val_days: int, step_days: int
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = timestamps.min()
    end = timestamps.max()
    step = pd.Timedelta(days=step_days)
    train_delta = pd.Timedelta(days=train_days)
    val_delta = pd.Timedelta(days=val_days)

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_delta
        val_start = train_end
        val_end = val_start + val_delta
        if val_end > end:
            break
        yield train_start, train_end, val_start, val_end
        cursor = cursor + step


def _clip_targets(y_train: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    std = float(np.std(y_train))
    if std <= 0 or np.isnan(std):
        return y_train, y_val, std
    limit = 3.0 * std
    return np.clip(y_train, -limit, limit), np.clip(y_val, -limit, limit), std


def _select_params(
    X: np.ndarray,
    y: np.ndarray,
    alphas: List[float],
    l1_ratios: List[float],
    n_splits: int,
) -> Tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best = None
    best_mse = np.inf

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_mse = []
            for tr_idx, va_idx in tscv.split(X):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=42,
                )
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_va)
                fold_mse.append(mean_squared_error(y_va, y_hat))

            mse = float(np.mean(fold_mse)) if fold_mse else np.inf
            if mse < best_mse:
                best_mse = mse
                best = (alpha, l1_ratio)

    if best is None:
        raise ValueError("No valid hyperparameters found.")
    return best[0], best[1]


def _corr(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_hat) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_hat)[0, 1])


def _directional_accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_hat)))


def walk_forward_train(
    dataset: pd.DataFrame,
    artifacts_dir: str | Path,
    horizon_min: int,
    train_days: int = 60,
    val_days: int = 7,
    step_days: int = 7,
    alphas: List[float] | None = None,
    l1_ratios: List[float] | None = None,
    cv_splits: int = 5,
) -> Tuple[List[WindowResult], ElasticNet, StandardScaler, List[str]]:
    if alphas is None:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if l1_ratios is None:
        l1_ratios = [0.5, 0.7, 0.9, 0.95, 0.99]

    df = dataset.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    feature_cols = [c for c in df.columns if c not in ["timestamp", "target"]]
    X_all = df[feature_cols].values
    y_all = df["target"].values

    windows = list(_time_windows(df["timestamp"], train_days, val_days, step_days))
    if not windows:
        raise ValueError("No walk-forward windows available.")

    results: List[WindowResult] = []
    final_model = None
    final_scaler = None

    for train_start, train_end, val_start, val_end in windows:
        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] < val_end)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        y_train_clip, y_val_clip, _ = _clip_targets(y_train, y_val)

        best_alpha, best_l1_ratio = _select_params(
            X_train, y_train_clip, alphas, l1_ratios, cv_splits
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1_ratio,
            max_iter=20000,
            tol=1e-4,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_clip)

        y_hat = model.predict(X_val_scaled)

        mse = float(mean_squared_error(y_val_clip, y_hat))
        mae = float(mean_absolute_error(y_val_clip, y_hat))
        corr = _corr(y_val_clip, y_hat)
        directional_acc = _directional_accuracy(y_val_clip, y_hat)
        nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))

        results.append(
            WindowResult(
                train_start=str(train_start),
                train_end=str(train_end),
                val_start=str(val_start),
                val_end=str(val_end),
                n_train=len(X_train),
                n_val=len(X_val),
                best_alpha=float(best_alpha),
                best_l1_ratio=float(best_l1_ratio),
                mse=mse,
                mae=mae,
                corr=corr,
                directional_acc=directional_acc,
                nonzero=nonzero,
            )
        )

        final_model = model
        final_scaler = scaler

    if final_model is None or final_scaler is None:
        raise ValueError("Training failed; no windows produced a model.")

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, artifacts_path / "model.joblib")
    joblib.dump(final_scaler, artifacts_path / "scaler.joblib")

    meta = {
        "horizon_min": horizon_min,
        "train_days": train_days,
        "val_days": val_days,
        "step_days": step_days,
        "feature_list": feature_cols,
        "windows": [r.__dict__ for r in results],
        "last_window": results[-1].__dict__ if results else None,
    }
    (artifacts_path / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    metrics_df = pd.DataFrame([r.__dict__ for r in results])
    metrics_df.to_csv(artifacts_path / "metrics_windows.csv", index=False)

    return results, final_model, final_scaler, feature_cols
