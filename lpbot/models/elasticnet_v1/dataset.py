from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def _validate_vol_target_alignment(
    r1: pd.Series,
    sigma_fwd: pd.Series,
    horizon_min: int,
    sample_size: int = 5,
) -> None:
    if not __debug__:
        return
    if len(sigma_fwd) == 0:
        return
    if np.nanstd(r1.values) == 0:
        return

    valid_idx = np.where(~np.isnan(sigma_fwd.values))[0]
    if len(valid_idx) == 0:
        return

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        valid_idx, size=min(sample_size, len(valid_idx)), replace=False
    )

    for idx in sample_idx:
        manual = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        vec = sigma_fwd.iloc[idx]
        if not np.isclose(manual, vec, rtol=1e-6, atol=1e-12, equal_nan=True):
            raise ValueError(
                "Vol target alignment mismatch: forward sigma does not match "
                "vectorized computation. Check shift direction and window."
            )

    past_candidates = [i for i in sample_idx if i >= horizon_min - 1]
    if not past_candidates:
        return

    all_past_match = True
    for idx in past_candidates:
        forward = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        past = r1.iloc[idx - horizon_min + 1 : idx + 1].std()
        if not np.isclose(forward, past, rtol=1e-6, atol=1e-12, equal_nan=True):
            all_past_match = False
            break

    if all_past_match:
        raise ValueError(
            "Vol target alignment looks reversed: forward sigma matches past-window "
            "sigma for all sampled points."
        )


def build_supervised_dataset(
    path_1m: str | Path,
    horizon_min: int = 15,
    include_time_features: bool = True,
    atr_window: int = 14,
    regime_path: str | Path | None = None,
    target_type: str = "return",
) -> pd.DataFrame:
    if horizon_min <= 0:
        raise ValueError("horizon_min must be positive.")
    if target_type not in {"return", "vol"}:
        raise ValueError("target_type must be 'return' or 'vol'.")

    df_1m = load_ohlcv_1m(path_1m)

    regime_df = None
    if regime_path is not None:
        regime_df = pd.read_csv(Path(regime_path))

    features = build_features(
        df_1m=df_1m,
        include_time_features=include_time_features,
        atr_window=atr_window,
        regime_df=regime_df,
    )

    if target_type == "return":
        log_close = np.log(df_1m["close"])
        target = log_close.shift(-horizon_min) - log_close
    else:
        close_series = df_1m["close"]
        r1 = np.log(close_series / close_series.shift(1))
        forward_returns = r1.shift(-1)
        sigma_fwd = (
            forward_returns.rolling(window=horizon_min, min_periods=horizon_min)
            .std()
            .shift(-(horizon_min - 1))
        )
        _validate_vol_target_alignment(r1, sigma_fwd, horizon_min)
        target = np.log1p(sigma_fwd)
    target_df = pd.DataFrame({"timestamp": df_1m["timestamp"], "target": target})
    target_df = target_df.dropna()

    dataset = features.merge(target_df, on="timestamp", how="inner")
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def build_dataset(
    path_1m: str | Path,
    horizon_min: int = 15,
    include_time_features: bool = True,
    atr_window: int = 14,
    regime_path: str | Path | None = None,
    target_type: str = "return",
) -> pd.DataFrame:
    return build_supervised_dataset(
        path_1m=path_1m,
        horizon_min=horizon_min,
        include_time_features=include_time_features,
        atr_window=atr_window,
        regime_path=regime_path,
        target_type=target_type,
    )


def save_dataset(dataset: pd.DataFrame, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)
