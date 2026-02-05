from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def build_dataset(
    path_1m: str | Path,
    horizon_min: int = 15,
    include_time_features: bool = True,
    atr_window: int = 14,
    regime_path: str | Path | None = None,
) -> pd.DataFrame:
    if horizon_min <= 0:
        raise ValueError("horizon_min must be positive.")

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

    log_close = np.log(df_1m["close"])
    target = log_close.shift(-horizon_min) - log_close
    target_df = pd.DataFrame({"timestamp": df_1m["timestamp"], "target": target})
    target_df = target_df.dropna()

    dataset = features.merge(target_df, on="timestamp", how="inner")
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def save_dataset(dataset: pd.DataFrame, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)
