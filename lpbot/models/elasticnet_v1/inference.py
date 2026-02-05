from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def load_artifacts(artifacts_dir: str | Path):
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / "model.joblib")
    scaler = joblib.load(artifacts_path / "scaler.joblib")
    meta = json.loads((artifacts_path / "metadata.json").read_text(encoding="utf-8"))
    feature_list = meta.get("feature_list", [])
    return model, scaler, feature_list, meta


def predict_latest(
    path_1m: str | Path,
    artifacts_dir: str | Path,
    regime_path: str | Path | None = None,
    include_time_features: bool = True,
    atr_window: int = 14,
    n_latest: int = 1,
) -> pd.DataFrame:
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

    model, scaler, feature_list, _ = load_artifacts(artifacts_dir)
    features = features.set_index("timestamp")
    features = features[feature_list]
    latest = features.tail(n_latest)
    X = scaler.transform(latest.values)
    y_hat = model.predict(X)

    out = pd.DataFrame(
        {
            "timestamp": latest.index,
            "yhat": y_hat,
        }
    )
    return out.reset_index(drop=True)
