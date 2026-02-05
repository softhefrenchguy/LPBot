from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Missing timestamp column. Columns: {list(df.columns)}")


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        median = int(s_num.dropna().median()) if s_num.dropna().empty is False else 0
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(s_num, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_ohlcv_1m(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    ts_col = _timestamp_column(df)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.reset_index(drop=True)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def _merge_regime_features(
    base: pd.DataFrame, regime_df: pd.DataFrame | None
) -> pd.DataFrame:
    if regime_df is None:
        return base

    ts_col = _timestamp_column(regime_df)
    reg = regime_df.rename(columns={ts_col: "timestamp"}).copy()
    reg["timestamp"] = _parse_timestamp_series(reg["timestamp"])
    reg = reg.dropna(subset=["timestamp"])
    reg = reg.sort_values("timestamp")
    reg = reg.drop_duplicates(subset=["timestamp"], keep="last")

    if "regime_label" in reg.columns:
        labels = reg["regime_label"]
        if not pd.api.types.is_numeric_dtype(labels):
            labels = labels.astype(str)
        dummies = pd.get_dummies(labels, prefix="regime_label")
        reg = pd.concat([reg.drop(columns=["regime_label"]), dummies], axis=1)

    feature_cols = [c for c in reg.columns if c != "timestamp"]
    if not feature_cols:
        return base

    for col in feature_cols:
        reg[col] = pd.to_numeric(reg[col], errors="coerce")

    reg = reg.dropna(subset=feature_cols)
    reg = reg.rename(columns={c: f"regime_{c}" for c in feature_cols})

    merged = base.merge(reg, on="timestamp", how="inner")
    return merged


def build_features(
    df_1m: pd.DataFrame,
    include_time_features: bool = True,
    atr_window: int = 14,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df_1m.copy()
    log_close = np.log(df["close"])

    df["r1"] = log_close.diff(1)
    df["r5"] = log_close.diff(5)
    df["r15"] = log_close.diff(15)
    df["r60"] = log_close.diff(60)

    df["vol_5"] = df["r1"].rolling(window=5, min_periods=5).std()
    df["vol_15"] = df["r1"].rolling(window=15, min_periods=15).std()
    df["vol_60"] = df["r1"].rolling(window=60, min_periods=60).std()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    df["atr_pct"] = atr / df["close"]

    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    df["dist_ema20"] = (df["close"] - ema20) / df["close"]
    df["dist_ema50"] = (df["close"] - ema50) / df["close"]
    df["ema20_slope"] = ema20.pct_change()
    df["ema50_slope"] = ema50.pct_change()

    if include_time_features:
        df = _add_time_features(df)

    feature_cols = [
        "r1",
        "r5",
        "r15",
        "r60",
        "vol_5",
        "vol_15",
        "vol_60",
        "atr_pct",
        "dist_ema20",
        "dist_ema50",
        "ema20_slope",
        "ema50_slope",
    ]

    if include_time_features:
        feature_cols += ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]

    features = df[["timestamp", *feature_cols]].copy()
    features = features.dropna()
    features = _merge_regime_features(features, regime_df)
    features = features.sort_values("timestamp").reset_index(drop=True)
    return features
