from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def per_bar_rate_from_annual(rate_ann: float, bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return float(rate_ann) / bars_per_year


def _build_lp_on(
    raw_on: pd.Series, stable_on: pd.Series, lp_cooldown_bars: int
) -> pd.Series:
    lp_on = np.zeros(len(raw_on), dtype=bool)
    cooldown = 0
    is_on = False
    for i in range(len(raw_on)):
        raw = bool(raw_on.iat[i])
        stable = bool(stable_on.iat[i])
        if is_on and not raw:
            is_on = False
            cooldown = lp_cooldown_bars
        if cooldown > 0:
            cooldown -= 1
            is_on = False
        elif not is_on and stable:
            is_on = True
        lp_on[i] = is_on
    return pd.Series(lp_on, index=raw_on.index)


def compute_lp_overlay_returns(
    df: pd.DataFrame,
    bar_minutes: int,
    fee_rate_ann: float,
    il_k: float,
    lp_vol_on: float,
    lp_scale: float,
    lp_weight_max: float,
    lp_min_on_bars: int = 6,
    lp_cooldown_bars: int = 12,
) -> pd.DataFrame:
    required = {"timestamp", "close", "weight", "sigma_ann_smooth", "gate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    r = np.log(close / close.shift(1))
    out["r"] = r

    gate_bool = out["gate"].fillna(False).astype(bool)
    weight = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    sigma_ann_smooth = pd.to_numeric(out["sigma_ann_smooth"], errors="coerce")

    raw_on = (~gate_bool) & (weight > 0) & (sigma_ann_smooth < lp_vol_on)
    stable_on = (
        raw_on.rolling(lp_min_on_bars, min_periods=lp_min_on_bars)
        .min()
        .fillna(False)
        .astype(bool)
    )
    lp_on = _build_lp_on(raw_on, stable_on, lp_cooldown_bars)

    lp_weight = (weight * lp_scale).clip(lower=0.0, upper=lp_weight_max)
    lp_weight = lp_weight * lp_on.astype(float)

    fee_per_bar = per_bar_rate_from_annual(fee_rate_ann, bar_minutes)
    lp_weight_lag = lp_weight.shift(1)
    fee_r = lp_weight_lag * fee_per_bar
    il_r = -lp_weight_lag * il_k * (r**2)
    overlay_r = fee_r + il_r

    core_r = weight.shift(1) * r
    combined_r = core_r + overlay_r

    out["raw_on"] = raw_on
    out["stable_on"] = stable_on
    out["lp_on"] = lp_on
    out["lp_weight"] = lp_weight
    out["core_r"] = core_r
    out["fee_r"] = fee_r
    out["il_r"] = il_r
    out["overlay_r"] = overlay_r
    out["combined_r"] = combined_r

    out = out[
        [
            "timestamp",
            "r",
            "weight",
            "sigma_ann_smooth",
            "gate",
            "raw_on",
            "stable_on",
            "lp_on",
            "lp_weight",
            "core_r",
            "fee_r",
            "il_r",
            "overlay_r",
            "combined_r",
        ]
    ]
    return out
