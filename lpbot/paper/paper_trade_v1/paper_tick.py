from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def per_bar_rate_from_annual(rate_ann: float, bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return float(rate_ann) / bars_per_year


def compute_in_range_frac_eff(
    in_range_frac: float,
    range_sigma: float,
    range_sigma_ref: float,
    min_frac: float,
    max_frac: float,
) -> float:
    if range_sigma <= 0:
        raise ValueError("range_sigma must be positive.")
    raw = float(in_range_frac) * (float(range_sigma_ref) / float(range_sigma))
    return float(min(max(raw, min_frac), max_frac))


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


def compute_paper_rows(
    exposure_df: pd.DataFrame,
    price_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    bar_minutes: int,
    fee_tier: float,
    pool_tvl_usd: float,
    in_range_frac: float,
    range_sigma: float,
    range_sigma_ref: float,
    min_in_range_frac: float,
    max_in_range_frac: float,
    churn_k: float,
    il_k: float,
    lp_vol_on: float,
    lp_scale: float,
    lp_weight_max: float,
    lp_min_on_bars: int,
    lp_cooldown_bars: int,
) -> pd.DataFrame:
    required_exposure = {"timestamp", "weight", "sigma_ann_smooth", "gate"}
    required_price = {"timestamp", "close"}
    required_vol = {"timestamp", "volume_usd"}

    missing = required_exposure - set(exposure_df.columns)
    if missing:
        raise ValueError(f"Missing exposure columns: {sorted(missing)}")
    missing = required_price - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    missing = required_vol - set(vol_df.columns)
    if missing:
        raise ValueError(f"Missing volume columns: {sorted(missing)}")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    if pool_tvl_usd <= 0:
        raise ValueError("pool_tvl_usd must be positive.")

    exp = exposure_df.copy()
    px = price_df.copy()
    vol = vol_df.copy()

    exp["timestamp"] = pd.to_datetime(exp["timestamp"], utc=True, errors="coerce")
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
    vol["timestamp"] = pd.to_datetime(vol["timestamp"], utc=True, errors="coerce")

    exp = exp.dropna(subset=["timestamp"])
    px = px.dropna(subset=["timestamp"])
    vol = vol.dropna(subset=["timestamp"])

    merged = exp.merge(px, on="timestamp", how="inner", suffixes=("_exp", "_px"))
    merged = merged.merge(vol, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    close_col = "close"
    if close_col not in merged.columns:
        if "close_px" in merged.columns:
            close_col = "close_px"
        elif "close_exp" in merged.columns:
            close_col = "close_exp"
        else:
            raise ValueError("Missing close column after merge.")

    close = pd.to_numeric(merged[close_col], errors="coerce")
    r = np.log(close / close.shift(1))

    gate_bool = merged["gate"].fillna(False).astype(bool)
    weight = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    weight_raw = pd.to_numeric(merged.get("weight_raw"), errors="coerce")
    sigma_ann_smooth = pd.to_numeric(merged["sigma_ann_smooth"], errors="coerce")
    sigma_ann_raw = pd.to_numeric(
        merged.get("sigma_ann_raw", merged.get("sigma_ann")), errors="coerce"
    )
    trend_riskoff = merged.get("trend_riskoff", pd.Series(False, index=merged.index))
    regime_riskoff = merged.get("regime_riskoff", pd.Series(False, index=merged.index))
    panic_triggered = merged.get(
        "panic_triggered", pd.Series(False, index=merged.index)
    ).astype(bool)

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

    in_range_frac_eff = compute_in_range_frac_eff(
        in_range_frac,
        range_sigma,
        range_sigma_ref,
        min_in_range_frac,
        max_in_range_frac,
    )
    in_range_frac_eff_series = pd.Series(
        in_range_frac_eff, index=merged.index, dtype=float
    )

    volume_usd = pd.to_numeric(merged["volume_usd"], errors="coerce").fillna(0.0)
    lp_weight_lag = lp_weight.shift(1)
    fee_r = lp_weight_lag * (volume_usd * fee_tier * in_range_frac_eff_series) / pool_tvl_usd

    il_r = -lp_weight_lag * il_k * (r**2)

    bars_per_year = 365 * 24 * (60 / bar_minutes)
    sigma_bar = sigma_ann_smooth / math.sqrt(bars_per_year)
    band = range_sigma * sigma_bar
    churn_intensity = (r.abs() / (band + 1e-12)) - 1.0
    churn_intensity = churn_intensity.clip(lower=0.0)
    churn_r = -lp_weight_lag * churn_k * churn_intensity

    overlay_r = fee_r + il_r + churn_r
    core_r = weight.shift(1) * r
    combined_r = core_r + overlay_r

    out = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "close": close,
            "r": r,
            "gate": gate_bool,
            "weight": weight,
            "weight_raw": weight_raw,
            "trend_riskoff": trend_riskoff.astype(bool),
            "regime_riskoff": regime_riskoff.astype(bool),
            "panic_triggered": panic_triggered,
            "sigma_ann_raw": sigma_ann_raw,
            "sigma_ann_smooth": sigma_ann_smooth,
            "lp_raw_on": raw_on,
            "lp_on": lp_on,
            "lp_weight": lp_weight,
            "in_range_frac_eff": in_range_frac_eff_series,
            "churn_intensity": churn_intensity,
            "fee_r": fee_r,
            "il_r": il_r,
            "churn_r": churn_r,
            "overlay_r": overlay_r,
            "core_r": core_r,
            "combined_r": combined_r,
        }
    )

    return out[
        [
            "timestamp",
            "close",
            "r",
            "gate",
            "weight",
            "weight_raw",
            "trend_riskoff",
            "regime_riskoff",
            "panic_triggered",
            "sigma_ann_raw",
            "sigma_ann_smooth",
            "lp_raw_on",
            "lp_on",
            "lp_weight",
            "in_range_frac_eff",
            "churn_intensity",
            "fee_r",
            "il_r",
            "churn_r",
            "overlay_r",
            "core_r",
            "combined_r",
        ]
    ]
