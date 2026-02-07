from __future__ import annotations

import numpy as np


def horizon_bars(horizon_minutes: int, bar_minutes: int) -> float:
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive.")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    return horizon_minutes / bar_minutes


def sigma_fwd_from_yhat(yhat: float | np.ndarray) -> float | np.ndarray:
    yhat_arr = np.asarray(yhat, dtype=float)
    sigma_fwd = np.expm1(yhat_arr)
    if np.ndim(yhat_arr) == 0:
        return float(sigma_fwd)
    return sigma_fwd


def sigma_ann_from_sigma_fwd(
    sigma_fwd: float | np.ndarray,
    horizon_bars: float,
    bar_minutes: int,
) -> float | np.ndarray:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")

    sigma_fwd_arr = np.asarray(sigma_fwd, dtype=float)
    if horizon_bars <= 0:
        sigma_ann = np.full_like(sigma_fwd_arr, np.nan, dtype=float)
    else:
        sigma_per_bar = sigma_fwd_arr / np.sqrt(horizon_bars)
        bars_per_year = 365 * 24 * (60 / bar_minutes)
        sigma_ann = sigma_per_bar * np.sqrt(bars_per_year)

    sigma_ann = np.where(np.isfinite(sigma_ann), sigma_ann, np.nan)
    sigma_ann = np.where(sigma_ann == 0, np.nan, sigma_ann)
    if np.ndim(sigma_fwd_arr) == 0:
        return float(sigma_ann)
    return sigma_ann
