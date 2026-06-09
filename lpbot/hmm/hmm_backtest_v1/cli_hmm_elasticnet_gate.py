from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _parse_riskoff_values(raw: str) -> Tuple[set[str], set[float]]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    str_vals = set()
    num_vals = set()
    for p in parts:
        str_vals.add(p.lower())
        try:
            num_vals.add(float(p))
        except Exception:
            pass
    return str_vals, num_vals


def _match_values(series: pd.Series, str_vals: set[str], num_vals: set[float]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    if str_vals:
        mask = mask | series.astype(str).str.lower().isin(str_vals)
    if num_vals:
        num = pd.to_numeric(series, errors="coerce")
        mask = mask | num.isin(num_vals)
    return mask


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must contain timestamp, close")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def _load_exposure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "weight" not in df.columns:
        raise ValueError("exposure csv must contain timestamp, weight")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df[["timestamp", "weight"]]


def _load_regimes(path: Path, regime_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("regime csv must contain timestamp")
    if regime_col is None:
        if "state" in df.columns:
            regime_col = "state"
        elif "regime_label" in df.columns:
            regime_col = "regime_label"
        else:
            raise ValueError("regime csv must contain state or regime_label")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df[["timestamp", regime_col]].rename(columns={regime_col: "regime"})
    return df


def _merge_asof(base_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        base_df.sort_values("timestamp"),
        regime_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.dropna(subset=["regime"])
    return merged


def _compute_metrics(log_r: np.ndarray, bar_minutes: int) -> tuple[float, float, float, float]:
    log_r = np.asarray(log_r, float)
    log_r = log_r[np.isfinite(log_r)]
    if len(log_r) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(log_r))
    cagr = eq[-1] ** (bars_per_year / len(log_r)) - 1
    ann_vol = np.std(log_r) * np.sqrt(bars_per_year)
    sharpe = (np.mean(log_r) * bars_per_year) / (ann_vol + 1e-12)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return float(cagr), float(ann_vol), float(sharpe), float(mdd)


def _resampled_close_no_lookahead(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
) -> pd.Series:
    tf_map = {"same": f"{bar_minutes}min", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = tf_map[timeframe]
    tmp = pd.DataFrame({"close": close.to_numpy(dtype=float)}, index=ts)
    if tf == f"{bar_minutes}min":
        return tmp["close"]
    # Only completed HTF bars to avoid lookahead.
    return tmp["close"].resample(tf).last().shift(1).ffill()


def _hysteresis_state(
    on_cond: pd.Series,
    off_cond: pd.Series,
    persist_on_bars: int,
    persist_off_bars: int,
) -> pd.Series:
    on_arr = on_cond.fillna(False).to_numpy(dtype=bool)
    off_arr = off_cond.fillna(False).to_numpy(dtype=bool)

    n = len(on_arr)
    state = np.zeros(n, dtype=bool)
    is_on = False
    on_streak = 0
    off_streak = 0

    on_need = max(1, int(persist_on_bars))
    off_need = max(1, int(persist_off_bars))

    for i in range(n):
        on_streak = on_streak + 1 if on_arr[i] else 0
        off_streak = off_streak + 1 if off_arr[i] else 0

        if not is_on and on_streak >= on_need:
            is_on = True
            off_streak = 0
        elif is_on and off_streak >= off_need:
            is_on = False
            on_streak = 0

        state[i] = is_on

    return pd.Series(state, index=on_cond.index)


def _bull_features(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
    ema: int,
) -> tuple[pd.Series, pd.Series]:
    tclose = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=timeframe)
    level = tclose.ewm(span=max(2, int(ema)), adjust=False).mean()
    ratio = (tclose / level) - 1.0
    slope = level.pct_change()

    ratio = ratio.reindex(ts, method="ffill").fillna(0.0)
    slope = slope.reindex(ts, method="ffill").fillna(0.0)
    return ratio, slope


def _chop_signal(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
    fast_ema: int,
    slow_ema: int,
    strength_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    tclose = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=timeframe)
    ema_fast = tclose.ewm(span=max(2, int(fast_ema)), adjust=False).mean()
    ema_slow = tclose.ewm(span=max(3, int(slow_ema)), adjust=False).mean()
    strength = (ema_fast - ema_slow).abs() / (ema_slow.abs() + 1e-12)
    chop = strength < float(strength_threshold)

    strength = strength.reindex(ts, method="ffill").fillna(0.0)
    chop = chop.reindex(ts, method="ffill").fillna(True)
    return strength, chop


def _fast_derisk_signal(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
    ema_fast: int,
    ema_slow: int,
) -> pd.Series:
    tclose = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=timeframe)
    ef = tclose.ewm(span=max(2, int(ema_fast)), adjust=False).mean()
    es = tclose.ewm(span=max(3, int(ema_slow)), adjust=False).mean()
    sig = (ef < es) | (ef.diff() < 0)
    return sig.reindex(ts, method="ffill").fillna(False)


def _ema_on_timeframe(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
    span: int,
) -> pd.Series:
    tclose = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=timeframe)
    ema = tclose.ewm(span=max(2, int(span)), adjust=False).mean()
    return ema.reindex(ts, method="ffill").ffill().fillna(0.0)


def _vote_regime_state(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    tf1: str,
    tf2: str,
    ema_fast: int,
    ema_slow: int,
    neutral_ema_gap: float,
    neutral_price_gap: float,
) -> tuple[pd.Series, pd.Series]:
    close_tf1 = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=tf1)
    close_tf1 = close_tf1.reindex(ts, method="ffill").ffill().fillna(0.0)
    ema_fast_tf1 = _ema_on_timeframe(ts, close, bar_minutes, tf1, ema_fast)
    ema_slow_tf1 = _ema_on_timeframe(ts, close, bar_minutes, tf1, ema_slow)
    ema_slow_tf2 = _ema_on_timeframe(ts, close, bar_minutes, tf2, ema_slow)
    close_tf2 = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=tf2)
    close_tf2 = close_tf2.reindex(ts, method="ffill").ffill().fillna(0.0)

    sig1 = close_tf1 > ema_slow_tf1              # close > EMA200 (tf1)
    sig2 = ema_fast_tf1 > ema_slow_tf1           # EMA50 > EMA200 (tf1)
    sig3 = close_tf2 > ema_slow_tf2              # close > EMA200 (tf2)

    vote_count = sig1.astype(int) + sig2.astype(int) + sig3.astype(int)

    ema_gap = ((ema_fast_tf1 - ema_slow_tf1).abs() / (ema_slow_tf1.abs() + 1e-12))
    price_gap = ((close_tf1 - ema_slow_tf1).abs() / (ema_slow_tf1.abs() + 1e-12))
    neutral = (ema_gap < float(neutral_ema_gap)) | (price_gap < float(neutral_price_gap))

    state = pd.Series("bear", index=ts)
    state = state.mask(vote_count >= 2, "bull")
    state = state.mask(neutral, "neutral")
    return vote_count.astype(int), state


def _prob_bull_features(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    tf1: str,
    tf2: str,
    ema_fast: int,
    ema_slow: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close_tf1 = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=tf1)
    close_tf1 = close_tf1.reindex(ts, method="ffill").ffill().fillna(0.0)
    close_tf2 = _resampled_close_no_lookahead(ts=ts, close=close, bar_minutes=bar_minutes, timeframe=tf2)
    close_tf2 = close_tf2.reindex(ts, method="ffill").ffill().fillna(0.0)

    ema_fast_tf1 = _ema_on_timeframe(ts, close, bar_minutes, tf1, ema_fast)
    ema_slow_tf1 = _ema_on_timeframe(ts, close, bar_minutes, tf1, ema_slow)
    ema_slow_tf2 = _ema_on_timeframe(ts, close, bar_minutes, tf2, ema_slow)

    ema_gap_1h = (ema_fast_tf1 - ema_slow_tf1) / (ema_slow_tf1.abs() + 1e-12)
    price_gap_1h = (close_tf1 - ema_slow_tf1) / (ema_slow_tf1.abs() + 1e-12)
    price_gap_4h = (close_tf2 - ema_slow_tf2) / (ema_slow_tf2.abs() + 1e-12)

    vote_score = (
        (close_tf1 > ema_slow_tf1).astype(float)
        + (ema_fast_tf1 > ema_slow_tf1).astype(float)
        + (close_tf2 > ema_slow_tf2).astype(float)
    ) / 3.0
    return vote_score, ema_gap_1h, price_gap_1h, price_gap_4h


def main() -> None:
    p = argparse.ArgumentParser(description="ElasticNet exposure gated by HMM regimes.")
    p.add_argument("--price-csv", required=True)
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--regime-csv", required=True)
    p.add_argument("--regime-col", default=None)
    p.add_argument("--mode", choices=["gate", "scale"], default="gate")
    p.add_argument("--riskon-values", default="")
    p.add_argument("--neutral-values", default="")
    p.add_argument("--riskoff-values", default="bear,0,2")
    p.add_argument("--riskon-scale", type=float, default=1.0)
    p.add_argument("--neutral-scale", type=float, default=0.6)
    p.add_argument("--riskoff-scale", type=float, default=0.3)
    p.add_argument("--bull-override", action="store_true")
    p.add_argument("--bull-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--bull-ema", type=int, default=200)
    # Backward compatibility only (kept to avoid breaking old commands).
    p.add_argument("--bull-slope-ema", type=int, default=50)
    p.add_argument("--bull-on-buffer", type=float, default=0.005)
    p.add_argument("--bull-off-buffer", type=float, default=0.002)
    p.add_argument("--bull-slope-min", type=float, default=0.0)
    p.add_argument("--bull-persist-on-bars", type=int, default=3)
    p.add_argument("--bull-persist-off-bars", type=int, default=3)
    p.add_argument("--bull-min-weight", type=float, default=0.35)
    p.add_argument("--chop-cap-weight", type=float, default=0.35)
    p.add_argument("--chop-strength-threshold", type=float, default=0.003)
    p.add_argument("--chop-fast-ema", type=int, default=50)
    p.add_argument("--chop-slow-ema", type=int, default=200)
    p.add_argument("--disable-chop-cap", action="store_true")
    p.add_argument("--bull-fast-derisk", action="store_true")
    p.add_argument("--bull-derisk-timeframe", choices=["same", "1h", "4h", "1d"], default="same")
    p.add_argument("--bull-derisk-ema-fast", type=int, default=20)
    p.add_argument("--bull-derisk-ema-slow", type=int, default=50)
    p.add_argument("--bull-derisk-max-weight", type=float, default=0.45)
    p.add_argument("--vote-override", action="store_true")
    p.add_argument("--vote-tf1", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--vote-tf2", choices=["same", "1h", "4h", "1d"], default="4h")
    p.add_argument("--vote-ema-fast", type=int, default=50)
    p.add_argument("--vote-ema-slow", type=int, default=200)
    p.add_argument("--vote-neutral-ema-gap", type=float, default=0.003)
    p.add_argument("--vote-neutral-price-gap", type=float, default=0.002)
    p.add_argument("--vote-neutral-cap", type=float, default=0.35)
    p.add_argument("--vote-bear-cap", type=float, default=0.15)
    p.add_argument("--vote-bull-min", type=float, default=0.0)
    p.add_argument("--prob-override", action="store_true")
    p.add_argument("--prob-tf1", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--prob-tf2", choices=["same", "1h", "4h", "1d"], default="4h")
    p.add_argument("--prob-ema-fast", type=int, default=50)
    p.add_argument("--prob-ema-slow", type=int, default=200)
    p.add_argument("--prob-logit-k", type=float, default=3.0)
    p.add_argument("--prob-w-vote", type=float, default=1.0)
    p.add_argument("--prob-w-ema-gap", type=float, default=1.0)
    p.add_argument("--prob-w-price1", type=float, default=1.0)
    p.add_argument("--prob-w-price2", type=float, default=1.0)
    p.add_argument("--prob-w-regime", type=float, default=0.5)
    p.add_argument("--prob-scale-ema-gap", type=float, default=0.003)
    p.add_argument("--prob-scale-price-gap", type=float, default=0.004)
    p.add_argument("--prob-bull-threshold", type=float, default=0.62)
    p.add_argument("--prob-bear-threshold", type=float, default=0.38)
    p.add_argument("--prob-neutral-cap", type=float, default=0.35)
    p.add_argument("--prob-bear-cap", type=float, default=0.12)
    p.add_argument("--prob-bull-min", type=float, default=0.0)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    price_df = _load_price(Path(args.price_csv))
    exposure_df = _load_exposure(Path(args.exposure_csv))
    regime_df = _load_regimes(Path(args.regime_csv), args.regime_col)

    base = price_df.merge(exposure_df, on="timestamp", how="inner")
    merged = _merge_asof(base, regime_df)

    r = np.log(merged["close"] / merged["close"].shift(1))

    riskon_str, riskon_num = _parse_riskoff_values(args.riskon_values)
    neutral_str, neutral_num = _parse_riskoff_values(args.neutral_values)
    riskoff_str, riskoff_num = _parse_riskoff_values(args.riskoff_values)
    reg = merged["regime"]
    riskoff = _match_values(reg, riskoff_str, riskoff_num)
    neutral = _match_values(reg, neutral_str, neutral_num)
    riskon = _match_values(reg, riskon_str, riskon_num)

    if args.mode == "gate":
        regime_scale = (~riskoff).astype(float)
    else:
        # Default fallback is risk-on scale, then apply neutral/risk-off overrides.
        regime_scale = pd.Series(float(args.riskon_scale), index=merged.index)
        if args.riskon_values.strip():
            regime_scale[:] = float(args.neutral_scale)
            regime_scale = regime_scale.mask(riskon, float(args.riskon_scale))
        regime_scale = regime_scale.mask(neutral, float(args.neutral_scale))
        regime_scale = regime_scale.mask(riskoff, float(args.riskoff_scale))

    weight = merged["weight"] * regime_scale

    bull_on = pd.Series(False, index=merged.index)
    bull_ratio = pd.Series(0.0, index=merged.index)
    bull_slope = pd.Series(0.0, index=merged.index)
    trend_strength = pd.Series(0.0, index=merged.index)
    chop_on = pd.Series(False, index=merged.index)
    bull_derisk_on = pd.Series(False, index=merged.index)
    vote_count = pd.Series(0, index=merged.index, dtype=int)
    vote_state = pd.Series("na", index=merged.index)
    p_bull = pd.Series(0.5, index=merged.index)
    prob_state = pd.Series("neutral", index=merged.index)

    if args.bull_override:
        bull_ratio, bull_slope = _bull_features(
            ts=merged["timestamp"],
            close=merged["close"],
            bar_minutes=args.bar_minutes,
            timeframe=args.bull_timeframe,
            ema=int(args.bull_ema),
        )
        on_cond = (bull_ratio > float(args.bull_on_buffer)) & (bull_slope > float(args.bull_slope_min))
        off_cond = (bull_ratio < -float(args.bull_off_buffer)) | (bull_slope <= 0.0)
        bull_on = _hysteresis_state(
            on_cond=on_cond,
            off_cond=off_cond,
            persist_on_bars=int(args.bull_persist_on_bars),
            persist_off_bars=int(args.bull_persist_off_bars),
        )

        trend_strength, chop_on = _chop_signal(
            ts=merged["timestamp"],
            close=merged["close"],
            bar_minutes=args.bar_minutes,
            timeframe=args.bull_timeframe,
            fast_ema=int(args.chop_fast_ema),
            slow_ema=int(args.chop_slow_ema),
            strength_threshold=float(args.chop_strength_threshold),
        )

        w_floor = float(args.bull_min_weight)
        floor_mask = bull_on.to_numpy() & (~chop_on.to_numpy())
        weight = np.where(floor_mask, np.maximum(weight.to_numpy(), w_floor), weight.to_numpy())
        if not args.disable_chop_cap:
            weight = np.where(chop_on.to_numpy(), np.minimum(weight, float(args.chop_cap_weight)), weight)

        if args.bull_fast_derisk:
            bull_derisk_on = _fast_derisk_signal(
                ts=merged["timestamp"],
                close=merged["close"],
                bar_minutes=args.bar_minutes,
                timeframe=args.bull_derisk_timeframe,
                ema_fast=int(args.bull_derisk_ema_fast),
                ema_slow=int(args.bull_derisk_ema_slow),
            )
            mask = bull_on.to_numpy() & bull_derisk_on.to_numpy()
            weight = np.where(mask, np.minimum(weight, float(args.bull_derisk_max_weight)), weight)

        weight = pd.Series(weight, index=merged.index).clip(lower=0.0, upper=1.0)

    if args.vote_override:
        vote_count, vote_state = _vote_regime_state(
            ts=merged["timestamp"],
            close=merged["close"],
            bar_minutes=args.bar_minutes,
            tf1=args.vote_tf1,
            tf2=args.vote_tf2,
            ema_fast=int(args.vote_ema_fast),
            ema_slow=int(args.vote_ema_slow),
            neutral_ema_gap=float(args.vote_neutral_ema_gap),
            neutral_price_gap=float(args.vote_neutral_price_gap),
        )
        w = pd.to_numeric(weight, errors="coerce").fillna(0.0).to_numpy()
        is_bull = (vote_state == "bull").to_numpy()
        is_neutral = (vote_state == "neutral").to_numpy()
        is_bear = (vote_state == "bear").to_numpy()

        w = np.where(is_neutral, np.minimum(w, float(args.vote_neutral_cap)), w)
        w = np.where(is_bear, np.minimum(w, float(args.vote_bear_cap)), w)
        if float(args.vote_bull_min) > 0:
            w = np.where(is_bull, np.maximum(w, float(args.vote_bull_min)), w)
        weight = pd.Series(w, index=merged.index).clip(lower=0.0, upper=1.0)

    if args.prob_override:
        vote_score, ema_gap_1h, price_gap_1h, price_gap_4h = _prob_bull_features(
            ts=merged["timestamp"],
            close=merged["close"],
            bar_minutes=args.bar_minutes,
            tf1=args.prob_tf1,
            tf2=args.prob_tf2,
            ema_fast=int(args.prob_ema_fast),
            ema_slow=int(args.prob_ema_slow),
        )

        z_vote = (pd.to_numeric(vote_score, errors="coerce").fillna(0.5).to_numpy() - 0.5) * 2.0
        z_ema = np.tanh(pd.to_numeric(ema_gap_1h, errors="coerce").fillna(0.0).to_numpy() / max(1e-9, float(args.prob_scale_ema_gap)))
        z_p1 = np.tanh(pd.to_numeric(price_gap_1h, errors="coerce").fillna(0.0).to_numpy() / max(1e-9, float(args.prob_scale_price_gap)))
        z_p2 = np.tanh(pd.to_numeric(price_gap_4h, errors="coerce").fillna(0.0).to_numpy() / max(1e-9, float(args.prob_scale_price_gap)))
        z_reg = (pd.to_numeric(regime_scale, errors="coerce").fillna(0.5).to_numpy() - 0.5) * 2.0

        raw = (
            float(args.prob_w_vote) * z_vote
            + float(args.prob_w_ema_gap) * z_ema
            + float(args.prob_w_price1) * z_p1
            + float(args.prob_w_price2) * z_p2
            + float(args.prob_w_regime) * z_reg
        )
        p_bull_arr = 1.0 / (1.0 + np.exp(-float(args.prob_logit_k) * raw))
        p_bull_arr = np.nan_to_num(np.asarray(p_bull_arr, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
        p_bull = pd.Series(p_bull_arr, index=merged.index)

        bull_th = float(args.prob_bull_threshold)
        bear_th = float(args.prob_bear_threshold)
        if bear_th >= bull_th:
            raise ValueError("prob_bear_threshold must be < prob_bull_threshold")

        prob_state = pd.Series("neutral", index=merged.index)
        prob_state = prob_state.mask(p_bull >= bull_th, "bull")
        prob_state = prob_state.mask(p_bull <= bear_th, "bear")

        w = pd.to_numeric(weight, errors="coerce").fillna(0.0).to_numpy()
        p = p_bull.to_numpy()
        max_w = np.where(
            p <= bear_th,
            float(args.prob_bear_cap),
            np.where(
                p >= bull_th,
                1.0,
                float(args.prob_bear_cap)
                + (p - bear_th) / (bull_th - bear_th) * (1.0 - float(args.prob_bear_cap)),
            ),
        )
        w = np.minimum(w, max_w)
        if float(args.prob_neutral_cap) < 1.0:
            w = np.where((p > bear_th) & (p < bull_th), np.minimum(w, float(args.prob_neutral_cap)), w)
        if float(args.prob_bull_min) > 0:
            w = np.where(p >= bull_th, np.maximum(w, float(args.prob_bull_min)), w)
        weight = pd.Series(w, index=merged.index).clip(lower=0.0, upper=1.0)

    strat_r = weight.shift(1) * r

    out_df = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "close": merged["close"],
            "r": r,
            "regime": reg,
            "riskoff": riskoff.to_numpy(dtype=bool),
            "regime_scale": pd.to_numeric(regime_scale, errors="coerce").to_numpy(),
            "bull_on": bull_on.to_numpy(dtype=bool),
            "bull_ratio": pd.to_numeric(bull_ratio, errors="coerce").to_numpy(),
            "bull_slope": pd.to_numeric(bull_slope, errors="coerce").to_numpy(),
            "trend_strength": pd.to_numeric(trend_strength, errors="coerce").to_numpy(),
            "chop_on": chop_on.to_numpy(dtype=bool),
            "bull_derisk_on": bull_derisk_on.to_numpy(dtype=bool),
            "vote_count": pd.to_numeric(vote_count, errors="coerce").fillna(0).astype(int).to_numpy(),
            "vote_state": vote_state.astype(str).to_numpy(),
            "p_bull": pd.to_numeric(p_bull, errors="coerce").fillna(0.5).to_numpy(),
            "prob_state": prob_state.astype(str).to_numpy(),
            "weight": pd.to_numeric(weight, errors="coerce").to_numpy(),
            "strat_r": strat_r,
        }
    )
    out_df = out_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    cagr, ann_vol, sharpe, mdd = _compute_metrics(out_df["strat_r"].to_numpy(), args.bar_minutes)
    print(f"CAGR={cagr:.4f} | ann_vol={ann_vol:.4f} | Sharpe={sharpe:.4f} | max_drawdown={mdd:.4f}")
    print(
        f"rows={len(out_df)} mode={args.mode} "
        f"riskoff_pct={float(riskoff.mean()*100.0):.2f}% "
        f"avg_regime_scale={float(regime_scale.mean()):.3f} "
        f"bull_on_pct={float(bull_on.mean()*100.0):.2f}% "
        f"chop_on_pct={float(chop_on.mean()*100.0):.2f}% "
        f"bull_derisk_on_pct={float(bull_derisk_on.mean()*100.0):.2f}% "
        f"vote_override={args.vote_override} "
        f"prob_override={args.prob_override} "
        f"p_bull_mean={float(p_bull.mean()):.3f} "
        f"weight_zero_pct={float((weight==0).mean()*100.0):.2f}%"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"out={out_path}")


if __name__ == "__main__":
    main()
