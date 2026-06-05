from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m
from lpbot.models.elasticnet_v1.inference import load_artifacts
from lpbot.models.elasticnet_v1.vol_mapping import sigma_ann_from_sigma_fwd, sigma_fwd_from_yhat


def _parse_riskoff_values(raw: str) -> tuple[set[str], set[float]]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    str_vals = {t.lower() for t in tokens}
    num_vals: set[float] = set()
    for token in tokens:
        try:
            num_vals.add(float(token))
        except ValueError:
            continue
    return str_vals, num_vals


def _ema_span_bars(ema_span_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    span = ema_span_minutes / bar_minutes
    span_int = int(round(span))
    return max(1, span_int)


def _horizon_bars(horizon_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars = horizon_minutes / bar_minutes
    return max(1, int(round(bars)))


def _trend_riskoff_with_hyst(
    close: pd.Series, ema: pd.Series, hyst_on: float, hyst_off: float
) -> pd.Series:
    if hyst_on <= 0 and hyst_off <= 0:
        return close < ema
    upper = ema * (1.0 + max(hyst_on, 0.0))
    lower = ema * (1.0 - max(hyst_off, 0.0))
    out = np.zeros(len(close), dtype=bool)
    is_riskoff = False
    for i in range(len(close)):
        c = close.iat[i]
        lo = lower.iat[i]
        hi = upper.iat[i]
        if np.isnan(c) or np.isnan(lo) or np.isnan(hi):
            out[i] = is_riskoff
            continue
        if is_riskoff:
            if c > hi:
                is_riskoff = False
        else:
            if c < lo:
                is_riskoff = True
        out[i] = is_riskoff
    return pd.Series(out, index=close.index)


def _apply_ramp(weight: pd.Series, trend_riskoff: pd.Series, ramp_bars: int) -> pd.Series:
    if ramp_bars <= 0:
        return weight
    out = weight.to_numpy().copy()
    ramp = 0
    in_ramp = False
    for i in range(len(out)):
        if bool(trend_riskoff.iat[i]):
            in_ramp = False
            ramp = 0
            continue
        if i > 0 and bool(trend_riskoff.iat[i - 1]):
            in_ramp = True
            ramp = 0
        if in_ramp:
            ramp += 1
            factor = min(1.0, ramp / float(ramp_bars))
            out[i] = out[i] * factor
            if factor >= 1.0:
                in_ramp = False
    return pd.Series(out, index=weight.index)


def _apply_weight_controls(
    weight: pd.Series, max_dw_per_bar: float, min_rebalance_delta: float
) -> pd.Series:
    if max_dw_per_bar <= 0 and min_rebalance_delta <= 0:
        return weight
    out = weight.to_numpy().copy()
    if len(out) == 0:
        return weight
    prev = float(out[0])
    for i in range(1, len(out)):
        target = float(out[i])
        if min_rebalance_delta > 0 and abs(target - prev) < min_rebalance_delta:
            target = prev
        if max_dw_per_bar > 0:
            delta = target - prev
            if delta > max_dw_per_bar:
                target = prev + max_dw_per_bar
            elif delta < -max_dw_per_bar:
                target = prev - max_dw_per_bar
        out[i] = target
        prev = target
    return pd.Series(out, index=weight.index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--model-dir", default="models/elasticnet-v1.0")
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--ema-span-minutes", type=float, default=None)
    p.add_argument("--trend-filter", choices=["none", "ema"], default="ema")
    p.add_argument("--trend-timeframe", choices=["same", "1h", "1d"], default="1h")
    p.add_argument("--trend-ema", type=int, default=200)
    p.add_argument(
        "--trend-hyst",
        type=float,
        default=0.0,
        help="Symmetric hysteresis band around EMA as fraction (e.g. 0.002 = 0.2%).",
    )
    p.add_argument(
        "--trend-hyst-on",
        type=float,
        default=None,
        help="Turn risk-off -> risk-on when price > EMA*(1+hyst_on).",
    )
    p.add_argument(
        "--trend-hyst-off",
        type=float,
        default=None,
        help="Turn risk-on -> risk-off when price < EMA*(1-hyst_off).",
    )
    p.add_argument("--riskoff-mode", choices=["none", "regime", "trend", "both"], default="both")
    p.add_argument(
        "--trend-scale",
        type=float,
        default=0.0,
        help="If >0 and trend risk-off, scale weight by this factor instead of forcing to 0.",
    )
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--target-vol", type=float, required=True)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--panic-vol", type=float, default=1.0)
    p.add_argument(
        "--max-dw-per-bar",
        type=float,
        default=0.0,
        help="Cap per-bar weight change magnitude (0 disables).",
    )
    p.add_argument(
        "--min-rebalance-delta",
        type=float,
        default=0.0,
        help="Ignore weight changes smaller than this threshold (0 disables).",
    )
    p.add_argument(
        "--ramp-bars",
        type=int,
        default=0,
        help="After trend flips risk-on, ramp weight over N bars (0 disables).",
    )
    p.add_argument(
        "--riskoff-values",
        default="bear,0,2",
        help="Comma-separated labels treated as risk-off (e.g. bear,0,2).",
    )
    p.add_argument("--out", default=None)

    args = p.parse_args()

    model, scaler, feature_list, meta = load_artifacts(args.model_dir)
    target_type = meta.get("target_type", "return")
    if target_type != "vol":
        raise SystemExit("cli_exposure expects a vol model (target_type='vol').")

    horizon_min = args.horizon if args.horizon is not None else meta.get("horizon_min")
    if horizon_min is None or int(horizon_min) <= 0:
        raise SystemExit("Provide a valid --horizon or ensure metadata has horizon_min.")
    horizon_min = int(horizon_min)
    horizon_bars = _horizon_bars(horizon_min, args.bar_minutes)

    df_1m = load_ohlcv_1m(args.input_1m)
    regime_df = None
    if args.regime_path is not None:
        regime_df = pd.read_csv(Path(args.regime_path))

    close_series = df_1m.set_index("timestamp")["close"].sort_index()
    trend_hyst_on = args.trend_hyst_on if args.trend_hyst_on is not None else args.trend_hyst
    trend_hyst_off = args.trend_hyst_off if args.trend_hyst_off is not None else args.trend_hyst

    if args.trend_filter == "none":
        trend_riskoff = pd.Series(False, index=close_series.index)
    else:
        if args.trend_timeframe == "same":
            ema_trend = close_series.ewm(span=args.trend_ema, adjust=False).mean()
            trend_riskoff = _trend_riskoff_with_hyst(
                close_series, ema_trend, float(trend_hyst_on), float(trend_hyst_off)
            )
        else:
            rule = "1h" if args.trend_timeframe == "1h" else "1d"
            close_resampled = close_series.resample(rule).last().dropna()
            ema_resampled = close_resampled.ewm(span=args.trend_ema, adjust=False).mean()
            ema_trend = ema_resampled.reindex(close_series.index, method="ffill")
            trend_riskoff = _trend_riskoff_with_hyst(
                close_series, ema_trend, float(trend_hyst_on), float(trend_hyst_off)
            )

    features = build_features(
        df_1m=df_1m,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        regime_df=regime_df,
    )

    features = features.set_index("timestamp")
    if feature_list:
        features = features[feature_list]

    X = scaler.transform(features.values)
    yhat = model.predict(X)

    out_df = pd.DataFrame(
        {
            "timestamp": features.index,
            "yhat": yhat,
        }
    )

    sigma_fwd = sigma_fwd_from_yhat(out_df["yhat"].astype(float))
    sigma_ann = sigma_ann_from_sigma_fwd(
        sigma_fwd, horizon_bars=horizon_bars, bar_minutes=args.bar_minutes
    )

    out_df["sigma_fwd"] = sigma_fwd
    out_df["sigma_ann"] = sigma_ann

    ema_span_minutes = (
        float(args.ema_span_minutes) if args.ema_span_minutes is not None else horizon_min
    )
    span_bars = _ema_span_bars(ema_span_minutes, args.bar_minutes)
    out_df["sigma_ann_smooth"] = (
        out_df["sigma_ann"].ewm(span=span_bars, adjust=False).mean()
    )

    regime_series = None
    if "regime_label" in df_1m.columns:
        regime_series = df_1m.set_index("timestamp")["regime_label"]
    elif regime_df is not None and "regime_label" in regime_df.columns and "timestamp" in regime_df.columns:
        reg = regime_df.copy()
        reg["timestamp"] = pd.to_datetime(reg["timestamp"], utc=True)
        regime_series = reg.set_index("timestamp")["regime_label"]

    riskoff_str, riskoff_num = _parse_riskoff_values(args.riskoff_values)
    if regime_series is not None:
        regime_mask = pd.Series(False, index=regime_series.index)
        if riskoff_str:
            regime_mask = regime_mask | regime_series.astype(str).str.lower().isin(riskoff_str)
        if riskoff_num:
            regime_num = pd.to_numeric(regime_series, errors="coerce")
            regime_mask = regime_mask | regime_num.isin(riskoff_num)
        regime_riskoff = regime_mask.reindex(out_df["timestamp"]).fillna(False)
    else:
        regime_riskoff = pd.Series(False, index=out_df["timestamp"])

    trend_riskoff = trend_riskoff.reindex(out_df["timestamp"]).fillna(False)

    use_regime = args.riskoff_mode in {"regime", "both"}
    use_trend = args.riskoff_mode in {"trend", "both"}
    gate_regime = regime_riskoff if use_regime else pd.Series(False, index=out_df["timestamp"])
    gate_trend = trend_riskoff if use_trend else pd.Series(False, index=out_df["timestamp"])

    denom = out_df["sigma_ann_smooth"]
    weight_raw = args.target_vol / denom
    weight_raw = weight_raw.where(denom > 0)
    weight_raw = weight_raw.replace([np.inf, -np.inf], np.nan)
    forced_zero = weight_raw.isna()
    weight_raw = weight_raw.fillna(0.0)
    weight_raw = weight_raw.clip(lower=0.0, upper=args.w_max)

    panic_mask = out_df["sigma_ann_smooth"] > args.panic_vol
    weight_raw = weight_raw.mask(panic_mask, 0.0)
    forced_zero = forced_zero | panic_mask

    weight = weight_raw.copy()
    if use_trend:
        if args.trend_scale > 0:
            weight = weight.where(~gate_trend.values, weight * args.trend_scale)
        else:
            weight = weight.mask(gate_trend.values, 0.0)
            forced_zero = forced_zero | gate_trend.values
    if use_regime:
        weight = weight.mask(gate_regime.values, 0.0)
        forced_zero = forced_zero | gate_regime.values

    weight = _apply_ramp(weight, gate_trend, args.ramp_bars)
    weight_pre_smooth = weight.copy()
    weight = _apply_weight_controls(weight, args.max_dw_per_bar, args.min_rebalance_delta)
    weight = weight.clip(lower=0.0, upper=args.w_max)

    if args.min_weight > 0:
        weight = weight.mask(~forced_zero, weight.clip(lower=args.min_weight, upper=args.w_max))

    out_df["weight"] = weight
    out_df["weight_pre_smooth"] = weight_pre_smooth
    out_df["weight_raw"] = weight_raw
    out_df["panic_triggered"] = panic_mask.to_numpy()
    out_df["sigma_ann_raw"] = out_df["sigma_ann"]
    out_df["close"] = close_series.reindex(out_df["timestamp"]).to_numpy()
    out_df["trend_riskoff"] = trend_riskoff.to_numpy()
    out_df["regime_riskoff"] = regime_riskoff.to_numpy()
    gate = gate_regime | gate_trend
    out_df["gate"] = gate.to_numpy()
    out_df = out_df[
        [
            "timestamp",
            "close",
            "weight",
            "weight_pre_smooth",
            "weight_raw",
            "yhat",
            "sigma_ann_raw",
            "sigma_ann_smooth",
            "trend_riskoff",
            "regime_riskoff",
            "gate",
            "panic_triggered",
        ]
    ]

    print(weight.describe().to_string())
    pct_zero = float((weight == 0).mean() * 100.0)
    pct_max = float((weight == args.w_max).mean() * 100.0)
    avg_weight = float(weight.mean())
    print(f"pct_weight_zero: {pct_zero:.2f}%")
    print(f"pct_weight_w_max: {pct_max:.2f}%")
    print(f"avg_weight: {avg_weight:.6f}")
    print(f"frac_trend_riskoff: {float(trend_riskoff.mean() * 100.0):.2f}%")
    if regime_series is not None:
        print(f"frac_regime_riskoff: {float(regime_riskoff.mean() * 100.0):.2f}%")
    else:
        print("frac_regime_riskoff: N/A")

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
