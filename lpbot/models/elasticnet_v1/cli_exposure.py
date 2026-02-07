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
    p.add_argument("--riskoff-mode", choices=["none", "regime", "trend", "both"], default="both")
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--target-vol", type=float, required=True)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--panic-vol", type=float, default=1.0)
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
    if args.trend_filter == "none":
        trend_riskoff = pd.Series(False, index=close_series.index)
    else:
        if args.trend_timeframe == "same":
            ema_trend = close_series.ewm(span=args.trend_ema, adjust=False).mean()
            trend_riskoff = close_series < ema_trend
        else:
            rule = "1h" if args.trend_timeframe == "1h" else "1d"
            close_resampled = close_series.resample(rule).last().dropna()
            ema_resampled = close_resampled.ewm(span=args.trend_ema, adjust=False).mean()
            ema_trend = ema_resampled.reindex(close_series.index, method="ffill")
            trend_riskoff = close_series < ema_trend

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

    if args.riskoff_mode == "none":
        gate = pd.Series(False, index=out_df["timestamp"])
    elif args.riskoff_mode == "regime":
        gate = regime_riskoff
    elif args.riskoff_mode == "trend":
        gate = trend_riskoff
    else:
        gate = regime_riskoff | trend_riskoff

    denom = out_df["sigma_ann_smooth"]
    weight = args.target_vol / denom
    weight = weight.where(denom > 0)
    weight = weight.replace([np.inf, -np.inf], np.nan)
    forced_zero = weight.isna()
    weight = weight.fillna(0.0)
    weight = weight.clip(lower=0.0, upper=args.w_max)

    panic_mask = out_df["sigma_ann_smooth"] > args.panic_vol
    weight = weight.mask(panic_mask, 0.0)
    forced_zero = forced_zero | panic_mask

    weight = weight.mask(gate.values, 0.0)
    forced_zero = forced_zero | gate.values

    if args.min_weight > 0:
        weight = weight.mask(~forced_zero, weight.clip(lower=args.min_weight, upper=args.w_max))

    out_df["weight"] = weight
    out_df["close"] = close_series.reindex(out_df["timestamp"]).to_numpy()
    out_df["trend_riskoff"] = trend_riskoff.to_numpy()
    out_df["regime_riskoff"] = regime_riskoff.to_numpy()
    out_df["gate"] = gate.to_numpy()
    out_df = out_df[
        [
            "timestamp",
            "close",
            "weight",
            "yhat",
            "sigma_ann_smooth",
            "trend_riskoff",
            "regime_riskoff",
            "gate",
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
