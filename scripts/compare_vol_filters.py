from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset


def _stats(r: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan}
    eq = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=0) * np.sqrt(252.0))
    excess = r - (0.05 / 252.0)
    sharpe = float(excess.mean() / excess.std(ddof=0) * np.sqrt(252.0)) if excess.std(ddof=0) > 0 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "ann_vol": ann_vol}


def _pct_rank_last(x: pd.Series) -> float:
    if x.isna().all():
        return np.nan
    return float(x.rank(pct=True).iloc[-1])


def _mult_from_percentile(pct: pd.Series) -> pd.Series:
    out = pd.Series(1.0, index=pct.index)
    out.loc[pct > 0.75] = 0.5
    out.loc[pct < 0.25] = 1.2
    return out


def _load_strategy_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d["combined_return"] = pd.to_numeric(d["combined_return"], errors="coerce").fillna(0.0)
    d["eth_spot_return"] = pd.to_numeric(d["eth_spot_return"], errors="coerce")
    return d.dropna(subset=["day"]).sort_values("day")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare realised-vol and ElasticNet predicted-vol filters.")
    ap.add_argument("--ohlcv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--bar-minutes", type=int, default=5)
    ap.add_argument("--horizon-bars", type=int, default=3, help="3 bars on 5m data = 15 minutes.")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--strategy-daily", default="artifacts/backtest/eth_btc_portfolio_daily_20bps.csv")
    ap.add_argument("--out-summary", default="artifacts/backtest/vol_filter_comparison.csv")
    ap.add_argument("--out-daily", default="artifacts/backtest/vol_filter_comparison_daily.csv")
    args = ap.parse_args()

    ohlcv = Path(args.ohlcv)
    if not ohlcv.exists():
        raise SystemExit(f"Missing OHLCV file: {ohlcv}")

    dataset = build_supervised_dataset(
        path_1m=ohlcv,
        horizon_min=int(args.horizon_bars),
        target_type="vol",
    )
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["timestamp"]).sort_values("timestamp")

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    test_start = pd.Timestamp(args.test_start, tz="UTC")
    train = dataset[dataset["timestamp"] <= train_end].copy()
    test = dataset[dataset["timestamp"] >= test_start].copy()
    if train.empty or test.empty:
        raise SystemExit(f"Insufficient dataset rows. train={len(train)} test={len(test)}")

    feature_cols = [c for c in dataset.columns if c not in {"timestamp", "target"}]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols].values)
    y_train = train["target"].to_numpy()
    x_test = scaler.transform(test[feature_cols].values)
    y_test = test["target"].to_numpy()

    model = ElasticNet(alpha=1e-4, l1_ratio=0.1, max_iter=10000, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    test_pred = test[["timestamp", "target"]].copy()
    test_pred["pred_target"] = pred
    test_pred["sigma_fwd"] = np.expm1(pd.to_numeric(test_pred["target"], errors="coerce"))
    test_pred["sigma_pred"] = np.expm1(pd.to_numeric(test_pred["pred_target"], errors="coerce"))
    horizon_bars = float(args.horizon_bars)
    bars_per_year = 365.0 * 24.0 * (60.0 / float(args.bar_minutes))
    test_pred["pred_ann_vol"] = (test_pred["sigma_pred"] / np.sqrt(horizon_bars)) * np.sqrt(bars_per_year)
    test_pred["actual_ann_vol"] = (test_pred["sigma_fwd"] / np.sqrt(horizon_bars)) * np.sqrt(bars_per_year)
    test_pred["day"] = test_pred["timestamp"].dt.floor("D")

    pred_daily = (
        test_pred.groupby("day")
        .agg(pred_ann_vol=("pred_ann_vol", "median"), actual_ann_vol=("actual_ann_vol", "median"))
        .reset_index()
    )
    pred_daily["elastic_vol_percentile"] = pred_daily["pred_ann_vol"].rolling(252, min_periods=60).apply(_pct_rank_last, raw=False)
    pred_daily["elastic_multiplier"] = _mult_from_percentile(pred_daily["elastic_vol_percentile"])

    strat = _load_strategy_daily(Path(args.strategy_daily))
    strat["realized_vol_20d"] = strat["eth_spot_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    strat["realized_vol_percentile"] = strat["realized_vol_20d"].rolling(252, min_periods=60).apply(_pct_rank_last, raw=False)
    strat["realized_multiplier"] = _mult_from_percentile(strat["realized_vol_percentile"])

    daily = strat.merge(pred_daily[["day", "pred_ann_vol", "actual_ann_vol", "elastic_vol_percentile", "elastic_multiplier"]], on="day", how="left")
    daily["elastic_multiplier"] = pd.to_numeric(daily["elastic_multiplier"], errors="coerce").fillna(1.0)
    daily["realized_multiplier"] = pd.to_numeric(daily["realized_multiplier"], errors="coerce").fillna(1.0)

    daily["baseline_return"] = daily["combined_return"]
    daily["realized_filter_return"] = daily["combined_return"] * daily["realized_multiplier"]
    daily["elastic_filter_return"] = daily["combined_return"] * daily["elastic_multiplier"]

    rows = []
    for name, col in [
        ("baseline_20bps", "baseline_return"),
        ("realized_vol_filter_proxy", "realized_filter_return"),
        ("elasticnet_vol_filter_proxy", "elastic_filter_return"),
    ]:
        st = _stats(daily[col])
        rows.append({"configuration": name, **st})

    mse = float(mean_squared_error(y_test, pred))
    mae = float(mean_absolute_error(y_test, pred))
    corr = float(pd.Series(y_test).corr(pd.Series(pred)))
    summary = pd.DataFrame(rows)
    summary["model_mse"] = mse
    summary["model_mae"] = mae
    summary["model_corr"] = corr
    summary["train_rows"] = len(train)
    summary["test_rows"] = len(test)
    summary["source_ohlcv"] = str(ohlcv)
    summary["bar_minutes"] = int(args.bar_minutes)
    summary["horizon_bars"] = int(args.horizon_bars)

    out_summary = Path(args.out_summary)
    out_daily = Path(args.out_daily)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    daily.to_csv(out_daily, index=False)

    print("================================================================")
    print("VOL FILTER COMPARISON")
    print("================================================================")
    print(f"OHLCV: {ohlcv} ({args.bar_minutes}m proxy; horizon={args.horizon_bars} bars)")
    print(f"ElasticNet test mse={mse:.8f} mae={mae:.8f} corr={corr:.4f}")
    print(summary[["configuration", "cagr", "sharpe", "maxdd", "ann_vol"]].to_string(index=False))
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_daily}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
