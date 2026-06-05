from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_regime_daily(path: Path, regime_col: str) -> pd.DataFrame:
    r = pd.read_csv(path)
    if regime_col not in r.columns:
        raise ValueError(f"missing {regime_col} in {path}")
    if "day" in r.columns:
        r["day"] = pd.to_datetime(r["day"], utc=True, errors="coerce").dt.floor("D")
    elif "timestamp" in r.columns:
        r["day"] = pd.to_datetime(r["timestamp"], utc=True, errors="coerce").dt.floor("D")
    else:
        raise ValueError(f"{path} must contain day or timestamp")
    out = r.dropna(subset=["day"])[["day", regime_col]].drop_duplicates(subset=["day"], keep="last")
    return out.rename(columns={regime_col: "regime_label"})


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bars_per_year)),
        "sharpe": float((np.mean(x) * bars_per_year) / (np.std(x) * np.sqrt(bars_per_year) + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def build_regime_flags(spot_r: pd.Series, spot_close: pd.Series, horizon_days: int = 90) -> pd.DataFrame:
    bars_per_day = 24 * 12  # 5m bars
    h = horizon_days * bars_per_day
    fwd = np.log(spot_close.shift(-h) / spot_close)
    fwd_ret = np.exp(fwd) - 1.0

    vol_30d = spot_r.rolling(30 * bars_per_day, min_periods=10 * bars_per_day).std() * np.sqrt(365 * 24 * 12)
    vol_thr = float(vol_30d.quantile(0.90))

    out = pd.DataFrame(
        {
            "fwd_90d_ret": fwd_ret,
            "vol_30d_ann": vol_30d,
        }
    )
    out["uptrend_90d"] = out["fwd_90d_ret"] >= 0.20
    out["downtrend_90d"] = out["fwd_90d_ret"] <= -0.20
    out["sideways_90d"] = out["fwd_90d_ret"].abs() <= 0.10
    out["high_vol_spike"] = out["vol_30d_ann"] >= vol_thr
    out["vol_90p_threshold"] = vol_thr
    return out


def monthly_walkforward_summary(df: pd.DataFrame, strat_col: str, spot_col: str) -> pd.DataFrame:
    # This is rolling train/test evaluation of realized OOS returns.
    # Train window: 12 months, test window: 1 month.
    d = df.copy()
    d["month"] = d["timestamp"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    months = sorted(d["month"].dropna().unique())
    rows: list[dict[str, float | int | str]] = []
    for i in range(12, len(months)):
        train_months = set(months[i - 12 : i])
        test_month = months[i]
        tr = d[d["month"].isin(train_months)].copy()
        te = d[d["month"] == test_month].copy()
        if len(tr) == 0 or len(te) == 0:
            continue
        tr_m = perf(tr[strat_col], 5)
        te_m = perf(te[strat_col], 5)
        te_spot = perf(te[spot_col], 5)
        rows.append(
            {
                "test_month": str(pd.Timestamp(test_month).date()),
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_sharpe": tr_m["sharpe"],
                "test_ret": te_m["ret"],
                "test_sharpe": te_m["sharpe"],
                "test_spot_ret": te_spot["ret"],
                "test_excess": te_m["ret"] - te_spot["ret"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="3-way split + walk-forward + regime coverage protocol check.")
    ap.add_argument("--strategy-csv", required=True)
    ap.add_argument("--price-csv", required=True)
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--strat-ret-col", default="strat_r")
    ap.add_argument("--spot-ret-col", default="spot_r")
    ap.add_argument("--spot-close-col", default="eth_close")
    ap.add_argument("--regime-csv", default="", help="Optional daily regime csv (with day or timestamp)")
    ap.add_argument("--regime-col", default="regime_v2", help="Regime column in regime-csv")
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--valid-years", type=int, default=1)
    ap.add_argument("--test-years", type=int, default=1)
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/oos_protocol_summary.csv")
    ap.add_argument("--out-wf-csv", default="artifacts/backtest/oos_protocol_walkforward_monthly.csv")
    args = ap.parse_args()

    s = pd.read_csv(args.strategy_csv)
    s[args.timestamp_col] = pd.to_datetime(s[args.timestamp_col], utc=True, errors="coerce")
    for c in [args.strat_ret_col, args.spot_ret_col]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=[args.timestamp_col, args.strat_ret_col, args.spot_ret_col]).sort_values(args.timestamp_col).copy()
    s = s.rename(columns={args.timestamp_col: "timestamp", args.strat_ret_col: "strat_r", args.spot_ret_col: "spot_r"})
    if args.spot_close_col not in s.columns:
        raise ValueError(f"missing {args.spot_close_col} in strategy csv")
    s["spot_close"] = pd.to_numeric(s[args.spot_close_col], errors="coerce")

    p = pd.read_csv(args.price_csv, usecols=[args.timestamp_col])
    p[args.timestamp_col] = pd.to_datetime(p[args.timestamp_col], utc=True, errors="coerce")
    p = p.dropna(subset=[args.timestamp_col]).sort_values(args.timestamp_col)

    if args.regime_csv:
        reg = load_regime_daily(Path(args.regime_csv), args.regime_col)
        s["day"] = s["timestamp"].dt.floor("D")
        s = s.merge(reg, on="day", how="left")
        s["regime_label"] = s["regime_label"].fillna("CHOP")
    else:
        s["regime_label"] = np.nan

    start = s["timestamp"].iloc[0]
    t1 = start + pd.DateOffset(years=int(args.train_years))
    t2 = t1 + pd.DateOffset(years=int(args.valid_years))
    t3 = t2 + pd.DateOffset(years=int(args.test_years))

    # clamp test end to available data
    data_end = s["timestamp"].iloc[-1]
    if t3 > data_end:
        t3 = data_end

    train = s[(s["timestamp"] >= start) & (s["timestamp"] < t1)].copy()
    valid = s[(s["timestamp"] >= t1) & (s["timestamp"] < t2)].copy()
    test = s[(s["timestamp"] >= t2) & (s["timestamp"] <= t3)].copy()

    parts = [("train", train), ("validate", valid), ("test", test)]
    rows = []
    for name, d in parts:
        m = perf(d["strat_r"], 5) if len(d) else {k: np.nan for k in ["ret", "cagr", "ann_vol", "sharpe", "max_dd"]}
        ms = perf(d["spot_r"], 5) if len(d) else {k: np.nan for k in ["ret", "cagr", "ann_vol", "sharpe", "max_dd"]}
        rows.append(
            {
                "split": name,
                "rows": int(len(d)),
                "start": str(d["timestamp"].iloc[0]) if len(d) else "",
                "end": str(d["timestamp"].iloc[-1]) if len(d) else "",
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "excess_vs_spot": m["ret"] - ms["ret"] if np.isfinite(m["ret"]) and np.isfinite(ms["ret"]) else np.nan,
                "regime_bull_pct": float((d["regime_label"] == "BULL").mean() * 100.0) if len(d) and d["regime_label"].notna().any() else np.nan,
                "regime_chop_pct": float((d["regime_label"] == "CHOP").mean() * 100.0) if len(d) and d["regime_label"].notna().any() else np.nan,
                "regime_bear_pct": float((d["regime_label"] == "BEAR").mean() * 100.0) if len(d) and d["regime_label"].notna().any() else np.nan,
            }
        )

    # OOS ratio vs raw market rows across same calendar range.
    overall_start = start
    overall_end = t3
    p_rng = p[(p[args.timestamp_col] >= overall_start) & (p[args.timestamp_col] <= overall_end)]
    oos_ratio = float(len(s[(s["timestamp"] >= overall_start) & (s["timestamp"] <= overall_end)]) / max(1, len(p_rng)))

    # Regime coverage on TEST split.
    reg = build_regime_flags(test["spot_r"], test["spot_close"], horizon_days=90) if len(test) else pd.DataFrame()
    regime_row = {
        "split": "test_regime_coverage",
        "rows": int(len(test)),
        "start": str(test["timestamp"].iloc[0]) if len(test) else "",
        "end": str(test["timestamp"].iloc[-1]) if len(test) else "",
        "uptrend_90d_present": bool(reg["uptrend_90d"].any()) if len(reg) else False,
        "downtrend_90d_present": bool(reg["downtrend_90d"].any()) if len(reg) else False,
        "sideways_90d_present": bool(reg["sideways_90d"].any()) if len(reg) else False,
        "high_vol_spike_present": bool(reg["high_vol_spike"].any()) if len(reg) else False,
        "vol_90p_threshold": float(reg["vol_90p_threshold"].iloc[0]) if len(reg) else np.nan,
    }

    summary = pd.DataFrame(rows)
    summary["oos_ratio_vs_price_rows"] = oos_ratio
    summary["oos_ratio_pass_30pct"] = summary["oos_ratio_vs_price_rows"] >= 0.30
    summary = pd.concat([summary, pd.DataFrame([regime_row])], ignore_index=True)

    wf = monthly_walkforward_summary(s[(s["timestamp"] >= overall_start) & (s["timestamp"] <= overall_end)], "strat_r", "spot_r")
    wf_stats = {
        "wf_test_months": int(len(wf)),
        "wf_positive_months_pct": float((wf["test_ret"] > 0).mean() * 100.0) if len(wf) else np.nan,
        "wf_avg_test_ret": float(wf["test_ret"].mean()) if len(wf) else np.nan,
        "wf_avg_test_excess": float(wf["test_excess"].mean()) if len(wf) else np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([{"split": "walkforward_summary", **wf_stats}])], ignore_index=True)

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    out_wf = Path(args.out_wf_csv)
    out_wf.parent.mkdir(parents=True, exist_ok=True)
    wf.to_csv(out_wf, index=False)

    print(f"wrote {out_summary}")
    print(f"wrote {out_wf}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
