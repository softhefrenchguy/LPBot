from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_btc_full_stack import (
    _download_binance_daily,
    classify_regime_v2_on_btc,
    load_eth_defensive_proxy,
    perf,
)


def grade_from_sharpe(sharpe: float) -> str:
    if np.isfinite(sharpe) and sharpe >= 0.8:
        return "PASS"
    if np.isfinite(sharpe) and sharpe >= 0.6:
        return "MARGINAL"
    return "FAIL"


def run_btc_backtest(
    price_df: pd.DataFrame,
    def_proxy_daily: pd.DataFrame,
    *,
    confirm_days: int = 5,
    cost_bps: float = 10.0,
    lag_days: int = 1,
    test_start: str | None = None,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    d = price_df.copy()
    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)

    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema55"] = d["close"].ewm(span=55, adjust=False).mean()
    d["ema144"] = d["close"].ewm(span=144, adjust=False).mean()
    d["stack_aligned"] = (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])

    conf = max(1, int(confirm_days))
    entry = (d["stack_aligned"].rolling(conf, min_periods=conf).min() == 1).fillna(False)
    exit_ = ((~d["stack_aligned"]).rolling(conf, min_periods=conf).min() == 1).fillna(False)

    rv = d["close"].pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    d["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(d), dtype=int)
    active = 0
    for i in range(len(d)):
        if active == 0 and bool(entry.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
        pos[i] = active

    d["off_active"] = pos
    d["off_raw_signal"] = np.where(d["off_active"] == 1, d["vol_scalar"], 0.0)
    off_scale_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    d["off_scale"] = d["regime_v2"].map(off_scale_map).fillna(0.0)
    d["off_target"] = d["off_raw_signal"] * d["off_scale"]

    d = d.merge(def_proxy_daily, on="day", how="left")
    d["def_proxy_signal"] = pd.to_numeric(d["def_proxy_signal"], errors="coerce").fillna(0.0)
    def_scale_map = {"BULL": 0.0, "CHOP": 0.4, "BEAR": 1.0}
    d["def_scale"] = d["regime_v2"].map(def_scale_map).fillna(0.0)
    d["def_target"] = d["def_proxy_signal"] * d["def_scale"]

    d["weight_target"] = (d["off_target"] + d["def_target"]).clip(lower=0.0, upper=1.0)
    d["exec_return"] = d["open"].shift(-1) / d["open"] - 1.0
    d["weight_exec"] = d["weight_target"].shift(int(lag_days)).fillna(0.0)
    d["turnover"] = (d["weight_exec"] - d["weight_exec"].shift(1).fillna(0.0)).abs()
    d["cost"] = d["turnover"] * (float(cost_bps) / 10000.0)
    d["strategy_return"] = d["weight_exec"] * d["exec_return"].fillna(0.0) - d["cost"]
    d = d.dropna(subset=["exec_return"]).copy()

    if test_start is not None:
        d = d[d["timestamp"] >= pd.Timestamp(test_start, tz="UTC")].copy()
    if test_end is not None:
        d = d[d["timestamp"] <= pd.Timestamp(test_end, tz="UTC")].copy()

    d["eq"] = (1.0 + d["strategy_return"]).cumprod()
    m = perf(d["strategy_return"])
    return d, m


def main() -> int:
    start = "2019-01-01"
    end = "2024-12-31"
    confirm_days = 5
    eth_def_csv = Path("artifacts/backtest/direction_event_model_v1_flat_defensive_6y_gapfilled.csv")
    out_csv = Path("artifacts/backtest/btc_stress_summary.csv")

    btc = _download_binance_daily("BTCUSDC", start, end)
    if btc.empty:
        raise RuntimeError("No BTCUSDC data downloaded.")

    def_proxy = load_eth_defensive_proxy(eth_def_csv)

    rows: list[dict[str, object]] = []

    # Test 1: Walk-forward expanding windows (evaluate on test slices).
    wf_tests = [
        ("Walk-forward 2022", "2019-01-01", "2022-12-31", "2022-01-01", "2022-12-31"),
        ("Walk-forward 2023", "2019-01-01", "2023-12-31", "2023-01-01", "2023-12-31"),
        ("Walk-forward 2024", "2019-01-01", "2024-12-31", "2024-01-01", "2024-12-31"),
    ]
    wf_grades: list[str] = []
    for name, train_start, train_end, test_start, test_end in wf_tests:
        train_slice = btc[(btc["timestamp"] >= pd.Timestamp(train_start, tz="UTC")) & (btc["timestamp"] <= pd.Timestamp(train_end, tz="UTC"))].copy()
        _, m = run_btc_backtest(
            train_slice,
            def_proxy,
            confirm_days=confirm_days,
            cost_bps=10.0,
            lag_days=1,
            test_start=test_start,
            test_end=test_end,
        )
        sharpe = float(m["sharpe"])
        grade = "PASS" if sharpe > 0.8 else ("FAIL" if sharpe < 0.5 else "MARGINAL")
        wf_grades.append(grade)
        rows.append({"test": name, "metric": "sharpe", "value": sharpe, "grade": grade})

    # Test 2: Cost sensitivity.
    for bps in [10, 20, 30, 50]:
        _, m = run_btc_backtest(btc, def_proxy, confirm_days=confirm_days, cost_bps=float(bps), lag_days=1)
        sharpe = float(m["sharpe"])
        if bps == 20:
            grade = "PASS" if sharpe >= 0.6 else "FAIL"
        elif bps == 30:
            grade = "PASS" if sharpe > 0.8 else "FAIL"
        else:
            grade = grade_from_sharpe(sharpe)
        rows.append({"test": f"Cost {bps}bps", "metric": "sharpe", "value": sharpe, "grade": grade})

    # Test 3: Subperiod stability.
    subperiods = [
        ("Period 2019-2020", "2019-01-01", "2020-12-31"),
        ("Period 2021-2022", "2021-01-01", "2022-12-31"),
        ("Period 2023-2024", "2023-01-01", "2024-12-31"),
    ]
    sub_sharpes: list[float] = []
    for label, s, e in subperiods:
        d_sub, m = run_btc_backtest(btc, def_proxy, confirm_days=confirm_days, cost_bps=10.0, lag_days=1, test_start=s, test_end=e)
        sharpe = float(m["sharpe"])
        sub_sharpes.append(sharpe)
        grade = "PASS" if sharpe > 0 else "FAIL"
        rows.append({"test": label, "metric": "sharpe", "value": sharpe, "grade": grade})
        rows.append({"test": label, "metric": "cagr", "value": float(m["cagr"]), "grade": grade})
        rows.append({"test": label, "metric": "max_dd", "value": float(m["max_dd"]), "grade": grade})
        rows.append({"test": label, "metric": "n_days", "value": int(len(d_sub)), "grade": grade})

    # Test 4: Execution lag.
    lag_sharpes: dict[int, float] = {}
    for lag in [0, 1, 2]:
        _, m = run_btc_backtest(btc, def_proxy, confirm_days=confirm_days, cost_bps=10.0, lag_days=lag)
        lag_sharpes[lag] = float(m["sharpe"])
        rows.append({"test": f"Lag {lag}-day", "metric": "sharpe", "value": lag_sharpes[lag], "grade": ""})
    lag_gap = lag_sharpes[0] - lag_sharpes[1]
    lag_grade = "PASS" if lag_gap < 0.5 else ("FAIL" if lag_gap > 1.0 else "MARGINAL")
    rows.append({"test": "Lag gap (0d-1d)", "metric": "delta_sharpe", "value": lag_gap, "grade": lag_grade})

    # Overall grade.
    wf_any_fail = any(g == "FAIL" for g in wf_grades)
    cost20 = next(r for r in rows if r["test"] == "Cost 20bps" and r["metric"] == "sharpe")
    cost30 = next(r for r in rows if r["test"] == "Cost 30bps" and r["metric"] == "sharpe")
    sub_any_fail = any(s <= 0 for s in sub_sharpes)
    if (not wf_any_fail) and (cost20["grade"] == "PASS") and (cost30["grade"] == "PASS") and (not sub_any_fail) and (lag_grade == "PASS"):
        overall = "PASS"
    elif wf_any_fail or cost20["grade"] == "FAIL" or sub_any_fail or lag_grade == "FAIL":
        overall = "FAIL"
    else:
        overall = "MARGINAL"

    rows.append({"test": "Overall", "metric": "grade", "value": np.nan, "grade": overall})

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    def fmt(x: float) -> str:
        return f"{x:.3f}" if np.isfinite(x) else "nan"

    print("===============================================")
    print("BTC STRESS TEST RESULTS")
    print("===============================================")
    for t in ["Walk-forward 2022", "Walk-forward 2023", "Walk-forward 2024", "Cost 20bps", "Cost 30bps", "Cost 50bps"]:
        r = out[(out["test"] == t) & (out["metric"] == "sharpe")].iloc[0]
        print(f"{t:<18} {fmt(float(r['value'])):<8} {r['grade']}")
    for t in ["Period 2019-2020", "Period 2021-2022", "Period 2023-2024"]:
        r = out[(out["test"] == t) & (out["metric"] == "sharpe")].iloc[0]
        print(f"{t:<18} {fmt(float(r['value'])):<8} {r['grade']}")
    print(f"Lag 0-day         {fmt(lag_sharpes[0])}")
    print(f"Lag 1-day         {fmt(lag_sharpes[1])}   {lag_grade}")
    print(f"Lag 2-day         {fmt(lag_sharpes[2])}")
    print("-----------------------------------------------")
    print(f"Overall: {overall}")
    print("===============================================")
    print(f"Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
