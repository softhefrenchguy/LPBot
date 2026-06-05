from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_btc_full_stack import load_eth_defensive_proxy, perf
from backtest_eth_btc_portfolio import run_sleeve


def _slice(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    x = d.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    return x[(x["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (x["timestamp"] <= pd.Timestamp(end, tz="UTC"))].copy()


def _run_combined(
    def_proxy: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_bps: float,
    cost_mode: str,
    lag_days: int,
    allocation_mode: str,
    gross_cap: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, float]]:
    eth = run_sleeve(
        symbol="ETHUSDC",
        confirm_days=3,
        def_proxy=def_proxy,
        cost_bps=cost_bps,
        cost_mode=cost_mode,
        start=start,
        end=end,
        out_price_csv=Path("data/eth_daily.csv"),
    )
    btc = run_sleeve(
        symbol="BTCUSDC",
        confirm_days=5,
        def_proxy=def_proxy,
        cost_bps=cost_bps,
        cost_mode=cost_mode,
        start=start,
        end=end,
        out_price_csv=Path("data/btc_daily.csv"),
    )

    if lag_days != 1:
        for x in (eth, btc):
            x["weight_exec"] = x["weight_target"].shift(int(lag_days)).fillna(0.0)
            prev_w = x["weight_exec"].shift(1).fillna(0.0)
            if str(cost_mode).lower() == "entry_exit":
                crossed = ((x["weight_exec"] > 0).astype(int) != (prev_w > 0).astype(int))
                x["turnover"] = np.where(crossed, (x["weight_exec"] - prev_w).abs(), 0.0)
            else:
                x["turnover"] = (x["weight_exec"] - prev_w).abs()
            x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
            x["strategy_return"] = x["weight_exec"] * x["exec_return"].fillna(0.0) - x["cost"]
            x["eq"] = (1.0 + x["strategy_return"]).cumprod()

    keep = ["timestamp", "weight_exec", "strategy_return", "exec_return"]
    e = eth[keep].rename(
        columns={
            "weight_exec": "eth_weight_exec",
            "strategy_return": "eth_strategy_return",
            "exec_return": "eth_spot_return",
        }
    )
    b = btc[keep].rename(
        columns={
            "weight_exec": "btc_weight_exec",
            "strategy_return": "btc_strategy_return",
            "exec_return": "btc_spot_return",
        }
    )
    m = e.merge(b, on="timestamp", how="inner").sort_values("timestamp").copy()
    if str(allocation_mode) == "signal_weighted":
        score_sum = m["eth_weight_exec"] + m["btc_weight_exec"]
        active = score_sum > 0
        gross = np.where(active, float(gross_cap), 0.0)
        m["alloc_eth"] = np.where(active, gross * (m["eth_weight_exec"] / score_sum), 0.0)
        m["alloc_btc"] = np.where(active, gross * (m["btc_weight_exec"] / score_sum), 0.0)
    else:
        m["alloc_eth"] = 0.5
        m["alloc_btc"] = 0.5
    m["combined_return"] = m["alloc_eth"] * m["eth_strategy_return"] + m["alloc_btc"] * m["btc_strategy_return"]
    m["combined_spot_return"] = 0.5 * m["eth_spot_return"] + 0.5 * m["btc_spot_return"]
    m["combined_time_in_market"] = ((m["eth_weight_exec"] > 0).astype(float) + (m["btc_weight_exec"] > 0).astype(float)) / 2.0
    return m, perf(m["eth_strategy_return"]), perf(m["btc_strategy_return"]), perf(m["combined_return"])


def main() -> int:
    cost_mode = "weight_change"
    allocation_mode = "fixed"
    gross_cap = 0.8
    out_csv = Path("artifacts/backtest/eth_btc_stress_summary.csv")
    def_proxy = load_eth_defensive_proxy(Path("artifacts/backtest/direction_event_model_v1_flat_defensive_6y_gapfilled.csv"))

    rows: list[dict[str, object]] = []

    # Test 1: Walk-forward windows.
    wf = [
        ("Walk-forward 2022", "2019-01-01", "2022-12-31", "2022-01-01", "2022-12-31"),
        ("Walk-forward 2023", "2019-01-01", "2023-12-31", "2023-01-01", "2023-12-31"),
        ("Walk-forward 2024", "2019-01-01", "2024-12-31", "2024-01-01", "2024-12-31"),
    ]
    wf_comp: list[float] = []
    wf_table: list[tuple[str, float, float, float]] = []
    for name, full_start, full_end, test_start, test_end in wf:
        m, me, mb, mc = _run_combined(
            def_proxy,
            start=full_start,
            end=full_end,
            cost_bps=10.0,
            cost_mode=cost_mode,
            lag_days=1,
            allocation_mode=allocation_mode,
            gross_cap=gross_cap,
        )
        s = _slice(m, test_start, test_end)
        pe = perf(s["eth_strategy_return"])
        pb = perf(s["btc_strategy_return"])
        pc = perf(s["combined_return"])
        grade = "PASS" if pc["sharpe"] > 0.8 else ("FAIL" if pc["sharpe"] < 0.5 else "MARGINAL")
        rows.append({"test": name, "metric": "combined_sharpe", "value": float(pc["sharpe"]), "grade": grade})
        rows.append({"test": name, "metric": "eth_sharpe", "value": float(pe["sharpe"]), "grade": ""})
        rows.append({"test": name, "metric": "btc_sharpe", "value": float(pb["sharpe"]), "grade": ""})
        wf_comp.append(float(pc["sharpe"]))
        wf_table.append((name, float(pe["sharpe"]), float(pb["sharpe"]), float(pc["sharpe"])))

    # Test 2: Cost sensitivity.
    for bps in [10, 20, 30, 50]:
        m, _, _, mc = _run_combined(
            def_proxy,
            start="2019-01-01",
            end="2024-12-31",
            cost_bps=float(bps),
            cost_mode=cost_mode,
            lag_days=1,
            allocation_mode=allocation_mode,
            gross_cap=gross_cap,
        )
        grade = "PASS" if (bps != 30 and mc["sharpe"] >= 0.8) else ("PASS" if (bps == 30 and mc["sharpe"] > 1.0) else "FAIL")
        rows.append({"test": f"Cost {bps}bps", "metric": "combined_sharpe", "value": float(mc["sharpe"]), "grade": grade})

    # Test 3: Subperiod stability.
    for label, s0, s1 in [
        ("Period 2019-2020", "2019-01-01", "2020-12-31"),
        ("Period 2021-2022", "2021-01-01", "2022-12-31"),
        ("Period 2023-2024", "2023-01-01", "2024-12-31"),
    ]:
        m, _, _, _ = _run_combined(
            def_proxy,
            start="2019-01-01",
            end="2024-12-31",
            cost_bps=10.0,
            cost_mode=cost_mode,
            lag_days=1,
            allocation_mode=allocation_mode,
            gross_cap=gross_cap,
        )
        p = perf(_slice(m, s0, s1)["combined_return"])
        grade = "PASS" if p["sharpe"] > 0 else "FAIL"
        rows.append({"test": label, "metric": "combined_sharpe", "value": float(p["sharpe"]), "grade": grade})
        rows.append({"test": label, "metric": "combined_cagr", "value": float(p["cagr"]), "grade": grade})
        rows.append({"test": label, "metric": "combined_max_dd", "value": float(p["max_dd"]), "grade": grade})

    # Test 4: Execution lag.
    lag_sharpes: dict[int, float] = {}
    for lag in [0, 1, 2]:
        _, _, _, mc = _run_combined(
            def_proxy,
            start="2019-01-01",
            end="2024-12-31",
            cost_bps=10.0,
            cost_mode=cost_mode,
            lag_days=lag,
            allocation_mode=allocation_mode,
            gross_cap=gross_cap,
        )
        lag_sharpes[lag] = float(mc["sharpe"])
        rows.append({"test": f"Lag {lag}-day", "metric": "combined_sharpe", "value": lag_sharpes[lag], "grade": ""})
    lag_gap = lag_sharpes[0] - lag_sharpes[1]
    lag_grade = "PASS" if lag_gap < 0.5 else "FAIL"
    rows.append({"test": "Lag gap (0d-1d)", "metric": "delta_sharpe", "value": lag_gap, "grade": lag_grade})

    wf_fail = any(v < 0.5 for v in wf_comp)
    c30 = next(r for r in rows if r["test"] == "Cost 30bps" and r["metric"] == "combined_sharpe")
    sub_fails = [r for r in rows if r["test"].startswith("Period ") and r["metric"] == "combined_sharpe" and float(r["value"]) <= 0]
    if (not wf_fail) and c30["grade"] == "PASS" and (not sub_fails) and lag_grade == "PASS":
        overall = "PASS"
    elif wf_fail or c30["grade"] == "FAIL" or sub_fails or lag_grade == "FAIL":
        overall = "FAIL"
    else:
        overall = "MARGINAL"
    rows.append({"test": "Overall", "metric": "grade", "value": np.nan, "grade": overall})

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("===============================================")
    print("ETH+BTC COMBINED STRESS TEST")
    print("===============================================")
    for t in ["Walk-forward 2022", "Walk-forward 2023", "Walk-forward 2024", "Cost 20bps", "Cost 30bps", "Cost 50bps"]:
        r = out[(out["test"] == t) & (out["metric"] == "combined_sharpe")].iloc[0]
        print(f"{t:<20} {float(r['value']):>6.3f}   {r['grade']}")
    for t in ["Period 2019-2020", "Period 2021-2022", "Period 2023-2024"]:
        r = out[(out["test"] == t) & (out["metric"] == "combined_sharpe")].iloc[0]
        print(f"{t:<20} {float(r['value']):>6.3f}   {r['grade']}")
    print(f"Lag 0-day            {lag_sharpes[0]:>6.3f}")
    print(f"Lag 1-day            {lag_sharpes[1]:>6.3f}   {lag_grade}")
    print(f"Lag 2-day            {lag_sharpes[2]:>6.3f}")
    print("-----------------------------------------------")
    print(f"Overall: {overall}")
    print("-----------------------------------------------")
    print("WF comparison (ETH / BTC / Combined)")
    for n, e, b, c in wf_table:
        lbl = n.replace("Walk-forward ", "WF ")
        print(f"{lbl:<10} {e:>6.3f}  {b:>6.3f}  {c:>6.3f}")
    print("===============================================")
    print(f"Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
