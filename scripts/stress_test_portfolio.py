from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency yfinance. Install with: pip install yfinance pandas numpy --break-system-packages"
    ) from exc


ASSETS = [
    {"name": "ETH", "ticker": "ETH-USD", "alloc": 0.25},
    {"name": "BTC", "ticker": "BTC-USD", "alloc": 0.25},
    {"name": "Oil/WTI", "ticker": "CL=F", "alloc": 0.25},
    {"name": "Nat Gas", "ticker": "NG=F", "alloc": 0.25},
]


def _grade(sharpe: float) -> str:
    if np.isfinite(sharpe) and sharpe >= 1.5:
        return "PASS"
    if np.isfinite(sharpe) and sharpe >= 1.0:
        return "MARGINAL"
    return "FAIL"


def _download_asset(ticker: str, start: str, end: str, yf_cache_dir: Path) -> pd.DataFrame:
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(yf_cache_dir))
    except Exception:
        pass
    d = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if d is None or len(d) == 0:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in d.columns]
    d = d.reset_index()
    if "Date" not in d.columns:
        if "index" in d.columns:
            d = d.rename(columns={"index": "Date"})
        elif "Datetime" in d.columns:
            d = d.rename(columns={"Datetime": "Date"})
        elif len(d.columns) > 0:
            d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    if "Adj Close" in d.columns:
        d["price"] = pd.to_numeric(d["Adj Close"], errors="coerce")
    elif "Close" in d.columns:
        d["price"] = pd.to_numeric(d["Close"], errors="coerce")
    else:
        return pd.DataFrame()
    if "Open" in d.columns:
        d["open"] = pd.to_numeric(d["Open"], errors="coerce")
    else:
        d["open"] = pd.to_numeric(d["price"], errors="coerce")
    d["price"] = d["price"].ffill(limit=3)
    d["open"] = d["open"].ffill(limit=3)
    d = d.dropna(subset=["price", "open"])
    return d[["Date", "price", "open"]].copy()


def _compute_regime(price: pd.Series) -> pd.Series:
    ret20 = price / price.shift(20) - 1.0
    dd20 = price / price.rolling(20, min_periods=20).max() - 1.0
    regime = pd.Series("CHOP", index=price.index, dtype=object)
    regime[(ret20 > 0.05) & (dd20 > -0.10)] = "BULL"
    regime[dd20 < -0.15] = "BEAR"
    return regime


def _build_sleeve_from_df(raw_df: pd.DataFrame, alloc: float, cost_bps: float, exec_lag_days: int) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.set_index("Date").sort_index()
    df["daily_return"] = df["price"].pct_change().fillna(0.0)
    df["exec_return"] = df["open"].shift(-1) / df["open"] - 1.0
    df["ema21"] = df["price"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["price"].ewm(span=55, adjust=False).mean()
    df["ema144"] = df["price"].ewm(span=144, adjust=False).mean()
    stack = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema144"])
    entry = (stack.rolling(3, min_periods=3).min() == 1).fillna(False)
    exit_ = ((~stack).rolling(3, min_periods=3).min() == 1).fillna(False)
    regime = _compute_regime(df["price"])
    off = regime.replace({"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}).astype(float)
    rv = df["daily_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(df), dtype=int)
    active = 0
    for i in range(len(df)):
        if active == 0 and bool(entry.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
        pos[i] = active
    raw_w = np.where(pos == 1, vol_scalar * off, 0.0)
    w_exec = pd.Series(raw_w, index=df.index).shift(int(exec_lag_days)).fillna(0.0)
    turnover = (w_exec - w_exec.shift(1).fillna(0.0)).abs()
    cost = turnover * (cost_bps / 10000.0)
    strat_r = w_exec * df["exec_return"].fillna(0.0) - cost

    out = pd.DataFrame(index=df.index)
    out["strategy_return"] = pd.to_numeric(strat_r, errors="coerce").fillna(0.0)
    out["portfolio_contrib"] = float(alloc) * out["strategy_return"]
    out["strategy_weight_exec"] = w_exec
    return out


def _metrics(r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
    years = len(x) / 252.0 if len(x) else np.nan
    eq = (1.0 + x).cumprod()
    total = float(eq.iloc[-1]) if len(eq) else np.nan
    cagr = (total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(np.mean(ex) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else np.nan
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "ann_vol": ann_vol}


def _run_portfolio(
    cache: dict[str, pd.DataFrame],
    cost_bps: float,
    exec_lag_days: int,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleeves: list[pd.DataFrame] = []
    for a in ASSETS:
        d = cache[a["name"]].copy()
        if end_date is not None:
            d = d[d["Date"] <= pd.Timestamp(end_date, tz="UTC")]
        d = d.sort_values("Date").copy()
        s = _build_sleeve_from_df(raw_df=d, alloc=float(a["alloc"]), cost_bps=float(cost_bps), exec_lag_days=int(exec_lag_days))
        s = s.iloc[144:].copy()
        s = s.iloc[:-1].copy()
        s.columns = [f"{a['name']}_{c}" for c in s.columns]
        sleeves.append(s)

    merged = pd.concat(sleeves, axis=1).sort_index()
    merged = merged.fillna(0.0)
    contrib_cols = [c for c in merged.columns if c.endswith("_portfolio_contrib")]
    weight_cols = [c for c in merged.columns if c.endswith("_strategy_weight_exec")]
    strat_cols = [c for c in merged.columns if c.endswith("_strategy_return")]
    merged["portfolio_return"] = merged[contrib_cols].sum(axis=1)
    # gross exposure uses allocated sleeve weights.
    gross = pd.Series(0.0, index=merged.index)
    for a in ASSETS:
        gross = gross + float(a["alloc"]) * merged[f"{a['name']}_strategy_weight_exec"]
    merged["portfolio_gross_exposure"] = gross
    merged["portfolio_eq"] = (1.0 + merged["portfolio_return"]).cumprod()

    corr = merged[strat_cols].corr()
    corr.index = [i.replace("_strategy_return", "") for i in corr.index]
    corr.columns = [i.replace("_strategy_return", "") for i in corr.columns]
    return merged, corr


def _slice_metrics(df: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    x = df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))]
    if len(x) == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "ann_vol": np.nan, "n_days": 0}
    m = _metrics(x["portfolio_return"])
    m["n_days"] = int(len(x))
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="Stress tests for independent multi-asset portfolio")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out-dir", default="artifacts/backtest/stress_tests")
    p.add_argument("--out-summary", default="artifacts/backtest/stress_tests/stress_summary.csv")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = out_dir / ".yf_tz_cache"

    # Download once.
    cache: dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        d = _download_asset(a["ticker"], args.start, args.end, yf_cache_dir)
        if len(d) == 0:
            print(f"Warning: empty download for {a['name']}")
            return 1
        cache[a["name"]] = d

    rows: list[dict[str, Any]] = []

    # Baseline run (10bps, realistic 1-day lag after close signal).
    base_df, base_corr = _run_portfolio(cache=cache, cost_bps=10.0, exec_lag_days=1, end_date=args.end)
    base_m = _metrics(base_df["portfolio_return"])
    rows.append(
        {
            "test": "baseline",
            "scenario": "cost_10bps_lag1_realistic",
            "cagr": base_m["cagr"],
            "sharpe": base_m["sharpe"],
            "max_dd": base_m["max_dd"],
            "ann_vol": base_m["ann_vol"],
            "n_days": int(len(base_df)),
            "grade": _grade(base_m["sharpe"]),
        }
    )
    # Diagnostic row to quantify optimistic bias if someone runs same-day execution.
    diag0_df, _ = _run_portfolio(cache=cache, cost_bps=10.0, exec_lag_days=0, end_date=args.end)
    diag0_m = _metrics(diag0_df["portfolio_return"])
    rows.append(
        {
            "test": "timing_diagnostic",
            "scenario": "cost_10bps_lag0_optimistic",
            "cagr": diag0_m["cagr"],
            "sharpe": diag0_m["sharpe"],
            "max_dd": diag0_m["max_dd"],
            "ann_vol": diag0_m["ann_vol"],
            "n_days": int(len(diag0_df)),
            "grade": _grade(diag0_m["sharpe"]),
        }
    )
    base_df.to_csv(out_dir / "baseline_daily.csv")
    base_corr.to_csv(out_dir / "baseline_corr.csv")

    # Test 1: Walk-forward expanding windows (test-year OOS slices).
    wf_folds = [
        ("wf_test_2022", "2019-01-01", "2022-12-31", "2022-01-01", "2022-12-31"),
        ("wf_test_2023", "2019-01-01", "2023-12-31", "2023-01-01", "2023-12-31"),
        ("wf_test_2024", "2019-01-01", "2024-12-31", "2024-01-01", "2024-12-31"),
    ]
    for label, _train_start, end_train_test, eval_start, eval_end in wf_folds:
        df_fold, _ = _run_portfolio(cache=cache, cost_bps=10.0, exec_lag_days=1, end_date=end_train_test)
        m = _slice_metrics(df_fold, eval_start, eval_end)
        rows.append(
            {
                "test": "walk_forward",
                "scenario": label,
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "ann_vol": m["ann_vol"],
                "n_days": m["n_days"],
                "grade": _grade(m["sharpe"]),
            }
        )

    # Test 2: Cost sensitivity.
    for c in [10.0, 20.0, 30.0, 50.0]:
        df_c, _ = _run_portfolio(cache=cache, cost_bps=c, exec_lag_days=1, end_date=args.end)
        m = _metrics(df_c["portfolio_return"])
        rows.append(
            {
                "test": "cost_sensitivity",
                "scenario": f"cost_{int(c)}bps",
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "ann_vol": m["ann_vol"],
                "n_days": int(len(df_c)),
                "grade": _grade(m["sharpe"]),
            }
        )

    # Test 3: Subperiod stability (combined + per-asset sleeves from baseline).
    subperiods = [
        ("2019_2020", "2019-01-01", "2020-12-31"),
        ("2021_2022", "2021-01-01", "2022-12-31"),
        ("2023_2024", "2023-01-01", "2024-12-31"),
    ]
    for lbl, s, e in subperiods:
        m = _slice_metrics(base_df, s, e)
        rows.append(
            {
                "test": "subperiod_combined",
                "scenario": lbl,
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "ann_vol": m["ann_vol"],
                "n_days": m["n_days"],
                "grade": _grade(m["sharpe"]),
            }
        )
        for a in ASSETS:
            col = f"{a['name']}_strategy_return"
            part = base_df[(base_df.index >= pd.Timestamp(s, tz="UTC")) & (base_df.index <= pd.Timestamp(e, tz="UTC"))]
            if len(part) == 0:
                continue
            mm = _metrics(part[col])
            rows.append(
                {
                    "test": "subperiod_asset",
                    "scenario": f"{lbl}_{a['name']}",
                    "cagr": mm["cagr"],
                    "sharpe": mm["sharpe"],
                    "max_dd": mm["max_dd"],
                    "ann_vol": mm["ann_vol"],
                    "n_days": int(len(part)),
                    "grade": _grade(mm["sharpe"]),
                }
            )

    # Test 4: Execution constraints (additional delay beyond realistic baseline).
    lag_df, _ = _run_portfolio(cache=cache, cost_bps=10.0, exec_lag_days=2, end_date=args.end)
    lag_m = _metrics(lag_df["portfolio_return"])
    rows.append(
        {
            "test": "execution_lag",
            "scenario": "lag_2day",
            "cagr": lag_m["cagr"],
            "sharpe": lag_m["sharpe"],
            "max_dd": lag_m["max_dd"],
            "ann_vol": lag_m["ann_vol"],
            "n_days": int(len(lag_df)),
            "grade": _grade(lag_m["sharpe"]),
        }
    )
    rows.append(
        {
            "test": "execution_lag_delta",
            "scenario": "lag2_vs_base_lag1",
            "cagr": lag_m["cagr"] - base_m["cagr"],
            "sharpe": lag_m["sharpe"] - base_m["sharpe"],
            "max_dd": lag_m["max_dd"] - base_m["max_dd"],
            "ann_vol": lag_m["ann_vol"] - base_m["ann_vol"],
            "n_days": int(len(lag_df)),
            "grade": _grade(lag_m["sharpe"]),
        }
    )

    stress = pd.DataFrame(rows)
    stress.to_csv(args.out_summary, index=False)

    # Console report.
    print("===================================================")
    print("STRESS TEST SUMMARY")
    print("PASS >=1.5 | MARGINAL 1.0-1.5 | FAIL <1.0 (Sharpe)")
    print("===================================================")
    view = stress[["test", "scenario", "cagr", "sharpe", "max_dd", "grade"]].copy()
    print(view.to_string(index=False))
    print("===================================================")
    print(f"Saved summary: {args.out_summary}")
    print(f"Saved details: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
