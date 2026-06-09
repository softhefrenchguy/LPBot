from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _stats(r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if x.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan}
    eq = (1.0 + x).cumprod()
    years = len(x) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "ann_vol": ann_vol}


def _load_local_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    ts_col = next((c for c in ["timestamp", "date", "day", "datetime", "Date"] if c in d.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp/date column")
    d["timestamp"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce").dt.floor("D")
    rename = {c: c.lower() for c in d.columns if str(c).lower() in {"open", "high", "low", "close", "volume"}}
    d = d.rename(columns=rename)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "volume" not in d.columns:
        d["volume"] = 0.0
    d = d.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return d[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _download_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    x = yf.download(
        ticker,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if x is None or x.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.reset_index()
    date_col = "Date" if "Date" in x.columns else "index"
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(x[date_col], utc=True, errors="coerce").dt.floor("D"),
            "open": pd.to_numeric(x["Open"], errors="coerce"),
            "high": pd.to_numeric(x["High"], errors="coerce"),
            "low": pd.to_numeric(x["Low"], errors="coerce"),
            "close": pd.to_numeric(x["Close"], errors="coerce"),
            "volume": pd.to_numeric(x.get("Volume", 0.0), errors="coerce"),
        }
    )
    return out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def _normalize_to_binance(yf_df: pd.DataFrame, binance_df: pd.DataFrame, merge_date: str) -> tuple[pd.DataFrame, float]:
    merge_ts = pd.to_datetime(merge_date, utc=True)
    y = yf_df[yf_df["timestamp"] < merge_ts].copy()
    b = binance_df[binance_df["timestamp"] >= merge_ts].copy()
    if y.empty or b.empty:
        raise RuntimeError("Cannot normalize: pre-merge or post-merge segment is empty")
    y_last = float(y["close"].iloc[-1])
    b_first = float(b["open"].iloc[0])
    factor = b_first / y_last if y_last > 0 else 1.0
    for c in ["open", "high", "low", "close"]:
        y[c] = y[c] * factor
    out = pd.concat([y, b], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True), factor


def build_extended_data(start: str, end: str, merge_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    eth_binance = _load_local_daily(Path("data/eth_daily.csv"))
    btc_binance = _load_local_daily(Path("data/btc_daily.csv"))
    eth_yf = _download_yf("ETH-USD", start, end)
    btc_yf = _download_yf("BTC-USD", "2015-01-01", end)
    eth_ext, eth_factor = _normalize_to_binance(eth_yf, eth_binance, merge_date)
    btc_ext, btc_factor = _normalize_to_binance(btc_yf, btc_binance, merge_date)
    eth_ext = eth_ext[(eth_ext["timestamp"] >= pd.to_datetime(start, utc=True)) & (eth_ext["timestamp"] <= pd.to_datetime(end, utc=True))]
    # Keep the full BTC-USD history from 2015 in the saved file; the backtest loader filters to --start.
    btc_ext = btc_ext[(btc_ext["timestamp"] >= pd.Timestamp("2015-01-01", tz="UTC")) & (btc_ext["timestamp"] <= pd.to_datetime(end, utc=True))]
    Path("data").mkdir(exist_ok=True)
    eth_ext.to_csv("data/eth_daily_extended.csv", index=False)
    btc_ext.to_csv("data/btc_daily_extended.csv", index=False)
    return eth_ext, btc_ext, {"eth_normalization_factor": eth_factor, "btc_normalization_factor": btc_factor}


def _run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_overlay_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d["return_mean_reversion"] = pd.to_numeric(d["return_mean_reversion"], errors="coerce").fillna(0.0)
    d["baseline_return"] = pd.to_numeric(d["baseline_return"], errors="coerce").fillna(0.0)
    return d.dropna(subset=["day"]).sort_values("day")


def _period_row(name: str, d: pd.DataFrame, ret_col: str = "return_mean_reversion") -> dict[str, float | str | int]:
    m = _stats(d[ret_col])
    return {
        "period": name,
        "start": str(d["day"].min().date()) if not d.empty else "",
        "end": str(d["day"].max().date()) if not d.empty else "",
        "days": int(len(d)),
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_dd": m["maxdd"],
        "ann_vol": m["ann_vol"],
    }


def _yearly(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in d.groupby(d["day"].dt.year):
        rows.append(_period_row(str(int(year)), g))
    return pd.DataFrame(rows)


def _crash_2018(d: pd.DataFrame) -> dict[str, object]:
    y = d[(d["day"] >= pd.Timestamp("2018-01-01", tz="UTC")) & (d["day"] <= pd.Timestamp("2018-12-31", tz="UTC"))].copy()
    if y.empty:
        return {}
    strat = float((1.0 + y["return_mean_reversion"]).prod() - 1.0)
    eth_spot = float((1.0 + pd.to_numeric(y["eth_spot_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
    m = _stats(y["return_mean_reversion"])
    avoided = float(strat - eth_spot)
    protected = bool(strat > eth_spot and m["maxdd"] > eth_spot)

    eth = y[["day", "eth_close", "eth_regime", "eth_weight_exec"]].copy()
    eth["eth_close"] = pd.to_numeric(eth["eth_close"], errors="coerce")
    peak_i = eth["eth_close"].idxmax()
    peak_day = eth.loc[peak_i, "day"]
    after = eth[eth["day"] >= peak_day].copy()
    bear = after[after["eth_regime"].astype(str).eq("BEAR")]
    flat = after[pd.to_numeric(after["eth_weight_exec"], errors="coerce").fillna(0.0) <= 1e-9]
    bear_day = bear["day"].iloc[0] if not bear.empty else pd.NaT
    flat_day = flat["day"].iloc[0] if not flat.empty else pd.NaT
    return {
        "strategy_return_2018": strat,
        "eth_spot_return_2018": eth_spot,
        "strategy_max_dd_2018": m["maxdd"],
        "protected": protected,
        "avoided_vs_eth": avoided,
        "eth_peak_day_2018": str(peak_day.date()) if pd.notna(peak_day) else "",
        "first_bear_after_peak": str(bear_day.date()) if pd.notna(bear_day) else "",
        "days_peak_to_bear": int((bear_day - peak_day).days) if pd.notna(bear_day) and pd.notna(peak_day) else -1,
        "first_flat_after_peak": str(flat_day.date()) if pd.notna(flat_day) else "",
        "days_peak_to_flat": int((flat_day - peak_day).days) if pd.notna(flat_day) and pd.notna(peak_day) else -1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run extended ETH/BTC validated portfolio backtest from 2017.")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--merge-date", default="2019-01-01")
    args = ap.parse_args()

    eth_ext, btc_ext, norm = build_extended_data(args.start, args.end, args.merge_date)
    print(f"ETH extended rows: {len(eth_ext)} | BTC extended rows: {len(btc_ext)}")
    print(f"Normalization: {norm}")

    base_daily = Path("artifacts/backtest/extended_base_daily.csv")
    base_summary = Path("artifacts/backtest/extended_base_summary.csv")
    overlay_daily = Path("artifacts/backtest/extended_overlay_daily.csv")
    overlay_summary = Path("artifacts/backtest/extended_overlay_summary.csv")

    _run_cmd([
        sys.executable,
        "scripts/backtest_eth_btc_portfolio.py",
        "--eth-data", "data/eth_daily_extended.csv",
        "--btc-data", "data/btc_daily_extended.csv",
        "--start", args.start,
        "--end", args.end,
        "--vol-filter",
        "--transition-momentum",
        "--asymmetric-sizing",
        "--allocation-mode", "signal_weighted",
        "--cost-mode", "weight_change",
        "--cost-bps", "20",
        "--eth-confirm-days", "3",
        "--btc-confirm-days", "5",
        "--eth-ema", "21,55,144",
        "--btc-ema", "15,40,120",
        "--out-summary-csv", str(base_summary),
        "--out-daily-csv", str(base_daily),
    ])
    _run_cmd([
        sys.executable,
        "scripts/backtest_overlay_strategies.py",
        "--baseline-daily", str(base_daily),
        "--out-summary", str(overlay_summary),
        "--out-daily", str(overlay_daily),
    ])

    d = _load_overlay_daily(overlay_daily)
    yearly = _yearly(d)
    periods = pd.DataFrame([
        _period_row("2017-2018", d[(d["day"] >= pd.Timestamp("2017-01-01", tz="UTC")) & (d["day"] <= pd.Timestamp("2018-12-31", tz="UTC"))]),
        _period_row("2019-2024", d[(d["day"] >= pd.Timestamp("2019-01-01", tz="UTC")) & (d["day"] <= pd.Timestamp("2024-12-31", tz="UTC"))]),
        _period_row("2017-2024", d),
    ])
    crash = _crash_2018(d)

    out_summary = Path("artifacts/backtest/extended_backtest_summary.csv")
    out_yearly = Path("artifacts/backtest/extended_backtest_yearly.csv")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = periods.copy()
    for k, v in crash.items():
        summary_rows[k] = v
    for k, v in norm.items():
        summary_rows[k] = v
    summary_rows.to_csv(out_summary, index=False)
    yearly.to_csv(out_yearly, index=False)

    print("\nEXTENDED BACKTEST RESULTS")
    print("================================================")
    print("Period          Sharpe   MaxDD     CAGR")
    print("------------------------------------------------")
    for _, r in periods.iterrows():
        print(f"{str(r['period']):<14} {r['sharpe']:>7.3f} {r['max_dd']*100:>7.2f}% {r['cagr']*100:>7.2f}%")
    print("------------------------------------------------")
    print("Year by year:")
    for _, r in yearly.iterrows():
        print(f"{str(r['period']):<14} {r['sharpe']:>7.3f} {r['max_dd']*100:>7.2f}% {r['cagr']*100:>7.2f}%")
    print("------------------------------------------------")
    print("2018 crash test:")
    print(f"  Strategy:      {crash.get('strategy_return_2018', np.nan)*100:.2f}%")
    print(f"  ETH spot:      {crash.get('eth_spot_return_2018', np.nan)*100:.2f}%")
    print(f"  MaxDD:         {crash.get('strategy_max_dd_2018', np.nan)*100:.2f}%")
    print(f"  Protected:     {'YES' if crash.get('protected') else 'NO'}")
    print(f"  Peak to BEAR:  {crash.get('days_peak_to_bear', 'NA')} days ({crash.get('first_bear_after_peak', '')})")
    print(f"  Peak to flat:  {crash.get('days_peak_to_flat', 'NA')} days ({crash.get('first_flat_after_peak', '')})")
    print("================================================")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_yearly}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

