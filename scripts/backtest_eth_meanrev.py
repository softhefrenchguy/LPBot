from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_btc_full_stack import _download_binance_daily, classify_regime_v2_on_btc, perf


def load_eth_daily(path: Path, start: str, end: str) -> pd.DataFrame:
    if path.exists():
        d = pd.read_csv(path)
        if "timestamp" not in d.columns and "day" in d.columns:
            d["timestamp"] = d["day"]
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["timestamp", "open", "close"]).sort_values("timestamp")
    else:
        d = _download_binance_daily("ETHUSDC", start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(path, index=False)
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return d[(d["timestamp"] >= s) & (d["timestamp"] <= e)].copy().reset_index(drop=True)


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_hold": np.nan}
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades["return_pct"] > 0).mean()),
        "avg_hold": float(trades["days_held"].mean()),
    }


def load_trend_returns(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "day" in d.columns:
        d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    elif "timestamp" in d.columns:
        d["day"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.floor("D")
    else:
        raise ValueError(f"{path} must contain day or timestamp")
    ret_col = "combined_return" if "combined_return" in d.columns else "strategy_return"
    d["trend_return"] = pd.to_numeric(d[ret_col], errors="coerce").fillna(0.0)
    return d[["day", "trend_return"]].dropna(subset=["day"]).copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH daily mean-reversion backtest")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--eth-daily-csv", default="data/ETHUSDC_daily.csv")
    ap.add_argument("--trend-daily-csv", default="artifacts/backtest/eth_btc_portfolio_daily_20bps.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/eth_meanrev_summary.csv")
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/eth_meanrev_daily.csv")
    args = ap.parse_args()

    d = load_eth_daily(Path(args.eth_daily_csv), args.start, args.end)
    d = classify_regime_v2_on_btc(d)
    d["day"] = d["timestamp"].dt.floor("D")
    d["rolling_mean"] = d["close"].rolling(20, min_periods=20).mean()
    d["rolling_std"] = d["close"].rolling(20, min_periods=20).std(ddof=0)
    d["z_score"] = (d["close"] - d["rolling_mean"]) / d["rolling_std"].replace(0.0, np.nan)
    d["daily_return_close"] = d["close"].pct_change()
    d["exec_return"] = d["open"].shift(-1) / d["open"] - 1.0

    position = np.zeros(len(d), dtype=float)
    entry_idx: int | None = None
    trade_rows: list[dict[str, Any]] = []
    active = False

    for i in range(len(d)):
        z = float(d["z_score"].iloc[i]) if pd.notna(d["z_score"].iloc[i]) else np.nan
        r = float(d["daily_return_close"].iloc[i]) if pd.notna(d["daily_return_close"].iloc[i]) else np.nan
        if not active:
            if pd.notna(z) and pd.notna(r) and z < -1.5 and r < -0.03:
                active = True
                entry_idx = i
        else:
            held = i - int(entry_idx) if entry_idx is not None else 0
            exit_now = (pd.notna(z) and z > -0.5) or (pd.notna(z) and z < -3.0) or held >= 10
            if exit_now:
                if entry_idx is not None and i > entry_idx:
                    entry_open = float(d["open"].iloc[entry_idx + 1]) if entry_idx + 1 < len(d) else np.nan
                    exit_open = float(d["open"].iloc[i + 1]) if i + 1 < len(d) else float(d["close"].iloc[i])
                    ret = (exit_open / entry_open - 1.0) * 0.5 - 0.002 if entry_open and np.isfinite(entry_open) else np.nan
                    trade_rows.append(
                        {
                            "entry_date": str(d["day"].iloc[entry_idx].date()),
                            "exit_date": str(d["day"].iloc[i].date()),
                            "regime": d["regime_v2"].iloc[entry_idx],
                            "days_held": int(max(1, held)),
                            "return_pct": ret,
                        }
                    )
                active = False
                entry_idx = None
        position[i] = 0.5 if active else 0.0

    d["weight_target"] = position
    d["weight_exec"] = d["weight_target"].shift(1).fillna(0.0)
    prev_w = d["weight_exec"].shift(1).fillna(0.0)
    d["turnover"] = (d["weight_exec"] - prev_w).abs()
    d["cost"] = d["turnover"] * 0.002
    d["strategy_return"] = d["weight_exec"] * d["exec_return"].fillna(0.0) - d["cost"]
    d = d.dropna(subset=["exec_return"]).copy()
    d["equity"] = (1.0 + d["strategy_return"]).cumprod()

    trades = pd.DataFrame(trade_rows)
    m = perf(d["strategy_return"])
    ts = trade_stats(trades)

    regime_lines: list[str] = []
    regime_summary: dict[str, dict[str, float]] = {}
    for regime in ["BULL", "CHOP", "BEAR"]:
        x = trades[trades["regime"].astype(str).eq(regime)] if not trades.empty else pd.DataFrame()
        avg = float(x["return_pct"].mean()) if not x.empty else np.nan
        regime_summary[regime] = {"trades": int(len(x)), "avg_return": avg}
        regime_lines.append(f"{regime} trades: {len(x)}  avg return: {avg*100:.2f}%" if np.isfinite(avg) else f"{regime} trades: {len(x)}  avg return: n/a")

    trend = load_trend_returns(Path(args.trend_daily_csv))
    merged = d[["day", "strategy_return"]].merge(trend, on="day", how="inner")
    corr = float(merged["strategy_return"].corr(merged["trend_return"])) if len(merged) > 2 else np.nan
    trend_m = perf(merged["trend_return"])
    merged["combined_70_30"] = 0.7 * merged["trend_return"] + 0.3 * merged["strategy_return"]
    comb_m = perf(merged["combined_70_30"])

    periods = [("2019-2020", "2019-01-01", "2020-12-31"), ("2021-2022", "2021-01-01", "2022-12-31"), ("2023-2024", "2023-01-01", "2024-12-31")]
    period_metrics: dict[str, float] = {}
    for label, s, e in periods:
        mask = (d["day"] >= pd.Timestamp(s, tz="UTC")) & (d["day"] <= pd.Timestamp(e, tz="UTC"))
        period_metrics[label] = perf(d.loc[mask, "strategy_return"])["cagr"]

    verdict = "PASS" if m["sharpe"] >= 0.7 and corr < 0.3 and comb_m["sharpe"] > trend_m["sharpe"] else "MARGINAL" if m["sharpe"] >= 0.4 else "FAIL"
    worth = "YES" if comb_m["sharpe"] > trend_m["sharpe"] and corr < 0.5 else "NO"

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_daily, index=False)

    summary = pd.DataFrame(
        [
            {
                "strategy": "eth_mean_reversion",
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "trades": ts["trades"],
                "win_rate": ts["win_rate"],
                "avg_hold_days": ts["avg_hold"],
                "corr_vs_trend": corr,
                "trend_sharpe": trend_m["sharpe"],
                "combined_70_30_sharpe": comb_m["sharpe"],
                "combined_70_30_max_dd": comb_m["max_dd"],
                "verdict": verdict,
                "worth_combining": worth,
                "bull_trades": regime_summary["BULL"]["trades"],
                "chop_trades": regime_summary["CHOP"]["trades"],
                "bear_trades": regime_summary["BEAR"]["trades"],
            }
        ]
    )
    summary.to_csv(args.out_summary_csv, index=False)

    print("=" * 64)
    print("ETH MEAN-REVERSION BACKTEST")
    print("2019-2024 | Lag-1 | 20bps")
    print("=" * 64)
    print(f"CAGR:        {m['cagr']*100:.2f}%")
    print(f"Sharpe:      {m['sharpe']:.3f}")
    print(f"MaxDD:       {m['max_dd']*100:.2f}%")
    print(f"Trades:      {ts['trades']}")
    print(f"Win rate:    {ts['win_rate']*100:.2f}%" if np.isfinite(ts["win_rate"]) else "Win rate:    n/a")
    print(f"Avg hold:    {ts['avg_hold']:.2f} days" if np.isfinite(ts["avg_hold"]) else "Avg hold:    n/a")
    print("")
    print("Regime breakdown:")
    for line in regime_lines:
        print(line)
    print("")
    print(f"Correlation vs trend strategy: {corr:.3f}")
    print("")
    print("Combined 70/30 portfolio:")
    print(f"Sharpe: {comb_m['sharpe']:.3f}  MaxDD: {comb_m['max_dd']*100:.2f}%")
    print(f"vs trend only: {trend_m['sharpe']:.3f}")
    print("")
    print("Period CAGR:")
    for label, _, _ in periods:
        print(f"{label}: {period_metrics[label]*100:.2f}%")
    print("")
    print(f"Verdict: {verdict}")
    print(f"Worth combining: {worth}")
    print("=" * 64)
    print(f"Saved: {args.out_summary_csv}")
    print(f"Saved: {args.out_daily_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
