from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_forex_optimised import _crypto_main, _stats


def _load_daily(path: Path, start: str, end: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(columns={c: str(c).strip().lower() for c in d.columns})
    if "timestamp" not in d.columns:
        date_col = next((c for c in ["day", "date", "datetime", "time"] if c in d.columns), None)
        if date_col is None:
            raise ValueError(f"{path} needs timestamp/day/date column. Columns: {list(d.columns)}")
        d["timestamp"] = d[date_col]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.floor("D")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    return d[(d["timestamp"] >= s) & (d["timestamp"] <= e)].drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _trade_stats(d: pd.DataFrame) -> dict[str, float | int]:
    active = (d["eth_weight"].abs() + d["btc_weight"].abs()) > 0
    ret = pd.to_numeric(d["strategy_return"], errors="coerce").fillna(0.0)
    side = pd.to_numeric(d["eth_weight"], errors="coerce").fillna(0.0)
    rows: list[dict[str, float | int]] = []
    in_trade = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_trade:
            start = i
            in_trade = True
        if in_trade and ((not on) or i == len(active) - 1):
            end = i - 1 if not on else i
            rows.append(
                {
                    "return": float((1.0 + ret.iloc[start : end + 1]).prod() - 1.0),
                    "days": end - start + 1,
                    "side": int(np.sign(side.iloc[start : end + 1].replace(0.0, np.nan).dropna().iloc[0]))
                    if side.iloc[start : end + 1].abs().sum()
                    else 0,
                }
            )
            in_trade = False
    if not rows:
        return {"trades": 0, "win_rate": np.nan, "avg_hold": np.nan, "avg_trade_return": np.nan}
    t = pd.DataFrame(rows)
    return {
        "trades": int(len(t)),
        "win_rate": float((t["return"] > 0).mean()),
        "avg_hold": float(t["days"].mean()),
        "avg_trade_return": float(t["return"].mean()),
        "long_eth_short_btc": int((t["side"] > 0).sum()),
        "short_eth_long_btc": int((t["side"] < 0).sum()),
    }


def _backtest(eth: pd.DataFrame, btc: pd.DataFrame, cost_bps: float, window: int, entry_z: float, stop_z: float, max_hold: int) -> pd.DataFrame:
    d = eth[["timestamp", "close"]].rename(columns={"close": "eth_close"}).merge(
        btc[["timestamp", "close"]].rename(columns={"close": "btc_close"}), on="timestamp", how="inner"
    )
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["day"] = d["timestamp"]
    d["ratio"] = d["eth_close"] / d["btc_close"]
    d["ratio_mean"] = d["ratio"].rolling(window, min_periods=window).mean()
    d["ratio_std"] = d["ratio"].rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    d["z_score"] = ((d["ratio"] - d["ratio_mean"]) / d["ratio_std"]).replace([np.inf, -np.inf], np.nan)
    d["eth_ret"] = d["eth_close"].pct_change().fillna(0.0)
    d["btc_ret"] = d["btc_close"].pct_change().fillna(0.0)

    target_side: list[int] = []
    active = 0
    held = 0
    for z in d["z_score"]:
        zf = float(z) if pd.notna(z) else np.nan
        if active == 0:
            if np.isfinite(zf) and zf < -abs(entry_z):
                active = 1
                held = 0
            elif np.isfinite(zf) and zf > abs(entry_z):
                active = -1
                held = 0
        else:
            exit_mean = (active == 1 and np.isfinite(zf) and zf >= 0.0) or (active == -1 and np.isfinite(zf) and zf <= 0.0)
            exit_stop = np.isfinite(zf) and abs(zf) >= abs(stop_z)
            exit_time = held >= int(max_hold)
            if exit_mean or exit_stop or exit_time:
                active = 0
                held = 0
        target_side.append(active)
        if active != 0:
            held += 1

    d["side_target"] = target_side
    d["side_exec"] = d["side_target"].shift(1).fillna(0.0)
    d["eth_weight"] = 0.5 * d["side_exec"]
    d["btc_weight"] = -0.5 * d["side_exec"]
    d["turnover"] = d["eth_weight"].diff().abs().fillna(d["eth_weight"].abs()) + d["btc_weight"].diff().abs().fillna(d["btc_weight"].abs())
    d["cost"] = d["turnover"] * (float(cost_bps) / 10000.0)
    d["strategy_return"] = d["eth_weight"] * d["eth_ret"] + d["btc_weight"] * d["btc_ret"] - d["cost"]
    d["equity"] = (1.0 + d["strategy_return"]).cumprod()
    return d


def _regression_beta(y: pd.Series, x: pd.Series) -> float:
    a = pd.concat([pd.to_numeric(y, errors="coerce"), pd.to_numeric(x, errors="coerce")], axis=1).dropna()
    if len(a) < 3:
        return np.nan
    xv = a.iloc[:, 1]
    var = float(xv.var(ddof=0))
    return float(a.iloc[:, 0].cov(xv, ddof=0) / var) if var > 1e-12 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH/BTC ratio statistical arbitrage backtest.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--eth-data", default="data/eth_daily_extended.csv")
    ap.add_argument("--btc-data", default="data/btc_daily_extended.csv")
    ap.add_argument("--paper-start", default="2026-03-21")
    ap.add_argument("--paper-end", default="2026-08-27")
    ap.add_argument("--paper-eth-data", default="artifacts/tmp_server_compare/data/eth_daily_from_5m_live.csv")
    ap.add_argument("--paper-btc-data", default="artifacts/tmp_server_compare/data/btc_daily.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--entry-z", type=float, default=1.5)
    ap.add_argument("--stop-z", type=float, default=3.0)
    ap.add_argument("--max-hold", type=int, default=20)
    ap.add_argument("--out-summary", default="artifacts/backtest/eth_btc_statarb_summary.csv")
    ap.add_argument("--out-daily", default="artifacts/backtest/eth_btc_statarb_daily.csv")
    args = ap.parse_args()

    eth = _load_daily(Path(args.eth_data), args.start, args.end)
    btc = _load_daily(Path(args.btc_data), args.start, args.end)
    d = _backtest(eth, btc, args.cost_bps, args.window, args.entry_z, args.stop_z, args.max_hold)
    main = _crypto_main(args.start, args.end, Path("artifacts/backtest/statarb_main"), 20.0, False)
    aligned = main.merge(d[["day", "strategy_return", "btc_ret"]], on="day", how="inner")
    st = _stats(aligned["strategy_return"])
    tm = _trade_stats(d)
    corr = float(aligned["main_return"].corr(aligned["strategy_return"])) if len(aligned) > 2 else np.nan
    beta = _regression_beta(aligned["strategy_return"], aligned["btc_ret"])

    paper_return = np.nan
    paper_trading = False
    if Path(args.paper_eth_data).exists() and Path(args.paper_btc_data).exists():
        warm_start = (pd.Timestamp(args.paper_start) - pd.Timedelta(days=max(args.window * 3, 240))).strftime("%Y-%m-%d")
        pe = _load_daily(Path(args.paper_eth_data), warm_start, args.paper_end)
        pb = _load_daily(Path(args.paper_btc_data), warm_start, args.paper_end)
        pdaily = _backtest(pe, pb, args.cost_bps, args.window, args.entry_z, args.stop_z, args.max_hold)
        s = pd.Timestamp(args.paper_start, tz="UTC")
        e = pd.Timestamp(args.paper_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        pwin = pdaily[(pdaily["day"] >= s) & (pdaily["day"] <= e)].copy()
        paper_return = float((1.0 + pwin["strategy_return"]).prod() - 1.0) if len(pwin) else np.nan
        paper_trading = bool(((pwin["eth_weight"].abs() + pwin["btc_weight"].abs()) > 0).any())
        pdaily.to_csv("artifacts/backtest/eth_btc_statarb_paper_window_daily.csv", index=False)

    out = pd.DataFrame(
        [
            {
                "strategy": "eth_btc_statarb",
                "period": f"{args.start} to {args.end}",
                **st,
                **tm,
                "corr_vs_main": corr,
                "beta_to_btc": beta,
                "paper_window_trading": paper_trading,
                "paper_window_return": paper_return,
            }
        ]
    )
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_summary, index=False)
    d.to_csv(args.out_daily, index=False)

    print("=" * 64)
    print("ETH/BTC STAT ARB RESULTS")
    print("=" * 64)
    print(f"Trades: {int(tm['trades'])}")
    print(f"Win rate: {tm['win_rate']*100:.1f}%" if np.isfinite(tm["win_rate"]) else "Win rate: n/a")
    print(f"Avg hold: {tm['avg_hold']:.1f} days" if np.isfinite(tm["avg_hold"]) else "Avg hold: n/a")
    print(f"Avg return/trade: {tm['avg_trade_return']*100:.2f}%" if np.isfinite(tm["avg_trade_return"]) else "Avg return/trade: n/a")
    print(f"Sharpe: {st['sharpe']:.3f} | Raw Sharpe: {st['raw_sharpe']:.3f}")
    print(f"CAGR: {st['cagr']*100:.2f}% | MaxDD: {st['maxdd']*100:.2f}%")
    print(f"Corr vs main: {corr:.3f} | BTC beta: {beta:.3f}")
    print(f"Paper window trading: {'YES' if paper_trading else 'NO'} | Return: {paper_return*100:.2f}%")
    print(f"Saved: {args.out_summary}")
    print(f"Saved: {args.out_daily}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
