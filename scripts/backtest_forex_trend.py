from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_overlay_strategies import _mean_reversion_overlay


PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
}

EMA_GRID = {
    "20/50/200": (20, 50, 200),
    "50/120/300": (50, 120, 300),
    "100/200/400": (100, 200, 400),
}


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _stats(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return {"return": np.nan, "cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan}
    eq = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=0) * np.sqrt(252.0))
    raw_sd = float(r.std(ddof=0))
    raw_sharpe = float(r.mean() / raw_sd * np.sqrt(252.0)) if raw_sd > 1e-12 else np.nan
    ex = r - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {
        "return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "raw_sharpe": raw_sharpe,
        "maxdd": maxdd,
        "ann_vol": ann_vol,
    }


def _download_yfinance(symbol: str, start: str, end: str, out: Path, refresh: bool) -> pd.DataFrame:
    if out.exists() and not refresh:
        return _load_price(out)
    try:
        import yfinance as yf
    except Exception as exc:
        raise SystemExit("yfinance is required for forex downloads. Install with: pip install yfinance") from exc

    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbol, start=start, end=end_plus, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit(f"No yfinance data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"date", "datetime", "index"}), None)
    if date_col is None:
        raise SystemExit(f"Could not identify date column for {symbol}. Columns: {list(raw.columns)}")
    raw = raw.rename(columns={date_col: "timestamp"})
    rename = {"adj_close": "adj_close"}
    raw = raw.rename(columns=rename)
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in raw.columns:
            raw[col] = np.nan if col == "volume" else raw.get("close", np.nan)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[keep].to_csv(out, index=False)
    return _load_price(out)


def _load_price(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns and "day" in d.columns:
        d["timestamp"] = d["day"]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.floor("D")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if "open" not in d.columns:
        d["open"] = d["close"]
    d["open"] = pd.to_numeric(d["open"], errors="coerce").fillna(d["close"])
    return d.reset_index(drop=True)


def _trade_stats(d: pd.DataFrame) -> dict[str, float | int]:
    active = pd.to_numeric(d["weight_exec"], errors="coerce").fillna(0.0).abs() > 0
    ret = pd.to_numeric(d["strategy_return"], errors="coerce").fillna(0.0)
    side = pd.to_numeric(d["signal_exec"], errors="coerce").fillna(0.0)
    trades: list[dict[str, float | int]] = []
    in_trade = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_trade:
            start = i
            in_trade = True
        if in_trade and ((not on) or i == len(active) - 1):
            end = i - 1 if not on else i
            r = float((1.0 + ret.iloc[start : end + 1]).prod() - 1.0)
            s = float(np.sign(side.iloc[start : end + 1].replace(0.0, np.nan).dropna().iloc[0])) if side.iloc[start : end + 1].abs().sum() else 0.0
            trades.append({"return": r, "days": end - start + 1, "side": int(s)})
            in_trade = False
    if not trades:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": np.nan,
            "short_win_rate": np.nan,
            "avg_trade_return": np.nan,
        }
    t = pd.DataFrame(trades)
    longs = t[t["side"] > 0]
    shorts = t[t["side"] < 0]
    return {
        "trades": int(len(t)),
        "win_rate": float((t["return"] > 0).mean()),
        "long_trades": int(len(longs)),
        "short_trades": int(len(shorts)),
        "long_win_rate": float((longs["return"] > 0).mean()) if len(longs) else np.nan,
        "short_win_rate": float((shorts["return"] > 0).mean()) if len(shorts) else np.nan,
        "avg_trade_return": float(t["return"].mean()),
    }


def _backtest_pair(d: pd.DataFrame, ema: tuple[int, int, int], confirm_days: int, weight: float, cost_bps: float) -> pd.DataFrame:
    x = d.copy()
    f, m, s = ema
    x["ema_fast"] = x["close"].ewm(span=f, adjust=False).mean()
    x["ema_mid"] = x["close"].ewm(span=m, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=s, adjust=False).mean()
    bull = (x["ema_fast"] > x["ema_mid"]) & (x["ema_mid"] > x["ema_slow"])
    bear = (x["ema_fast"] < x["ema_mid"]) & (x["ema_mid"] < x["ema_slow"])
    raw = np.select([bull, bear], [1.0, -1.0], default=0.0)
    confirmed = np.zeros(len(x), dtype=float)
    streak_side = 0.0
    streak = 0
    for i, side in enumerate(raw):
        if side != 0 and side == streak_side:
            streak += 1
        elif side != 0:
            streak_side = float(side)
            streak = 1
        else:
            streak_side = 0.0
            streak = 0
        confirmed[i] = float(side) if side != 0 and streak >= int(confirm_days) else 0.0
    x["signal_target"] = confirmed
    x["signal_exec"] = x["signal_target"].shift(1).fillna(0.0)
    x["weight_exec"] = x["signal_exec"] * float(weight)
    x["pair_return"] = x["close"].pct_change().fillna(0.0)
    x["turnover"] = (x["weight_exec"] - x["weight_exec"].shift(1).fillna(0.0)).abs()
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["pair_return"] - x["cost"]
    return x


def _run_validated_crypto(start: str, end: str, work: Path, cost_bps: float, refresh: bool) -> pd.DataFrame:
    daily = work / "validated_crypto_daily.csv"
    summary = work / "validated_crypto_summary.csv"
    if refresh or not daily.exists():
        _run(
            [
                sys.executable,
                "scripts/backtest_eth_btc_portfolio.py",
                "--start",
                start,
                "--end",
                end,
                "--vol-filter",
                "--transition-momentum",
                "--asymmetric-sizing",
                "--allocation-mode",
                "signal_weighted",
                "--cost-mode",
                "weight_change",
                "--cost-bps",
                str(float(cost_bps)),
                "--gross-cap",
                "0.8",
                "--eth-confirm-days",
                "3",
                "--btc-confirm-days",
                "5",
                "--eth-ema",
                "50,120,300",
                "--btc-ema",
                "15,40,120",
                "--include-gold",
                "--gold-symbol",
                "PAXG-USD",
                "--gold-ema",
                "25,65,180",
                "--gold-cap",
                "0.3",
                "--gold-cost-bps",
                str(float(cost_bps)),
                "--out-summary-csv",
                str(summary),
                "--out-daily-csv",
                str(daily),
            ]
        )
    base = pd.read_csv(daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    mr = _mean_reversion_overlay(base, gross_cap=0.8, cost_bps=float(cost_bps), z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr["main_return"] = pd.to_numeric(mr["combined_return"], errors="coerce").fillna(0.0) + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    return mr[["day", "main_return"]].dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Forex long/short trend following as a parallel sleeve to LPBot crypto.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--data-dir", default="data/forex")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_trend")
    ap.add_argument("--out-summary", default="artifacts/backtest/forex_trend_summary.csv")
    ap.add_argument("--out-combined", default="artifacts/backtest/forex_combined_summary.csv")
    ap.add_argument("--position-size", type=float, default=0.15)
    ap.add_argument("--combined-fx-weight", type=float, default=0.10)
    ap.add_argument("--main-weight", type=float, default=0.70)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--crypto-cost-bps", type=float, default=20.0)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--refresh-crypto", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    main = _run_validated_crypto(args.start, args.end, work, float(args.crypto_cost_bps), bool(args.refresh_crypto))
    main_stats = _stats(main["main_return"])

    rows: list[dict[str, object]] = []
    best_frames: dict[str, pd.DataFrame] = {}
    for pair, symbol in PAIRS.items():
        px = _download_yfinance(symbol, args.start, args.end, data_dir / f"{pair}_daily.csv", bool(args.refresh_data))
        best_row: dict[str, object] | None = None
        best_df: pd.DataFrame | None = None
        for ema_label, ema in EMA_GRID.items():
            for confirm in [3, 5]:
                bt = _backtest_pair(px, ema, confirm, float(args.position_size), float(args.cost_bps))
                st = _stats(bt["strategy_return"])
                tm = _trade_stats(bt)
                aligned = main.merge(bt[["timestamp", "strategy_return"]].rename(columns={"timestamp": "day", "strategy_return": "fx_return"}), on="day", how="inner")
                corr = float(aligned["main_return"].corr(aligned["fx_return"])) if len(aligned) > 2 else np.nan
                row = {
                    "pair": pair,
                    "symbol": symbol,
                    "ema": ema_label,
                    "confirm_days": confirm,
                    **st,
                    **tm,
                    "corr_vs_main": corr,
                    "daily_csv": str(work / f"{pair}_{ema_label.replace('/', '_')}_c{confirm}_daily.csv"),
                }
                rows.append(row)
                bt.to_csv(row["daily_csv"], index=False)
                if best_row is None or (float(row["raw_sharpe"]) > float(best_row["raw_sharpe"])):
                    best_row = row
                    best_df = bt
        assert best_row is not None and best_df is not None
        best_frames[pair] = best_df.copy()

    summary = pd.DataFrame(rows)
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    best = summary.sort_values(["pair", "raw_sharpe", "maxdd"], ascending=[True, False, False]).groupby("pair", as_index=False).head(1)
    combined = main.copy()
    combined["portfolio_return"] = float(args.main_weight) * pd.to_numeric(combined["main_return"], errors="coerce").fillna(0.0)
    corr_values: list[float] = []
    for _, row in best.iterrows():
        pair = str(row["pair"])
        bt = pd.read_csv(str(row["daily_csv"]))
        bt["day"] = pd.to_datetime(bt["timestamp"], utc=True, errors="coerce").dt.floor("D")
        bt["fx_unit_return"] = pd.to_numeric(bt["strategy_return"], errors="coerce").fillna(0.0) / float(args.position_size)
        combined = combined.merge(bt[["day", "fx_unit_return"]].rename(columns={"fx_unit_return": f"{pair}_return"}), on="day", how="left")
        combined[f"{pair}_return"] = pd.to_numeric(combined[f"{pair}_return"], errors="coerce").fillna(0.0)
        combined["portfolio_return"] += float(args.combined_fx_weight) * combined[f"{pair}_return"]
        corr_values.append(float(row["corr_vs_main"]))

    combined_stats = _stats(combined["portfolio_return"])
    combined_rows: list[dict[str, object]] = [
        {
            "configuration": "main_only",
            "weighting": "100% crypto",
            **main_stats,
            "avg_fx_corr_vs_main": np.nan,
        },
        {
            "configuration": "main_70_fx_10_each",
            "weighting": "70% crypto + 10% EURUSD + 10% GBPUSD + 10% USDJPY",
            **combined_stats,
            "avg_fx_corr_vs_main": float(np.nanmean(corr_values)) if corr_values else np.nan,
        },
    ]

    usdjpy_best = best[best["pair"].astype(str).eq("USDJPY")].iloc[0]
    usdjpy = pd.read_csv(str(usdjpy_best["daily_csv"]))
    usdjpy["day"] = pd.to_datetime(usdjpy["timestamp"], utc=True, errors="coerce").dt.floor("D")
    usdjpy["fx_unit_return"] = pd.to_numeric(usdjpy["strategy_return"], errors="coerce").fillna(0.0) / float(args.position_size)
    usd_base = main.merge(usdjpy[["day", "fx_unit_return"]], on="day", how="left")
    usd_base["fx_unit_return"] = pd.to_numeric(usd_base["fx_unit_return"], errors="coerce").fillna(0.0)
    usd_corr = float(usdjpy_best["corr_vs_main"])
    for main_w, fx_w in [(0.90, 0.10), (0.80, 0.20), (0.70, 0.30)]:
        r = main_w * pd.to_numeric(usd_base["main_return"], errors="coerce").fillna(0.0) + fx_w * usd_base["fx_unit_return"]
        st = _stats(r)
        combined_rows.append(
            {
                "configuration": f"main_{int(main_w*100)}_usdjpy_{int(fx_w*100)}",
                "weighting": f"{main_w:.0%} crypto + {fx_w:.0%} USDJPY",
                **st,
                "avg_fx_corr_vs_main": usd_corr,
            }
        )
        tmp = usd_base[["day", "main_return", "fx_unit_return"]].copy()
        tmp["portfolio_return"] = r
        tmp.to_csv(work / f"combined_usdjpy_{int(main_w*100)}_{int(fx_w*100)}_daily.csv", index=False)

    comb = pd.DataFrame(combined_rows)
    out_combined = Path(args.out_combined)
    comb.to_csv(out_combined, index=False)
    combined.to_csv(work / "combined_daily.csv", index=False)

    print("")
    print("=" * 72)
    print("FOREX TREND FOLLOWING RESULTS")
    print("2019-2024 | Long+Short | 2bps costs")
    print("=" * 72)
    print("Pair      EMA          Cnf Sharpe RawShp   CAGR   MaxDD  Corr")
    print("-" * 72)
    for _, r in summary.sort_values(["pair", "ema", "confirm_days"]).iterrows():
        print(
            f"{str(r['pair']):<9} {str(r['ema']):<12} {int(r['confirm_days']):>3} "
            f"{float(r['sharpe']):>6.3f} {float(r['raw_sharpe']):>6.3f} {float(r['cagr'])*100:>6.2f}% "
            f"{float(r['maxdd'])*100:>6.2f}% {float(r['corr_vs_main']):>6.3f}"
        )
    print("-" * 72)
    print("Best per pair:")
    for _, r in best.sort_values("pair").iterrows():
        print(
            f"  {r['pair']}: EMA {r['ema']} confirm {int(r['confirm_days'])} "
            f"excess Sharpe {float(r['sharpe']):.3f}, raw Sharpe {float(r['raw_sharpe']):.3f}"
        )
    print("-" * 72)
    print("Combined portfolio (70/10/10/10):")
    print(f"  Main only: Sharpe {main_stats['sharpe']:.3f}  MaxDD {main_stats['maxdd']*100:.2f}%  CAGR {main_stats['cagr']*100:.2f}%")
    print(f"  + Forex:   Sharpe {combined_stats['sharpe']:.3f}  MaxDD {combined_stats['maxdd']*100:.2f}%  CAGR {combined_stats['cagr']*100:.2f}%")
    print(f"  MaxDD change: {(combined_stats['maxdd'] - main_stats['maxdd'])*100:+.2f}%")
    print("")
    print("USDJPY-only allocation sweep:")
    for _, r in comb[comb["configuration"].astype(str).str.contains("usdjpy")].iterrows():
        print(
            f"  {r['weighting']:<24} Sharpe {float(r['sharpe']):.3f}  "
            f"MaxDD {float(r['maxdd'])*100:.2f}%  CAGR {float(r['cagr'])*100:.2f}%"
        )
    all_pair_sharpes_ok = bool((best["raw_sharpe"] > 0.5).all())
    max_corr = float(best["corr_vs_main"].max())
    decision = "ADD" if combined_stats["sharpe"] > 1.80 and max_corr < 0.3 and all_pair_sharpes_ok else "SKIP"
    print(f"Decision: {decision}")
    print("=" * 72)
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
