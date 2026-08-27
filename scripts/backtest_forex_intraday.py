from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_forex_optimised import _crypto_main, _stats


PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
}


def _load_hourly(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return d.reset_index(drop=True)


def _download_hourly(symbol: str, out: Path, refresh: bool) -> pd.DataFrame:
    if out.exists() and not refresh:
        return _load_hourly(out)
    try:
        import yfinance as yf
    except Exception as exc:
        raise SystemExit("yfinance is required. Install with: pip install yfinance") from exc
    raw = yf.download(symbol, period="729d", interval="1h", auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit(f"No hourly yfinance data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"datetime", "date", "index"}), None)
    if date_col is None:
        raise SystemExit(f"Could not identify datetime column for {symbol}. Columns: {list(raw.columns)}")
    raw = raw.rename(columns={date_col: "timestamp"})
    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raw[col] = raw.get("close", np.nan)
    if "volume" not in raw.columns:
        raw["volume"] = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out, index=False)
    return _load_hourly(out)


def _atr(d: pd.DataFrame, window: int) -> pd.Series:
    prev = d["close"].shift(1)
    tr = pd.concat([(d["high"] - d["low"]).abs(), (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _trade_stats(d: pd.DataFrame) -> dict[str, float | int]:
    active = pd.to_numeric(d["weight_exec"], errors="coerce").fillna(0.0).abs() > 0
    ret = pd.to_numeric(d["strategy_return"], errors="coerce").fillna(0.0)
    side = pd.to_numeric(d["signal_exec"], errors="coerce").fillna(0.0)
    rows: list[dict[str, float | int]] = []
    in_trade = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_trade:
            start = i
            in_trade = True
        if in_trade and ((not on) or i == len(active) - 1):
            end = i - 1 if not on else i
            nonzero = side.iloc[start : end + 1][side.iloc[start : end + 1].abs() > 0]
            rows.append(
                {
                    "return": float((1.0 + ret.iloc[start : end + 1]).prod() - 1.0),
                    "hours": end - start + 1,
                    "side": int(np.sign(nonzero.iloc[0])) if len(nonzero) else 0,
                }
            )
            in_trade = False
    if not rows:
        return {"trades": 0, "win_rate": np.nan, "avg_hold_hours": np.nan, "trades_per_week": 0.0, "avg_trade_return": np.nan}
    t = pd.DataFrame(rows)
    weeks = max((d["timestamp"].max() - d["timestamp"].min()).days / 7.0, 1e-9)
    return {
        "trades": int(len(t)),
        "win_rate": float((t["return"] > 0).mean()),
        "avg_hold_hours": float(t["hours"].mean()),
        "trades_per_week": float(len(t) / weeks),
        "avg_trade_return": float(t["return"].mean()),
    }


def _backtest_pair(d: pd.DataFrame, cost_bps: float, weight: float) -> pd.DataFrame:
    x = d.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["atr14"] = _atr(x, 14).replace(0.0, np.nan)
    x["mid"] = (x["ema20"] + x["ema50"]) / 2.0
    x["dev"] = ((x["close"] - x["mid"]) / x["atr14"]).replace([np.inf, -np.inf], np.nan)
    x["liquid_hours"] = x["timestamp"].dt.hour.between(7, 17, inclusive="both")

    active = 0
    held = 0
    entry_dev = np.nan
    target: list[int] = []
    for _, row in x.iterrows():
        dev = float(row["dev"]) if pd.notna(row["dev"]) else np.nan
        liquid = bool(row["liquid_hours"])
        if active == 0:
            if liquid and np.isfinite(dev) and dev < -1.5:
                active = 1
                held = 0
                entry_dev = dev
            elif liquid and np.isfinite(dev) and dev > 1.5:
                active = -1
                held = 0
                entry_dev = dev
        else:
            recovered = np.isfinite(dev) and abs(dev) <= 0.5
            stop = (active == 1 and np.isfinite(dev) and dev < entry_dev - 2.0) or (active == -1 and np.isfinite(dev) and dev > entry_dev + 2.0)
            timeout = held >= 24
            if recovered or stop or timeout:
                active = 0
                held = 0
                entry_dev = np.nan
        target.append(active)
        if active:
            held += 1

    x["signal_target"] = target
    x["signal_exec"] = x["signal_target"].shift(1).fillna(0.0)
    x["weight_exec"] = x["signal_exec"] * float(weight)
    x["pair_return"] = x["close"].pct_change().fillna(0.0)
    x["turnover"] = x["weight_exec"].diff().abs().fillna(x["weight_exec"].abs())
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["pair_return"] - x["cost"]
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday forex mean-reversion on 1h bars.")
    ap.add_argument("--data-dir", default="data/forex_intraday")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_intraday")
    ap.add_argument("--out-summary", default="artifacts/backtest/forex_intraday_summary.csv")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--weight", type=float, default=0.15)
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    daily_returns: dict[str, pd.DataFrame] = {}
    start_seen: pd.Timestamp | None = None
    end_seen: pd.Timestamp | None = None
    for pair, symbol in PAIRS.items():
        px = _download_hourly(symbol, data_dir / f"{pair}_1h.csv", bool(args.refresh_data))
        start_seen = px["timestamp"].min() if start_seen is None else min(start_seen, px["timestamp"].min())
        end_seen = px["timestamp"].max() if end_seen is None else max(end_seen, px["timestamp"].max())
        bt = _backtest_pair(px, float(args.cost_bps), float(args.weight))
        st = _stats(bt["strategy_return"])
        tm = _trade_stats(bt)
        bt["day"] = bt["timestamp"].dt.floor("D")
        daily = bt.groupby("day", as_index=False)["strategy_return"].apply(lambda s: (1.0 + s).prod() - 1.0)
        daily_returns[pair] = daily.rename(columns={"strategy_return": f"{pair}_return"})
        bt.to_csv(work / f"{pair}_intraday_mr_daily.csv", index=False)
        rows.append({"pair": pair, "symbol": symbol, "start": str(px["timestamp"].min()), "end": str(px["timestamp"].max()), **st, **tm})

    start = str(start_seen.date()) if start_seen is not None else "2024-01-01"
    end = str(end_seen.date()) if end_seen is not None else "2026-08-27"
    main = _crypto_main(start, end, Path("artifacts/backtest/forex_intraday_main"), 20.0, False)
    for i, row in enumerate(rows):
        pair = str(row["pair"])
        aligned = main.merge(daily_returns[pair].rename(columns={f"{pair}_return": "fx_return"}), on="day", how="inner")
        rows[i]["corr_vs_crypto"] = float(aligned["main_return"].corr(aligned["fx_return"])) if len(aligned) > 2 else np.nan

        s = pd.Timestamp("2026-03-21", tz="UTC")
        e = pd.Timestamp("2026-08-27", tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        bt = pd.read_csv(work / f"{pair}_intraday_mr_daily.csv")
        bt["timestamp"] = pd.to_datetime(bt["timestamp"], utc=True, errors="coerce")
        pwin = bt[(bt["timestamp"] >= s) & (bt["timestamp"] <= e)]
        rows[i]["paper_window_trades"] = _trade_stats(pwin)["trades"] if len(pwin) else 0
        rows[i]["paper_window_return"] = float((1.0 + pd.to_numeric(pwin["strategy_return"], errors="coerce").fillna(0.0)).prod() - 1.0) if len(pwin) else np.nan

    summary = pd.DataFrame(rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    best = summary.sort_values("raw_sharpe", ascending=False).iloc[0]

    print("=" * 64)
    print("FOREX INTRADAY MEAN-REVERSION RESULTS")
    print("=" * 64)
    for _, r in summary.sort_values("raw_sharpe", ascending=False).iterrows():
        print(
            f"{r['pair']:<6} Sharpe {float(r['sharpe']):>7.3f} raw {float(r['raw_sharpe']):>6.3f} "
            f"CAGR {float(r['cagr'])*100:>6.2f}% MaxDD {float(r['maxdd'])*100:>6.2f}% "
            f"trades/wk {float(r['trades_per_week']):>5.2f} win {float(r['win_rate'])*100:>5.1f}% "
            f"corr {float(r['corr_vs_crypto']):>6.3f}"
        )
    print(f"Best pair: {best['pair']}")
    print(f"Saved: {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
