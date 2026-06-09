from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    {"name": "BTC", "ticker": "BTC-USD"},
    {"name": "ETH", "ticker": "ETH-USD"},
    {"name": "Oil/WTI", "ticker": "CL=F"},
    {"name": "Nat Gas", "ticker": "NG=F"},
    {"name": "Gold", "ticker": "GC=F"},
    {"name": "Silver", "ticker": "SI=F"},
    {"name": "S&P 500", "ticker": "^GSPC"},
    {"name": "FTSE 100", "ticker": "^FTSE"},
    {"name": "TLT", "ticker": "TLT"},
    {"name": "DXY", "ticker": "DX-Y.NYB"},
    {"name": "Wheat", "ticker": "ZW=F"},
]


@dataclass
class AssetResult:
    asset: str
    cagr: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    time_in_market: float
    buyhold_cagr: float
    excess_cagr: float
    grade: str


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


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
        # yfinance may return MultiIndex columns even for a single ticker.
        d.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in d.columns]
    d = d.reset_index()
    if "Date" not in d.columns:
        if "index" in d.columns:
            d = d.rename(columns={"index": "Date"})
        elif "Datetime" in d.columns:
            d = d.rename(columns={"Datetime": "Date"})
        elif len(d.columns) > 0:
            # Fallback for provider/schema changes.
            d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    if "Adj Close" in d.columns:
        d["price"] = pd.to_numeric(d["Adj Close"], errors="coerce")
    elif "Close" in d.columns:
        d["price"] = pd.to_numeric(d["Close"], errors="coerce")
    else:
        return pd.DataFrame()
    return d[["Date", "price"]].copy()


def _max_consecutive_true(mask: pd.Series) -> int:
    if len(mask) == 0:
        return 0
    grp = (~mask).cumsum()
    run = mask.groupby(grp).cumcount() + 1
    run = run.where(mask, 0)
    return int(run.max()) if len(run) else 0


def _prep_prices(d: pd.DataFrame, name: str) -> pd.DataFrame:
    raw_nan = d["price"].isna()
    d["price"] = d["price"].ffill(limit=3)
    still_nan = d["price"].isna()
    long_gap_days = _max_consecutive_true(still_nan)
    if long_gap_days > 0:
        print(f"Warning: Data quality issue for {name}: gaps >3 days detected (max unresolved gap {long_gap_days}d). Dropping unresolved rows.")
    d = d.dropna(subset=["price"]).copy()
    if raw_nan.mean() > 0.05:
        print(f"Warning: Data quality for {name} may be poor ({raw_nan.mean()*100:.1f}% raw NaN).")
    return d


def _compute_regime(price: pd.Series) -> pd.Series:
    ret20 = price / price.shift(20) - 1.0
    dd20 = price / price.rolling(20, min_periods=20).max() - 1.0
    regime = pd.Series("CHOP", index=price.index, dtype=object)
    regime[(ret20 > 0.05) & (dd20 > -0.10)] = "BULL"
    regime[dd20 < -0.15] = "BEAR"
    return regime


def _grade(sharpe: float, max_dd: float) -> str:
    if np.isfinite(sharpe) and sharpe >= 0.7 and np.isfinite(max_dd) and max_dd > -0.35:
        return "PASS"
    if np.isfinite(sharpe) and sharpe >= 0.5 and np.isfinite(max_dd) and max_dd > -0.40:
        return "MARGINAL"
    return "FAIL"


def _backtest_one(name: str, ticker: str, data: pd.DataFrame, out_dir: Path) -> AssetResult | None:
    if len(data) < 200:
        print(f"Warning: Insufficient data for {name}")
        return None

    df = data.copy().reset_index(drop=True)
    df["daily_return"] = pd.to_numeric(df["price"], errors="coerce").pct_change().fillna(0.0)

    df["ema21"] = df["price"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["price"].ewm(span=55, adjust=False).mean()
    df["ema144"] = df["price"].ewm(span=144, adjust=False).mean()

    df["stack_aligned"] = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema144"])
    grp = (~df["stack_aligned"]).cumsum()
    days_aligned = df["stack_aligned"].groupby(grp).cumcount() + 1
    df["days_aligned"] = days_aligned.where(df["stack_aligned"], 0).astype(int)

    entry_confirm = (df["stack_aligned"].rolling(3, min_periods=3).min() == 1).fillna(False)
    exit_confirm = ((~df["stack_aligned"]).rolling(3, min_periods=3).min() == 1).fillna(False)

    df["regime"] = _compute_regime(df["price"])
    off_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}

    rv = df["daily_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    vol_scalar = vol_scalar.clip(lower=0.25, upper=1.0).fillna(0.25)
    df["vol_scalar"] = vol_scalar

    pos = np.zeros(len(df), dtype=int)
    active = 0
    for i in range(len(df)):
        if active == 0 and bool(entry_confirm.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_confirm.iloc[i]):
            active = 0
        pos[i] = active
    df["position_active"] = pos

    df["raw_signal"] = np.where(df["position_active"] == 1, df["vol_scalar"], 0.0)
    df["off_weight"] = df["regime"].map(off_map).fillna(0.0)
    df["scaled_weight"] = df["raw_signal"] * df["off_weight"]

    prev_w = df["scaled_weight"].shift(1).fillna(0.0)
    delta_w = (df["scaled_weight"] - prev_w).abs()
    cost_ret = 0.001 * delta_w
    df["strategy_return"] = df["scaled_weight"] * df["daily_return"] - cost_ret

    # Warmup rule: start from day 145 (EMA144 warmup)
    df = df.iloc[144:].copy()
    if len(df) < 200:
        print(f"Warning: Insufficient post-warmup data for {name}")
        return None

    df["cumulative_strategy"] = (1.0 + df["strategy_return"]).cumprod()
    df["cumulative_spot"] = (1.0 + df["daily_return"]).cumprod()

    years = len(df) / 252.0
    total_ret = float(df["cumulative_strategy"].iloc[-1])
    cagr = total_ret ** (1.0 / years) - 1.0 if years > 0 else np.nan

    daily_excess = df["strategy_return"] - (0.05 / 252.0)
    sd = float(daily_excess.std(ddof=0))
    sharpe = float(np.mean(daily_excess) / sd * np.sqrt(252.0)) if sd > 0 else np.nan

    rolling_max = df["cumulative_strategy"].cummax()
    dd = (df["cumulative_strategy"] - rolling_max) / rolling_max
    max_dd = float(dd.min()) if len(dd) else np.nan

    # Count transitions flat<->long using active state
    prev_pos = pd.Series(df["position_active"]).shift(1).fillna(0).astype(int)
    curr_pos = pd.Series(df["position_active"]).astype(int)
    n_trades = int((prev_pos != curr_pos).sum())

    # Completed trade returns: cumulative strategy return while position_active==1 between entry and exit
    trade_returns: list[float] = []
    in_trade = False
    entry_idx = None
    for i in range(len(df)):
        p = int(df["position_active"].iloc[i])
        if not in_trade and p == 1:
            in_trade = True
            entry_idx = i
        elif in_trade and p == 0:
            if entry_idx is not None and i > entry_idx:
                tr = (1.0 + df["strategy_return"].iloc[entry_idx:i]).prod() - 1.0
                trade_returns.append(float(tr))
            in_trade = False
            entry_idx = None
    if in_trade and entry_idx is not None and entry_idx < len(df) - 1:
        tr = (1.0 + df["strategy_return"].iloc[entry_idx:]).prod() - 1.0
        trade_returns.append(float(tr))

    win_rate = float(np.mean([x > 0 for x in trade_returns])) if trade_returns else np.nan
    time_in_market = float((df["scaled_weight"] > 0).mean())

    bh_total = float(df["price"].iloc[-1] / df["price"].iloc[0])
    buyhold_cagr = bh_total ** (1.0 / years) - 1.0 if years > 0 else np.nan
    excess_cagr = cagr - buyhold_cagr if np.isfinite(cagr) and np.isfinite(buyhold_cagr) else np.nan

    grade = _grade(sharpe, max_dd)

    out = df[
        [
            "Date",
            "price",
            "ema21",
            "ema55",
            "ema144",
            "stack_aligned",
            "days_aligned",
            "regime",
            "vol_scalar",
            "scaled_weight",
            "daily_return",
            "strategy_return",
            "cumulative_strategy",
            "cumulative_spot",
        ]
    ].copy()
    out = out.rename(columns={"Date": "date"})
    out_path = out_dir / f"{_sanitize_name(name)}_backtest.csv"
    out.to_csv(out_path, index=False)

    return AssetResult(
        asset=name,
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        n_trades=int(n_trades),
        win_rate=float(win_rate) if np.isfinite(win_rate) else np.nan,
        time_in_market=float(time_in_market),
        buyhold_cagr=float(buyhold_cagr),
        excess_cagr=float(excess_cagr),
        grade=grade,
    )


def _print_summary(summary: pd.DataFrame) -> None:
    print("===================================================")
    print("MULTI-ASSET BACKTEST RESULTS (2019-2024)")
    print("EMA 21/55/144 Trend Follow | Vol Target 50%")
    print("===================================================")
    print("Rank  Asset        CAGR    Sharpe  MaxDD    Trades")
    print("---------------------------------------------------")
    for i, row in enumerate(summary.itertuples(index=False), start=1):
        print(
            f"{i:<5} {row.asset:<12} {row.cagr*100:>6.1f}%  {row.sharpe:>6.3f}  {row.max_dd*100:>6.1f}%  {int(row.n_trades):>6}"
        )
    print("---------------------------------------------------")
    p = int((summary["grade"] == "PASS").sum())
    m = int((summary["grade"] == "MARGINAL").sum())
    f = int((summary["grade"] == "FAIL").sum())
    print(f"Pass (Sharpe > 0.7):  {p} assets")
    print(f"Marginal (0.5-0.7):   {m} assets")
    print(f"Fail (< 0.5):         {f} assets")
    print("===================================================")
    print()
    print("BUY AND HOLD COMPARISON (2019-2024)")
    print("-------------------------------------")
    print("Asset        B&H CAGR   Strategy   Excess")
    for row in summary.itertuples(index=False):
        print(f"{row.asset:<12} {row.buyhold_cagr*100:>7.1f}%   {row.cagr*100:>7.1f}%   {row.excess_cagr*100:+6.1f}%")
    print()
    passed = summary.loc[summary["grade"] == "PASS", "asset"].tolist()
    if passed:
        print("Assets cleared for paper trading:")
        print("  " + ", ".join(passed))
    else:
        print("Assets cleared for paper trading:")
        print("  None")


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest multi-asset EMA trend-follow strategy")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out-dir", default="artifacts/backtest/multi_asset")
    p.add_argument("--summary-csv", default="artifacts/backtest/multi_asset_summary.csv")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = out_dir / ".yf_tz_cache"
    summary_rows: list[dict[str, Any]] = []

    for a in ASSETS:
        name = a["name"]
        ticker = a["ticker"]
        try:
            raw = _download_asset(ticker=ticker, start=args.start, end=args.end, yf_cache_dir=yf_cache_dir)
        except Exception as exc:
            print(f"Warning: Could not download {name}: {exc}")
            continue
        if len(raw) == 0:
            print(f"Warning: Could not download {name}: empty dataset")
            continue
        try:
            d = _prep_prices(raw, name=name)
            res = _backtest_one(name=name, ticker=ticker, data=d, out_dir=out_dir)
        except Exception as exc:
            print(f"Warning: Failed processing {name}: {exc}")
            continue
        if res is None:
            continue
        summary_rows.append(
            {
                "asset": res.asset,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "max_dd": res.max_dd,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate,
                "time_in_market": res.time_in_market,
                "buyhold_cagr": res.buyhold_cagr,
                "excess_cagr": res.excess_cagr,
                "grade": res.grade,
            }
        )

    if not summary_rows:
        print("No assets produced valid backtest results.")
        return 0

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    _print_summary(summary)
    print(f"\nSaved summary: {args.summary_csv}")
    print(f"Saved daily files: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
