from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib import request

import numpy as np
import pandas as pd


ASSETS: list[dict[str, str]] = [
    {"name": "BTC", "ticker": "BTC-USD", "source": "yfinance"},
    {"name": "Oil/WTI", "ticker": "CL=F", "source": "yfinance"},
    {"name": "Nat Gas", "ticker": "NG=F", "source": "yfinance"},
    {"name": "Gold", "ticker": "GC=F", "source": "yfinance"},
    {"name": "Silver", "ticker": "SI=F", "source": "yfinance"},
    {"name": "S&P 500", "ticker": "^GSPC", "source": "yfinance"},
    {"name": "FTSE 100", "ticker": "^FTSE", "source": "yfinance"},
    {"name": "TLT", "ticker": "TLT", "source": "yfinance"},
    {"name": "DXY", "ticker": "DX-Y.NYB", "source": "yfinance"},
    {"name": "Wheat", "ticker": "ZW=F", "source": "yfinance"},
]


def _safe_file_token(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def _fetch_yahoo_daily(ticker: str, timeout_sec: float, range_period: str) -> pd.DataFrame:
    enc = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
        f"?interval=1d&range={range_period}&includePrePost=false&events=div%2Csplits"
    )
    req = request.Request(url, headers={"User-Agent": "LPBot-MarketTracker/1.0"})
    with request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    r0 = result[0]
    ts = r0.get("timestamp", []) or []
    q = (r0.get("indicators", {}) or {}).get("quote", [{}])[0] or {}
    if not ts:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, unit="s", utc=True),
            "open": pd.to_numeric(q.get("open", []), errors="coerce"),
            "high": pd.to_numeric(q.get("high", []), errors="coerce"),
            "low": pd.to_numeric(q.get("low", []), errors="coerce"),
            "close": pd.to_numeric(q.get("close", []), errors="coerce"),
            "volume": pd.to_numeric(q.get("volume", []), errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _compute_metrics(asset_name: str, ticker: str, df: pd.DataFrame, asof_date: dt.date) -> dict[str, Any] | None:
    if len(df) < 2:
        return None
    c = pd.to_numeric(df["close"], errors="coerce")
    c = c.dropna()
    if len(c) < 2:
        return None
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema55 = c.ewm(span=55, adjust=False).mean()
    ema144 = c.ewm(span=144, adjust=False).mean()
    stack = (ema21 > ema55) & (ema55 > ema144)
    stack_rev = stack.astype(int).iloc[::-1]
    days_aligned = int(stack_rev.cumprod().sum()) if len(stack_rev) else 0
    trend = "CHOP"
    if float(ema21.iloc[-1]) > float(ema55.iloc[-1]) > float(ema144.iloc[-1]):
        trend = "BULL"
    elif float(ema21.iloc[-1]) < float(ema55.iloc[-1]) < float(ema144.iloc[-1]):
        trend = "BEAR"
    pct_change = float(c.iloc[-1] / c.iloc[-2] - 1.0)
    return {
        "date": asof_date.isoformat(),
        "asset": asset_name,
        "ticker": ticker,
        "price": float(c.iloc[-1]),
        "pct_change": pct_change,
        "ema21": float(ema21.iloc[-1]),
        "ema55": float(ema55.iloc[-1]),
        "ema144": float(ema144.iloc[-1]),
        "stack_aligned": bool(stack.iloc[-1]),
        "days_aligned": days_aligned,
        "trend": trend,
    }


def _write_asset_daily(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _upsert_tracker(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    dnew = pd.DataFrame(rows)
    cols = [
        "date",
        "asset",
        "ticker",
        "price",
        "pct_change",
        "ema21",
        "ema55",
        "ema144",
        "stack_aligned",
        "days_aligned",
        "trend",
    ]
    if not path.exists():
        dout = dnew[cols].copy()
        dout.to_csv(path, index=False)
        return dout
    dold = pd.read_csv(path, low_memory=False)
    for c in cols:
        if c not in dold.columns:
            dold[c] = np.nan
    dold = dold[cols]
    # replace same date+asset rows
    key = set((str(r["date"]), str(r["asset"])) for r in rows)
    mask = ~dold.apply(lambda x: (str(x.get("date", "")), str(x.get("asset", ""))) in key, axis=1)
    dout = pd.concat([dold[mask], dnew[cols]], ignore_index=True).sort_values(["date", "asset"])
    dout.to_csv(path, index=False)
    return dout


def _fmt_price(asset: str, px: float) -> str:
    if not np.isfinite(px):
        return "n/a"
    if asset in {"S&P 500", "FTSE 100", "DXY", "Wheat"}:
        return f"{px:,.1f}"
    if asset in {"BTC"}:
        return f"{px:,.0f}"
    return f"{px:,.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Track daily multi-asset market state via Yahoo Finance")
    p.add_argument("--out-csv", default="artifacts/markets/market_tracker.csv")
    p.add_argument("--data-dir", default="data/markets")
    p.add_argument("--timeout-sec", type=float, default=15.0)
    p.add_argument("--range", default="2y", help="Yahoo range for daily history, e.g. 1y/2y/5y/max")
    args = p.parse_args()

    now_utc = dt.datetime.now(dt.timezone.utc)
    asof = now_utc.date()
    rows: list[dict[str, Any]] = []
    for a in ASSETS:
        name = a["name"]
        ticker = a["ticker"]
        try:
            d = _fetch_yahoo_daily(ticker=ticker, timeout_sec=float(args.timeout_sec), range_period=str(args.range))
        except Exception:
            continue
        if len(d) == 0:
            continue
        fname = f"{_safe_file_token(ticker)}_daily.csv"
        _write_asset_daily(Path(args.data_dir) / fname, d)
        m = _compute_metrics(name, ticker, d, asof_date=asof)
        if m is not None:
            rows.append(m)

    if not rows:
        print("No market rows produced")
        return 0
    _upsert_tracker(Path(args.out_csv), rows)

    # Human summary for logs
    order = [a["name"] for a in ASSETS]
    by_name = {r["asset"]: r for r in rows}
    for name in order:
        if name not in by_name:
            continue
        r = by_name[name]
        marker = " <- stack aligned" if bool(r["stack_aligned"]) else ""
        print(f"{name:<9} {_fmt_price(name, float(r['price'])):>10}  {float(r['pct_change'])*100:+5.1f}%  {r['trend']}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
