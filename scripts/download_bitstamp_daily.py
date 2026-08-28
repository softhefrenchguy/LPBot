from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily OHLCV candles from Bitstamp public API.")
    p.add_argument("--pair", default="btcusd", help="Bitstamp pair, e.g. btcusd.")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default="2015-03-01")
    p.add_argument("--out", default="data/BTCUSD_bitstamp_daily.csv")
    return p.parse_args()


def _to_unix(date_str: str) -> int:
    ts = pd.to_datetime(date_str, utc=True)
    return int(ts.timestamp())


def download_daily(pair: str, start: str, end: str) -> pd.DataFrame:
    params = urlencode(
        {
            "step": 86400,
            "limit": 1000,
            "start": _to_unix(start),
            "end": _to_unix(end),
            "exclude_current_candle": "true",
        }
    )
    url = f"{BITSTAMP_OHLC_URL.format(pair=pair.lower())}?{params}"
    req = Request(url, headers={"User-Agent": "LPBot research downloader"})
    raw = urlopen(req, timeout=30).read().decode("utf-8")
    payload = json.loads(raw)
    rows = payload.get("data", {}).get("ohlc", [])
    if not rows:
        raise RuntimeError(f"No Bitstamp OHLC rows returned for {pair} {start}..{end}")

    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(pd.to_numeric(out["timestamp"], errors="coerce"), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out[["timestamp", "open", "high", "low", "close", "volume"]]


def main() -> int:
    args = _parse_args()
    out = download_daily(args.pair, args.start, args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"rows={len(out)} first={out['timestamp'].min().date()} last={out['timestamp'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
