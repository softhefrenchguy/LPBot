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


def _fetch_chunk(pair: str, start_ts: int, end_ts: int) -> list[dict[str, object]]:
    params = urlencode(
        {
            "step": 86400,
            "limit": 1000,
            "start": start_ts,
            "end": end_ts,
            "exclude_current_candle": "true",
        }
    )
    url = f"{BITSTAMP_OHLC_URL.format(pair=pair.lower())}?{params}"
    req = Request(url, headers={"User-Agent": "LPBot research downloader"})
    raw = urlopen(req, timeout=30).read().decode("utf-8")
    payload = json.loads(raw)
    return payload.get("data", {}).get("ohlc", [])


def download_daily(pair: str, start: str, end: str) -> pd.DataFrame:
    start_ts = _to_unix(start)
    end_ts = _to_unix(end)
    # Bitstamp's OHLC endpoint returns at most `limit` (1000) rows, and when the
    # requested [start, end] window spans more than that, it silently returns the
    # MOST RECENT 1000 rows ending at `end` -- not the earliest 1000 from `start`.
    # A single non-paginated request (the previous behaviour) silently truncated the
    # front of any range longer than ~2.7 years with no error or warning. Page
    # backward from `end` until coverage reaches `start` or a request stops making
    # progress.
    frames: list[pd.DataFrame] = []
    cursor_end = end_ts
    prev_earliest: int | None = None
    while True:
        rows = _fetch_chunk(pair, start_ts, cursor_end)
        if not rows:
            break
        chunk = pd.DataFrame(rows)
        raw_ts = pd.to_numeric(chunk["timestamp"], errors="coerce").dropna()
        if raw_ts.empty:
            break
        earliest = int(raw_ts.min())
        frames.append(chunk)
        if earliest <= start_ts or (prev_earliest is not None and earliest >= prev_earliest):
            break
        prev_earliest = earliest
        cursor_end = earliest - 1

    if not frames:
        raise RuntimeError(f"No Bitstamp OHLC rows returned for {pair} {start}..{end}")

    out = pd.concat(frames, ignore_index=True)
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
