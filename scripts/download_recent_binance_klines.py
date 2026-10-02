from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib import parse, request

import pandas as pd


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _request_klines(
    symbol: str,
    interval: str,
    limit: int,
    timeout_sec: float,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> list[list[object]]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
    }
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    qs = parse.urlencode(params)
    req = request.Request(
        f"https://api.binance.com/api/v3/klines?{qs}",
        headers={"User-Agent": "LPBot-RecentKlines/1.0"},
    )
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("unexpected Binance response")
    return payload


def _fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    timeout_sec: float,
    days: int | None = None,
) -> pd.DataFrame:
    payload: list[list[object]] = []
    if days is None or int(days) <= 0:
        payload = _request_klines(symbol, interval, min(int(limit), 1000), timeout_sec)
    else:
        if interval not in INTERVAL_MS:
            raise RuntimeError(f"unsupported interval for paged download: {interval}")
        end_ms = int(datetime.now(UTC).timestamp() * 1000)
        cursor = int((datetime.now(UTC) - timedelta(days=int(days))).timestamp() * 1000)
        step = INTERVAL_MS[interval]
        while cursor <= end_ms:
            chunk = _request_klines(
                symbol=symbol,
                interval=interval,
                limit=min(int(limit), 1000),
                timeout_sec=timeout_sec,
                start_time_ms=cursor,
                end_time_ms=end_ms,
            )
            if not chunk:
                break
            payload.extend(chunk)
            last_open = int(chunk[-1][0])
            next_cursor = last_open + step
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(chunk) < min(int(limit), 1000):
                break
            time.sleep(0.05)

    rows = []
    for k in payload:
        if not isinstance(k, list) or len(k) < 6:
            continue
        rows.append(
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "is_gap": 0,
            }
        )
    if not rows:
        raise RuntimeError("no klines downloaded")
    return pd.DataFrame(rows)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "is_gap"])
    d = pd.read_csv(path, low_memory=False)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "is_gap" not in d.columns:
        d["is_gap"] = 0
    d["is_gap"] = pd.to_numeric(d["is_gap"], errors="coerce").fillna(0).astype(int)
    return d.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])


def main() -> int:
    p = argparse.ArgumentParser(description="Merge recent Binance klines into an existing OHLCV CSV.")
    p.add_argument("--symbol", default="ETHUSDC")
    p.add_argument("--interval", default="1m")
    p.add_argument("--out", default="data/ETHUSDC_1m.csv")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--days", type=int, default=0, help="If >0, page through this many recent days.")
    p.add_argument("--timestamp-format", choices=["ms", "iso"], default="ms")
    p.add_argument("--count-1m-rows", type=int, default=0, help="Optional constant count_1m_rows column for resampled files.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite instead of merging with existing CSV.")
    p.add_argument("--timeout-sec", type=float, default=20.0)
    args = p.parse_args()

    out = Path(args.out)
    existing = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "is_gap"]) if args.overwrite else _load_existing(out)
    recent = _fetch_klines(
        str(args.symbol),
        str(args.interval),
        int(args.limit),
        float(args.timeout_sec),
        days=int(args.days) if int(args.days) > 0 else None,
    )
    merged = pd.concat([existing, recent], ignore_index=True)
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"])
    merged["timestamp"] = merged["timestamp"].astype("int64")
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if str(args.timestamp_format) == "iso":
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True).astype(str)
        merged = merged.drop(columns=["is_gap"], errors="ignore")
        if int(args.count_1m_rows) > 0:
            merged["count_1m_rows"] = int(args.count_1m_rows)
            merged = merged[["timestamp", "open", "high", "low", "close", "volume", "count_1m_rows"]]
        else:
            merged = merged[["timestamp", "open", "high", "low", "close", "volume"]]
    else:
        merged = merged[["timestamp", "open", "high", "low", "close", "volume", "is_gap"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: this file is read moments later by a separate process in the same cron
    # sequence (the defensive-model refresh, then the main checklist), each its own `docker-compose
    # exec` invocation. A direct to_csv() on a ~38MB file leaves a window where a reader can open a
    # partially-written file mid-write. Seen in production: the defensive refresh failed with
    # "No valid timestamps for walk-forward" and, in the same run, the main checklist reported the
    # price feed as 5371 hours stale (~224 days) immediately after this script logged a successful
    # fresh write -- both symptoms of reading a half-written file, not an actual feed outage. Writing
    # to a temp file in the same directory then atomically replacing the target closes that window:
    # any reader always sees either the complete old file or the complete new one, never a partial.
    tmp_out = out.with_name(f"{out.name}.tmp{os.getpid()}")
    merged.to_csv(tmp_out, index=False)
    os.replace(tmp_out, out)
    if str(args.timestamp_format) == "iso":
        first = pd.to_datetime(merged["timestamp"].iloc[0], utc=True)
        last = pd.to_datetime(merged["timestamp"].iloc[-1], utc=True)
    else:
        first = pd.to_datetime(merged["timestamp"].iloc[0], unit="ms", utc=True)
        last = pd.to_datetime(merged["timestamp"].iloc[-1], unit="ms", utc=True)
    print(f"wrote {out} rows={len(merged)} first={first} last={last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
