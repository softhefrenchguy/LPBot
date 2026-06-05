from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


API_ROOT = "https://api.geckoterminal.com/api/v2"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GeckoTerminalClient/1.0"


def _parse_date_utc(val: str) -> pd.Timestamp:
    ts = pd.to_datetime(val, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _fetch_page(
    network: str,
    pool: str,
    timeframe: str,
    params: dict,
    max_retries: int,
    retry_wait: float,
) -> list:
    url = f"{API_ROOT}/networks/{network}/pools/{pool}/ohlcv/{timeframe}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    attempt = 0
    while True:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            payload = resp.json()
            try:
                return payload["data"]["attributes"]["ohlcv_list"]
            except Exception as exc:  # noqa: BLE001
                raise SystemExit("Unexpected API response shape.") from exc
        if resp.status_code == 429 and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = retry_wait
            else:
                wait = retry_wait
            time.sleep(max(wait, retry_wait))
            attempt += 1
            continue
        raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:200]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-volume", required=True)
    p.add_argument("--sleep-seconds", type=float, default=2.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-wait", type=float, default=30.0)
    args = p.parse_args()

    if args.bar_minutes != 5:
        raise SystemExit("Only --bar-minutes 5 is supported for now.")

    start_ts = _parse_date_utc(args.start)
    end_ts = _parse_date_utc(args.end)
    if end_ts <= start_ts:
        raise SystemExit("--end must be after --start.")

    timeframe = "minute"
    aggregate = args.bar_minutes

    before_ts = int(end_ts.timestamp())
    all_rows: list[tuple[int, float, float]] = []
    seen: set[int] = set()

    while True:
        params = {
            "aggregate": aggregate,
            "before_timestamp": before_ts,
            "limit": 1000,
            "currency": "usd",
            "token": "base",
            "include_empty_intervals": "true",
        }
        ohlcv_list = _fetch_page(
            args.network,
            args.pool,
            timeframe,
            params,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        if not ohlcv_list:
            break

        min_ts = None
        for row in ohlcv_list:
            if len(row) < 6:
                continue
            ts, _open, _high, _low, close, volume = row[:6]
            if ts in seen:
                continue
            seen.add(ts)
            ts_dt = pd.to_datetime(ts, unit="s", utc=True)
            if ts_dt < start_ts or ts_dt > end_ts:
                continue
            all_rows.append((ts, float(close), float(volume)))
            if min_ts is None or ts < min_ts:
                min_ts = ts

        if min_ts is None:
            break
        if pd.to_datetime(min_ts, unit="s", utc=True) < start_ts:
            break

        before_ts = int(min_ts - 1)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not all_rows:
        raise SystemExit("No OHLCV rows fetched for the requested range.")

    df = pd.DataFrame(all_rows, columns=["timestamp", "close", "volume_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    out_vol_path = Path(args.out_volume)
    out_vol_path.parent.mkdir(parents=True, exist_ok=True)
    df[["timestamp", "volume_usd"]].to_csv(out_vol_path, index=False)

    print(f"rows={len(df)}")
    print(f"out={out_path}")
    print(f"out_volume={out_vol_path}")


if __name__ == "__main__":
    main()
