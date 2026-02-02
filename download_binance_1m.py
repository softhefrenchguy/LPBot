import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
LIMIT = 1000
SLEEP_MIN = 0.2
SLEEP_MAX = 0.5
MAX_RETRIES = 5
TIMEOUT_SEC = 20

SYMBOLS = ["ETHUSDC", "BTCUSDC", "SOLUSDC"]
START_UTC = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def utc_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def request_with_retries(session: requests.Session, params: dict) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                else:
                    time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            if resp.status_code >= 500:
                time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.5 * attempt + random.uniform(0, 0.5))

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {last_err}")


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    session = requests.Session()
    all_rows = []
    current_start = start_ms

    while current_start <= end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": LIMIT,
            "startTime": current_start,
            "endTime": end_ms,
        }
        resp = request_with_retries(session, params)
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        last_open = data[-1][0]
        if last_open >= end_ms:
            break

        next_start = last_open + 60_000
        if next_start <= current_start:
            raise RuntimeError("Pagination stalled; start time did not advance.")

        current_start = next_start
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    return all_rows


def klines_to_dataframe(rows: list) -> pd.DataFrame:
    trimmed = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]
    df = pd.DataFrame(trimmed, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna()
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with NaN values.")

    dupes = df.duplicated(subset="timestamp").sum()
    if dupes:
        print(f"Removed {dupes} duplicate timestamps.")
    df = df.drop_duplicates(subset="timestamp")

    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df


def validate_and_report(symbol: str, df: pd.DataFrame, start_ms: int, end_ms: int) -> None:
    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol}.")

    first_ts = int(df["timestamp"].iloc[0])
    last_ts = int(df["timestamp"].iloc[-1])
    total_rows = len(df)

    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > 60_000]
    missing_minutes = int(((gaps // 60_000) - 1).sum()) if not gaps.empty else 0

    expected_total = int((end_ms - start_ms) // 60_000) + 1
    years_approx = (end_ms - start_ms) / (365 * 24 * 60 * 60 * 1000)
    expected_per_year = 365 * 24 * 60

    print(f"\nSymbol: {symbol}")
    print(f"Total rows: {total_rows}")
    print(f"First timestamp: {first_ts} ({ms_to_utc_str(first_ts)})")
    print(f"Last timestamp: {last_ts} ({ms_to_utc_str(last_ts)})")
    print(f"Expected minutes (approx): {expected_total} (~{expected_per_year} per year x {years_approx:.2f} years)")
    print(f"Missing minute count (gaps > 1m): {missing_minutes}")


def main() -> None:
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_ms = utc_to_ms(START_UTC)
    end_ms = utc_to_ms(end_dt)

    print(f"Downloading 1m klines from {START_UTC.strftime('%Y-%m-%d %H:%M:%S UTC')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        print(f"\nFetching {symbol}...")
        rows = fetch_klines(symbol, start_ms, end_ms)
        print(f"Downloaded rows (raw): {len(rows)}")

        df = klines_to_dataframe(rows)
        validate_and_report(symbol, df, start_ms, end_ms)

        out_path = data_dir / f"{symbol}_1m.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()