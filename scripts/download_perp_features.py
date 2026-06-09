from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


SPOT_BASE = "https://api.binance.com"
FUT_BASE = "https://fapi.binance.com"


@dataclass
class KlineRow:
    timestamp: pd.Timestamp
    close: float


def _binance_interval_to_pandas_freq(interval: str) -> str:
    iv = interval.strip()
    if iv.endswith("m"):
        return f"{int(iv[:-1])}min"
    if iv.endswith("h"):
        return f"{int(iv[:-1])}h"
    if iv.endswith("d"):
        return f"{int(iv[:-1])}d"
    if iv.endswith("w"):
        return f"{int(iv[:-1])}w"
    return iv


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def _safe_get(url: str, params: Dict, timeout: int = 20, max_retries: int = 8, sleep_base: float = 0.5):
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (418, 429):
                time.sleep(max(sleep_base, min(8.0, sleep_base * (2 ** i))))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(max(sleep_base, min(8.0, sleep_base * (2 ** i))))
    if last_err is not None:
        raise last_err
    raise RuntimeError("request failed with no exception")


def fetch_klines(base_url: str, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, market: str) -> pd.DataFrame:
    if market == "spot":
        url = f"{base_url}/api/v3/klines"
    elif market == "futures":
        url = f"{base_url}/fapi/v1/klines"
    else:
        raise ValueError(f"unknown market={market}")
    out: List[KlineRow] = []
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    interval_td = pd.Timedelta(_binance_interval_to_pandas_freq(interval))
    interval_ms = int(interval_td.total_seconds() * 1000)

    cur = start_ms
    while cur < end_ms:
        data = _safe_get(
            url,
            {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
        )
        if not data:
            break
        for row in data:
            ts = pd.to_datetime(int(row[0]), unit="ms", utc=True)
            out.append(KlineRow(ts, float(row[4])))
        last_open_ms = int(data[-1][0])
        nxt = last_open_ms + interval_ms
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.02)

    df = pd.DataFrame({"timestamp": [x.timestamp for x in out], "close": [x.close for x in out]})
    if len(df) == 0:
        return df
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = f"{FUT_BASE}/fapi/v1/fundingRate"
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    cur = start_ms
    rows = []
    while cur < end_ms:
        data = _safe_get(url, {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        last_ms = int(data[-1]["fundingTime"])
        nxt = last_ms + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "funding_rate"])
    return df[["timestamp", "funding_rate"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_oi_hist(symbol: str, period: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = f"{FUT_BASE}/futures/data/openInterestHist"
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    cur_end = end_ms
    rows = []
    while cur_end > start_ms:
        data = _safe_get(
            url,
            {"symbol": symbol, "period": period, "endTime": cur_end, "limit": 500},
        )
        if not data:
            break
        rows.extend(data)
        first_ms = int(data[0]["timestamp"])
        nxt_end = first_ms - 1
        if nxt_end >= cur_end:
            break
        cur_end = nxt_end
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "oi", "oi_value"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    df["oi"] = pd.to_numeric(df.get("sumOpenInterest"), errors="coerce")
    df["oi_value"] = pd.to_numeric(df.get("sumOpenInterestValue"), errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[["timestamp", "oi", "oi_value"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[df["timestamp"] >= start].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Download Binance spot/perp positioning features")
    p.add_argument("--spot-symbol", default="ETHUSDC")
    p.add_argument("--perp-symbol", default="ETHUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", default="data/backtest/ETH_perp_features_5m.csv")
    p.add_argument("--cache-dir", default="data/backtest/.cache_perp_features", help="Cache raw downloaded legs so reruns resume quickly.")
    args = p.parse_args()

    end = pd.Timestamp.now(tz="UTC").floor(_binance_interval_to_pandas_freq(args.interval))
    start = end - pd.Timedelta(days=args.days)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{args.spot_symbol}_{args.perp_symbol}_{args.interval}_{args.days}d"
    spot_cache = cache_dir / f"{cache_key}_spot.csv"
    perp_cache = cache_dir / f"{cache_key}_perp.csv"
    funding_cache = cache_dir / f"{cache_key}_funding.csv"
    oi_cache = cache_dir / f"{cache_key}_oi.csv"

    if spot_cache.exists():
        spot = pd.read_csv(spot_cache)
        spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True, errors="coerce")
        spot = spot.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Loaded cached spot rows={len(spot)} from {spot_cache}")
    else:
        print(f"Downloading spot klines: {args.spot_symbol} {args.interval} {start} -> {end}")
        spot = fetch_klines(SPOT_BASE, args.spot_symbol, args.interval, start, end, market="spot").rename(columns={"close": "spot_close"})
        spot.to_csv(spot_cache, index=False)
        print(f"spot rows={len(spot)} (cached: {spot_cache})")

    if perp_cache.exists():
        perp = pd.read_csv(perp_cache)
        perp["timestamp"] = pd.to_datetime(perp["timestamp"], utc=True, errors="coerce")
        perp = perp.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Loaded cached perp rows={len(perp)} from {perp_cache}")
    else:
        print(f"Downloading perp klines: {args.perp_symbol} {args.interval} {start} -> {end}")
        perp = fetch_klines(FUT_BASE, args.perp_symbol, args.interval, start, end, market="futures").rename(columns={"close": "perp_close"})
        perp.to_csv(perp_cache, index=False)
        print(f"perp rows={len(perp)} (cached: {perp_cache})")

    if funding_cache.exists():
        funding = pd.read_csv(funding_cache)
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True, errors="coerce")
        funding = funding.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Loaded cached funding rows={len(funding)} from {funding_cache}")
    else:
        print(f"Downloading funding: {args.perp_symbol}")
        funding = fetch_funding(args.perp_symbol, start, end)
        funding.to_csv(funding_cache, index=False)
        print(f"funding rows={len(funding)} (cached: {funding_cache})")

    if oi_cache.exists():
        oi = pd.read_csv(oi_cache)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True, errors="coerce")
        oi = oi.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Loaded cached oi rows={len(oi)} from {oi_cache}")
    else:
        print(f"Downloading OI history: {args.perp_symbol}")
        oi = fetch_oi_hist(args.perp_symbol, args.interval, start, end)
        oi.to_csv(oi_cache, index=False)
        print(f"oi rows={len(oi)} (cached: {oi_cache})")

    if len(spot) == 0 or len(perp) == 0:
        raise RuntimeError("No spot/perp kline rows downloaded")

    base = pd.merge(spot, perp, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    base["basis"] = base["perp_close"] / base["spot_close"] - 1.0

    if len(funding):
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            funding.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        base["funding_rate"] = base["funding_rate"].ffill().fillna(0.0)
    else:
        base["funding_rate"] = 0.0

    if len(oi):
        oi2 = oi.copy()
        if oi2["oi_value"].notna().sum() > 0:
            oi2["oi_metric"] = oi2["oi_value"]
        else:
            oi2["oi_metric"] = oi2["oi"]
        oi2 = oi2.rename(columns={"timestamp": "oi_timestamp"})

        base = pd.merge_asof(
            base.sort_values("timestamp"),
            oi2[["oi_timestamp", "oi_metric"]].sort_values("oi_timestamp"),
            left_on="timestamp",
            right_on="oi_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        base["oi_metric"] = base["oi_metric"].ffill()
    else:
        base["oi_metric"] = np.nan
        base["oi_timestamp"] = pd.NaT

    # Feature-ready fields
    base["ret_5m"] = np.log(base["spot_close"] / base["spot_close"].shift(1)).fillna(0.0)
    base["ret_1h"] = np.log(base["spot_close"] / base["spot_close"].shift(12)).fillna(0.0)
    base["oi_chg"] = np.log(base["oi_metric"] / base["oi_metric"].shift(1))
    base["oi_chg"] = base["oi_chg"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # OI freshness and event-based OI changes (more robust than ffill-series diffs).
    if "oi_timestamp" in base.columns:
        base["oi_age_sec"] = (
            (pd.to_datetime(base["timestamp"], utc=True, errors="coerce") - pd.to_datetime(base["oi_timestamp"], utc=True, errors="coerce"))
            .dt.total_seconds()
            .clip(lower=0)
        )
        base["oi_age_bars"] = (base["oi_age_sec"] / max(1, int(pd.Timedelta(_binance_interval_to_pandas_freq(args.interval)).total_seconds()))).fillna(0.0)
        oi_event = base["oi_timestamp"].ne(base["oi_timestamp"].shift(1))
        prev_event_oi = base["oi_metric"].where(oi_event).ffill().shift(1)
        base["oi_chg_event"] = np.log(base["oi_metric"] / prev_event_oi)
        base["oi_chg_event"] = base["oi_chg_event"].replace([np.inf, -np.inf], np.nan)
        base["oi_is_fresh"] = (base["oi_age_bars"] <= 1).astype(int)
    else:
        base["oi_age_sec"] = np.nan
        base["oi_age_bars"] = np.nan
        base["oi_chg_event"] = np.nan
        base["oi_is_fresh"] = 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out, index=False)
    print(f"wrote {out} rows={len(base)} first={base['timestamp'].iloc[0]} last={base['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
