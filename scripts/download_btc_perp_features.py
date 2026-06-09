from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request
from urllib.parse import urlencode

import pandas as pd


def _fetch_funding(symbol: str, start_ms: int | None, end_ms: int | None, limit: int, timeout_sec: float) -> list[dict]:
    base = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": int(limit)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    url = f"{base}?{urlencode(params)}"
    req = request.Request(url, headers={"User-Agent": "LPBot-BTCFunding/1.0"})
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("unexpected response from Binance funding API")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Download BTC perpetual funding history from Binance.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default="data/btc_perp_features.csv")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--timeout-sec", type=float, default=20.0)
    args = p.parse_args()

    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - pd.Timedelta(days=int(args.days))
    cur_start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    rows: list[dict[str, object]] = []
    while cur_start_ms < end_ms:
        batch = _fetch_funding(args.symbol, cur_start_ms, end_ms, int(args.limit), float(args.timeout_sec))
        if not batch:
            break
        for r in batch:
            try:
                ts = pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True)
                fr = float(r["fundingRate"])
            except Exception:
                continue
            rows.append({"timestamp": ts, "funding_rate": fr})
        last_ms = int(batch[-1]["fundingTime"])
        nxt = last_ms + 1
        if nxt <= cur_start_ms:
            break
        cur_start_ms = nxt

    if not rows:
        raise RuntimeError("no funding rows downloaded")

    d = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} rows={len(d)} "
        f"start={pd.to_datetime(d['timestamp'].iloc[0]).date()} end={pd.to_datetime(d['timestamp'].iloc[-1]).date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
