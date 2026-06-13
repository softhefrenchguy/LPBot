from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import parse, request

import pandas as pd


def _fetch_klines(symbol: str, interval: str, limit: int, timeout_sec: float) -> pd.DataFrame:
    qs = parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
        }
    )
    req = request.Request(
        f"https://api.binance.com/api/v3/klines?{qs}",
        headers={"User-Agent": "LPBot-RecentKlines/1.0"},
    )
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("unexpected Binance response")
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
    p.add_argument("--timeout-sec", type=float, default=20.0)
    args = p.parse_args()

    out = Path(args.out)
    existing = _load_existing(out)
    recent = _fetch_klines(str(args.symbol), str(args.interval), int(args.limit), float(args.timeout_sec))
    merged = pd.concat([existing, recent], ignore_index=True)
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"])
    merged["timestamp"] = merged["timestamp"].astype("int64")
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    merged = merged[["timestamp", "open", "high", "low", "close", "volume", "is_gap"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    first = pd.to_datetime(merged["timestamp"].iloc[0], unit="ms", utc=True)
    last = pd.to_datetime(merged["timestamp"].iloc[-1], unit="ms", utc=True)
    print(f"wrote {out} rows={len(merged)} first={first} last={last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
