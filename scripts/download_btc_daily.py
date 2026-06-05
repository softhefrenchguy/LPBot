from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib import request


def _fetch_klines(symbol: str, limit: int, timeout_sec: float) -> list[list[object]]:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={int(limit)}"
    req = request.Request(url, headers={"User-Agent": "LPBot-BTCFeed/1.0"})
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("unexpected response from Binance API")
    return payload


def _row_from_kline(k: list[object]) -> dict[str, str]:
    ts_ms = int(k[0])
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).date().isoformat()
    return {
        "Date": dt,
        "Open": str(k[1]),
        "High": str(k[2]),
        "Low": str(k[3]),
        "Close": str(k[4]),
        "Volume": str(k[5]),
    }


def _read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = str(row.get("Date", "")).strip()
            if not d:
                continue
            out[d] = {
                "Date": d,
                "Open": str(row.get("Open", "")),
                "High": str(row.get("High", "")),
                "Low": str(row.get("Low", "")),
                "Close": str(row.get("Close", "")),
                "Volume": str(row.get("Volume", "")),
            }
    return out


def _write_rows(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Date", "Open", "High", "Low", "Close", "Volume"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in sorted(rows.keys()):
            w.writerow(rows[d])


def main() -> int:
    p = argparse.ArgumentParser(description="Download BTC daily bars from Binance and write CSV.")
    p.add_argument("--symbol", default="BTCUSDC")
    p.add_argument("--out", default="data/btc_daily.csv")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--timeout-sec", type=float, default=15.0)
    args = p.parse_args()

    out_path = Path(args.out)
    rows = _read_existing(out_path)
    klines = _fetch_klines(symbol=str(args.symbol), limit=int(args.limit), timeout_sec=float(args.timeout_sec))
    for k in klines:
        try:
            row = _row_from_kline(k)
        except Exception:
            continue
        rows[row["Date"]] = row
    if not rows:
        raise RuntimeError("no rows downloaded")
    _write_rows(out_path, rows)
    all_dates = sorted(rows.keys())
    print(f"wrote {out_path} rows={len(rows)} start={all_dates[0]} end={all_dates[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
