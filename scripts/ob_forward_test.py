from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import requests

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"


def _now_ts() -> float:
    return time.time()


def _fetch_depth(symbol: str, limit: int) -> dict:
    resp = requests.get(
        BINANCE_DEPTH_URL,
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _book_stats(depth: dict) -> tuple[float, float, float, float, float]:
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if not bids or not asks:
        raise ValueError("Empty order book snapshot.")

    bid0 = float(bids[0][0])
    ask0 = float(asks[0][0])
    mid = (bid0 + ask0) / 2.0
    spread = ask0 - bid0
    spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0

    bid_qty = sum(float(q) for _, q in bids)
    ask_qty = sum(float(q) for _, q in asks)
    denom = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / denom if denom > 0 else 0.0

    return mid, spread, spread_bps, bid_qty, ask_qty, imbalance


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Forward-test order book imbalance signal.")
    p.add_argument("--symbol", default="ETHUSDC")
    p.add_argument("--limit", type=int, default=50, help="Depth levels per side.")
    p.add_argument("--interval-sec", type=int, default=10, help="Sampling interval.")
    p.add_argument("--horizon-sec", type=int, default=300, help="Return horizon.")
    p.add_argument("--out", default="artifacts/ob/ob_forward.csv")
    p.add_argument("--print-every", type=int, default=50, help="Print stats every N rows.")
    args = p.parse_args()

    out_path = Path(args.out)
    _ensure_parent(out_path)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    f = out_path.open("a", newline="")
    writer = csv.writer(f)

    if write_header:
        writer.writerow(
            [
                "ts",
                "symbol",
                "mid",
                "spread",
                "spread_bps",
                "bid_qty",
                "ask_qty",
                "imbalance",
                "horizon_sec",
                "mid_fwd",
                "ret_fwd",
            ]
        )
        f.flush()

    samples: deque[tuple[float, float, float, float, float, float]] = deque()
    n_written = 0
    last_print = 0

    while True:
        t0 = _now_ts()
        try:
            depth = _fetch_depth(args.symbol, args.limit)
            mid, spread, spread_bps, bid_qty, ask_qty, imb = _book_stats(depth)
            samples.append((t0, mid, spread, spread_bps, bid_qty, ask_qty, imb))

            # Emit rows whose horizon has passed
            while samples and t0 - samples[0][0] >= args.horizon_sec:
                ts, mid0, spr0, spr_bps0, bidq0, askq0, imb0 = samples.popleft()
                ret = math.log(mid / mid0) if mid0 > 0 else 0.0
                writer.writerow(
                    [
                        int(ts),
                        args.symbol,
                        mid0,
                        spr0,
                        spr_bps0,
                        bidq0,
                        askq0,
                        imb0,
                        args.horizon_sec,
                        mid,
                        ret,
                    ]
                )
                n_written += 1

            if n_written - last_print >= args.print_every:
                last_print = n_written
                print(f"rows_written={n_written}", flush=True)

        except Exception as exc:
            print(f"error: {exc}", flush=True)

        sleep_for = max(0.0, args.interval_sec - (_now_ts() - t0))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
