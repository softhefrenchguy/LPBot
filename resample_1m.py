#!/usr/bin/env python3
"""
Resample Binance 1-minute OHLCV CSVs into higher timeframes.

Input:  data/{SYMBOL}_1m.csv
Output: data/{SYMBOL}_{TF}.csv  where TF in {5m,15m,1h}

Assumptions about input columns (common Binance klines export):
- timestamp in milliseconds OR ISO datetime. We try to detect.
- Must include: open, high, low, close, volume
Optional: quote_volume, trades, taker_base_volume, taker_quote_volume

This script:
- parses timestamps
- sorts + dedupes
- resamples with OHLC rules and volume sums
- drops empty bars
- reports missing-bar gaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_TFS = ["5m", "15m", "1h"]


def _normalize_tf(tf: str) -> str:
    """
    Normalize short timeframes like '5m' into pandas-compatible offsets.
    """
    tf = tf.strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{tf[:-1]}min"
    return tf


def _infer_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a pandas datetime series in UTC from common timestamp formats.
    """
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            s = df[col]
            break
    else:
        raise ValueError(
            f"Could not find a timestamp column. Columns: {list(df.columns)}"
        )

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.dropna().empty:
            raise ValueError("Timestamp column is numeric but couldn't parse numbers.")

        median = int(s_num.dropna().median())
        unit = "ms" if median > 10_000_000_000 else "s"
        dt = pd.to_datetime(s_num, unit=unit, utc=True)
        return dt

    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if dt.isna().mean() > 0.5:
        raise ValueError("Timestamp column exists but datetime parsing failed badly.")
    return dt


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical OHLCV column names exist. Accept common variants.
    """
    col_map = {}
    variants = {
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["volume", "vol"],
        "quote_volume": ["quote_volume", "quote asset volume", "quoteAssetVolume"],
        "trades": ["trades", "number_of_trades", "num_trades"],
        "taker_base_volume": [
            "taker_base_volume",
            "taker buy base asset volume",
            "takerBuyBaseAssetVolume",
        ],
        "taker_quote_volume": [
            "taker_quote_volume",
            "taker buy quote asset volume",
            "takerBuyQuoteAssetVolume",
        ],
    }

    lower_cols = {c.lower(): c for c in df.columns}

    for canonical, names in variants.items():
        for name in names:
            key = name.lower()
            if key in lower_cols:
                col_map[lower_cols[key]] = canonical
                break

    df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Columns: {list(df.columns)}"
        )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_base_volume",
        "taker_quote_volume",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample a 1m OHLCV dataframe (indexed by datetime) to tf.
    """
    rule = _normalize_tf(tf)

    agg: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    for opt in ["quote_volume", "taker_base_volume", "taker_quote_volume"]:
        if opt in df.columns:
            agg[opt] = "sum"
    if "trades" in df.columns:
        agg["trades"] = "sum"

    out = df.resample(rule, label="right", closed="right").agg(agg)

    out = out.dropna(subset=["open", "high", "low", "close"])

    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
    return out


def _missing_bar_stats(ts: pd.Series, tf: str) -> Tuple[int, int]:
    """
    Return (expected_bars, missing_gaps_count) based on time range.
    Counts gaps where delta > tf.
    """
    if ts.empty:
        return 0, 0
    ts_sorted = ts.sort_values()
    start, end = ts_sorted.iloc[0], ts_sorted.iloc[-1]

    offset = pd.tseries.frequencies.to_offset(_normalize_tf(tf))
    expected = int(((end - start) / offset) + 1)

    deltas = ts_sorted.diff().dropna()
    missing_gaps = int((deltas > offset).sum())
    return expected, missing_gaps


def process_symbol_csv(path_in: Path, tfs: List[str]) -> None:
    df = pd.read_csv(path_in)

    ts = _infer_timestamp_series(df)
    df["timestamp"] = ts

    df = _standardize_columns(df)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    df = df.set_index("timestamp")

    symbol = path_in.name.replace("_1m.csv", "")

    for tf in tfs:
        out = _resample(df, tf)

        exp, gaps = _missing_bar_stats(out["timestamp"], tf)
        rows = len(out)
        first_ts = out["timestamp"].iloc[0] if rows else None
        last_ts = out["timestamp"].iloc[-1] if rows else None

        path_out = path_in.with_name(f"{symbol}_{tf}.csv")
        out.to_csv(path_out, index=False)

        print(
            f"[{symbol} {tf}] wrote {path_out.name} | rows={rows:,} | "
            f"first={first_ts} | last={last_ts} | expected~={exp:,} | gap_count={gaps}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing {SYMBOL}_1m.csv files (default: data/)",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help=(
            "Symbols to process (e.g. ETHUSDC BTCUSDC). If omitted, "
            "process all *_1m.csv in data-dir."
        ),
    )
    p.add_argument(
        "--tfs",
        nargs="*",
        default=DEFAULT_TFS,
        help="Timeframes to write (default: 5m 15m 1h)",
    )

    args = p.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir.resolve()}")

    if args.symbols:
        paths = [data_dir / f"{s}_1m.csv" for s in args.symbols]
    else:
        paths = sorted(data_dir.glob("*_1m.csv"))

    if not paths:
        raise SystemExit(f"No *_1m.csv files found in {data_dir.resolve()}")

    for path_in in paths:
        if not path_in.exists():
            print(f"Skipping missing: {path_in}")
            continue
        process_symbol_csv(path_in, args.tfs)


if __name__ == "__main__":
    main()
