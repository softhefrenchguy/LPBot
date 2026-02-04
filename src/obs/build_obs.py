#!/usr/bin/env python3
"""Build HMM observation datasets from 1h OHLCV CSVs.

Rows with close <= 0 are dropped (log return invalid).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")


def _timestamp_sort_key(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().any():
        return dt
    return s.astype(str)


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ts_col = _timestamp_column(df)
    df = df.copy()
    df["_ts_key"] = _timestamp_sort_key(df[ts_col])
    df = df.dropna(subset=["_ts_key"])

    df = df.sort_values("_ts_key")
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.reset_index(drop=True)

    df = df.rename(columns={ts_col: "timestamp"})

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]

    df = df.drop(columns=["_ts_key"])
    df = df.reset_index(drop=True)
    df["_pos"] = np.arange(len(df), dtype=int)
    return df, len(df)


def _build_observations(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    log_close = np.log(df["close"])
    df["r"] = log_close.diff()
    df["abs_r"] = df["r"].abs()
    df["vol20"] = df["r"].rolling(window=20, min_periods=20).std()

    vol_mean = df["volume"].rolling(window=168, min_periods=168).mean()
    vol_std = df["volume"].rolling(window=168, min_periods=168).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std
    df.loc[vol_std == 0, "vol_z"] = 0.0

    df = df.dropna(subset=["r", "abs_r", "vol20", "vol_z"])
    df = df[df["_pos"] >= 168]

    min_pos = int(df["_pos"].iloc[0]) if len(df) else -1
    out = df[["timestamp", "r", "abs_r", "vol20", "vol_z"]].copy()
    return out, min_pos


def _sanity_checks(df_out: pd.DataFrame, input_rows: int, min_pos: int) -> None:
    expected_cols = ["timestamp", "r", "abs_r", "vol20", "vol_z"]
    if list(df_out.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {list(df_out.columns)}")

    if df_out["r"].isna().any():
        raise ValueError("r contains NaNs in output.")
    if (df_out["abs_r"] < 0).any():
        raise ValueError("abs_r contains negative values.")
    if (df_out["vol20"] < 0).any():
        raise ValueError("vol20 contains negative values.")
    if df_out.duplicated(subset=["timestamp"]).any():
        raise ValueError("Output has duplicated timestamps.")

    if input_rows >= 169 and min_pos >= 0 and min_pos < 168:
        raise ValueError("Output starts before enough history (>=168 bars + return).")


def _load_symbols_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Symbols file not found: {path.resolve()}")
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)
    return symbols


def _print_symbol_summary(requested: List[str], found: List[str], missing: List[str]) -> None:
    print(f"Symbols requested: {', '.join(requested) if requested else '(none)'}")
    print(f"Symbols found: {', '.join(found) if found else '(none)'}")
    if missing:
        print(f"Symbols skipped (missing files): {', '.join(missing)}")


def process_symbol_csv(path_in: Path, out_dir: Path, timeframe: str) -> None:
    df_raw = pd.read_csv(path_in)
    input_rows = len(df_raw)

    df, cleaned_rows = _prepare_dataframe(df_raw)
    df_out, min_pos = _build_observations(df)

    _sanity_checks(df_out, cleaned_rows, min_pos)

    suffix = f"_{timeframe}.csv"
    name = path_in.name
    symbol = name[: -len(suffix)] if name.endswith(suffix) else name.replace(".csv", "")
    out_path = out_dir / f"{symbol}_obs_{timeframe}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    dropped = input_rows - len(df_out)
    dropped_pct = (dropped / input_rows * 100) if input_rows else 0.0

    first_ts = df_out["timestamp"].iloc[0] if len(df_out) else None
    last_ts = df_out["timestamp"].iloc[-1] if len(df_out) else None

    print(
        f"[{symbol}] input={input_rows:,} output={len(df_out):,} "
        f"dropped={dropped_pct:.2f}% first={first_ts} last={last_ts}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing {SYMBOL}_1h.csv files (default: data/)",
    )
    p.add_argument(
        "--out-dir",
        default="data/obs",
        help="Output directory for observation CSVs (default: data/obs)",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help=(
            "Symbols to process (e.g. ETHUSDC BTCUSDC). If omitted, "
            "process all *_1h.csv in data-dir."
        ),
    )
    p.add_argument(
        "--symbols-file",
        default=None,
        help="Path to a newline-delimited symbols file (e.g. config/symbols.txt).",
    )
    p.add_argument(
        "--timeframe",
        default="1h",
        help="Timeframe suffix to read (default: 1h)",
    )

    args = p.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir.resolve()}")

    print(f"Scanning input directory: {data_dir.resolve()}")

    if args.symbols_file or args.symbols:
        requested = args.symbols or _load_symbols_file(Path(args.symbols_file))
        candidates: List[Path] = [data_dir / f"{s}_{args.timeframe}.csv" for s in requested]

        obs_dir = out_dir.resolve()
        paths = []
        for path in candidates:
            name = path.name
            if "_obs_" in name:
                continue
            try:
                if obs_dir in path.resolve().parents:
                    continue
            except OSError:
                continue
            if path.exists():
                paths.append(path)

        missing = [p.name.replace(f"_{args.timeframe}.csv", "") for p in candidates if not p.exists()]
        _print_symbol_summary(
            requested,
            [p.name.replace(f"_{args.timeframe}.csv", "") for p in paths],
            missing,
        )
    else:
        candidates = sorted(data_dir.glob(f"*_{args.timeframe}.csv"))
        print(f"Found {len(candidates)} candidate files for timeframe {args.timeframe}.")

        obs_dir = out_dir.resolve()
        paths = []
        for path in candidates:
            name = path.name
            if "_obs_" in name:
                continue
            try:
                if obs_dir in path.resolve().parents:
                    continue
            except OSError:
                continue
            paths.append(path)
        print(f"Accepted {len(paths)} input files after filtering.")

    if not paths:
        raise SystemExit(f"No *_{args.timeframe}.csv files found in {data_dir.resolve()}")

    for path_in in paths:
        if not path_in.exists():
            print(f"Skipping missing: {path_in}")
            continue
        process_symbol_csv(path_in, out_dir, args.timeframe)


if __name__ == "__main__":
    main()
