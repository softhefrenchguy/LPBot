from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def _get_json(endpoint: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    url = f"{DERIBIT_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"Deribit API error for {endpoint}: {payload['error']}")
    return payload


def _parse_instrument_name(name: str) -> dict[str, Any]:
    # Example: ETH-30JUN26-2000-C
    parts = str(name).split("-")
    if len(parts) < 4:
        return {"parsed_expiry": pd.NaT, "parsed_strike": np.nan, "parsed_option_type": ""}
    expiry_raw = parts[1]
    strike_raw = parts[2]
    type_raw = parts[3].upper()
    expiry = pd.to_datetime(expiry_raw, format="%d%b%y", errors="coerce", utc=True)
    option_type = "call" if type_raw == "C" else "put" if type_raw == "P" else ""
    return {
        "parsed_expiry": expiry,
        "parsed_strike": pd.to_numeric(strike_raw, errors="coerce"),
        "parsed_option_type": option_type,
    }


def _load_instruments(currency: str) -> pd.DataFrame:
    payload = _get_json(
        "get_instruments",
        {"currency": currency, "kind": "option", "expired": "false"},
    )
    rows = payload.get("result", [])
    if not rows:
        raise RuntimeError(f"No active {currency} option instruments returned by Deribit.")
    df = pd.DataFrame(rows)
    keep = [c for c in ["instrument_name", "strike", "expiration_timestamp", "option_type"] if c in df.columns]
    df = df[keep].copy()
    df["expiry_date"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True, errors="coerce").dt.floor("D")
    df["option_type"] = df["option_type"].astype(str).str.lower()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df


def _load_book_summary(currency: str) -> pd.DataFrame:
    payload = _get_json(
        "get_book_summary_by_currency",
        {"currency": currency, "kind": "option"},
    )
    rows = payload.get("result", [])
    if not rows:
        raise RuntimeError(f"No {currency} option book summaries returned by Deribit.")
    df = pd.DataFrame(rows)
    cols = [
        "instrument_name",
        "mark_iv",
        "underlying_price",
        "open_interest",
        "bid_price",
        "ask_price",
        "volume",
    ]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    for col in ["mark_iv", "underlying_price", "open_interest", "bid_price", "ask_price", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _nearest_atm_by_target(df: pd.DataFrame, target_days: int) -> float:
    if df.empty:
        return float("nan")
    x = df.copy()
    x["target_dist"] = (x["days_to_expiry"] - float(target_days)).abs()
    x["atm_dist"] = (x["moneyness"] - 1.0).abs()
    x = x.sort_values(["target_dist", "atm_dist", "open_interest"], ascending=[True, True, False])
    if x.empty:
        return float("nan")
    return float(x.iloc[0]["mark_iv"])


def _term_structure_summary(df: pd.DataFrame, targets: list[int]) -> dict[int, float]:
    return {days: _nearest_atm_by_target(df, days) for days in targets}


def build_surface(currency: str = "ETH") -> pd.DataFrame:
    instruments = _load_instruments(currency)
    book = _load_book_summary(currency)
    df = book.merge(instruments, on="instrument_name", how="left", suffixes=("", "_instrument"))

    # Fallback parser for any missing instrument metadata.
    parsed = df["instrument_name"].map(_parse_instrument_name).apply(pd.Series)
    df["expiry_date"] = df["expiry_date"].fillna(parsed["parsed_expiry"].dt.floor("D"))
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(parsed["parsed_strike"])
    df["option_type"] = df["option_type"].replace("", np.nan).fillna(parsed["parsed_option_type"])

    now = pd.Timestamp.now(tz="UTC")
    df["days_to_expiry"] = (df["expiry_date"] - now.floor("D")).dt.total_seconds() / 86400.0
    df["moneyness"] = df["strike"] / df["underlying_price"]
    df["snapshot_ts"] = now.isoformat()

    filtered = df[
        (df["option_type"].astype(str).str.lower() == "call")
        & (pd.to_numeric(df["mark_iv"], errors="coerce") > 0)
        & (pd.to_numeric(df["open_interest"], errors="coerce") > 0)
        & (df["days_to_expiry"].between(1, 180, inclusive="both"))
        & (df["moneyness"].between(0.5, 2.0, inclusive="both"))
    ].copy()

    out_cols = [
        "snapshot_ts",
        "instrument_name",
        "strike",
        "expiry_date",
        "days_to_expiry",
        "option_type",
        "mark_iv",
        "underlying_price",
        "moneyness",
        "open_interest",
        "bid_price",
        "ask_price",
        "volume",
    ]
    for col in out_cols:
        if col not in filtered.columns:
            filtered[col] = np.nan
    return filtered[out_cols].sort_values(["expiry_date", "strike"]).reset_index(drop=True)


def print_summary(df: pd.DataFrame, raw_count: int | None = None) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    price = float(df["underlying_price"].dropna().median()) if not df.empty else float("nan")
    targets = [7, 14, 30, 60, 90]
    atm = _term_structure_summary(df, targets)
    valid_atm = [(k, v) for k, v in atm.items() if np.isfinite(v)]
    upward = "NA"
    if len(valid_atm) >= 2:
        upward = "YES" if valid_atm[-1][1] > valid_atm[0][1] else "NO"

    print("=" * 48)
    print("ETH OPTIONS VOL SURFACE SNAPSHOT")
    print(now_str)
    print(f"Underlying: ${price:,.2f}" if np.isfinite(price) else "Underlying: NA")
    print("=" * 48)
    print(f"Options loaded: {len(df):,}" + (f" filtered from {raw_count:,}" if raw_count is not None else ""))
    if df.empty:
        print("No liquid call options passed filters.")
        print("=" * 48)
        return
    expiries = df["expiry_date"].dropna().nunique()
    print(f"Expiries: {expiries} dates")
    print(f"Strike range: ${df['strike'].min():,.0f} - ${df['strike'].max():,.0f}")
    print("")
    print("ATM IV by expiry:")
    for days in targets:
        val = atm.get(days, float("nan"))
        print(f"{days}d: {val:5.1f}%" if np.isfinite(val) else f"{days}d:    NA")
    print("")
    print("Vol term structure:")
    print(f"Upward sloping: {upward}")
    print("=" * 48)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch ETH options implied volatility surface from Deribit.")
    ap.add_argument("--currency", default="ETH")
    ap.add_argument("--out-dir", default="artifacts/options")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    book_raw = _load_book_summary(str(args.currency).upper())
    surface = build_surface(str(args.currency).upper())
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    latest_path = out_dir / "vol_surface_latest.csv"
    dated_path = out_dir / f"vol_surface_{today}.csv"
    surface.to_csv(latest_path, index=False)
    surface.to_csv(dated_path, index=False)

    print_summary(surface, raw_count=len(book_raw))
    print(f"Saved: {latest_path}")
    print(f"Saved: {dated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
