from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

"""
Prepare no-look-ahead inputs for a later rules-based BEAR type classifier.

This script does not classify bear type and does not backtest any new logic. It
extracts each LPBot BEAR segment, aligns macro inputs exactly as they would have
been knowable on the segment start date, and appends hindsight scoring targets.

No-look-ahead input definitions:
- vix_at_start: latest available VIX close on or before bear_start only. It is
  not an average, max, or peak from inside the bear window.
- vix_trend_approx_20d_at_start: vix_at_start minus the latest VIX close on or
  before bear_start - 28 calendar days. This approximates 20 trading days.
- CPI: latest CPI observation whose assumed release date is <= bear_start. The
  release date is modelled conservatively as observation month-end + 14 days.
- Credit spread: latest FRED credit observation on or before bear_start.

Hindsight definitions:
- *_return_over_bear_period_hindsight: proxy asset close-to-close return from
  bear_start to bear_end.
- best_hindsight_asset_return_over_bear_period: asset with the highest of those
  hindsight returns. This is only for later rule scoring; it is not an input.
- underlying_peak_to_trough_pct: worst peak-to-trough drawdown of the LPBot
  underlying close series inside the BEAR segment. Used only for tagging
  major_bear vs minor_dip.
"""


DEFAULT_REGIME_PATHS = [
    Path("artifacts/backtest/extended_overlay_daily.csv"),
    Path("artifacts/backtest/forex_optimised/validated_crypto_daily.csv"),
]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL,AAA,BAA,BAMLH0A0HYM2"
YF_SYMBOLS = {
    "vix": "^VIX",
    "gold": "GLD",
    "energy": "USO",
    "usd": "UUP",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assemble no-look-ahead inputs for LPBot bear-market type classification. "
            "This only prepares labelled periods and macro/asset context; it does not "
            "fit rules or change trading logic."
        )
    )
    p.add_argument("--regime-csv", default=None, help="LPBot daily CSV with eth_regime/btc_regime columns.")
    p.add_argument("--regime-column", default="eth_regime", help="Regime column to segment, default eth_regime.")
    p.add_argument("--min-bear-days", type=int, default=3, help="Drop very short BEAR runs below this length.")
    p.add_argument("--major-min-days", type=int, default=30, help="Major bear requires at least this many days.")
    p.add_argument("--major-min-drawdown", type=float, default=0.10, help="Major bear requires at least this peak-to-trough drawdown.")
    p.add_argument("--start", default=None, help="Optional start date filter.")
    p.add_argument("--end", default=None, help="Optional end date filter.")
    p.add_argument("--refresh", action="store_true", help="Refresh cached yfinance/FRED downloads.")
    p.add_argument("--cache-dir", default="artifacts/bear_classifier/cache")
    p.add_argument("--out", default="artifacts/bear_classifier/bear_period_classifier_inputs.csv")
    p.add_argument("--dataset-tag", default="eth_btc_portfolio", help="Label for this regime source, e.g. btc_only_pre_eth.")
    p.add_argument("--source-note", default="", help="Free-form note carried into the output for provenance.")
    return p.parse_args()


def _pick_regime_csv(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"Regime CSV not found: {path}")
        return path
    for path in DEFAULT_REGIME_PATHS:
        if path.exists():
            return path
    raise SystemExit(
        "No default regime CSV found. Pass --regime-csv pointing at a daily LPBot backtest/history file."
    )


def _normalise_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"date", "datetime", "index"}), None)
    close_col = next((c for c in raw.columns if c in {"close", "adj_close"}), None)
    if date_col is None or close_col is None:
        raise ValueError(f"Could not parse yfinance frame columns: {list(raw.columns)}")
    out = raw.rename(columns={date_col: "date", close_col: "close"})[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.floor("D")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")


def _download_yfinance(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache: Path, refresh: bool) -> pd.DataFrame:
    if cache.exists() and not refresh:
        d = pd.read_csv(cache)
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.floor("D")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        return d.dropna(subset=["date", "close"])

    import yfinance as yf

    raw = yf.download(
        symbol,
        start=(start - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No yfinance data returned for {symbol}")
    out = _normalise_price_frame(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _download_fred(cache: Path, refresh: bool) -> pd.DataFrame:
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    raw = urlopen(FRED_URL, timeout=30).read()
    if raw.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = next((name for name in zf.namelist() if name.lower().endswith(".csv")), None)
            if csv_name is None:
                raise RuntimeError(f"FRED zip did not contain a CSV. Files: {zf.namelist()}")
            with zf.open(csv_name) as fh:
                fred = pd.read_csv(fh)
    else:
        fred = pd.read_csv(io.BytesIO(raw))
    cache.parent.mkdir(parents=True, exist_ok=True)
    fred.to_csv(cache, index=False)
    return fred


def _load_regimes(path: Path, regime_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    date_col = next((c for c in ["day", "timestamp", "date", "datetime"] if c in d.columns), None)
    if date_col is None:
        raise SystemExit(f"{path} has no day/timestamp/date column.")
    if regime_col not in d.columns:
        raise SystemExit(f"{path} has no regime column '{regime_col}'. Columns include: {list(d.columns)[:30]}")
    d["date"] = pd.to_datetime(d[date_col], utc=True, errors="coerce").dt.floor("D")
    d[regime_col] = d[regime_col].astype(str).str.upper()
    d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if start:
        d = d[d["date"] >= pd.to_datetime(start, utc=True)]
    if end:
        d = d[d["date"] <= pd.to_datetime(end, utc=True)]
    return d.reset_index(drop=True)


def _bear_periods(d: pd.DataFrame, regime_col: str, min_days: int) -> pd.DataFrame:
    is_bear = d[regime_col].eq("BEAR")
    groups = is_bear.ne(is_bear.shift(fill_value=False)).cumsum()
    rows = []
    for _, g in d[is_bear].groupby(groups[is_bear]):
        start = g["date"].iloc[0]
        end = g["date"].iloc[-1]
        days = int((end - start).days + 1)
        if days < min_days:
            continue
        rows.append({"bear_start": start, "bear_end": end, "bear_days": days})
    return pd.DataFrame(rows)


def _underlying_close_column(regime_col: str, d: pd.DataFrame) -> str | None:
    prefix = regime_col.removesuffix("_regime")
    preferred = f"{prefix}_close"
    if preferred in d.columns:
        return preferred
    for candidate in ["eth_close", "btc_close", "close"]:
        if candidate in d.columns:
            return candidate
    return None


def _underlying_magnitude(
    regimes: pd.DataFrame,
    close_col: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float | str]:
    if close_col is None:
        return {
            "underlying_close_column": "",
            "underlying_start_to_trough_pct": np.nan,
            "underlying_peak_to_trough_pct": np.nan,
        }
    window = regimes[(regimes["date"] >= start) & (regimes["date"] <= end)].copy()
    px = pd.to_numeric(window[close_col], errors="coerce").dropna()
    if px.empty:
        return {
            "underlying_close_column": close_col,
            "underlying_start_to_trough_pct": np.nan,
            "underlying_peak_to_trough_pct": np.nan,
        }

    start_px = float(px.iloc[0])
    trough_px = float(px.min())
    start_to_trough = trough_px / start_px - 1.0 if start_px > 0 else np.nan

    running_peak = px.cummax()
    dd = px / running_peak - 1.0
    peak_to_trough = float(dd.min()) if not dd.empty else np.nan
    return {
        "underlying_close_column": close_col,
        "underlying_start_to_trough_pct": float(start_to_trough),
        "underlying_peak_to_trough_pct": peak_to_trough,
    }


def _asof_value(d: pd.DataFrame, date: pd.Timestamp, col: str = "close") -> tuple[float, pd.Timestamp | pd.NaT]:
    x = d[d["date"] <= date]
    if x.empty:
        return np.nan, pd.NaT
    row = x.iloc[-1]
    return float(row[col]), row["date"]


def _price_return(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s, _ = _asof_value(d, start)
    e, _ = _asof_value(d, end)
    if not np.isfinite(s) or not np.isfinite(e) or s <= 0:
        return np.nan
    return float(e / s - 1.0)


def _prepare_fred(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fred = raw.copy()
    date_col = next((c for c in fred.columns if c.upper() in {"DATE", "OBSERVATION_DATE"} or c.lower() == "date"), fred.columns[0])
    fred = fred.rename(columns={date_col: "date"})
    fred["date"] = pd.to_datetime(fred["date"], utc=True, errors="coerce").dt.floor("D")
    for col in ["CPIAUCSL", "AAA", "BAA", "BAMLH0A0HYM2"]:
        if col in fred.columns:
            fred[col] = pd.to_numeric(fred[col], errors="coerce")
    fred = fred.dropna(subset=["date"]).sort_values("date")

    cpi = fred[["date", "CPIAUCSL"]].dropna().copy()
    cpi["release_date"] = (cpi["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=14)).dt.tz_convert("UTC")
    cpi["cpi_yoy"] = cpi["CPIAUCSL"] / cpi["CPIAUCSL"].shift(12) - 1.0
    cpi["cpi_yoy_trend_3m"] = cpi["cpi_yoy"] - cpi["cpi_yoy"].shift(3)

    credit = fred[["date"]].copy()
    if {"BAA", "AAA"}.issubset(fred.columns):
        credit["baa_aaa_spread"] = fred["BAA"] - fred["AAA"]
    else:
        credit["baa_aaa_spread"] = np.nan
    if "BAMLH0A0HYM2" in fred.columns:
        credit["hy_oas"] = fred["BAMLH0A0HYM2"]
    else:
        credit["hy_oas"] = np.nan
    credit["credit_spread"] = credit["hy_oas"].combine_first(credit["baa_aaa_spread"])
    credit = credit.dropna(subset=["credit_spread"]).copy()
    credit["credit_spread_trend_20obs"] = credit["credit_spread"] - credit["credit_spread"].shift(20)
    return cpi, credit


def _cpi_at_start(cpi: pd.DataFrame, start: pd.Timestamp) -> dict[str, object]:
    known = cpi[cpi["release_date"] <= start]
    if known.empty:
        return {
            "cpi_obs_date_known_at_start": "",
            "cpi_release_date_used": "",
            "cpi_yoy_known_at_start": np.nan,
            "cpi_yoy_trend_3m_known_at_start": np.nan,
            "cpi_no_lookahead": False,
        }
    row = known.iloc[-1]
    return {
        "cpi_obs_date_known_at_start": row["date"].date().isoformat(),
        "cpi_release_date_used": row["release_date"].date().isoformat(),
        "cpi_yoy_known_at_start": float(row["cpi_yoy"]),
        "cpi_yoy_trend_3m_known_at_start": float(row["cpi_yoy_trend_3m"]),
        "cpi_no_lookahead": bool(row["release_date"] <= start),
    }


def _credit_at_start(credit: pd.DataFrame, start: pd.Timestamp) -> dict[str, object]:
    known = credit[credit["date"] <= start]
    if known.empty:
        return {
            "credit_obs_date_known_at_start": "",
            "credit_spread_known_at_start": np.nan,
            "credit_spread_trend_20obs_known_at_start": np.nan,
            "credit_no_lookahead": False,
        }
    row = known.iloc[-1]
    return {
        "credit_obs_date_known_at_start": row["date"].date().isoformat(),
        "credit_spread_known_at_start": float(row["credit_spread"]),
        "credit_spread_trend_20obs_known_at_start": float(row["credit_spread_trend_20obs"]),
        "credit_no_lookahead": bool(row["date"] <= start),
    }


def _best_defensive_asset(returns: dict[str, float]) -> str:
    valid = {k: v for k, v in returns.items() if np.isfinite(v)}
    if not valid:
        return ""
    return max(valid, key=valid.get)


def main() -> int:
    args = _parse_args()
    regime_path = _pick_regime_csv(args.regime_csv)
    regimes = _load_regimes(regime_path, args.regime_column, args.start, args.end)
    periods = _bear_periods(regimes, args.regime_column, args.min_bear_days)
    if periods.empty:
        raise SystemExit(f"No BEAR periods found in {regime_path} using column {args.regime_column}.")

    min_date = periods["bear_start"].min()
    max_date = periods["bear_end"].max()
    cache_dir = Path(args.cache_dir)
    yf_data = {
        name: _download_yfinance(symbol, min_date, max_date, cache_dir / f"{name}_{symbol.replace('^', '')}.csv", args.refresh)
        for name, symbol in YF_SYMBOLS.items()
    }
    cpi, credit = _prepare_fred(_download_fred(cache_dir / "fred_cpi_credit.csv", args.refresh))
    close_col = _underlying_close_column(args.regime_column, regimes)

    rows = []
    for _, period in periods.iterrows():
        start = period["bear_start"]
        end = period["bear_end"]
        duration_days = int(period["bear_days"])
        magnitude = _underlying_magnitude(regimes, close_col, start, end)
        vix_at_start, vix_obs_date = _asof_value(yf_data["vix"], start)
        vix_20d_ago, _ = _asof_value(yf_data["vix"], start - pd.Timedelta(days=28))
        defensive_returns = {
            "gold": _price_return(yf_data["gold"], start, end),
            "energy": _price_return(yf_data["energy"], start, end),
            "usd": _price_return(yf_data["usd"], start, end),
            "vix": _price_return(yf_data["vix"], start, end),
        }
        row = {
            "bear_start": start.date().isoformat(),
            "bear_end": end.date().isoformat(),
            "duration_days": duration_days,
            "bear_days": duration_days,
            **magnitude,
            "bear_scale": (
                "major_bear"
                if duration_days >= args.major_min_days
                and np.isfinite(float(magnitude["underlying_peak_to_trough_pct"]))
                and abs(float(magnitude["underlying_peak_to_trough_pct"])) >= args.major_min_drawdown
                else "minor_dip"
            ),
            "source_regime_csv": str(regime_path),
            "source_regime_column": args.regime_column,
            "dataset_tag": str(args.dataset_tag),
            "source_note": str(args.source_note),
            "vix_obs_date_known_at_start": "" if pd.isna(vix_obs_date) else vix_obs_date.date().isoformat(),
            "vix_at_start": vix_at_start,
            "vix_trend_approx_20d_at_start": vix_at_start - vix_20d_ago if np.isfinite(vix_at_start) and np.isfinite(vix_20d_ago) else np.nan,
            "vix_no_lookahead": bool((not pd.isna(vix_obs_date)) and vix_obs_date <= start),
            **_cpi_at_start(cpi, start),
            **_credit_at_start(credit, start),
            "gold_return_over_bear_period_hindsight": defensive_returns["gold"],
            "energy_return_over_bear_period_hindsight": defensive_returns["energy"],
            "usd_return_over_bear_period_hindsight": defensive_returns["usd"],
            "vix_return_over_bear_period_hindsight": defensive_returns["vix"],
            "best_hindsight_asset_return_over_bear_period": _best_defensive_asset(defensive_returns),
        }
        row["all_inputs_no_lookahead"] = bool(
            row["vix_no_lookahead"] and row["cpi_no_lookahead"] and row["credit_no_lookahead"]
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", 30)
    print("=" * 100)
    print("BEAR CLASSIFIER DATA PREP")
    print("=" * 100)
    print(f"Regime source: {regime_path}")
    print(f"Rows: {len(out)}")
    print(f"Saved: {out_path}")
    print()
    cols = [
        "bear_start", "bear_end", "duration_days", "bear_scale",
        "underlying_start_to_trough_pct", "underlying_peak_to_trough_pct",
        "vix_at_start", "vix_trend_approx_20d_at_start",
        "cpi_yoy_known_at_start", "cpi_yoy_trend_3m_known_at_start",
        "credit_spread_known_at_start", "credit_spread_trend_20obs_known_at_start",
        "gold_return_over_bear_period_hindsight",
        "energy_return_over_bear_period_hindsight",
        "usd_return_over_bear_period_hindsight",
        "best_hindsight_asset_return_over_bear_period",
        "all_inputs_no_lookahead",
    ]
    print(out[cols].to_string(index=False))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
