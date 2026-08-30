from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_btc_full_stack import (
    _download_binance_daily,
    classify_regime_v2_on_btc,
    perf,
)


def _load_daily_price_csv(path: Path, start: str, end: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    ts_col = next((c for c in ["timestamp", "date", "day", "datetime", "Date"] if c in d.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} must contain a timestamp/date column.")
    d["timestamp"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    rename = {c: c.lower() for c in d.columns if str(c).lower() in {"open", "high", "low", "close", "volume"}}
    d = d.rename(columns=rename)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "volume" not in d.columns:
        d["volume"] = 0.0
    d = d.dropna(subset=["timestamp", "open", "high", "low", "close"])
    d = d.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True) + pd.Timedelta(days=1)
    d = d[(d["timestamp"] >= start_ts) & (d["timestamp"] < end_ts)].copy()
    return d[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _download_gold_daily(start: str, end: str, symbol: str) -> tuple[pd.DataFrame, str]:
    tickers = [symbol]
    if symbol.upper() != "GLD":
        tickers.append("GLD")
    for t in tickers:
        x = yf.download(
            t,
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if x is None or x.empty:
            continue
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x = x.reset_index()
        date_col = "Date" if "Date" in x.columns else "index"
        x["day"] = pd.to_datetime(x[date_col], errors="coerce").dt.floor("D")
        x["open"] = pd.to_numeric(x["Open"], errors="coerce")
        x["close"] = pd.to_numeric(x["Close"], errors="coerce")
        x = x.dropna(subset=["day", "open", "close"]).sort_values("day")
        return x[["day", "open", "close"]].copy(), t
    raise RuntimeError(f"Gold data unavailable for {symbol} and GLD fallback failed.")


def load_eth_defensive_proxy(path: Path, perp_path: Path | None = None, pup_fallback: bool = False) -> pd.DataFrame:
    d = pd.read_csv(path)
    if not {"timestamp", "weight"}.issubset(d.columns):
        raise ValueError(f"{path} must contain timestamp and weight")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    if "p_up" in d.columns:
        d["p_up"] = pd.to_numeric(d["p_up"], errors="coerce")
    else:
        d["p_up"] = np.nan
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

    out = d.set_index("timestamp")[["weight", "p_up"]].resample("1D").last().rename(columns={"weight": "def_proxy_signal"})
    out.index.name = "day"
    out = out.reset_index()
    out["def_proxy_signal"] = pd.to_numeric(out["def_proxy_signal"], errors="coerce").ffill().fillna(0.0)
    out["funding_z"] = np.nan

    if perp_path is not None and perp_path.exists():
        p = pd.read_csv(perp_path)
        if {"timestamp", "funding_rate"}.issubset(p.columns):
            p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
            p["funding_rate"] = pd.to_numeric(p["funding_rate"], errors="coerce")
            p = p.dropna(subset=["timestamp"]).sort_values("timestamp")
            f = p.set_index("timestamp")["funding_rate"].resample("1D").mean().to_frame("funding_rate")
            mu = f["funding_rate"].rolling(60, min_periods=20).mean()
            sd = f["funding_rate"].rolling(60, min_periods=20).std(ddof=0)
            f["funding_z"] = ((f["funding_rate"] - mu) / sd.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
            f.index.name = "day"
            out = out.drop(columns=["funding_z"]).merge(f[["funding_z"]].reset_index(), on="day", how="left")

    out["p_up_fallback"] = np.nan
    z = pd.to_numeric(out["funding_z"], errors="coerce")
    out.loc[z > 1.5, "p_up_fallback"] = 0.60
    out.loc[(z > 0.5) & (z <= 1.5), "p_up_fallback"] = 0.52
    out.loc[z < -1.5, "p_up_fallback"] = 0.40
    out.loc[(z < -0.5) & (z >= -1.5), "p_up_fallback"] = 0.48
    out.loc[z.between(-0.5, 0.5, inclusive="both"), "p_up_fallback"] = 0.50
    out["pup_fallback_used"] = bool(pup_fallback) & out["p_up"].isna() & out["p_up_fallback"].notna()
    out["p_up_eff"] = out["p_up"]
    if pup_fallback:
        out.loc[out["pup_fallback_used"], "p_up_eff"] = out.loc[out["pup_fallback_used"], "p_up_fallback"]

    # Conservative fallback: only synthesize a defensive long signal when the
    # fallback is at least directionally bullish. The fallback is intentionally
    # capped below the original model's strongest signals.
    out["def_proxy_signal_original"] = out["def_proxy_signal"]
    if pup_fallback:
        mask = out["pup_fallback_used"] & (out["def_proxy_signal"] <= 0.0) & (out["p_up_fallback"] > 0.50)
        out.loc[mask, "def_proxy_signal"] = out.loc[mask, "p_up_fallback"]
    return out


def _derisk_multiplier(p_up_eff: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lows: list[int] = []
    mults: list[float] = []
    counter = 0
    for v in pd.to_numeric(p_up_eff, errors="coerce"):
        if pd.isna(v):
            pass
        elif float(v) < 0.35:
            counter += 1
        else:
            counter = 0
        lows.append(counter)
        if counter >= 3:
            mults.append(0.50)
        elif counter == 2:
            mults.append(0.75)
        else:
            mults.append(1.0)
    return np.asarray(lows, dtype=int), np.asarray(mults, dtype=float)


def _download_yfinance_daily(ticker: str, start: str, end: str, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        d = pd.read_csv(out_csv)
        date_col = "Date" if "Date" in d.columns else "day" if "day" in d.columns else "timestamp"
        d["day"] = pd.to_datetime(d[date_col], utc=True, errors="coerce").dt.floor("D")
        close_col = "Close" if "Close" in d.columns else "close"
        d["close"] = pd.to_numeric(d[close_col], errors="coerce")
        return d.dropna(subset=["day", "close"])[["day", "close"]].sort_values("day").copy()
    x = yf.download(
        ticker,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if x is None or x.empty:
        raise RuntimeError(f"No yfinance data for {ticker}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.reset_index()
    x["day"] = pd.to_datetime(x["Date"], utc=True, errors="coerce").dt.floor("D")
    x["close"] = pd.to_numeric(x["Close"], errors="coerce")
    out = x.dropna(subset=["day", "close"])[["day", "close"]].sort_values("day").copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _sp500_macro_frame(start: str, end: str, out_csv: Path) -> pd.DataFrame:
    s = _download_yfinance_daily("^GSPC", start, end, out_csv)
    s["sp_ema21"] = s["close"].ewm(span=21, adjust=False).mean()
    s["sp_ema55"] = s["close"].ewm(span=55, adjust=False).mean()
    s["sp_ema144"] = s["close"].ewm(span=144, adjust=False).mean()
    s["sp_regime"] = np.where(
        (s["sp_ema21"] > s["sp_ema55"]) & (s["sp_ema55"] > s["sp_ema144"]),
        "BULL",
        np.where((s["sp_ema21"] < s["sp_ema55"]) & (s["sp_ema55"] < s["sp_ema144"]), "BEAR", "CHOP"),
    )
    s["macro_multiplier"] = np.where(s["sp_regime"] == "BEAR", 0.5, 1.0)
    # sp_regime/macro_multiplier are derived from day t's own S&P close, so they
    # aren't knowable until after day t's close -- shift so they only apply
    # starting day t+1 (not currently enabled in the production reference config,
    # but fixed for consistency with the vol-filter/asymmetric-sizing fix above).
    s["sp_regime"] = s["sp_regime"].shift(1).fillna("CHOP")
    s["macro_multiplier"] = s["macro_multiplier"].shift(1).fillna(1.0)
    return s[["day", "sp_regime", "macro_multiplier"]].copy()


def _load_external_regime(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    day_col = "day" if "day" in d.columns else "date" if "date" in d.columns else "timestamp"
    regime_col = None
    for col in ["regime_v2", "hmm_regime", "regime", "label"]:
        if col in d.columns:
            regime_col = col
            break
    if regime_col is None:
        raise ValueError(f"{path} must contain a regime column.")
    d["day"] = pd.to_datetime(d[day_col], utc=True, errors="coerce").dt.floor("D")
    d["regime_override"] = d[regime_col].astype(str).str.upper()
    d = d[d["regime_override"].isin(["BULL", "CHOP", "BEAR"])]
    d = d.dropna(subset=["day"]).sort_values("day").drop_duplicates(subset=["day"], keep="last")
    if d.empty:
        raise ValueError(f"{path} did not contain any usable BULL/CHOP/BEAR regimes.")
    return d[["day", "regime_override"]].copy()


def _apply_resize_deadband(alloc: pd.Series, active: pd.Series, threshold: float) -> pd.Series:
    """Only update the EXECUTED allocation when the target moves more than `threshold`
    (absolute, e.g. 0.05 = 5 percentage points of gross_cap) away from the last executed
    value -- unless the sleeve is entering (active flips 0->1) or exiting (active flips
    1->0) that day, which always execute the new value immediately regardless of size.
    This throttles the continuous vol-filter/asymmetric-sizing-driven resizing that drives
    most of alloc_turnover_cost while a position is held, without delaying real entry/exit
    decisions. threshold=0.0 disables the deadband entirely (every target executes as-is,
    identical to no deadband)."""
    if threshold <= 0.0:
        return alloc.copy()
    alloc_vals = alloc.reset_index(drop=True).to_numpy(dtype=float)
    active_vals = active.reset_index(drop=True).to_numpy(dtype=float)
    out = np.zeros(len(alloc_vals))
    executed = 0.0
    prev_active = 0
    for i in range(len(alloc_vals)):
        target = alloc_vals[i]
        cur_active = int(active_vals[i] > 0)
        entering_or_exiting = cur_active != prev_active
        if entering_or_exiting or abs(target - executed) > threshold:
            executed = target
        out[i] = executed
        prev_active = cur_active
    return pd.Series(out, index=alloc.index)


def _eth_vol_frame(eth_daily: pd.DataFrame) -> pd.DataFrame:
    v = eth_daily[["day", "close"]].copy()
    ret = v["close"].pct_change()
    v["rolling_vol_20d"] = ret.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)

    def pct_rank(x: pd.Series) -> float:
        if x.isna().all():
            return np.nan
        return float(x.rank(pct=True).iloc[-1])

    v["vol_percentile"] = v["rolling_vol_20d"].rolling(252, min_periods=60).apply(pct_rank, raw=False)
    v["vol_multiplier"] = 1.0
    v.loc[v["vol_percentile"] > 0.75, "vol_multiplier"] = 0.5
    v.loc[v["vol_percentile"] < 0.25, "vol_multiplier"] = 1.2
    # These stats are only knowable after day t's own close, so they can only be
    # applied to sizing starting day t+1 (same causal convention as weight_exec
    # = weight_target.shift(1) below) -- without this shift, day t's vol_multiplier
    # was being computed from day t's own close and applied to day t's position.
    v["rolling_vol_20d"] = v["rolling_vol_20d"].shift(1)
    v["vol_percentile"] = v["vol_percentile"].shift(1)
    v["vol_multiplier"] = v["vol_multiplier"].shift(1)
    return v[["day", "rolling_vol_20d", "vol_percentile", "vol_multiplier"]].copy()


def _intraday_timing_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["day", "eth_open_5m", "eth_low_4h", "eth_timing_improvement"])
    d = pd.read_csv(path, low_memory=False)
    if "timestamp" not in d.columns:
        return pd.DataFrame(columns=["day", "eth_open_5m", "eth_low_4h", "eth_timing_improvement"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    close_col = "close" if "close" in d.columns else "Close" if "Close" in d.columns else None
    low_col = "low" if "low" in d.columns else "Low" if "Low" in d.columns else close_col
    open_col = "open" if "open" in d.columns else "Open" if "Open" in d.columns else close_col
    if close_col is None or low_col is None or open_col is None:
        return pd.DataFrame(columns=["day", "eth_open_5m", "eth_low_4h", "eth_timing_improvement"])
    d["open"] = pd.to_numeric(d[open_col], errors="coerce")
    d["low"] = pd.to_numeric(d[low_col], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "low"]).sort_values("timestamp")
    d["day"] = d["timestamp"].dt.floor("D")
    d["minute_of_day"] = (d["timestamp"] - d["day"]).dt.total_seconds() / 60.0
    first4 = d[d["minute_of_day"] < 240].copy()
    if first4.empty:
        return pd.DataFrame(columns=["day", "eth_open_5m", "eth_low_4h", "eth_timing_improvement"])
    out = first4.groupby("day").agg(eth_open_5m=("open", "first"), eth_low_4h=("low", "min")).reset_index()
    out["eth_timing_improvement"] = ((out["eth_open_5m"] - out["eth_low_4h"]) / out["eth_open_5m"]).clip(lower=0.0)
    return out


def _parse_ema_spans(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(value).replace("/", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("EMA spans must be fast,mid,slow, for example 21,55,144")
    spans = tuple(int(p) for p in parts)
    if spans[0] <= 0 or spans[1] <= spans[0] or spans[2] <= spans[1]:
        raise argparse.ArgumentTypeError("EMA spans must be positive and increasing")
    return spans  # type: ignore[return-value]


def _signal_context(symbol: str, start: str, end: str, ema_spans: tuple[int, int, int]) -> pd.DataFrame:
    d = _download_binance_daily(symbol=symbol, start=start, end=end)
    if d.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")
    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)
    e1, e2, e3 = ema_spans
    d["ema_fast"] = d["close"].ewm(span=e1, adjust=False).mean()
    d["ema_mid"] = d["close"].ewm(span=e2, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=e3, adjust=False).mean()
    d["stack_aligned"] = (d["ema_fast"] > d["ema_mid"]) & (d["ema_mid"] > d["ema_slow"])
    return d[["day", "stack_aligned", "regime_v2"]].copy()


def _cross_confirm_entry(
    base_entry: pd.Series,
    secondary_ok: pd.Series,
    variant: str,
    max_delay_days: int = 5,
) -> tuple[pd.Series, dict[str, float]]:
    variant = str(variant).lower()
    if variant in {"", "off", "none"}:
        return base_entry.astype(bool), {"trades_removed": 0, "trades_delayed": 0}
    base = base_entry.fillna(False).astype(bool).to_numpy()
    ok = secondary_ok.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(base), dtype=bool)
    removed = 0
    delayed = 0
    pending = False
    pending_age = 0
    pending_i = -1
    for i in range(len(base)):
        if not pending and base[i]:
            if ok[i]:
                out[i] = True
            elif variant == "strict":
                removed += 1
            else:
                pending = True
                pending_age = 0
                pending_i = i
        elif pending:
            pending_age += 1
            if ok[i]:
                out[i] = True
                delayed += 1
                pending = False
            elif pending_age >= max_delay_days:
                out[i] = True
                delayed += 1
                pending = False
        if out[i]:
            pending = False
    if pending:
        removed += 1
    return pd.Series(out, index=base_entry.index), {"trades_removed": removed, "trades_delayed": delayed}


def _fmt_pct(v: float) -> str:
    return f"{float(v) * 100:.2f}%"


def _run_child(args: argparse.Namespace, name: str, extra: list[str]) -> dict[str, Any]:
    out_dir = Path("artifacts/backtest/improvement_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / f"{name}_summary.csv"
    daily = out_dir / f"{name}_daily.csv"
    cmd = [
        sys.executable,
        str(Path(__file__)),
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--eth-symbol",
        str(args.eth_symbol),
        "--btc-symbol",
        str(args.btc_symbol),
        "--cost-bps",
        "20",
        "--cost-mode",
        "weight_change",
        "--allocation-mode",
        "signal_weighted",
        "--gross-cap",
        "0.8",
        "--vol-filter",
        "--transition-momentum",
        "--eth-defensive-csv",
        str(args.eth_defensive_csv),
        "--eth-perp-csv",
        str(args.eth_perp_csv),
        "--out-summary-csv",
        str(summary),
        "--out-daily-csv",
        str(daily),
    ] + extra
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    row = pd.read_csv(summary).iloc[0].to_dict()
    row["name"] = name
    row["summary_csv"] = str(summary)
    row["daily_csv"] = str(daily)
    return row


def run_round2_suite(args: argparse.Namespace) -> int:
    out_dir = Path("artifacts/backtest/improvement_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, list[str], str]] = [
        ("vol_transition", [], "Vol+Transition baseline"),
    ]

    ema_defs = [
        ("ema_10_30_90", ["--eth-ema", "10,30,90", "--btc-ema", "10,30,90"], "10/30/90"),
        ("ema_15_40_120", ["--eth-ema", "15,40,120", "--btc-ema", "15,40,120"], "15/40/120"),
        ("ema_21_55_144", ["--eth-ema", "21,55,144", "--btc-ema", "21,55,144"], "21/55/144"),
        ("ema_25_65_150", ["--eth-ema", "25,65,150", "--btc-ema", "25,65,150"], "25/65/150"),
        ("ema_30_80_200", ["--eth-ema", "30,80,200", "--btc-ema", "30,80,200"], "30/80/200"),
        ("ema_13_34_89", ["--eth-ema", "13,34,89", "--btc-ema", "13,34,89"], "13/34/89"),
        ("ema_20_50_100", ["--eth-ema", "20,50,100", "--btc-ema", "20,50,100"], "20/50/100"),
        (
            "ema_asset_specific_h",
            ["--eth-ema", "21,55,144", "--btc-ema", "15,40,120"],
            "ETH 21/55/144 + BTC 15/40/120",
        ),
    ]
    for name, extra, label in ema_defs:
        cases.append((name, extra, label))
    cases.extend(
        [
            ("cross_confirm_strict", ["--cross-confirm", "strict"], "Cross-confirm strict"),
            ("cross_confirm_loose", ["--cross-confirm", "loose"], "Cross-confirm loose"),
            ("cross_confirm_one_way", ["--cross-confirm", "one-way"], "Cross-confirm one-way"),
            ("dd_aware_gradual", ["--dd-aware", "gradual"], "DD-aware gradual"),
            ("dd_aware_binary", ["--dd-aware", "binary"], "DD-aware binary"),
            ("asymmetric_sizing", ["--asymmetric-sizing"], "Asymmetric sizing"),
        ]
    )

    rows: list[dict[str, Any]] = []
    for name, extra, label in cases:
        row = _run_child(args, name, extra)
        row["configuration"] = label
        rows.append(row)

    all_df = pd.DataFrame(rows)
    ema_df = all_df[all_df["name"].astype(str).str.startswith("ema_")].copy()
    ema_df = ema_df.sort_values("combined_sharpe", ascending=False)
    ema_df.to_csv(out_dir / "ema_grid_summary.csv", index=False)

    cross_df = all_df[all_df["name"].astype(str).str.startswith("cross_confirm_")].copy()
    cross_df.to_csv(out_dir / "cross_confirm_summary.csv", index=False)

    dd_df = all_df[all_df["name"].astype(str).str.startswith("dd_aware_")].copy()
    dd_df.to_csv(out_dir / "dd_aware_summary.csv", index=False)

    asym_df = all_df[all_df["name"].astype(str).eq("asymmetric_sizing")].copy()
    asym_df.to_csv(out_dir / "asymmetric_sizing_summary.csv", index=False)

    final_cols = [
        "configuration",
        "combined_sharpe",
        "combined_max_dd",
        "combined_cagr",
        "combined_ann_vol",
        "eth_trades",
        "btc_trades",
        "eth_cross_trades_removed",
        "btc_cross_trades_removed",
        "eth_cross_trades_delayed",
        "btc_cross_trades_delayed",
    ]
    for c in final_cols:
        if c not in all_df.columns:
            all_df[c] = np.nan
    all_df[final_cols].to_csv(out_dir / "round2_full_comparison.csv", index=False)

    baseline = all_df[all_df["name"].eq("vol_transition")].iloc[0]
    best = all_df.sort_values("combined_sharpe", ascending=False).iloc[0]
    print("========================================================")
    print("ROUND 2 IMPROVEMENT RESULTS")
    print("Baseline: vol_filter + transition_momentum")
    print("========================================================")
    print(f"{'Configuration':34s} {'Sharpe':>7s} {'MaxDD':>9s} {'CAGR':>8s}")
    print("-" * 62)
    wanted = [
        "vol_transition",
        ema_df.iloc[0]["name"] if not ema_df.empty else "",
        "cross_confirm_strict",
        "cross_confirm_loose",
        "dd_aware_gradual",
        "dd_aware_binary",
        "asymmetric_sizing",
    ]
    printed = set()
    for name in wanted:
        if not name or name in printed:
            continue
        r = all_df[all_df["name"].eq(name)]
        if r.empty:
            continue
        rr = r.iloc[0]
        label = str(rr["configuration"])
        if name == str(ema_df.iloc[0]["name"]):
            label = "+ EMA best combo"
        print(f"{label[:34]:34s} {rr['combined_sharpe']:7.3f} {_fmt_pct(rr['combined_max_dd']):>9s} {_fmt_pct(rr['combined_cagr']):>8s}")
        printed.add(name)
    print("-" * 62)
    print(f"{'Best combination of all:':34s} {best['combined_sharpe']:7.3f} {_fmt_pct(best['combined_max_dd']):>9s} {_fmt_pct(best['combined_cagr']):>8s}")
    print("-" * 62)
    print("EMA grid full ranking:")
    for i, (_, r) in enumerate(ema_df.iterrows(), start=1):
        print(f"  Rank {i}: {r['configuration']}  Sharpe {r['combined_sharpe']:.3f}")
    print("========================================================")
    print(f"Saved: {out_dir / 'ema_grid_summary.csv'}")
    print(f"Saved: {out_dir / 'cross_confirm_summary.csv'}")
    print(f"Saved: {out_dir / 'dd_aware_summary.csv'}")
    print(f"Saved: {out_dir / 'asymmetric_sizing_summary.csv'}")
    print(f"Saved: {out_dir / 'round2_full_comparison.csv'}")
    return 0


def run_sleeve(
    symbol: str,
    confirm_days: int,
    def_proxy: pd.DataFrame,
    cost_bps: float,
    cost_mode: str,
    derisking: bool,
    stop_loss: bool,
    stop_loss_pct: float,
    transition_momentum: bool,
    start: str,
    end: str,
    out_price_csv: Path,
    ema_spans: tuple[int, int, int] = (21, 55, 144),
    secondary_confirm: pd.DataFrame | None = None,
    cross_confirm_variant: str = "off",
    regime_override: pd.DataFrame | None = None,
    price_data: pd.DataFrame | None = None,
    reversal_cooldown_days: int = 0,
    reversal_cooldown_threshold_days: int = 10,
) -> pd.DataFrame:
    d = price_data.copy() if price_data is not None else _download_binance_daily(symbol=symbol, start=start, end=end)
    if d.empty:
        raise RuntimeError(f"No price data available for {symbol}")
    out_price_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_price_csv, index=False)

    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)
    if regime_override is not None and not regime_override.empty:
        d = d.merge(regime_override, on="day", how="left")
        d["regime_v2"] = d["regime_override"].ffill().fillna(d["regime_v2"])
        d = d.drop(columns=["regime_override"], errors="ignore")

    e1, e2, e3 = ema_spans
    d["ema21"] = d["close"].ewm(span=e1, adjust=False).mean()
    d["ema55"] = d["close"].ewm(span=e2, adjust=False).mean()
    d["ema144"] = d["close"].ewm(span=e3, adjust=False).mean()
    d["ema_fast_span"] = e1
    d["ema_mid_span"] = e2
    d["ema_slow_span"] = e3
    d["stack_aligned"] = (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])
    conf = max(1, int(confirm_days))
    entry_base = (d["stack_aligned"].rolling(conf, min_periods=conf).min() == 1).fillna(False)
    d["cross_trades_removed"] = 0
    d["cross_trades_delayed"] = 0
    if secondary_confirm is not None and str(cross_confirm_variant).lower() not in {"", "off", "none"}:
        sec = secondary_confirm.rename(
            columns={"stack_aligned": "secondary_stack_aligned", "regime_v2": "secondary_regime"}
        )[["day", "secondary_stack_aligned", "secondary_regime"]].copy()
        d = d.merge(sec, on="day", how="left")
        secondary_ok = d["secondary_stack_aligned"].fillna(False).astype(bool) & ~d["secondary_regime"].astype(str).eq("BEAR")
        entry, cross_stats = _cross_confirm_entry(entry_base, secondary_ok, str(cross_confirm_variant))
        d["cross_trades_removed"] = int(cross_stats["trades_removed"])
        d["cross_trades_delayed"] = int(cross_stats["trades_delayed"])
    else:
        entry = entry_base
    exit_ = ((~d["stack_aligned"]).rolling(conf, min_periods=conf).min() == 1).fillna(False)

    rv = d["close"].pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    d["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(d), dtype=int)
    trade_id = np.zeros(len(d), dtype=int)
    stop_loss_exit = np.zeros(len(d), dtype=int)
    stop_loss_whipsaw = np.zeros(len(d), dtype=int)
    transition_class = np.array(["NONE"] * len(d), dtype=object)
    transition_mult = np.ones(len(d), dtype=float)
    active = 0
    entry_i: int | None = None
    entry_open = np.nan
    cur_trade_id = 0
    cooldown_until_i = -1
    reversal_cooldown_blocks = 0
    for i in range(len(d)):
        if active == 0 and bool(entry.iloc[i]) and i > cooldown_until_i:
            active = 1
            entry_i = i
            cur_trade_id += 1
            entry_open = float(d["open"].iloc[i + 1]) if i + 1 < len(d) else float(d["close"].iloc[i])
            if transition_momentum:
                prev_reg = d["regime_v2"].shift(1)
                flip_mask = (d["regime_v2"] != prev_reg) & d["regime_v2"].isin(["BULL", "CHOP"])
                flips = np.flatnonzero(flip_mask.iloc[: i + 1].to_numpy())
                if len(flips):
                    fi = int(flips[-1])
                    tr = float(d["close"].pct_change().iloc[fi]) if pd.notna(d["close"].pct_change().iloc[fi]) else 0.0
                    if tr > 0.03:
                        transition_class[i] = "STRONG"
                    elif tr < 0.01:
                        transition_class[i] = "WEAK"
                        transition_mult[i] = 0.6
                    else:
                        transition_class[i] = "MID"
        elif active == 0 and bool(entry.iloc[i]) and i <= cooldown_until_i:
            reversal_cooldown_blocks += 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
            if reversal_cooldown_days > 0 and entry_i is not None and (i - entry_i) < int(reversal_cooldown_threshold_days):
                cooldown_until_i = i + int(reversal_cooldown_days) - 1
            entry_i = None
        elif active == 1 and stop_loss and entry_i is not None and np.isfinite(entry_open) and entry_open > 0:
            trade_ret = float(d["close"].iloc[i] / entry_open - 1.0)
            if trade_ret < -abs(float(stop_loss_pct)):
                active = 0
                stop_loss_exit[i] = 1
                future = d["close"].iloc[i + 1 : i + 11]
                if len(future) and bool((future > entry_open).any()):
                    stop_loss_whipsaw[i] = 1
                if reversal_cooldown_days > 0 and entry_i is not None and (i - entry_i) < int(reversal_cooldown_threshold_days):
                    cooldown_until_i = i + int(reversal_cooldown_days) - 1
                entry_i = None
        if active == 1 and transition_momentum and entry_i is not None:
            transition_class[i] = transition_class[entry_i]
            if transition_class[entry_i] == "WEAK":
                held = i - entry_i
                trade_positive = np.isfinite(entry_open) and entry_open > 0 and float(d["close"].iloc[i] / entry_open - 1.0) > 0.0
                transition_mult[i] = 1.0 if held >= 5 and trade_positive else 0.6
        pos[i] = active
        trade_id[i] = cur_trade_id if active else 0
    d["off_active"] = pos
    d["trade_id"] = trade_id
    d["stop_loss_exit"] = stop_loss_exit
    d["stop_loss_whipsaw"] = stop_loss_whipsaw
    d["reversal_cooldown_blocks"] = reversal_cooldown_blocks
    d["transition_class"] = transition_class
    d["transition_multiplier"] = transition_mult
    d["off_raw_signal"] = np.where(d["off_active"] == 1, d["vol_scalar"], 0.0)
    d["conviction"] = 0.0
    gap_pct = ((d["ema55"] - d["ema144"]) / d["close"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    d["conviction_gap_score"] = np.select([gap_pct > 0.02, gap_pct >= 0.01], [0.33, 0.20], default=0.10)
    d["conviction_regime_score"] = d["regime_v2"].map({"BULL": 0.33, "CHOP": 0.17, "BEAR": 0.0}).fillna(0.0)
    d["conviction"] = d["conviction_gap_score"] + d["conviction_regime_score"]
    # conviction_gap_score uses day t's own close/EMAs, so it isn't knowable until
    # after day t's close -- shift so asymmetric-sizing (which consumes this
    # column downstream) only ever applies it starting day t+1.
    d["conviction"] = d["conviction"].shift(1).fillna(0.0)

    off_scale_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    def_scale_map = {"BULL": 0.0, "CHOP": 0.4, "BEAR": 1.0}
    d["off_scale"] = d["regime_v2"].map(off_scale_map).fillna(0.0)
    d["off_target"] = d["off_raw_signal"] * d["off_scale"]
    if transition_momentum:
        d["off_target"] = d["off_target"] * d["transition_multiplier"]

    x = d.merge(def_proxy, on="day", how="left")
    x["def_proxy_signal"] = pd.to_numeric(x["def_proxy_signal"], errors="coerce").fillna(0.0)
    for c in ["p_up", "p_up_eff", "funding_z", "pup_fallback_used"]:
        if c not in x.columns:
            x[c] = np.nan if c != "pup_fallback_used" else False
    lows, mults = _derisk_multiplier(x["p_up_eff"])
    x["p_up_low_days"] = lows
    x["derisk_multiplier"] = mults
    if derisking:
        x["off_target_pre_derisk"] = x["off_target"]
        active_off = x["off_active"] == 1
        x.loc[active_off, "off_target"] = x.loc[active_off, "off_target"] * x.loc[active_off, "derisk_multiplier"]
    else:
        x["off_target_pre_derisk"] = x["off_target"]
        x["derisk_multiplier"] = 1.0
    x["def_scale"] = x["regime_v2"].map(def_scale_map).fillna(0.0)
    x["def_target"] = x["def_proxy_signal"] * x["def_scale"]

    x["weight_target"] = (x["off_target"] + x["def_target"]).clip(lower=0.0, upper=1.0)
    x["weight_exec"] = x["weight_target"].shift(1).fillna(0.0)
    x["exec_return"] = x["open"].shift(-1) / x["open"] - 1.0

    prev_w = x["weight_exec"].shift(1).fillna(0.0)
    if str(cost_mode).lower() == "entry_exit":
        # Legacy/optimistic: charge only when crossing flat <-> non-flat.
        crossed = ((x["weight_exec"] > 0).astype(int) != (prev_w > 0).astype(int))
        x["turnover"] = np.where(crossed, (x["weight_exec"] - prev_w).abs(), 0.0)
    else:
        # Realistic/default: charge on any weight change.
        x["turnover"] = (x["weight_exec"] - prev_w).abs()
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["exec_return"].fillna(0.0) - x["cost"]
    x = x.dropna(subset=["exec_return"]).copy()
    x["eq"] = (1.0 + x["strategy_return"]).cumprod()
    x["spot_eq"] = (1.0 + x["exec_return"]).cumprod()
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH+BTC parallel sleeve portfolio backtest (lag-1 realistic)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--eth-data", default="", help="Optional local ETH daily OHLCV CSV.")
    ap.add_argument("--btc-data", default="", help="Optional local BTC daily OHLCV CSV.")
    ap.add_argument("--eth-symbol", default="ETHUSDC")
    ap.add_argument("--btc-symbol", default="BTCUSDC")
    ap.add_argument("--eth-confirm-days", type=int, default=3)
    ap.add_argument("--btc-confirm-days", type=int, default=5)
    ap.add_argument("--eth-ema", type=_parse_ema_spans, default=(21, 55, 144))
    ap.add_argument("--btc-ema", type=_parse_ema_spans, default=(21, 55, 144))
    ap.add_argument("--cost-bps", type=float, default=60.0)  # matches the validated production reference; was 10.0 (understated), briefly 20.0, now corrected to real Kraken taker fees at ~$1k-10k/month volume
    ap.add_argument("--cost-mode", choices=["entry_exit", "weight_change"], default="weight_change")
    ap.add_argument("--allocation-mode", choices=["fixed", "signal_weighted"], default="fixed")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--derisking", action="store_true")
    ap.add_argument("--pup-fallback", action="store_true")
    ap.add_argument("--macro-filter", action="store_true")
    ap.add_argument("--stop-loss", action="store_true")
    ap.add_argument("--stop-loss-pct", type=float, default=0.08)
    ap.add_argument("--reversal-cooldown-days", type=int, default=0, help="After a trade exits having been held less than --reversal-cooldown-threshold-days, block re-entry on that sleeve for this many days. 0 (default) disables the cooldown entirely.")
    ap.add_argument("--reversal-cooldown-threshold-days", type=int, default=10, help="A trade held fewer than this many days is classified as a quick reversal and triggers the cooldown on exit.")
    ap.add_argument("--resize-deadband", type=float, default=0.0, help="Only update the executed alloc_eth/alloc_btc when the target moves more than this much (absolute, e.g. 0.05 = 5pp of gross_cap) from the last executed value; real entries/exits always execute immediately regardless. 0.0 (default) disables the deadband entirely.")
    ap.add_argument("--vol-filter", action="store_true")
    ap.add_argument("--transition-momentum", action="store_true")
    ap.add_argument("--cross-confirm", choices=["off", "strict", "loose", "one-way"], default="off")
    ap.add_argument("--dd-aware", choices=["off", "gradual", "binary"], default="off")
    ap.add_argument("--asymmetric-sizing", action="store_true")
    ap.add_argument("--ema-grid", action="store_true")
    ap.add_argument("--round2-suite", action="store_true")
    ap.add_argument("--intraday-timing", action="store_true")
    ap.add_argument("--eth-5m-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--include-gold", action="store_true")
    ap.add_argument("--gold-symbol", default="PAXG-USD")
    ap.add_argument("--gold-ema", type=_parse_ema_spans, default=(21, 55, 144))
    ap.add_argument("--gold-cap", type=float, default=0.3)
    ap.add_argument("--gold-cost-bps", type=float, default=60.0)  # matches the validated production reference; was 10.0, briefly 20.0
    ap.add_argument("--regime-source", choices=["ema", "hmm"], default="ema")
    ap.add_argument("--hmm-regime-csv", default="")
    ap.add_argument("--eth-defensive-csv", default="artifacts/backtest/direction_event_model_v1_flat_defensive_6y_gapfilled.csv")
    ap.add_argument("--eth-perp-csv", default="data/backtest/ETH_perp_features_5m_6y_gapfilled.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/eth_btc_portfolio_summary.csv")
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/eth_btc_portfolio_daily.csv")
    args = ap.parse_args()

    if bool(args.ema_grid) or bool(args.round2_suite):
        return run_round2_suite(args)

    regime_override = None
    if str(args.regime_source) == "hmm":
        if not str(args.hmm_regime_csv).strip():
            raise SystemExit("--hmm-regime-csv is required when --regime-source hmm")
        regime_override = _load_external_regime(Path(args.hmm_regime_csv))

    def_proxy = load_eth_defensive_proxy(Path(args.eth_defensive_csv), Path(args.eth_perp_csv), bool(args.pup_fallback))
    eth_price_data = _load_daily_price_csv(Path(args.eth_data), args.start, args.end) if str(args.eth_data).strip() else None
    btc_price_data = _load_daily_price_csv(Path(args.btc_data), args.start, args.end) if str(args.btc_data).strip() else None
    eth_secondary = None
    btc_secondary = None
    if str(args.cross_confirm) in {"strict", "loose", "one-way"}:
        eth_ctx = _signal_context(args.eth_symbol, args.start, args.end, tuple(args.eth_ema))
        btc_ctx = _signal_context(args.btc_symbol, args.start, args.end, tuple(args.btc_ema))
        btc_secondary = eth_ctx
        eth_secondary = btc_ctx if str(args.cross_confirm) != "one-way" else None

    eth = run_sleeve(
        symbol=args.eth_symbol,
        confirm_days=int(args.eth_confirm_days),
        def_proxy=def_proxy,
        cost_bps=float(args.cost_bps),
        cost_mode=str(args.cost_mode),
        derisking=bool(args.derisking),
        stop_loss=bool(args.stop_loss),
        stop_loss_pct=float(args.stop_loss_pct),
        transition_momentum=bool(args.transition_momentum),
        start=args.start,
        end=args.end,
        out_price_csv=Path("data/eth_daily.csv"),
        ema_spans=tuple(args.eth_ema),
        secondary_confirm=eth_secondary,
        cross_confirm_variant=str(args.cross_confirm),
        regime_override=regime_override,
        price_data=eth_price_data,
        reversal_cooldown_days=int(args.reversal_cooldown_days),
        reversal_cooldown_threshold_days=int(args.reversal_cooldown_threshold_days),
    )
    btc = run_sleeve(
        symbol=args.btc_symbol,
        confirm_days=int(args.btc_confirm_days),
        def_proxy=def_proxy,
        cost_bps=float(args.cost_bps),
        cost_mode=str(args.cost_mode),
        derisking=bool(args.derisking),
        stop_loss=bool(args.stop_loss),
        stop_loss_pct=float(args.stop_loss_pct),
        transition_momentum=bool(args.transition_momentum),
        start=args.start,
        end=args.end,
        out_price_csv=Path("data/btc_daily.csv"),
        ema_spans=tuple(args.btc_ema),
        secondary_confirm=btc_secondary,
        cross_confirm_variant=str(args.cross_confirm),
        regime_override=regime_override,
        price_data=btc_price_data,
        reversal_cooldown_days=int(args.reversal_cooldown_days),
        reversal_cooldown_threshold_days=int(args.reversal_cooldown_threshold_days),
    )

    keep = [
        "timestamp",
        "day",
        "open",
        "close",
        "regime_v2",
        "off_active",
        "off_target",
        "weight_exec",
        "exec_return",
        "strategy_return",
        "turnover",
        "p_up_eff",
        "p_up_low_days",
        "derisk_multiplier",
        "pup_fallback_used",
        "stop_loss_exit",
        "stop_loss_whipsaw",
        "transition_class",
        "transition_multiplier",
        "conviction",
        "cross_trades_removed",
        "cross_trades_delayed",
        "reversal_cooldown_blocks",
    ]
    e = eth[keep].copy().rename(
        columns={
            "open": "eth_open",
            "close": "eth_close",
            "regime_v2": "eth_regime",
            "off_active": "eth_off_active",
            "off_target": "eth_off_target",
            "weight_exec": "eth_weight_exec",
            "exec_return": "eth_spot_return",
            "strategy_return": "eth_strategy_return",
            "turnover": "eth_turnover",
            "p_up_eff": "eth_p_up_eff",
            "p_up_low_days": "eth_p_up_low_days",
            "derisk_multiplier": "eth_derisk_multiplier",
            "pup_fallback_used": "eth_pup_fallback_used",
            "stop_loss_exit": "eth_stop_loss_exit",
            "stop_loss_whipsaw": "eth_stop_loss_whipsaw",
            "transition_class": "eth_transition_class",
            "transition_multiplier": "eth_transition_multiplier",
            "conviction": "eth_conviction",
            "cross_trades_removed": "eth_cross_trades_removed",
            "cross_trades_delayed": "eth_cross_trades_delayed",
            "reversal_cooldown_blocks": "eth_reversal_cooldown_blocks",
        }
    )
    b = btc[keep].copy().rename(
        columns={
            "open": "btc_open",
            "close": "btc_close",
            "regime_v2": "btc_regime",
            "off_active": "btc_off_active",
            "off_target": "btc_off_target",
            "weight_exec": "btc_weight_exec",
            "exec_return": "btc_spot_return",
            "strategy_return": "btc_strategy_return",
            "turnover": "btc_turnover",
            "p_up_eff": "btc_p_up_eff",
            "p_up_low_days": "btc_p_up_low_days",
            "derisk_multiplier": "btc_derisk_multiplier",
            "pup_fallback_used": "btc_pup_fallback_used",
            "stop_loss_exit": "btc_stop_loss_exit",
            "stop_loss_whipsaw": "btc_stop_loss_whipsaw",
            "transition_class": "btc_transition_class",
            "transition_multiplier": "btc_transition_multiplier",
            "conviction": "btc_conviction",
            "cross_trades_removed": "btc_cross_trades_removed",
            "cross_trades_delayed": "btc_cross_trades_delayed",
            "reversal_cooldown_blocks": "btc_reversal_cooldown_blocks",
        }
    )

    merged = e.merge(b, on=["timestamp", "day"], how="inner").sort_values("timestamp").copy()
    if str(args.allocation_mode) == "signal_weighted":
        score_sum = merged["eth_weight_exec"] + merged["btc_weight_exec"]
        active = score_sum > 0
        gross = np.where(active, float(args.gross_cap), 0.0)
        merged["alloc_eth"] = np.where(active, gross * (merged["eth_weight_exec"] / score_sum), 0.0)
        merged["alloc_btc"] = np.where(active, gross * (merged["btc_weight_exec"] / score_sum), 0.0)
    else:
        merged["alloc_eth"] = 0.5
        merged["alloc_btc"] = 0.5

    merged["macro_multiplier"] = 1.0
    merged["sp_regime"] = ""
    if bool(args.macro_filter):
        sp = _sp500_macro_frame(args.start, args.end, Path("data/sp500_daily.csv"))
        merged = merged.drop(columns=["macro_multiplier", "sp_regime"], errors="ignore").merge(sp, on="day", how="left")
        merged["sp_regime"] = merged["sp_regime"].ffill().fillna("CHOP")
        merged["macro_multiplier"] = pd.to_numeric(merged["macro_multiplier"], errors="coerce").fillna(1.0)
        merged["alloc_eth"] = merged["alloc_eth"] * merged["macro_multiplier"]
        merged["alloc_btc"] = merged["alloc_btc"] * merged["macro_multiplier"]

    merged["vol_multiplier"] = 1.0
    merged["vol_percentile"] = np.nan
    if bool(args.vol_filter):
        vf = _eth_vol_frame(eth[["day", "close"]].copy())
        merged = merged.drop(columns=["vol_multiplier", "vol_percentile"], errors="ignore").merge(vf, on="day", how="left")
        merged["vol_multiplier"] = pd.to_numeric(merged["vol_multiplier"], errors="coerce").fillna(1.0)
        merged["vol_percentile"] = pd.to_numeric(merged["vol_percentile"], errors="coerce")
        merged["alloc_eth"] = merged["alloc_eth"] * merged["vol_multiplier"]
        merged["alloc_btc"] = merged["alloc_btc"] * merged["vol_multiplier"]
        gross = merged["alloc_eth"] + merged["alloc_btc"]
        over = gross > float(args.gross_cap)
        merged.loc[over, "alloc_eth"] = merged.loc[over, "alloc_eth"] * float(args.gross_cap) / gross.loc[over]
        merged.loc[over, "alloc_btc"] = merged.loc[over, "alloc_btc"] * float(args.gross_cap) / gross.loc[over]

    merged["asym_eth_multiplier"] = 1.0
    merged["asym_btc_multiplier"] = 1.0
    if bool(args.asymmetric_sizing):
        vol_score = np.select(
            [
                pd.to_numeric(merged["vol_percentile"], errors="coerce") < 0.25,
                pd.to_numeric(merged["vol_percentile"], errors="coerce") > 0.75,
            ],
            [0.33, 0.10],
            default=0.20,
        )
        merged["eth_conviction_total"] = pd.to_numeric(merged["eth_conviction"], errors="coerce").fillna(0.0) + vol_score
        merged["btc_conviction_total"] = pd.to_numeric(merged["btc_conviction"], errors="coerce").fillna(0.0) + vol_score

        def conv_mult(s: pd.Series) -> pd.Series:
            return pd.Series(
                np.select(
                    [s > 0.80, s >= 0.60, s >= 0.40],
                    [1.2, 1.0, 0.8],
                    default=0.6,
                ),
                index=s.index,
            )

        merged["asym_eth_multiplier"] = conv_mult(merged["eth_conviction_total"])
        merged["asym_btc_multiplier"] = conv_mult(merged["btc_conviction_total"])
        merged["alloc_eth"] = merged["alloc_eth"] * merged["asym_eth_multiplier"]
        merged["alloc_btc"] = merged["alloc_btc"] * merged["asym_btc_multiplier"]
        gross = merged["alloc_eth"] + merged["alloc_btc"]
        over = gross > float(args.gross_cap)
        merged.loc[over, "alloc_eth"] = merged.loc[over, "alloc_eth"] * float(args.gross_cap) / gross.loc[over]
        merged.loc[over, "alloc_btc"] = merged.loc[over, "alloc_btc"] * float(args.gross_cap) / gross.loc[over]

    merged["dd_multiplier"] = 1.0
    if str(args.dd_aware) != "off":
        dd_mults: list[float] = []
        eq = 1.0
        peak = 1.0
        for _, row in merged.iterrows():
            current_dd = (eq - peak) / peak if peak > 0 else 0.0
            if str(args.dd_aware) == "binary":
                mult = 0.5 if current_dd <= -0.10 else 1.0
            elif current_dd <= -0.15:
                mult = 0.25
            elif current_dd <= -0.10:
                mult = 0.50
            elif current_dd <= -0.05:
                mult = 0.75
            else:
                mult = 1.0
            dd_mults.append(mult)
            r = (
                float(row["alloc_eth"]) * float(row["eth_strategy_return"])
                + float(row["alloc_btc"]) * float(row["btc_strategy_return"])
            ) * mult
            eq *= 1.0 + r
            peak = max(peak, eq)
        merged["dd_multiplier"] = dd_mults
        active_any = (merged["alloc_eth"] > 0) | (merged["alloc_btc"] > 0)
        merged.loc[active_any, "alloc_eth"] = merged.loc[active_any, "alloc_eth"] * merged.loc[active_any, "dd_multiplier"]
        merged.loc[active_any, "alloc_btc"] = merged.loc[active_any, "alloc_btc"] * merged.loc[active_any, "dd_multiplier"]

    if float(args.resize_deadband) > 0.0:
        merged["alloc_eth"] = _apply_resize_deadband(merged["alloc_eth"], merged["eth_off_active"], float(args.resize_deadband))
        merged["alloc_btc"] = _apply_resize_deadband(merged["alloc_btc"], merged["btc_off_active"], float(args.resize_deadband))

    # Alloc-level turnover cost: vol_multiplier/asym_*_multiplier/macro_multiplier/dd_multiplier
    # all continuously resize alloc_eth/alloc_btc day to day (independent of whether the
    # underlying sleeve signal itself is changing), but until now nothing charged a transaction
    # cost for that resizing -- only the sleeve-level weight_exec turnover (pre-normalization,
    # baked into eth_strategy_return/btc_strategy_return) was costed, and that gets scaled down
    # by alloc_eth/alloc_btc when combined into the portfolio return, making it a small
    # second-order effect rather than a real proxy for the cost of actually moving portfolio
    # capital between sleeves. Quantified in check_baseline_rebalance_cost.py as ~40-51% of
    # 2022/2023's total cost; fixed at the source here using the same _alloc_turnover formula
    # already established (backtest_mode_a_validation.py) rather than as a post-hoc adjustment.
    merged["alloc_turnover"] = (
        (merged["alloc_eth"] - merged["alloc_eth"].shift(1).fillna(0.0)).abs()
        + (merged["alloc_btc"] - merged["alloc_btc"].shift(1).fillna(0.0)).abs()
    )
    merged["alloc_turnover_cost"] = merged["alloc_turnover"] * (float(args.cost_bps) / 10000.0)

    merged["eth_timing_improvement"] = 0.0
    merged["eth_timing_benefit"] = 0.0
    if bool(args.intraday_timing):
        timing = _intraday_timing_frame(Path(args.eth_5m_csv))
        if not timing.empty:
            merged = merged.drop(columns=["eth_timing_improvement"], errors="ignore").merge(timing[["day", "eth_timing_improvement"]], on="day", how="left")
            merged["eth_timing_improvement"] = pd.to_numeric(merged["eth_timing_improvement"], errors="coerce").fillna(0.0)
            eth_prev_w = merged["eth_weight_exec"].shift(1).fillna(0.0)
            eth_increase = (merged["eth_weight_exec"] - eth_prev_w).clip(lower=0.0)
            # Use half the theoretical best-first-4h fill improvement to avoid
            # treating every limit order as filled at the exact low.
            merged["eth_timing_benefit"] = eth_increase * merged["eth_timing_improvement"] * 0.5
            merged["eth_strategy_return"] = merged["eth_strategy_return"] + merged["eth_timing_benefit"]

    merged["gold_symbol_used"] = ""
    merged["gold_weight_exec"] = 0.0
    merged["gold_strategy_return"] = 0.0
    merged["gold_spot_return"] = 0.0
    if bool(args.include_gold):
        gold_px, gold_used = _download_gold_daily(args.start, args.end, args.gold_symbol)
        g = pd.DataFrame({"day": pd.to_datetime(merged["day"], errors="coerce")}).drop_duplicates().sort_values("day")
        g["day"] = g["day"].dt.tz_localize(None)
        gold_px = gold_px.copy()
        gold_px["day"] = pd.to_datetime(gold_px["day"], errors="coerce")
        if getattr(gold_px["day"].dt, "tz", None) is not None:
            gold_px["day"] = gold_px["day"].dt.tz_localize(None)
        g = g.merge(gold_px, on="day", how="left").sort_values("day")
        g["open"] = g["open"].ffill()
        g["close"] = g["close"].ffill()
        ge1, ge2, ge3 = tuple(args.gold_ema)
        g["gold_ema21"] = g["close"].ewm(span=ge1, adjust=False).mean()
        g["gold_ema55"] = g["close"].ewm(span=ge2, adjust=False).mean()
        g["gold_ema144"] = g["close"].ewm(span=ge3, adjust=False).mean()
        g["gold_aligned"] = (g["gold_ema21"] > g["gold_ema55"]) & (g["gold_ema55"] > g["gold_ema144"])
        g["gold_ret_exec"] = g["open"].pct_change().fillna(0.0)

        merged_tmp = merged.copy()
        merged_tmp["day_merge"] = pd.to_datetime(merged_tmp["day"], errors="coerce")
        if getattr(merged_tmp["day_merge"].dt, "tz", None) is not None:
            merged_tmp["day_merge"] = merged_tmp["day_merge"].dt.tz_localize(None)
        gm = merged_tmp.merge(g[["day", "gold_aligned", "gold_ret_exec"]], left_on="day_merge", right_on="day", how="left")
        gm = gm.drop(columns=["day_y", "day_merge"], errors="ignore").rename(columns={"day_x": "day"})
        gm["gold_aligned"] = gm["gold_aligned"].fillna(False)
        gm["gold_ret_exec"] = pd.to_numeric(gm["gold_ret_exec"], errors="coerce").fillna(0.0)
        idle_cap = (1.0 - (gm["alloc_eth"] + gm["alloc_btc"])).clip(lower=0.0)
        both_flat = (gm["eth_off_active"] <= 0) & (gm["btc_off_active"] <= 0)
        bear_gate = gm["eth_regime"].astype(str).eq("BEAR")
        gold_gate = both_flat & bear_gate & gm["gold_aligned"]
        gm["gold_weight_target"] = np.where(gold_gate, np.minimum(idle_cap, float(args.gold_cap)), 0.0)
        gm["gold_weight_exec"] = gm["gold_weight_target"].shift(1).fillna(0.0)
        gprev = gm["gold_weight_exec"].shift(1).fillna(0.0)
        if str(args.cost_mode).lower() == "entry_exit":
            crossed = ((gm["gold_weight_exec"] > 0).astype(int) != (gprev > 0).astype(int))
            gm["gold_turnover"] = np.where(crossed, (gm["gold_weight_exec"] - gprev).abs(), 0.0)
        else:
            gm["gold_turnover"] = (gm["gold_weight_exec"] - gprev).abs()
        gm["gold_cost"] = gm["gold_turnover"] * (float(args.gold_cost_bps) / 10000.0)
        gm["gold_strategy_return"] = gm["gold_weight_exec"] * gm["gold_ret_exec"] - gm["gold_cost"]
        gm["gold_spot_return"] = gm["gold_ret_exec"]
        gm["gold_symbol_used"] = gold_used
        merged = gm

    merged["combined_return"] = (
        merged["alloc_eth"] * merged["eth_strategy_return"]
        + merged["alloc_btc"] * merged["btc_strategy_return"]
        + merged["gold_strategy_return"]
        - merged["alloc_turnover_cost"]
    )
    merged["combined_spot_return"] = 0.5 * merged["eth_spot_return"] + 0.5 * merged["btc_spot_return"]
    merged["combined_eq"] = (1.0 + merged["combined_return"]).cumprod()
    merged["combined_spot_eq"] = (1.0 + merged["combined_spot_return"]).cumprod()
    merged["combined_time_in_market"] = ((merged["eth_weight_exec"] > 0).astype(int) + (merged["btc_weight_exec"] > 0).astype(int)) / 2.0

    m_eth = perf(merged["eth_strategy_return"])
    m_btc = perf(merged["btc_strategy_return"])
    m_comb = perf(merged["combined_return"])
    eth_trades = int(((merged["eth_weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())
    btc_trades = int(((merged["btc_weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())

    corr_strategy = float(merged["eth_strategy_return"].corr(merged["btc_strategy_return"]))
    corr_spot = float(merged["eth_spot_return"].corr(merged["btc_spot_return"]))
    macro_bear_days = int(merged["sp_regime"].astype(str).eq("BEAR").sum())
    macro_active_days = int((merged["sp_regime"].astype(str).eq("BEAR") & ((merged["eth_weight_exec"] > 0) | (merged["btc_weight_exec"] > 0))).sum())
    high_vol_days = int((pd.to_numeric(merged["vol_percentile"], errors="coerce") > 0.75).sum())
    low_vol_days = int((pd.to_numeric(merged["vol_percentile"], errors="coerce") < 0.25).sum())
    dd_mult = pd.to_numeric(merged["dd_multiplier"], errors="coerce").fillna(1.0)
    dd_days_075 = int((dd_mult == 0.75).sum())
    dd_days_050 = int((dd_mult == 0.50).sum())
    dd_days_025 = int((dd_mult == 0.25).sum())
    avg_dd_multiplier = float(dd_mult.mean())
    eth_cross_removed = int(pd.to_numeric(merged["eth_cross_trades_removed"], errors="coerce").fillna(0).max())
    btc_cross_removed = int(pd.to_numeric(merged["btc_cross_trades_removed"], errors="coerce").fillna(0).max())
    eth_cross_delayed = int(pd.to_numeric(merged["eth_cross_trades_delayed"], errors="coerce").fillna(0).max())
    btc_cross_delayed = int(pd.to_numeric(merged["btc_cross_trades_delayed"], errors="coerce").fillna(0).max())
    eth_stops = int(pd.to_numeric(merged["eth_stop_loss_exit"], errors="coerce").fillna(0).sum())
    btc_stops = int(pd.to_numeric(merged["btc_stop_loss_exit"], errors="coerce").fillna(0).sum())
    eth_whips = int(pd.to_numeric(merged["eth_stop_loss_whipsaw"], errors="coerce").fillna(0).sum())
    btc_whips = int(pd.to_numeric(merged["btc_stop_loss_whipsaw"], errors="coerce").fillna(0).sum())
    strong_transitions = int(
        ((merged["eth_transition_class"].astype(str) == "STRONG") & (merged["eth_weight_exec"] > 0)).sum()
        + ((merged["btc_transition_class"].astype(str) == "STRONG") & (merged["btc_weight_exec"] > 0)).sum()
    )
    weak_transitions = int(
        ((merged["eth_transition_class"].astype(str) == "WEAK") & (merged["eth_weight_exec"] > 0)).sum()
        + ((merged["btc_transition_class"].astype(str) == "WEAK") & (merged["btc_weight_exec"] > 0)).sum()
    )
    timing_entry_days = int((pd.to_numeric(merged["eth_timing_benefit"], errors="coerce").fillna(0.0) > 0).sum())
    avg_timing_bps = float(pd.to_numeric(merged.loc[merged["eth_timing_benefit"] > 0, "eth_timing_improvement"], errors="coerce").mean() * 10000.0) if timing_entry_days else 0.0
    annual_timing_impact = float(pd.to_numeric(merged["eth_timing_benefit"], errors="coerce").fillna(0.0).sum() / max(len(merged) / 252.0, 1e-9))

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_daily, index=False)

    summary = pd.DataFrame(
        [
            {
                "start": str(pd.to_datetime(merged["timestamp"].iloc[0]).date()),
                "end": str(pd.to_datetime(merged["timestamp"].iloc[-1]).date()),
                "eth_confirm_days": int(args.eth_confirm_days),
                "btc_confirm_days": int(args.btc_confirm_days),
                "cost_bps": float(args.cost_bps),
                "cost_mode": str(args.cost_mode),
                "allocation_mode": str(args.allocation_mode),
                "gross_cap": float(args.gross_cap),
                "derisking": bool(args.derisking),
                "pup_fallback": bool(args.pup_fallback),
                "macro_filter": bool(args.macro_filter),
                "stop_loss": bool(args.stop_loss),
                "stop_loss_pct": float(args.stop_loss_pct),
                "vol_filter": bool(args.vol_filter),
                "transition_momentum": bool(args.transition_momentum),
                "cross_confirm": str(args.cross_confirm),
                "dd_aware": str(args.dd_aware),
                "asymmetric_sizing": bool(args.asymmetric_sizing),
                "eth_ema": "/".join(str(x) for x in tuple(args.eth_ema)),
                "btc_ema": "/".join(str(x) for x in tuple(args.btc_ema)),
                "intraday_timing": bool(args.intraday_timing),
                "include_gold": bool(args.include_gold),
                "gold_symbol": str(merged["gold_symbol_used"].iloc[-1]) if bool(args.include_gold) else "",
                "gold_ema": "/".join(str(x) for x in tuple(args.gold_ema)) if bool(args.include_gold) else "",
                "gold_cap": float(args.gold_cap),
                "regime_source": str(args.regime_source),
                "hmm_regime_csv": str(args.hmm_regime_csv),
                "combined_cagr": m_comb["cagr"],
                "combined_sharpe": m_comb["sharpe"],
                "combined_max_dd": m_comb["max_dd"],
                "combined_ann_vol": m_comb["ann_vol"],
                "combined_time_in_market_pct": float(merged["combined_time_in_market"].mean() * 100.0),
                "eth_cagr": m_eth["cagr"],
                "eth_sharpe": m_eth["sharpe"],
                "eth_max_dd": m_eth["max_dd"],
                "eth_trades": eth_trades,
                "btc_cagr": m_btc["cagr"],
                "btc_sharpe": m_btc["sharpe"],
                "btc_max_dd": m_btc["max_dd"],
                "btc_trades": btc_trades,
                "corr_strategy_returns": corr_strategy,
                "corr_spot_returns": corr_spot,
                "gold_time_in_market_pct": float((merged["gold_weight_exec"] > 0).mean() * 100.0),
                "derisk_days": int(((merged["eth_derisk_multiplier"] < 1.0) | (merged["btc_derisk_multiplier"] < 1.0)).sum()),
                "avg_derisk_multiplier_when_fired": float(
                    pd.concat(
                        [
                            merged.loc[merged["eth_derisk_multiplier"] < 1.0, "eth_derisk_multiplier"],
                            merged.loc[merged["btc_derisk_multiplier"] < 1.0, "btc_derisk_multiplier"],
                        ]
                    ).mean()
                )
                if int(((merged["eth_derisk_multiplier"] < 1.0) | (merged["btc_derisk_multiplier"] < 1.0)).sum()) > 0
                else 1.0,
                "pup_fallback_days_replaced": int((merged["eth_pup_fallback_used"] | merged["btc_pup_fallback_used"]).sum()),
                "macro_bear_days": macro_bear_days,
                "macro_active_days_affected": macro_active_days,
                "high_vol_days": high_vol_days,
                "low_vol_days": low_vol_days,
                "dd_days_075": dd_days_075,
                "dd_days_050": dd_days_050,
                "dd_days_025": dd_days_025,
                "avg_dd_multiplier": avg_dd_multiplier,
                "eth_cross_trades_removed": eth_cross_removed,
                "btc_cross_trades_removed": btc_cross_removed,
                "eth_cross_trades_delayed": eth_cross_delayed,
                "btc_cross_trades_delayed": btc_cross_delayed,
                "avg_eth_asym_multiplier": float(pd.to_numeric(merged["asym_eth_multiplier"], errors="coerce").mean()),
                "avg_btc_asym_multiplier": float(pd.to_numeric(merged["asym_btc_multiplier"], errors="coerce").mean()),
                "stop_loss_total": eth_stops + btc_stops,
                "stop_loss_whipsaws": eth_whips + btc_whips,
                "stop_loss_whipsaw_pct": (eth_whips + btc_whips) / (eth_stops + btc_stops) if (eth_stops + btc_stops) else 0.0,
                "strong_transition_active_days": strong_transitions,
                "weak_transition_active_days": weak_transitions,
                "intraday_timing_entry_days": timing_entry_days,
                "avg_intraday_improvement_bps": avg_timing_bps,
                "annual_intraday_timing_impact": annual_timing_impact,
            }
        ]
    )

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print("===============================================")
    print("ETH+BTC PORTFOLIO vs STANDALONE")
    print("===============================================")
    print("Metric        ETH only  BTC only  Combined")
    print("-----------------------------------------------")
    print(f"CAGR          {m_eth['cagr']*100:>6.2f}%   {m_btc['cagr']*100:>6.2f}%   {m_comb['cagr']*100:>6.2f}%")
    print(f"Sharpe        {m_eth['sharpe']:>6.3f}   {m_btc['sharpe']:>6.3f}   {m_comb['sharpe']:>6.3f}")
    print(f"MaxDD         {m_eth['max_dd']*100:>6.2f}%  {m_btc['max_dd']*100:>6.2f}%  {m_comb['max_dd']*100:>6.2f}%")
    print("-----------------------------------------------")
    print(f"Combined ann vol:      {m_comb['ann_vol']*100:.2f}%")
    print(f"Combined time in mkt:  {float(merged['combined_time_in_market'].mean()*100.0):.2f}%")
    print(f"ETH trades: {eth_trades} | BTC trades: {btc_trades}")
    print(f"Corr strategy returns: {corr_strategy:.3f}")
    print(f"Corr spot returns:     {corr_spot:.3f}")
    print(f"Cost mode:             {args.cost_mode}")
    print(f"Allocation mode:       {args.allocation_mode} (gross cap={float(args.gross_cap):.2f})")
    print(f"Derisking:             {'ON' if bool(args.derisking) else 'OFF'}")
    print(f"P(up) fallback:        {'ON' if bool(args.pup_fallback) else 'OFF'}")
    print(f"Macro filter:          {'ON' if bool(args.macro_filter) else 'OFF'} | SP BEAR days={macro_bear_days} active affected={macro_active_days}")
    print(f"Stop loss:             {'ON' if bool(args.stop_loss) else 'OFF'} | stops={eth_stops + btc_stops} whipsaws={eth_whips + btc_whips}")
    print(f"Vol filter:            {'ON' if bool(args.vol_filter) else 'OFF'} | high={high_vol_days} low={low_vol_days}")
    print(f"Transition momentum:   {'ON' if bool(args.transition_momentum) else 'OFF'} | strong={strong_transitions} weak={weak_transitions}")
    print(f"Intraday timing:       {'ON' if bool(args.intraday_timing) else 'OFF'} | entries={timing_entry_days} avg improve={avg_timing_bps:.1f}bps")
    print(f"Derisk days:           {int(((merged['eth_derisk_multiplier'] < 1.0) | (merged['btc_derisk_multiplier'] < 1.0)).sum())}")
    print(f"P(up) fallback days:   {int((merged['eth_pup_fallback_used'] | merged['btc_pup_fallback_used']).sum())}")
    if bool(args.include_gold):
        print(f"Gold sleeve:           ON ({str(merged['gold_symbol_used'].iloc[-1])}) cap={float(args.gold_cap):.2f}")
        print(f"Gold time in market:   {float((merged['gold_weight_exec']>0).mean()*100.0):.2f}%")
    else:
        print("Gold sleeve:           OFF")
    print("===============================================")
    print(f"Saved: {args.out_summary_csv}")
    print(f"Saved: {args.out_daily_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
