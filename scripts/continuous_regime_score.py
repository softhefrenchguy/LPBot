from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["risk_on", "weakening", "risk_off", "panic"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Research-only BTC continuous regime score. This does not touch live logic. "
            "It converts BTC daily trend/momentum/drawdown/vol primitives into four states."
        )
    )
    p.add_argument("--btc-bitstamp", default="data/BTCUSD_bitstamp_daily.csv")
    p.add_argument("--btc-modern", default="data/btc_daily_extended.csv")
    p.add_argument("--eth-modern", default="data/eth_daily_extended.csv")
    p.add_argument("--out-daily", default="artifacts/backtest/continuous_regime_score_daily.csv")
    p.add_argument("--out-state-stats", default="artifacts/backtest/continuous_regime_score_state_stats.csv")
    p.add_argument("--out-spot-checks", default="artifacts/backtest/continuous_regime_score_spot_checks.csv")
    p.add_argument("--out-summary", default="artifacts/backtest/continuous_regime_score_summary.csv")
    return p.parse_args()


def _read_ohlcv(path: Path, prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    ts_col = next((c for c in ["timestamp", "date", "day", "datetime"] if c in d.columns), None)
    if ts_col is None or "close" not in d.columns:
        raise ValueError(f"{path} must contain timestamp/date and close columns")
    d["date"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce").dt.floor("D")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    out = d.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    return out[["date", "close"]].rename(columns={"close": f"{prefix}_close"})


def _splice_btc(bitstamp: pd.DataFrame, modern: pd.DataFrame) -> pd.DataFrame:
    b = bitstamp.copy()
    m = modern.copy()
    overlap = b.merge(m, on="date", how="inner", suffixes=("_bitstamp", "_modern"))
    overlap = overlap.dropna(subset=["btc_close_bitstamp", "btc_close_modern"])
    factor = 1.0
    if not overlap.empty:
        ratios = overlap["btc_close_modern"] / overlap["btc_close_bitstamp"]
        factor = float(ratios.replace([np.inf, -np.inf], np.nan).dropna().median())
    b["btc_close"] = b["btc_close"] * factor
    cutoff = m["date"].min()
    out = pd.concat([b[b["date"] < cutoff], m], ignore_index=True)
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    def pct(x: np.ndarray) -> float:
        last = x[-1]
        valid = x[np.isfinite(x)]
        if len(valid) < max(20, window // 4) or not np.isfinite(last):
            return np.nan
        return float((valid <= last).mean())

    return s.rolling(window, min_periods=max(20, window // 4)).apply(pct, raw=True)


def _compute_states(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    px = d["btc_close"]
    d["btc_ret_1d"] = px.pct_change()
    d["btc_ret_5d"] = px.pct_change(5)
    d["btc_ret_10d"] = px.pct_change(10)
    d["btc_ret_20d"] = px.pct_change(20)
    d["btc_ema50"] = px.ewm(span=50, adjust=False).mean()
    d["btc_ema120"] = px.ewm(span=120, adjust=False).mean()
    d["btc_ema300"] = px.ewm(span=300, adjust=False).mean()
    d["btc_dd_20d"] = px / px.rolling(20, min_periods=10).max() - 1.0
    d["btc_dd_from_ath"] = px / px.cummax() - 1.0
    d["btc_realized_vol_20d"] = d["btc_ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(365)
    d["btc_vol_pctile_365d"] = _percentile_rank(d["btc_realized_vol_20d"], 365)

    trend_score = pd.Series(0, index=d.index, dtype=float)
    trend_score += np.where((px > d["btc_ema50"]) & (d["btc_ema50"] > d["btc_ema120"]) & (d["btc_ema120"] > d["btc_ema300"]), 2, 0)
    trend_score += np.where((px > d["btc_ema120"]) & (d["btc_ret_20d"] > 0), 1, 0)
    trend_score -= np.where((px < d["btc_ema50"]) & (d["btc_ema50"] < d["btc_ema120"]), 2, 0)
    trend_score -= np.where((px < d["btc_ema120"]) & (d["btc_ret_20d"] < 0), 1, 0)
    trend_score -= np.where(d["btc_dd_from_ath"] < -0.35, 1, 0)
    d["regime_score"] = trend_score

    panic = (
        (d["btc_dd_20d"] <= -0.25)
        | (d["btc_ret_5d"] <= -0.18)
        | ((d["btc_vol_pctile_365d"] >= 0.90) & (d["btc_ret_10d"] <= -0.12))
        | ((d["btc_dd_from_ath"] <= -0.50) & (d["btc_ret_20d"] <= -0.20))
    )
    risk_off = (
        (d["regime_score"] <= -3)
        | (d["btc_dd_20d"] <= -0.12)
        | ((d["btc_dd_from_ath"] <= -0.35) & (d["btc_ret_20d"] < 0) & (px < d["btc_ema120"]))
    )
    risk_on = (
        (d["regime_score"] >= 2)
        & (px > d["btc_ema50"])
        & (d["btc_ema50"] > d["btc_ema120"])
        & (d["btc_ret_20d"] > 0)
        & (d["btc_vol_pctile_365d"].fillna(0.5) < 0.90)
    )

    d["state"] = np.select([panic, risk_off, risk_on], ["panic", "risk_off", "risk_on"], default="weakening")

    for n in [7, 20, 60]:
        d[f"btc_fwd_{n}d_ret"] = px.pct_change(n).shift(-n)
        if "eth_close" in d.columns:
            d[f"eth_fwd_{n}d_ret"] = d["eth_close"].pct_change(n).shift(-n)
    return d


def _state_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATES:
        g = d[d["state"].eq(state)]
        row = {
            "state": state,
            "days": len(g),
            "pct_days": len(g) / len(d) if len(d) else np.nan,
            "avg_run_days": _avg_run_length(d["state"], state),
        }
        for asset in ["btc", "eth"]:
            for n in [7, 20, 60]:
                col = f"{asset}_fwd_{n}d_ret"
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce").dropna()
                    row[f"{asset}_avg_fwd_{n}d_ret"] = float(vals.mean()) if not vals.empty else np.nan
                    row[f"{asset}_positive_fwd_{n}d_rate"] = float((vals > 0).mean()) if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _avg_run_length(states: pd.Series, state: str) -> float:
    s = states.astype(str)
    mask = s.eq(state)
    if not mask.any():
        return np.nan
    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    lengths = mask[mask].groupby(run_id[mask]).size()
    return float(lengths.mean()) if not lengths.empty else np.nan


def _spot_checks(d: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("mt_gox_peak_crash", "2013-12-01", "2014-02-28", "panic/risk_off expected after initial break"),
        ("mt_gox_grind_lower", "2014-08-01", "2015-02-15", "risk_off/panic expected"),
        ("covid_crash", "2020-02-20", "2020-03-25", "panic expected"),
        ("ftx_crash", "2022-11-06", "2022-11-30", "panic/risk_off expected"),
        ("btc_2017_bull", "2017-01-01", "2017-12-17", "risk_on expected"),
        ("btc_2020_2021_bull", "2020-10-01", "2021-04-14", "risk_on expected"),
        ("btc_2023_2024_recovery", "2023-01-01", "2024-03-14", "risk_on expected"),
    ]
    rows = []
    for name, start, end, expectation in checks:
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        g = d[(d["date"] >= start_ts) & (d["date"] <= end_ts)]
        counts = g["state"].value_counts(normalize=True).reindex(STATES, fill_value=0.0)
        dominant = str(counts.idxmax()) if not g.empty else ""
        rows.append(
            {
                "check": name,
                "start": start,
                "end": end,
                "days": len(g),
                "dominant_state": dominant,
                "risk_on_pct": counts.get("risk_on", np.nan),
                "weakening_pct": counts.get("weakening", np.nan),
                "risk_off_pct": counts.get("risk_off", np.nan),
                "panic_pct": counts.get("panic", np.nan),
                "expectation": expectation,
                "btc_return": _window_return(g, "btc_close"),
                "eth_return": _window_return(g, "eth_close") if "eth_close" in g.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _window_return(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return np.nan
    px = pd.to_numeric(g[col], errors="coerce").dropna()
    if len(px) < 2 or px.iloc[0] <= 0:
        return np.nan
    return float(px.iloc[-1] / px.iloc[0] - 1.0)


def _summary(d: pd.DataFrame) -> pd.DataFrame:
    run_id = d["state"].ne(d["state"].shift()).cumsum()
    runs = d.groupby(run_id).agg(state=("state", "first"), length=("state", "size"))
    avg_persist = float(runs["length"].mean()) if not runs.empty else np.nan
    return pd.DataFrame(
        [
            {
                "start": d["date"].min().date().isoformat(),
                "end": d["date"].max().date().isoformat(),
                "days": len(d),
                "state_runs": len(runs),
                "avg_regime_persistence_days": avg_persist,
                "effective_independent_observations_approx": len(d) / avg_persist if avg_persist and np.isfinite(avg_persist) else np.nan,
            }
        ]
    )


def main() -> int:
    args = _parse_args()
    bitstamp = _read_ohlcv(Path(args.btc_bitstamp), "btc")
    modern = _read_ohlcv(Path(args.btc_modern), "btc")
    btc = _splice_btc(bitstamp, modern)
    eth = _read_ohlcv(Path(args.eth_modern), "eth")
    d = btc.merge(eth, on="date", how="left")
    d = _compute_states(d)

    daily_path = Path(args.out_daily)
    stats_path = Path(args.out_state_stats)
    checks_path = Path(args.out_spot_checks)
    summary_path = Path(args.out_summary)
    for path in [daily_path, stats_path, checks_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    state_stats = _state_stats(d)
    spot_checks = _spot_checks(d)
    summary = _summary(d)
    d.to_csv(daily_path, index=False)
    state_stats.to_csv(stats_path, index=False)
    spot_checks.to_csv(checks_path, index=False)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 220)
    print("=" * 110)
    print("CONTINUOUS BTC REGIME SCORE")
    print("=" * 110)
    print(f"Saved daily: {daily_path}")
    print(f"Saved state stats: {stats_path}")
    print(f"Saved spot checks: {checks_path}")
    print(f"Saved summary: {summary_path}")
    print()
    print(summary.to_string(index=False))
    print()
    print("State distribution and forward returns:")
    print(state_stats.to_string(index=False))
    print()
    print("Historical spot checks:")
    print(spot_checks.to_string(index=False))
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
