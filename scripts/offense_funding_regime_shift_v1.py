from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def perf_from_log_returns(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bars_per_year)),
        "sharpe": float((np.mean(x) * bars_per_year) / (np.std(x) * np.sqrt(bars_per_year) + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def clamp_series(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(lower=lo, upper=hi)


def load_regime_daily(path: str, regime_col: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["day", "regime"])
    r = pd.read_csv(path)
    if regime_col not in r.columns:
        raise ValueError(f"missing {regime_col} in {path}")
    if "day" in r.columns:
        r["day"] = pd.to_datetime(r["day"], utc=True, errors="coerce").dt.floor("D")
    elif "timestamp" in r.columns:
        r["day"] = pd.to_datetime(r["timestamp"], utc=True, errors="coerce").dt.floor("D")
    else:
        raise ValueError("regime csv must contain day or timestamp")
    out = r.dropna(subset=["day"])[["day", regime_col]].drop_duplicates(subset=["day"], keep="last")
    return out.rename(columns={regime_col: "regime"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Funding regime-shift offensive sleeve (daily logic, 5m execution).")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--regime-col", default="regime_v2")
    ap.add_argument("--smooth-window", type=int, default=3, help="Funding smoothing window (days).")
    ap.add_argument("--hold-days-min", type=int, default=3, help="Funding must stay positive this many days to confirm entry.")
    ap.add_argument("--hold-days-max", type=int, default=20, help="Maximum trade holding days.")
    ap.add_argument("--funding-threshold", type=float, default=0.0, help="Positive funding threshold for entry state.")
    ap.add_argument("--exit-funding-threshold", type=float, default=-0.0001, help="Exit if smoothed funding drops below this threshold.")
    ap.add_argument("--stop-loss", type=float, default=-0.08, help="Exit if return from entry close drops below this value.")
    ap.add_argument("--funding-cap-abs", type=float, default=0.001, help="Cap absolute funding to reduce outlier impact (0.001 = 0.1%).")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/backtest/offense_funding_regime_shift_v1_400d.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/offense_funding_regime_shift_v1_400d_summary.csv")
    ap.add_argument("--out-signals-csv", default="artifacts/backtest/offense_funding_regime_shift_v1_400d_signals.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/offense_funding_regime_shift_v1_400d.html")
    args = ap.parse_args()

    price = pd.read_csv(args.price_csv, low_memory=False)
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, errors="coerce")
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    price = price[["timestamp", "close"]].copy()

    perp = pd.read_csv(args.perp_csv, low_memory=False)
    perp["timestamp"] = pd.to_datetime(perp["timestamp"], utc=True, errors="coerce")
    perp["funding_rate"] = pd.to_numeric(perp.get("funding_rate"), errors="coerce")
    perp["basis"] = pd.to_numeric(perp.get("basis"), errors="coerce")
    perp = perp.dropna(subset=["timestamp"]).sort_values("timestamp")
    perp = perp[["timestamp", "funding_rate", "basis"]].copy()

    df = pd.merge(price, perp, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No overlapping rows after merging price and perp features.")

    # Regularize perp features before daily aggregation.
    df["funding_rate"] = clamp_series(df["funding_rate"], -abs(args.funding_cap_abs), abs(args.funding_cap_abs)).ffill().fillna(0.0)
    df["basis"] = pd.to_numeric(df["basis"], errors="coerce").ffill().fillna(0.0)
    df["r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["day"] = df["timestamp"].dt.floor("D")

    daily = (
        df.set_index("timestamp")
        .resample("1D")
        .agg(
            close=("close", "last"),
            funding_rate=("funding_rate", "mean"),
            basis=("basis", "mean"),
        )
        .dropna(subset=["close"])
        .copy()
    )
    daily["funding_rate"] = clamp_series(daily["funding_rate"], -abs(args.funding_cap_abs), abs(args.funding_cap_abs))
    daily["funding_smooth"] = daily["funding_rate"].rolling(int(args.smooth_window), min_periods=1).mean()

    pos_state = daily["funding_smooth"] >= float(args.funding_threshold)
    cross_up = (~pos_state.shift(1).fillna(False)) & pos_state
    hold_positive = pos_state.rolling(int(args.hold_days_min), min_periods=int(args.hold_days_min)).sum() == int(args.hold_days_min)
    recent_cross = cross_up.rolling(int(args.hold_days_min), min_periods=1).max().fillna(0).astype(bool)
    daily["entry_setup"] = (hold_positive & recent_cross).astype(int)

    regime_daily = load_regime_daily(args.regime_csv, args.regime_col)
    if len(regime_daily) > 0:
        daily = daily.reset_index().rename(columns={"timestamp": "day"}).merge(regime_daily, on="day", how="left").set_index("day")
    else:
        daily = daily.reset_index().rename(columns={"timestamp": "day"}).set_index("day")
        daily["regime"] = "NA"
    daily["regime"] = daily["regime"].fillna("NA")

    # State machine: position is decided at day close and applied next day.
    pos_next = 0
    entry_close = np.nan
    days_in_trade = 0
    open_trade_start = pd.NaT
    trade_durations: list[int] = []
    entry_rows: list[dict[str, float | str]] = []
    exit_rows: list[dict[str, float | str]] = []
    position_for_day: list[int] = []

    for day, row in daily.iterrows():
        pos_today = int(pos_next)
        position_for_day.append(pos_today)

        if pos_today == 1:
            days_in_trade += 1
            ret_from_entry = float(row["close"] / entry_close - 1.0) if np.isfinite(entry_close) else 0.0
            exit_funding = float(row["funding_smooth"]) < float(args.exit_funding_threshold)
            exit_stop = ret_from_entry <= float(args.stop_loss)
            exit_time = days_in_trade >= int(args.hold_days_max)
            if exit_funding or exit_stop or exit_time:
                pos_next = 0
                exit_rows.append(
                    {
                        "day": str(day),
                        "funding_smooth": float(row["funding_smooth"]),
                        "exit_reason": "funding" if exit_funding else ("stop" if exit_stop else "time"),
                        "ret_from_entry": ret_from_entry,
                        "hold_days": int(days_in_trade),
                        "regime": str(row["regime"]),
                    }
                )
                if pd.notna(open_trade_start):
                    trade_durations.append(int((day - open_trade_start).days) + 1)
                days_in_trade = 0
                open_trade_start = pd.NaT
        else:
            if int(row["entry_setup"]) == 1:
                pos_next = 1
                entry_close = float(row["close"])
                open_trade_start = day
                days_in_trade = 0
                entry_rows.append(
                    {
                        "day": str(day),
                        "funding_smooth": float(row["funding_smooth"]),
                        "funding_rate": float(row["funding_rate"]),
                        "basis": float(row["basis"]),
                        "regime": str(row["regime"]),
                    }
                )

    # Add open trade duration if still open at end.
    if pd.notna(open_trade_start):
        trade_durations.append(int((daily.index[-1] - open_trade_start).days) + 1)

    daily["position_day"] = position_for_day
    daily["weight_exec"] = daily["position_day"].shift(1).fillna(0.0)

    intraday = df.merge(daily.reset_index()[["day", "weight_exec", "funding_smooth", "entry_setup", "regime"]], on="day", how="left")
    intraday["weight"] = pd.to_numeric(intraday["weight_exec"], errors="coerce").ffill().fillna(0.0)
    intraday["turnover"] = intraday["weight"].diff().abs().fillna(0.0)
    cost = intraday["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    intraday["strat_r"] = intraday["weight"].shift(1).fillna(0.0) * intraday["r"] - cost
    intraday["spot_r"] = intraday["r"]
    intraday["eq"] = np.exp(np.cumsum(intraday["strat_r"].to_numpy(dtype=float)))
    intraday["spot_eq"] = np.exp(np.cumsum(intraday["spot_r"].to_numpy(dtype=float)))
    intraday["eth_close"] = intraday["close"]

    strat_m = perf_from_log_returns(intraday["strat_r"], 5)
    spot_m = perf_from_log_returns(intraday["spot_r"], 5)

    entry_df = pd.DataFrame(entry_rows)
    exit_df = pd.DataFrame(exit_rows)

    regime_counts = {"BULL": 0.0, "CHOP": 0.0, "BEAR": 0.0}
    if len(entry_df) > 0 and "regime" in entry_df.columns:
        e_pct = (entry_df["regime"].value_counts(normalize=True) * 100.0).to_dict()
        for k in regime_counts:
            regime_counts[k] = float(e_pct.get(k, 0.0))

    avg_hold = float(np.mean(trade_durations)) if trade_durations else np.nan
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(intraday)),
                "start": str(intraday["timestamp"].iloc[0]),
                "end": str(intraday["timestamp"].iloc[-1]),
                "ret": strat_m["ret"],
                "cagr": strat_m["cagr"],
                "ann_vol": strat_m["ann_vol"],
                "sharpe": strat_m["sharpe"],
                "max_dd": strat_m["max_dd"],
                "spot_ret": spot_m["ret"],
                "excess_vs_spot": strat_m["ret"] - spot_m["ret"],
                "time_in_market_pct": float((intraday["weight"] > 0).mean() * 100.0),
                "avg_weight": float(intraday["weight"].mean()),
                "turnover": float(intraday["turnover"].sum()),
                "total_entries": int(len(entry_df)),
                "avg_hold_days": avg_hold,
                "entry_regime_bull_pct": regime_counts["BULL"],
                "entry_regime_chop_pct": regime_counts["CHOP"],
                "entry_regime_bear_pct": regime_counts["BEAR"],
                "entry_funding_mean": float(entry_df["funding_smooth"].mean()) if len(entry_df) else np.nan,
                "entry_funding_median": float(entry_df["funding_smooth"].median()) if len(entry_df) else np.nan,
                "exit_funding_mean": float(exit_df["funding_smooth"].mean()) if len(exit_df) else np.nan,
                "exit_funding_median": float(exit_df["funding_smooth"].median()) if len(exit_df) else np.nan,
            }
        ]
    )

    signals = daily.reset_index()[
        [
            "day",
            "close",
            "funding_rate",
            "funding_smooth",
            "basis",
            "entry_setup",
            "position_day",
            "weight_exec",
            "regime",
        ]
    ].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    intraday.to_csv(out_csv, index=False)

    out_sum = Path(args.out_summary_csv)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)

    out_sig = Path(args.out_signals_csv)
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_sig, index=False)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity", "Weight", "Price", "Funding Smooth"],
    )
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["close"], name="ETH Close", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=signals["day"], y=signals["funding_smooth"], name="Funding Smooth", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_hline(y=float(args.funding_threshold), line_dash="dash", line_color="gray", row=4, col=1)
    fig.add_hline(y=float(args.exit_funding_threshold), line_dash="dot", line_color="red", row=4, col=1)
    fig.update_layout(height=1200, title="Offense Funding Regime Shift v1")

    html = (
        "<html><head><meta charset='utf-8'><title>Offense Funding Regime Shift v1</title></head><body>"
        "<h3>Offense Funding Regime Shift v1</h3>"
        + summary.round(6).to_html(index=False, border=0)
        + fig.to_html(full_html=False, include_plotlyjs="cdn")
        + "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_sum}")
    print(f"wrote {out_sig}")
    print(f"wrote {out_html}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
