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


def holding_periods_from_weight(weight_daily: pd.Series) -> list[int]:
    on = (pd.to_numeric(weight_daily, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0)
    if len(on) == 0:
        return []
    prev_on = np.roll(on, 1)
    prev_on[0] = False
    next_on = np.roll(on, -1)
    next_on[-1] = False
    starts = on & (~prev_on)
    ends = on & (~next_on)
    s_idx = list(np.where(starts)[0])
    e_idx = list(np.where(ends)[0])
    out: list[int] = []
    j = 0
    for s in s_idx:
        while j < len(e_idx) and e_idx[j] < s:
            j += 1
        if j < len(e_idx):
            out.append(int(e_idx[j] - s + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offense trend-following v1 (triple EMA + vol sizing).")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--regime-col", default="regime_v2")
    ap.add_argument("--ema-fast", type=int, default=8)
    ap.add_argument("--ema-mid", type=int, default=21)
    ap.add_argument("--ema-slow", type=int, default=55)
    ap.add_argument("--confirm-days", type=int, default=3)
    ap.add_argument("--exit-confirm-days", type=int, default=1, help="Require trend break for N consecutive days before exit.")
    ap.add_argument("--min-hold-days", type=int, default=0, help="Minimum hold days after entry before exits are allowed.")
    ap.add_argument("--trailing-stop-pct", type=float, default=np.nan, help="Optional trailing stop from in-hold high (e.g. -0.15).")
    ap.add_argument("--stop-overrides-min-hold", action="store_true", help="If set, trailing stop can exit before min-hold-days.")
    ap.add_argument("--target-vol", type=float, default=0.50)
    ap.add_argument("--vol-floor", type=float, default=0.25)
    ap.add_argument("--vol-cap", type=float, default=1.00)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/backtest/offense_trend_follow_v1_6y.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/offense_trend_follow_v1_6y_summary.csv")
    ap.add_argument("--out-signals-csv", default="artifacts/backtest/offense_trend_follow_v1_6y_signals.csv")
    ap.add_argument("--out-diagnostics-csv", default="artifacts/backtest/offense_trend_follow_v1_6y_diagnostics.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/offense_trend_follow_v1_6y.html")
    args = ap.parse_args()

    px = pd.read_csv(args.price_csv, low_memory=False)
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    px["r"] = np.log(px["close"] / px["close"].shift(1)).fillna(0.0)
    px["day"] = px["timestamp"].dt.floor("D")

    daily = (
        px.set_index("timestamp")
        .resample("1D")
        .agg(close=("close", "last"), r=("r", "sum"))
        .dropna(subset=["close"])
        .copy()
    )
    daily["ema_fast"] = daily["close"].ewm(span=int(args.ema_fast), adjust=False).mean()
    daily["ema_mid"] = daily["close"].ewm(span=int(args.ema_mid), adjust=False).mean()
    daily["ema_slow"] = daily["close"].ewm(span=int(args.ema_slow), adjust=False).mean()
    daily["trend_aligned"] = (daily["ema_fast"] > daily["ema_mid"]) & (daily["ema_mid"] > daily["ema_slow"])
    daily["trend_confirmed"] = daily["trend_aligned"].rolling(int(args.confirm_days), min_periods=int(args.confirm_days)).min() == 1
    daily["trend_broken"] = ~daily["trend_aligned"]
    daily["exit_confirmed_signal"] = (
        daily["trend_broken"].rolling(int(args.exit_confirm_days), min_periods=int(args.exit_confirm_days)).min() == 1
    )

    rv = daily["r"].rolling(int(args.vol_window), min_periods=int(max(5, args.vol_window // 2))).std() * np.sqrt(365.0)
    rv = rv.replace([np.inf, -np.inf], np.nan)
    daily["realized_vol"] = rv
    scalar = float(args.target_vol) / rv
    scalar = scalar.replace([np.inf, -np.inf], np.nan)
    daily["vol_scalar"] = scalar.clip(lower=float(args.vol_floor), upper=float(args.vol_cap))
    daily["vol_scalar"] = daily["vol_scalar"].fillna(0.0)

    # State machine: entry uses trend_confirmed, exit uses exit_confirmed_signal (+ optional min-hold).
    pos_next = 0
    days_in_pos = 0
    hold_high = np.nan
    pos_series: list[int] = []
    days_in_pos_series: list[int] = []
    hold_high_series: list[float] = []
    trailing_dd_series: list[float] = []
    trailing_stop_hit_series: list[int] = []
    for _, row in daily.iterrows():
        pos_today = int(pos_next)
        pos_series.append(pos_today)
        if pos_today == 1:
            days_in_pos += 1
            if not np.isfinite(hold_high):
                hold_high = float(row["close"])
            hold_high = max(float(hold_high), float(row["close"]))
            trailing_dd = float(row["close"] / hold_high - 1.0) if np.isfinite(hold_high) and hold_high > 0 else np.nan
        else:
            days_in_pos = 0
            hold_high = np.nan
            trailing_dd = np.nan
        days_in_pos_series.append(int(days_in_pos))
        hold_high_series.append(float(hold_high) if np.isfinite(hold_high) else np.nan)
        trailing_dd_series.append(trailing_dd)

        trailing_stop_hit = False
        if pos_today == 1 and np.isfinite(args.trailing_stop_pct):
            trailing_stop_hit = np.isfinite(trailing_dd) and (float(trailing_dd) <= float(args.trailing_stop_pct))
        trailing_stop_hit_series.append(int(trailing_stop_hit))

        if pos_today == 0:
            if bool(row["trend_confirmed"]):
                pos_next = 1
                hold_high = float(row["close"])
        else:
            can_exit = days_in_pos >= int(args.min_hold_days)
            exit_by_trend = bool(row["exit_confirmed_signal"]) and can_exit
            if bool(args.stop_overrides_min_hold):
                exit_by_stop = trailing_stop_hit
            else:
                exit_by_stop = trailing_stop_hit and can_exit
            if exit_by_trend or exit_by_stop:
                pos_next = 0
                hold_high = np.nan

    daily["position_state"] = pd.Series(pos_series, index=daily.index).astype(int)
    daily["days_in_pos"] = pd.Series(days_in_pos_series, index=daily.index).astype(int)
    daily["hold_high"] = pd.Series(hold_high_series, index=daily.index)
    daily["trailing_dd"] = pd.Series(trailing_dd_series, index=daily.index)
    daily["trailing_stop_hit"] = pd.Series(trailing_stop_hit_series, index=daily.index).astype(int)
    daily["weight_daily"] = daily["position_state"].astype(float) * daily["vol_scalar"]
    daily["weight_exec"] = daily["weight_daily"].shift(1).fillna(0.0)
    daily["weight_exec"] = daily["weight_exec"].clip(lower=0.0, upper=1.0)

    reg = load_regime_daily(args.regime_csv, args.regime_col)
    daily2 = daily.reset_index().rename(columns={"timestamp": "day"}).merge(reg, on="day", how="left")
    daily2["regime"] = daily2["regime"].fillna("NA")

    run = px.merge(
        daily2[
            [
                "day",
                "ema_fast",
                "ema_mid",
                "ema_slow",
                "trend_aligned",
                "trend_confirmed",
                "trend_broken",
                "exit_confirmed_signal",
                "realized_vol",
                "vol_scalar",
                "position_state",
                "days_in_pos",
                "hold_high",
                "trailing_dd",
                "trailing_stop_hit",
                "weight_daily",
                "weight_exec",
                "regime",
            ]
        ],
        on="day",
        how="left",
    )
    run["weight"] = pd.to_numeric(run["weight_exec"], errors="coerce").ffill().fillna(0.0).clip(lower=0.0, upper=1.0)
    run["turnover"] = run["weight"].diff().abs().fillna(0.0)
    cost = run["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    run["strat_r"] = run["weight"].shift(1).fillna(0.0) * run["r"] - cost
    run["spot_r"] = run["r"]
    run["eq"] = np.exp(np.cumsum(run["strat_r"].to_numpy(dtype=float)))
    run["spot_eq"] = np.exp(np.cumsum(run["spot_r"].to_numpy(dtype=float)))
    run["eth_close"] = run["close"]

    pm = perf_from_log_returns(run["strat_r"], 5)
    sm = perf_from_log_returns(run["spot_r"], 5)

    hold_periods = holding_periods_from_weight(daily2["weight_exec"])
    n_periods = int(len(hold_periods))
    avg_hold = float(np.mean(hold_periods)) if hold_periods else np.nan

    tim_all = float((daily2["weight_exec"] > 0).mean() * 100.0)
    reg_tim = {}
    for rg in ["BULL", "CHOP", "BEAR", "NA"]:
        g = daily2[daily2["regime"] == rg]
        reg_tim[rg] = float((g["weight_exec"] > 0).mean() * 100.0) if len(g) else np.nan

    vol_on = daily2.loc[daily2["trend_confirmed"], "vol_scalar"]
    diagnostics = pd.DataFrame(
        [
            {"metric": "rows", "value": int(len(run))},
            {"metric": "window_start", "value": str(run["timestamp"].iloc[0])},
            {"metric": "window_end", "value": str(run["timestamp"].iloc[-1])},
            {"metric": "time_in_market_pct_overall", "value": tim_all},
            {"metric": "time_in_market_pct_bull", "value": reg_tim["BULL"]},
            {"metric": "time_in_market_pct_chop", "value": reg_tim["CHOP"]},
            {"metric": "time_in_market_pct_bear", "value": reg_tim["BEAR"]},
            {"metric": "avg_holding_days", "value": avg_hold},
            {"metric": "distinct_long_periods", "value": n_periods},
            {"metric": "vol_scalar_mean", "value": float(vol_on.mean()) if len(vol_on) else np.nan},
            {"metric": "vol_scalar_median", "value": float(vol_on.median()) if len(vol_on) else np.nan},
            {"metric": "vol_scalar_q25", "value": float(vol_on.quantile(0.25)) if len(vol_on) else np.nan},
            {"metric": "vol_scalar_q75", "value": float(vol_on.quantile(0.75)) if len(vol_on) else np.nan},
            {"metric": "exit_confirm_days", "value": int(args.exit_confirm_days)},
            {"metric": "min_hold_days", "value": int(args.min_hold_days)},
            {"metric": "trailing_stop_pct", "value": float(args.trailing_stop_pct) if np.isfinite(args.trailing_stop_pct) else np.nan},
            {"metric": "trailing_stop_hits", "value": int(daily["trailing_stop_hit"].sum())},
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(run)),
                "start": str(run["timestamp"].iloc[0]),
                "end": str(run["timestamp"].iloc[-1]),
                "ret": pm["ret"],
                "cagr": pm["cagr"],
                "ann_vol": pm["ann_vol"],
                "sharpe": pm["sharpe"],
                "max_dd": pm["max_dd"],
                "spot_ret": sm["ret"],
                "excess_vs_spot": pm["ret"] - sm["ret"],
                "time_in_market_pct": tim_all,
                "avg_weight": float(run["weight"].mean()),
                "turnover": float(run["turnover"].sum()),
                "distinct_long_periods": n_periods,
                "avg_holding_days": avg_hold,
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run.to_csv(out_csv, index=False)

    out_sum = Path(args.out_summary_csv)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)

    out_sig = Path(args.out_signals_csv)
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    daily2.to_csv(out_sig, index=False)

    out_diag = Path(args.out_diagnostics_csv)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_diag, index=False)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity", "Weight", "Price", "Vol Scalar / Trend"],
    )
    fig.add_trace(go.Scatter(x=run["timestamp"], y=run["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=run["timestamp"], y=run["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=run["timestamp"], y=run["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=run["timestamp"], y=run["close"], name="ETH Close", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily2["day"], y=daily2["vol_scalar"], name="Vol Scalar", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily2["day"], y=daily2["trend_confirmed"].astype(int), name="Trend Confirmed", line=dict(color="#17becf")), row=4, col=1)
    fig.update_layout(height=1200, title="Offense Trend Follow v1")

    html = (
        "<html><head><meta charset='utf-8'><title>Offense Trend Follow v1</title></head><body>"
        "<h3>Offense Trend Follow v1</h3>"
        + diagnostics.to_html(index=False, border=0)
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
    print(f"wrote {out_diag}")
    print(f"wrote {out_html}")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
