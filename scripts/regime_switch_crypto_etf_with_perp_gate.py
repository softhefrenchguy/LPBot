from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_etf_close(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = next((cols[c] for c in ("timestamp", "date", "datetime") if c in cols), None)
    pcol = next((cols[c] for c in ("close", "adj close", "adj_close") if c in cols), None)
    if dcol is None or pcol is None:
        raise ValueError(f"{path} missing date/close columns")
    out = pd.DataFrame(
        {"day": pd.to_datetime(df[dcol], utc=True, errors="coerce"), "close": pd.to_numeric(df[pcol], errors="coerce")}
    ).dropna()
    out = out.sort_values("day").drop_duplicates("day", keep="last")
    return out.set_index("day")["close"]


def _rolling_z(x: pd.Series, w: int) -> pd.Series:
    mu = x.rolling(w, min_periods=max(20, w // 4)).mean()
    sd = x.rolling(w, min_periods=max(20, w // 4)).std()
    return (x - mu) / (sd + 1e-12)


def _perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    bpy = 365.0
    ann_vol = float(np.std(x) * np.sqrt(bpy))
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bpy / len(eq)) - 1.0),
        "ann_vol": ann_vol,
        "sharpe": float((np.mean(x) * bpy) / (ann_vol + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def _trend_state(
    close: pd.Series, ema_fast: int, ema_slow: int, slope_lookback: int, persist_on: int, persist_off: int
) -> pd.Series:
    x = pd.DataFrame({"close": close})
    x["ema_fast"] = x["close"].ewm(span=int(ema_fast), adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=int(ema_slow), adjust=False).mean()
    x["slope_slow"] = x["ema_slow"].pct_change(int(slope_lookback))
    bull_raw = (x["close"] > x["ema_slow"]) & (x["ema_fast"] > x["ema_slow"]) & (x["slope_slow"] > 0)
    bear_raw = (x["close"] < x["ema_slow"]) & (x["ema_fast"] < x["ema_slow"]) & (x["slope_slow"] < 0)

    state = np.zeros(len(x), dtype=int)
    on = False
    on_streak = 0
    off_streak = 0
    for i in range(len(x)):
        on_streak = on_streak + 1 if bool(bull_raw.iat[i]) else 0
        off_streak = off_streak + 1 if bool(bear_raw.iat[i]) else 0
        if not on and on_streak >= int(persist_on):
            on = True
            off_streak = 0
        elif on and off_streak >= int(persist_off):
            on = False
            on_streak = 0
        state[i] = 1 if on else 0
    return pd.Series(state, index=close.index, name="trend_bull")


def _captures(strategy_r: pd.Series, spot_r: pd.Series) -> tuple[float, float]:
    up = spot_r > 0
    dn = spot_r < 0
    up_cap = float(strategy_r[up].sum() / (spot_r[up].sum() + 1e-12))
    dn_cap = float(strategy_r[dn].sum() / (spot_r[dn].sum() - 1e-12))
    return up_cap, dn_cap


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime-switch crypto/ETF with perp predictive gate")
    ap.add_argument("--crypto-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--slope-lookback", type=int, default=20)
    ap.add_argument("--persist-on", type=int, default=3)
    ap.add_argument("--persist-off", type=int, default=3)
    ap.add_argument("--pred-z-window-bars", type=int, default=288)
    ap.add_argument("--pred-score-ema-span", type=int, default=12)
    ap.add_argument("--use-oi-in-score", action="store_true", help="Include oi_chg_z multiplier. Off by default due sparse OI updates.")
    ap.add_argument("--oi-event-bonus-weight", type=float, default=0.0, help="Add event-only OI z-score bonus to predictive score.")
    ap.add_argument("--oi-max-age-bars", type=float, default=1.0, help="OI considered fresh when oi_age_bars <= this.")
    ap.add_argument("--pred-threshold", type=float, default=0.55, help="Threshold on chosen predictive metric")
    ap.add_argument(
        "--pred-metric",
        choices=["daily_mean_score", "daily_bull_frac"],
        default="daily_bull_frac",
        help="daily_mean_score: mean(score_n) by day; daily_bull_frac: fraction(score_n>0) by day",
    )
    ap.add_argument("--invert-pred", action="store_true")
    ap.add_argument("--bull-w", type=float, default=0.80)
    ap.add_argument("--bear-w", type=float, default=0.10)
    ap.add_argument("--switch-cost-bps", type=float, default=2.0)
    ap.add_argument("--etf-dir", default="data/etf")
    ap.add_argument("--etf-symbols", default="GLD,DBC,DBA,DBB")
    ap.add_argument("--etf-weights", default="0.50,0.1666666667,0.1666666667,0.1666666666")
    ap.add_argument("--out-csv", default="artifacts/backtest/regime_switch_perp_gate_1y.csv")
    ap.add_argument("--summary-csv", default="artifacts/backtest/regime_switch_perp_gate_1y_summary.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_switch_perp_gate_1y.html")
    args = ap.parse_args()

    # Crypto daily series.
    c = pd.read_csv(args.crypto_csv)
    for col in ("timestamp", "close", "r"):
        if col not in c.columns:
            raise ValueError(f"{args.crypto_csv} missing {col}")
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True, errors="coerce")
    c["close"] = pd.to_numeric(c["close"], errors="coerce")
    c["r"] = pd.to_numeric(c["r"], errors="coerce")
    c = c.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")
    c["day"] = c["timestamp"].dt.floor("D")
    daily = c.groupby("day", as_index=False).agg(close=("close", "last"), crypto_r=("r", "sum")).sort_values("day")

    # Perp predictive score (5m -> daily gate).
    p = pd.read_csv(args.perp_csv, low_memory=False)
    need = {"timestamp", "basis", "funding_rate"}
    if not need.issubset(p.columns):
        raise ValueError(f"{args.perp_csv} missing columns: {sorted(need)}")
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    for col in ("basis", "funding_rate"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    if "oi_chg" in p.columns:
        p["oi_chg"] = pd.to_numeric(p["oi_chg"], errors="coerce")
    else:
        p["oi_chg"] = np.nan
    if "oi_chg_event" in p.columns:
        p["oi_chg_event"] = pd.to_numeric(p["oi_chg_event"], errors="coerce")
    else:
        p["oi_chg_event"] = np.nan
    if "oi_age_bars" in p.columns:
        p["oi_age_bars"] = pd.to_numeric(p["oi_age_bars"], errors="coerce")
    else:
        p["oi_age_bars"] = np.nan
    p = p.dropna(subset=["timestamp"]).sort_values("timestamp")
    p = p.dropna(subset=["basis", "funding_rate"]).copy()

    w = int(args.pred_z_window_bars)
    p["funding_z"] = _rolling_z(p["funding_rate"].ffill(), w)
    p["basis_z"] = _rolling_z(p["basis"], w)
    p["oi_chg_z"] = _rolling_z(p["oi_chg"], w)
    p["oi_event_z"] = _rolling_z(p["oi_chg_event"], w)
    p["oi_is_fresh"] = (p["oi_age_bars"] <= float(args.oi_max_age_bars)).fillna(False)
    p["oi_event_bonus"] = np.where(p["oi_is_fresh"], p["oi_event_z"], 0.0)
    crowded = p["funding_z"] + p["basis_z"]
    if args.use_oi_in_score:
        p["score_raw"] = -(crowded * p["oi_chg_z"].fillna(0.0))
    else:
        p["score_raw"] = -crowded
    if float(args.oi_event_bonus_weight) != 0.0:
        p["score_raw"] = p["score_raw"] + float(args.oi_event_bonus_weight) * p["oi_event_bonus"].fillna(0.0)
    p["score"] = p["score_raw"].ewm(span=int(args.pred_score_ema_span), adjust=False).mean()
    if args.invert_pred:
        p["score"] = -p["score"]
    p["score_sd"] = p["score"].rolling(w, min_periods=max(20, w // 4)).std()
    p["score_n"] = p["score"] / (p["score_sd"] + 1e-12)
    p["day"] = p["timestamp"].dt.floor("D")
    if args.pred_metric == "daily_mean_score":
        pred_daily = p.groupby("day", as_index=False).agg(pred_score=("score_n", "mean")).sort_values("day")
    else:
        p["bull_bit"] = (p["score_n"] > 0).astype(float)
        pred_daily = p.groupby("day", as_index=False).agg(pred_score=("bull_bit", "mean")).sort_values("day")
    pred_daily["pred_on"] = (pred_daily["pred_score"] > float(args.pred_threshold)).astype(int)

    # ETF basket.
    syms = [x.strip().upper() for x in args.etf_symbols.split(",") if x.strip()]
    ws = np.array([float(x) for x in args.etf_weights.split(",")], dtype=float)
    if len(syms) != len(ws):
        raise ValueError("etf-symbols and etf-weights must have same length")
    if not np.isclose(ws.sum(), 1.0, atol=1e-9):
        raise ValueError("etf-weights must sum to 1.0")
    etf_parts: list[pd.Series] = []
    for s, ww in zip(syms, ws):
        ser = _load_etf_close(Path(args.etf_dir) / f"{s}.csv")
        etf_parts.append(np.log(ser / ser.shift(1)) * float(ww))
    etf_r = pd.concat(etf_parts, axis=1).sum(axis=1, min_count=1).rename("etf_r").reset_index()

    daily = daily.merge(pred_daily[["day", "pred_score", "pred_on"]], on="day", how="left")
    daily = daily.merge(etf_r, on="day", how="left").sort_values("day")
    daily["pred_score"] = pd.to_numeric(daily["pred_score"], errors="coerce")
    daily["pred_on"] = pd.to_numeric(daily["pred_on"], errors="coerce").fillna(0).astype(int)
    daily["etf_r"] = pd.to_numeric(daily["etf_r"], errors="coerce").fillna(0.0)

    # Window after merge (so we stay in available perp range).
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()

    # Trend regime.
    trend = _trend_state(
        daily.set_index("day")["close"],
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        slope_lookback=int(args.slope_lookback),
        persist_on=int(args.persist_on),
        persist_off=int(args.persist_off),
    )
    daily = daily.merge(trend.reset_index(), on="day", how="left")
    daily["trend_exec"] = daily["trend_bull"].shift(1).fillna(0).astype(int)
    daily["pred_exec"] = daily["pred_on"].shift(1).fillna(0).astype(int)
    daily["bull_exec"] = ((daily["trend_exec"] == 1) & (daily["pred_exec"] == 1)).astype(int)

    # Baseline trend-only and trend+pred gates.
    daily["w_trend_only"] = np.where(daily["trend_exec"] == 1, float(args.bull_w), float(args.bear_w))
    daily["w_trend_pred"] = np.where(daily["bull_exec"] == 1, float(args.bull_w), float(args.bear_w))

    cost_k = float(args.switch_cost_bps) / 10000.0
    daily["turn_trend"] = daily["w_trend_only"].diff().abs().fillna(0.0)
    daily["turn_pred"] = daily["w_trend_pred"].diff().abs().fillna(0.0)
    daily["r_trend_only"] = (
        daily["w_trend_only"] * daily["crypto_r"] + (1.0 - daily["w_trend_only"]) * daily["etf_r"] - daily["turn_trend"] * cost_k
    )
    daily["r_trend_pred"] = (
        daily["w_trend_pred"] * daily["crypto_r"] + (1.0 - daily["w_trend_pred"]) * daily["etf_r"] - daily["turn_pred"] * cost_k
    )

    # Equity
    daily["eq_trend_only"] = np.exp(daily["r_trend_only"].cumsum())
    daily["eq_trend_pred"] = np.exp(daily["r_trend_pred"].cumsum())
    daily["eq_spot"] = np.exp(daily["crypto_r"].cumsum())
    daily["eq_etf"] = np.exp(daily["etf_r"].cumsum())
    for ccol in ("eq_trend_only", "eq_trend_pred", "eq_spot", "eq_etf"):
        daily[ccol] = daily[ccol] / float(daily[ccol].iloc[0])

    # Metrics
    up_t, dn_t = _captures(daily["r_trend_only"], daily["crypto_r"])
    up_p, dn_p = _captures(daily["r_trend_pred"], daily["crypto_r"])
    m_t = _perf(daily["r_trend_only"])
    m_p = _perf(daily["r_trend_pred"])
    m_s = _perf(daily["crypto_r"])
    m_e = _perf(daily["etf_r"])
    summary = pd.DataFrame(
        [
            {
                "model": "trend_only",
                **m_t,
                "avg_w_crypto": float(daily["w_trend_only"].mean()),
                "bull_pct": float((daily["trend_exec"] == 1).mean()),
                "up_capture": up_t,
                "down_capture": dn_t,
                "turnover_total": float(daily["turn_trend"].sum()),
            },
            {
                "model": "trend_plus_pred",
                **m_p,
                "avg_w_crypto": float(daily["w_trend_pred"].mean()),
                "bull_pct": float((daily["bull_exec"] == 1).mean()),
                "up_capture": up_p,
                "down_capture": dn_p,
                "turnover_total": float(daily["turn_pred"].sum()),
            },
            {
                "model": "eth_spot",
                **m_s,
                "avg_w_crypto": 1.0,
                "bull_pct": 1.0,
                "up_capture": 1.0,
                "down_capture": 1.0,
                "turnover_total": 0.0,
            },
            {"model": "etf_basket", **m_e, "avg_w_crypto": 0.0, "bull_pct": 0.0, "up_capture": 0.0, "down_capture": 0.0, "turnover_total": 0.0},
        ]
    )
    summary["window_start"] = str(daily["day"].iloc[0])
    summary["window_end"] = str(daily["day"].iloc[-1])
    summary["rows"] = len(daily)
    summary["pred_threshold"] = float(args.pred_threshold)
    summary["pred_metric"] = args.pred_metric
    summary["use_oi_in_score"] = bool(args.use_oi_in_score)
    summary["oi_event_bonus_weight"] = float(args.oi_event_bonus_weight)
    summary["oi_max_age_bars"] = float(args.oi_max_age_bars)
    summary["bull_w"] = float(args.bull_w)
    summary["bear_w"] = float(args.bear_w)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity", "Price", "Predictive Score", "States", "Crypto Weight"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_trend_only"], name="Trend-only Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_trend_pred"], name="Trend+Pred Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_spot"], name="ETH Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_etf"], name="ETF Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["close"], name="ETH close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["pred_score"], name="pred_score"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["trend_exec"], name="trend_exec"), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["pred_exec"], name="pred_exec"), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["bull_exec"], name="bull_exec"), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_trend_only"], name="w_trend_only"), row=5, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_trend_pred"], name="w_trend_pred"), row=5, col=1)
    fig.update_layout(height=1650, title="Regime Switch + Perp Predictive Gate")

    html = (
        "<html><head><meta charset='utf-8'><title>Regime Switch + Perp Gate</title></head><body>"
        "<h3>Regime Switch + Perp Predictive Gate</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    Path(args.out_html).write_text(html, encoding="utf-8")

    print("wrote", args.out_csv)
    print("wrote", args.summary_csv)
    print("wrote", args.out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
