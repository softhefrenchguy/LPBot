from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path} must contain timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    return df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
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


def build_frame(
    price_csv: Path,
    btc_csv: Path | None,
    perp_csv: Path | None,
    window_days: int,
    window_end_ts: pd.Timestamp | None,
    horizon_bars: int,
    breakout_lookback: int,
    vol_event_k: float,
    basis_z_th: float,
    funding_z_th: float,
    oi_z_th: float,
) -> tuple[pd.DataFrame, list[str]]:
    eth = load_price(price_csv).rename(columns={"close": "eth_close"})
    end = eth["timestamp"].iloc[-1]
    if window_end_ts is not None and pd.notna(window_end_ts):
        end = min(end, pd.Timestamp(window_end_ts))
    start = end - pd.Timedelta(days=window_days)
    eth = eth[(eth["timestamp"] >= start) & (eth["timestamp"] <= end)].copy()
    if len(eth) == 0:
        raise RuntimeError("No price rows after window filter; check --window-days/--window-end-ts and source data freshness")
    eth["r"] = np.log(eth["eth_close"] / eth["eth_close"].shift(1))

    df = eth[["timestamp", "eth_close", "r"]].copy()
    df["ret_3"] = np.log(df["eth_close"] / df["eth_close"].shift(3))
    df["ret_12"] = np.log(df["eth_close"] / df["eth_close"].shift(12))
    df["ret_36"] = np.log(df["eth_close"] / df["eth_close"].shift(36))
    df["ret_72"] = np.log(df["eth_close"] / df["eth_close"].shift(72))
    df["vol_12"] = df["r"].rolling(12, min_periods=12).std()
    df["vol_36"] = df["r"].rolling(36, min_periods=36).std()
    df["vol_72"] = df["r"].rolling(72, min_periods=72).std()
    df["vol_ratio_12_72"] = df["vol_12"] / (df["vol_72"] + 1e-12)
    ema_fast = df["eth_close"].ewm(span=24, adjust=False).mean()
    ema_slow = df["eth_close"].ewm(span=144, adjust=False).mean()
    df["trend_gap"] = (ema_fast - ema_slow) / (ema_slow.abs() + 1e-12)
    df["price_gap"] = (df["eth_close"] - ema_slow) / (ema_slow.abs() + 1e-12)
    df["mom_x_trend"] = df["ret_12"] * df["trend_gap"]
    df["mom_x_vol"] = df["ret_12"] / (df["vol_36"] + 1e-12)

    feat_cols = [
        "ret_3",
        "ret_12",
        "ret_36",
        "ret_72",
        "vol_12",
        "vol_36",
        "vol_72",
        "vol_ratio_12_72",
        "trend_gap",
        "price_gap",
        "mom_x_trend",
        "mom_x_vol",
    ]

    if btc_csv is not None and btc_csv.exists():
        btc = load_price(btc_csv).rename(columns={"close": "btc_close"})
        btc["btc_r"] = np.log(btc["btc_close"] / btc["btc_close"].shift(1))
        btc["btc_ret_12"] = np.log(btc["btc_close"] / btc["btc_close"].shift(12))
        btc["btc_ret_36"] = np.log(btc["btc_close"] / btc["btc_close"].shift(36))
        bfast = btc["btc_close"].ewm(span=24, adjust=False).mean()
        bslow = btc["btc_close"].ewm(span=144, adjust=False).mean()
        btc["btc_trend_gap"] = (bfast - bslow) / (bslow.abs() + 1e-12)
        df = df.merge(
            btc[["timestamp", "btc_r", "btc_ret_12", "btc_ret_36", "btc_trend_gap"]],
            on="timestamp",
            how="left",
        )
        for c in ["btc_r", "btc_ret_12", "btc_ret_36", "btc_trend_gap"]:
            df[c] = df[c].ffill()
        df["rel_ret_12"] = df["ret_12"] - df["btc_ret_12"]
        feat_cols += ["btc_r", "btc_ret_12", "btc_ret_36", "btc_trend_gap", "rel_ret_12"]

    if perp_csv is not None and perp_csv.exists():
        pf = pd.read_csv(perp_csv)
        if "timestamp" in pf.columns:
            pf["timestamp"] = pd.to_datetime(pf["timestamp"], utc=True, errors="coerce")
            for c in ["basis", "funding_rate", "oi_chg"]:
                if c in pf.columns:
                    pf[c] = pd.to_numeric(pf[c], errors="coerce")
            keep = ["timestamp"] + [c for c in ["basis", "funding_rate", "oi_chg"] if c in pf.columns]
            pf = pf[keep].dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.merge(pf, on="timestamp", how="left")
            for c in ["basis", "funding_rate", "oi_chg"]:
                if c in df.columns:
                    df[c] = df[c].ffill()
                    df[f"{c}_z"] = rolling_z(df[c], 288)
                    feat_cols.append(f"{c}_z")

    h = max(1, int(horizon_bars))
    df["fwd_r"] = np.log(df["eth_close"].shift(-h) / df["eth_close"])
    df["y_up"] = (df["fwd_r"] > 0).astype(int)

    # Event masks
    lb = max(12, int(breakout_lookback))
    hh = df["eth_close"].rolling(lb, min_periods=lb).max()
    ll = df["eth_close"].rolling(lb, min_periods=lb).min()
    breakout_up = df["eth_close"] > hh.shift(1)
    breakout_dn = df["eth_close"] < ll.shift(1)
    vol_event = df["r"].abs() > (float(vol_event_k) * (df["vol_36"] + 1e-12))
    crowd_event = pd.Series(False, index=df.index)
    if "basis_z" in df.columns:
        crowd_event = crowd_event | (df["basis_z"].abs() >= float(basis_z_th))
    if "funding_rate_z" in df.columns:
        crowd_event = crowd_event | (df["funding_rate_z"].abs() >= float(funding_z_th))
    if "oi_chg_z" in df.columns:
        crowd_event = crowd_event | (df["oi_chg_z"].abs() >= float(oi_z_th))
    df["event_breakout"] = (breakout_up | breakout_dn).astype(int)
    df["event_vol"] = vol_event.astype(int)
    df["event_crowd"] = crowd_event.astype(int)
    df["event"] = ((df["event_breakout"] == 1) | (df["event_vol"] == 1) | (df["event_crowd"] == 1)).astype(int)

    keep = ["timestamp", "eth_close", "r", "fwd_r", "y_up", "event", "event_breakout", "event_vol", "event_crowd"] + feat_cols
    df = df[keep].dropna().reset_index(drop=True)
    return df, feat_cols


def walk_forward_event_probs(
    df: pd.DataFrame,
    feat_cols: list[str],
    train_days: int,
    test_days: int,
    step_days: int,
    tail_infer_unseen: bool = False,
) -> pd.DataFrame:
    rows = []
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise RuntimeError("No valid timestamps for walk-forward")
    fold = 0
    max_folds = 10000
    last_model: HistGradientBoostingClassifier | None = None
    last_fold: int | None = None
    while True:
        if fold > max_folds:
            raise RuntimeError("walk-forward exceeded max folds; likely insufficient data/window config")
        tr_start = t_min + pd.Timedelta(days=fold * step_days)
        tr_end = tr_start + pd.Timedelta(days=train_days)
        te_end = tr_end + pd.Timedelta(days=test_days)
        if te_end > t_max:
            break

        tr_all = df[(df["timestamp"] >= tr_start) & (df["timestamp"] < tr_end)].copy()
        te_all = df[(df["timestamp"] >= tr_end) & (df["timestamp"] < te_end)].copy()
        tr = tr_all[tr_all["event"] == 1].copy()
        te_event = te_all[te_all["event"] == 1].copy()
        # Keep fold eligibility loose enough for live rolling windows.
        if len(tr_all) < 2000 or len(te_all) < 500 or len(tr) < 100 or len(te_event) < 20:
            fold += 1
            continue

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=60,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(tr[feat_cols].to_numpy(), tr["y_up"].to_numpy(dtype=int))
        te_all["p_up"] = np.nan
        te_all["fold"] = fold
        te_all["signal_source"] = "wf_fold"
        p_ev = model.predict_proba(te_event[feat_cols].to_numpy())[:, 1]
        te_all.loc[te_event.index, "p_up"] = p_ev
        rows.append(
            te_all[
                [
                    "timestamp",
                    "eth_close",
                    "r",
                    "fwd_r",
                    "y_up",
                    "event",
                    "event_breakout",
                    "event_vol",
                    "event_crowd",
                    "p_up",
                    "fold",
                    "signal_source",
                ]
            ]
        )
        last_model = model
        last_fold = int(fold)
        fold += 1

    if not rows:
        raise RuntimeError("No walk-forward folds produced event predictions")
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    # Live tail inference: extend coverage to newest bars that are outside completed WF folds,
    # using the last trained fold model only (no retraining on tail data).
    if tail_infer_unseen and last_model is not None and len(out):
        last_wf_ts = out["timestamp"].max()
        tail_all = df[df["timestamp"] > last_wf_ts].copy()
        if len(tail_all):
            tail_all["p_up"] = np.nan
            tail_all["fold"] = int(last_fold + 1) if last_fold is not None else -1
            tail_all["signal_source"] = "tail"
            tail_event = tail_all[tail_all["event"] == 1].copy()
            if len(tail_event):
                p_tail = last_model.predict_proba(tail_event[feat_cols].to_numpy())[:, 1]
                tail_all.loc[tail_event.index, "p_up"] = p_tail
            tail_keep = tail_all[
                [
                    "timestamp",
                    "eth_close",
                    "r",
                    "fwd_r",
                    "y_up",
                    "event",
                    "event_breakout",
                    "event_vol",
                    "event_crowd",
                    "p_up",
                    "fold",
                    "signal_source",
                ]
            ]
            out = (
                pd.concat([out, tail_keep], ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
    return out


def run_event_strategy(
    pred: pd.DataFrame,
    long_th: float,
    short_th: float,
    hold_bars: int,
    max_weight: float,
    trade_cost_bps: float,
    no_shorts: bool = False,
) -> pd.DataFrame:
    out = pred.copy()
    p = out["p_up"]
    w = np.zeros(len(out), dtype=float)
    hold = 0
    cur_w = 0.0
    for i in range(len(out)):
        ev = bool(out["event"].iat[i]) and np.isfinite(p.iat[i])
        if ev:
            pi = float(p.iat[i])
            if pi >= long_th:
                conf = (pi - long_th) / (1.0 - long_th + 1e-12)
                cur_w = float(np.clip(conf, 0.0, 1.0) * max_weight)
                hold = int(hold_bars)
            elif pi <= short_th:
                conf = (short_th - pi) / (short_th + 1e-12)
                cur_w = -float(np.clip(conf, 0.0, 1.0) * max_weight)
                hold = int(hold_bars)
        if no_shorts and cur_w < 0.0:
            cur_w = 0.0
        if hold > 0:
            w[i] = cur_w
            hold -= 1
        else:
            w[i] = 0.0
            cur_w = 0.0

    out["weight"] = pd.Series(w, index=out.index)
    out["turnover"] = out["weight"].diff().abs().fillna(0.0)
    cost = out["turnover"] * (float(trade_cost_bps) / 10000.0)
    out["cost_r"] = -cost
    out["strat_r"] = out["weight"].shift(1).fillna(0.0) * out["r"] - cost
    out["spot_r"] = out["r"]
    out["eq"] = np.exp(np.cumsum(out["strat_r"].to_numpy()))
    out["spot_eq"] = np.exp(np.cumsum(out["spot_r"].to_numpy()))
    out["active"] = (out["weight"].abs() > 1e-12).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-driven direction model v1")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument(
        "--window-end-ts",
        default="",
        help="Optional UTC timestamp to anchor rolling window end (defaults to latest price timestamp).",
    )
    ap.add_argument("--horizon-bars", type=int, default=12)
    ap.add_argument("--breakout-lookback", type=int, default=96)
    ap.add_argument("--vol-event-k", type=float, default=1.5)
    ap.add_argument("--basis-z-th", type=float, default=1.5)
    ap.add_argument("--funding-z-th", type=float, default=1.5)
    ap.add_argument("--oi-z-th", type=float, default=1.5)
    ap.add_argument("--wf-train-days", type=int, default=240)
    ap.add_argument("--wf-test-days", type=int, default=30)
    ap.add_argument("--wf-step-days", type=int, default=30)
    ap.add_argument("--long-thresholds", default="0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--short-thresholds", default="0.45,0.42,0.40,0.38,0.35")
    ap.add_argument("--hold-bars-list", default="6,12,24")
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--no-shorts", action="store_true", help="Floor position weights at 0 (flat-only, no shorts)")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--min-time-in-market-pct", type=float, default=2.0)
    ap.add_argument("--out-preds-csv", default="artifacts/backtest/direction_event_model_v1_preds.csv")
    ap.add_argument("--out-sweep-csv", default="artifacts/backtest/direction_event_model_v1_sweep.csv")
    ap.add_argument("--out-best-csv", default="artifacts/backtest/direction_event_model_v1.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/direction_event_model_v1.html")
    ap.add_argument(
        "--tail-infer-unseen",
        action="store_true",
        help="Append predictions for bars after the last completed WF fold using the last fold model (no retraining).",
    )
    args = ap.parse_args()
    window_end_ts = pd.to_datetime(args.window_end_ts, utc=True, errors="coerce") if str(args.window_end_ts).strip() else None

    df, feat_cols = build_frame(
        price_csv=Path(args.price_csv),
        btc_csv=Path(args.btc_csv) if Path(args.btc_csv).exists() else None,
        perp_csv=Path(args.perp_csv) if Path(args.perp_csv).exists() else None,
        window_days=int(args.window_days),
        window_end_ts=window_end_ts,
        horizon_bars=int(args.horizon_bars),
        breakout_lookback=int(args.breakout_lookback),
        vol_event_k=float(args.vol_event_k),
        basis_z_th=float(args.basis_z_th),
        funding_z_th=float(args.funding_z_th),
        oi_z_th=float(args.oi_z_th),
    )
    pred = walk_forward_event_probs(
        df=df,
        feat_cols=feat_cols,
        train_days=int(args.wf_train_days),
        test_days=int(args.wf_test_days),
        step_days=int(args.wf_step_days),
        tail_infer_unseen=bool(args.tail_infer_unseen),
    )

    ev_pred = pred[pred["p_up"].notna()].copy()
    if len(ev_pred) < 100:
        raise RuntimeError("Too few event predictions")
    auc_ev = float(roc_auc_score(ev_pred["y_up"].to_numpy(dtype=int), ev_pred["p_up"].to_numpy(dtype=float)))
    acc_ev = float(accuracy_score(ev_pred["y_up"].to_numpy(dtype=int), (ev_pred["p_up"] >= 0.5).astype(int)))

    longs = sorted(set([float(x) for x in str(args.long_thresholds).split(",") if str(x).strip()]))
    shorts = sorted(set([float(x) for x in str(args.short_thresholds).split(",") if str(x).strip()]), reverse=True)
    holds = sorted(set([int(float(x)) for x in str(args.hold_bars_list).split(",") if str(x).strip()]))

    sweep_rows = []
    best_score = -1e18
    best_sim: pd.DataFrame | None = None
    best_cfg = None
    for h in holds:
        for lth in longs:
            for sth in shorts:
                if sth >= lth:
                    continue
                sim = run_event_strategy(
                    pred=pred,
                    long_th=lth,
                    short_th=sth,
                    hold_bars=int(h),
                    max_weight=float(args.max_weight),
                    trade_cost_bps=float(args.trade_cost_bps),
                    no_shorts=bool(args.no_shorts),
                )
                m = perf(sim["strat_r"], 5)
                ms = perf(sim["spot_r"], 5)
                tim = float(sim["active"].mean() * 100.0)
                active = sim["active"].astype(bool)
                hit_active = float((np.sign(sim.loc[active, "weight"]) == np.sign(sim.loc[active, "fwd_r"])).mean()) if active.any() else np.nan
                row = {
                    "hold_bars": int(h),
                    "long_th": float(lth),
                    "short_th": float(sth),
                    "ret": m["ret"],
                    "cagr": m["cagr"],
                    "ann_vol": m["ann_vol"],
                    "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"],
                    "spot_ret": ms["ret"],
                    "excess_vs_spot": m["ret"] - ms["ret"],
                    "time_in_market_pct": tim,
                    "avg_abs_weight": float(sim["weight"].abs().mean()),
                    "turnover": float(sim["turnover"].sum()),
                    "hit_active": hit_active,
                }
                sweep_rows.append(row)

                score = (0.0 if np.isnan(row["sharpe"]) else row["sharpe"]) + 1.2 * row["ret"] - 0.00015 * row["turnover"]
                if tim < float(args.min_time_in_market_pct):
                    score -= 3.0
                if score > best_score:
                    best_score = score
                    best_sim = sim
                    best_cfg = (h, lth, sth)

    if best_sim is None or best_cfg is None:
        raise RuntimeError("No event strategy configuration found")

    sweep = pd.DataFrame(sweep_rows).sort_values(["sharpe", "ret"], ascending=[False, False]).reset_index(drop=True)
    m_best = perf(best_sim["strat_r"], 5)
    m_spot = perf(best_sim["spot_r"], 5)

    Path(args.out_preds_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_preds_csv, index=False)
    Path(args.out_sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out_sweep_csv, index=False)
    Path(args.out_best_csv).parent.mkdir(parents=True, exist_ok=True)
    best_sim.to_csv(args.out_best_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Equity", "P(up) on Events", "Weight", "Event Flags", "Price"],
    )
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["p_up"], name="P(up)", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_hline(y=float(best_cfg[1]), line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.add_hline(y=float(best_cfg[2]), line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["weight"], name="Weight", line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_breakout"], name="Breakout Event", line=dict(color="#17becf")), row=4, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_crowd"], name="Crowd Event", line=dict(color="#8c564b")), row=4, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_vol"], name="Vol Event", line=dict(color="#bcbd22")), row=4, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=5, col=1)
    fig.update_layout(height=1450, title="Event Direction Model v1")

    summary = pd.DataFrame(
        [
            {
                "rows_oos": int(len(pred)),
                "rows_event_preds": int(len(ev_pred)),
                "rows_tail": int((pred.get("signal_source", pd.Series(index=pred.index, dtype=object)) == "tail").sum()),
                "event_rate_pct": float((pred["event"] == 1).mean() * 100.0),
                "oos_auc_event": auc_ev,
                "oos_acc_event": acc_ev,
                "best_hold_bars": int(best_cfg[0]),
                "best_long_th": float(best_cfg[1]),
                "best_short_th": float(best_cfg[2]),
                "ret": m_best["ret"],
                "cagr": m_best["cagr"],
                "ann_vol": m_best["ann_vol"],
                "sharpe": m_best["sharpe"],
                "max_dd": m_best["max_dd"],
                "spot_ret": m_spot["ret"],
                "excess_vs_spot": m_best["ret"] - m_spot["ret"],
                "time_in_market_pct": float(best_sim["active"].mean() * 100.0),
                "avg_abs_weight": float(best_sim["weight"].abs().mean()),
                "turnover": float(best_sim["turnover"].sum()),
                "trade_cost_bps": float(args.trade_cost_bps),
                "no_shorts": bool(args.no_shorts),
            }
        ]
    )

    html = (
        "<html><head><meta charset='utf-8'><title>Event Direction Model v1</title></head><body>"
        "<h3>Event Direction Model v1</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        "<h4>Top 10 configs</h4>"
        f"{sweep.head(10).round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text(html, encoding="utf-8")

    print(f"wrote {args.out_preds_csv}")
    print(f"wrote {args.out_sweep_csv}")
    print(f"wrote {args.out_best_csv}")
    print(f"wrote {args.out_html}")
    print(
        "oos_auc_event={:.4f} oos_acc_event={:.4f} best_hold={} best_long={:.2f} best_short={:.2f} ret={:.4f} sharpe={:.4f} tim={:.2f}% rows_tail={}".format(
            auc_ev,
            acc_ev,
            int(best_cfg[0]),
            float(best_cfg[1]),
            float(best_cfg[2]),
            m_best["ret"],
            m_best["sharpe"],
            float(best_sim["active"].mean() * 100.0),
            int((pred.get("signal_source", pd.Series(index=pred.index, dtype=object)) == "tail").sum()),
        )
    )


if __name__ == "__main__":
    main()
