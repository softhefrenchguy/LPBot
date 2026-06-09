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
    horizon_bars: int,
    breakout_lookback: int,
    vol_event_k: float,
    basis_z_th: float,
    funding_z_th: float,
    oi_z_th: float,
) -> tuple[pd.DataFrame, list[str]]:
    eth = load_price(price_csv).rename(columns={"close": "eth_close"})
    end = eth["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=window_days)
    eth = eth[(eth["timestamp"] >= start) & (eth["timestamp"] <= end)].copy()
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
            # Explicit crowding x leverage feature from prior diagnostics.
            if all(c in df.columns for c in ["basis_z", "funding_rate_z", "oi_chg_z"]):
                crowd = df["basis_z"] + df["funding_rate_z"]
                df["perp_pos_score"] = -(crowd * df["oi_chg_z"])
                df["perp_pos_ema"] = df["perp_pos_score"].ewm(span=12, adjust=False).mean()
                df["perp_pos_abs"] = df["perp_pos_score"].abs()
                feat_cols += ["perp_pos_score", "perp_pos_ema", "perp_pos_abs"]

    h = max(1, int(horizon_bars))
    df["fwd_r"] = np.log(df["eth_close"].shift(-h) / df["eth_close"])
    df["y_up"] = (df["fwd_r"] > 0).astype(int)

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

    keep = [
        "timestamp",
        "eth_close",
        "r",
        "fwd_r",
        "y_up",
        "event",
        "event_breakout",
        "event_vol",
        "event_crowd",
    ] + feat_cols
    df = df[keep].dropna().reset_index(drop=True)
    return df, feat_cols


def walk_forward_meta(
    df: pd.DataFrame,
    feat_cols: list[str],
    train_days: int,
    test_days: int,
    step_days: int,
    base_long_th: float,
    trade_cost_bps: float,
    min_meta_samples: int,
) -> pd.DataFrame:
    rows = []
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise RuntimeError("No valid timestamps for walk-forward")
    fold = 0
    max_folds = 10000
    cost_hurdle = float(trade_cost_bps) / 10000.0
    while True:
        if fold > max_folds:
            raise RuntimeError("walk-forward exceeded max folds")
        tr_start = t_min + pd.Timedelta(days=fold * step_days)
        tr_end = tr_start + pd.Timedelta(days=train_days)
        te_end = tr_end + pd.Timedelta(days=test_days)
        if te_end > t_max:
            break

        tr_all = df[(df["timestamp"] >= tr_start) & (df["timestamp"] < tr_end)].copy()
        te_all = df[(df["timestamp"] >= tr_end) & (df["timestamp"] < te_end)].copy()
        tr_event = tr_all[tr_all["event"] == 1].copy()
        te_event = te_all[te_all["event"] == 1].copy()
        if len(tr_all) < 5000 or len(te_all) < 1000 or len(tr_event) < 500 or len(te_event) < 50:
            fold += 1
            continue

        model_up = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=60,
            l2_regularization=1.0,
            random_state=42,
        )
        model_up.fit(tr_event[feat_cols].to_numpy(), tr_event["y_up"].to_numpy(dtype=int))
        tr_event = tr_event.copy()
        te_event = te_event.copy()
        tr_event["p_up"] = model_up.predict_proba(tr_event[feat_cols].to_numpy())[:, 1]
        te_event["p_up"] = model_up.predict_proba(te_event[feat_cols].to_numpy())[:, 1]

        tr_cand = tr_event[tr_event["p_up"] >= float(base_long_th)].copy()
        tr_cand["y_meta"] = (tr_cand["fwd_r"] > cost_hurdle).astype(int)
        meta_feats = feat_cols + ["p_up", "event_breakout", "event_vol", "event_crowd"]
        meta_ok = (
            len(tr_cand) >= int(min_meta_samples)
            and tr_cand["y_meta"].nunique() == 2
            and tr_cand["y_meta"].mean() > 0.02
            and tr_cand["y_meta"].mean() < 0.98
        )

        te_all = te_all.copy()
        te_all["fold"] = fold
        te_all["p_up"] = np.nan
        te_all["p_meta"] = np.nan
        te_all.loc[te_event.index, "p_up"] = te_event["p_up"].to_numpy(dtype=float)

        if meta_ok:
            model_meta = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.05,
                max_depth=3,
                max_iter=220,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=43,
            )
            model_meta.fit(tr_cand[meta_feats].to_numpy(), tr_cand["y_meta"].to_numpy(dtype=int))
            te_cand = te_event[te_event["p_up"] >= float(base_long_th)].copy()
            if len(te_cand) > 0:
                p_meta = model_meta.predict_proba(te_cand[meta_feats].to_numpy())[:, 1]
                te_all.loc[te_cand.index, "p_meta"] = p_meta

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
                    "p_meta",
                    "fold",
                ]
            ]
        )
        fold += 1

    if not rows:
        raise RuntimeError("No walk-forward folds produced event predictions")
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def run_meta_strategy(
    pred: pd.DataFrame,
    long_th: float,
    meta_th: float,
    hold_bars: int,
    max_weight: float,
    trade_cost_bps: float,
) -> pd.DataFrame:
    out = pred.copy()
    p_up = out["p_up"]
    p_meta = out["p_meta"]
    w = np.zeros(len(out), dtype=float)
    hold = 0
    cur_w = 0.0
    for i in range(len(out)):
        ev = bool(out["event"].iat[i]) and np.isfinite(p_up.iat[i]) and np.isfinite(p_meta.iat[i])
        if ev:
            pu = float(p_up.iat[i])
            pm = float(p_meta.iat[i])
            if pu >= long_th and pm >= meta_th:
                conf_up = (pu - long_th) / (1.0 - long_th + 1e-12)
                conf_meta = (pm - meta_th) / (1.0 - meta_th + 1e-12)
                cur_w = float(np.clip(conf_up * conf_meta, 0.0, 1.0) * max_weight)
                hold = int(hold_bars)
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
    ap = argparse.ArgumentParser(description="Event direction model v2 (meta-label filtered)")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--horizon-bars", type=int, default=12)
    ap.add_argument("--breakout-lookback", type=int, default=96)
    ap.add_argument("--vol-event-k", type=float, default=1.5)
    ap.add_argument("--basis-z-th", type=float, default=1.5)
    ap.add_argument("--funding-z-th", type=float, default=1.5)
    ap.add_argument("--oi-z-th", type=float, default=1.5)
    ap.add_argument("--wf-train-days", type=int, default=240)
    ap.add_argument("--wf-test-days", type=int, default=30)
    ap.add_argument("--wf-step-days", type=int, default=30)
    ap.add_argument("--base-long-th", type=float, default=0.58)
    ap.add_argument("--long-thresholds", default="0.58,0.60,0.62,0.65")
    ap.add_argument("--meta-thresholds", default="0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--hold-bars-list", default="6,12,24")
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--min-time-in-market-pct", type=float, default=1.0)
    ap.add_argument("--min-meta-samples", type=int, default=300)
    ap.add_argument("--out-preds-csv", default="artifacts/backtest/direction_event_model_v2_meta_preds.csv")
    ap.add_argument("--out-sweep-csv", default="artifacts/backtest/direction_event_model_v2_meta_sweep.csv")
    ap.add_argument("--out-best-csv", default="artifacts/backtest/direction_event_model_v2_meta.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/direction_event_model_v2_meta.html")
    args = ap.parse_args()

    df, feat_cols = build_frame(
        price_csv=Path(args.price_csv),
        btc_csv=Path(args.btc_csv) if Path(args.btc_csv).exists() else None,
        perp_csv=Path(args.perp_csv) if Path(args.perp_csv).exists() else None,
        window_days=int(args.window_days),
        horizon_bars=int(args.horizon_bars),
        breakout_lookback=int(args.breakout_lookback),
        vol_event_k=float(args.vol_event_k),
        basis_z_th=float(args.basis_z_th),
        funding_z_th=float(args.funding_z_th),
        oi_z_th=float(args.oi_z_th),
    )
    pred = walk_forward_meta(
        df=df,
        feat_cols=feat_cols,
        train_days=int(args.wf_train_days),
        test_days=int(args.wf_test_days),
        step_days=int(args.wf_step_days),
        base_long_th=float(args.base_long_th),
        trade_cost_bps=float(args.trade_cost_bps),
        min_meta_samples=int(args.min_meta_samples),
    )

    ev_pred = pred[pred["p_up"].notna()].copy()
    meta_pred = pred[pred["p_meta"].notna()].copy()
    if len(ev_pred) < 100:
        raise RuntimeError("Too few event predictions")
    auc_ev = float(roc_auc_score(ev_pred["y_up"].to_numpy(dtype=int), ev_pred["p_up"].to_numpy(dtype=float)))
    acc_ev = float(accuracy_score(ev_pred["y_up"].to_numpy(dtype=int), (ev_pred["p_up"] >= 0.5).astype(int)))
    if len(meta_pred) >= 100 and meta_pred["y_up"].nunique() == 2:
        auc_meta_proxy = float(roc_auc_score(meta_pred["y_up"].to_numpy(dtype=int), meta_pred["p_meta"].to_numpy(dtype=float)))
    else:
        auc_meta_proxy = np.nan

    longs = sorted(set([float(x) for x in str(args.long_thresholds).split(",") if str(x).strip()]))
    metas = sorted(set([float(x) for x in str(args.meta_thresholds).split(",") if str(x).strip()]))
    holds = sorted(set([int(float(x)) for x in str(args.hold_bars_list).split(",") if str(x).strip()]))

    sweep_rows = []
    best_score = -1e18
    best_sim: pd.DataFrame | None = None
    best_cfg = None
    for h in holds:
        for lth in longs:
            for mth in metas:
                sim = run_meta_strategy(
                    pred=pred,
                    long_th=lth,
                    meta_th=mth,
                    hold_bars=int(h),
                    max_weight=float(args.max_weight),
                    trade_cost_bps=float(args.trade_cost_bps),
                )
                m = perf(sim["strat_r"], 5)
                ms = perf(sim["spot_r"], 5)
                tim = float(sim["active"].mean() * 100.0)
                active = sim["active"].astype(bool)
                hit_active = float((np.sign(sim.loc[active, "fwd_r"]) > 0).mean()) if active.any() else np.nan
                row = {
                    "hold_bars": int(h),
                    "long_th": float(lth),
                    "meta_th": float(mth),
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
                score = (0.0 if np.isnan(row["sharpe"]) else row["sharpe"]) + 1.0 * row["ret"] - 0.0002 * row["turnover"]
                if tim < float(args.min_time_in_market_pct):
                    score -= 2.5
                if score > best_score:
                    best_score = score
                    best_sim = sim
                    best_cfg = (h, lth, mth)

    if best_sim is None or best_cfg is None:
        raise RuntimeError("No valid strategy configuration found")

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
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity", "P(up)", "P(meta)", "Weight", "Event Flags", "Price"],
    )
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["p_up"], name="P(up)", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_hline(y=float(best_cfg[1]), line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["p_meta"], name="P(meta)", line=dict(color="#17becf")), row=3, col=1)
    fig.add_hline(y=float(best_cfg[2]), line_dash="dash", line_color="#d62728", row=3, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["weight"], name="Weight", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_breakout"], name="Breakout Event", line=dict(color="#17becf")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_crowd"], name="Crowd Event", line=dict(color="#8c564b")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_vol"], name="Vol Event", line=dict(color="#bcbd22")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=6, col=1)
    fig.update_layout(height=1600, title="Event Direction Model v2 Meta")

    summary = pd.DataFrame(
        [
            {
                "rows_oos": int(len(pred)),
                "rows_event_preds": int(len(ev_pred)),
                "rows_meta_preds": int(len(meta_pred)),
                "event_rate_pct": float((pred["event"] == 1).mean() * 100.0),
                "oos_auc_event": auc_ev,
                "oos_acc_event": acc_ev,
                "oos_auc_meta_proxy": auc_meta_proxy,
                "best_hold_bars": int(best_cfg[0]),
                "best_long_th": float(best_cfg[1]),
                "best_meta_th": float(best_cfg[2]),
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
            }
        ]
    )

    html = (
        "<html><head><meta charset='utf-8'><title>Event Direction Model v2 Meta</title></head><body>"
        "<h3>Event Direction Model v2 Meta</h3>"
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
        "oos_auc_event={:.4f} oos_acc_event={:.4f} rows_meta={} best_hold={} best_long={:.2f} best_meta={:.2f} ret={:.4f} sharpe={:.4f} tim={:.2f}%".format(
            auc_ev,
            acc_ev,
            int(len(meta_pred)),
            int(best_cfg[0]),
            float(best_cfg[1]),
            float(best_cfg[2]),
            m_best["ret"],
            m_best["sharpe"],
            float(best_sim["active"].mean() * 100.0),
        )
    )


if __name__ == "__main__":
    main()
