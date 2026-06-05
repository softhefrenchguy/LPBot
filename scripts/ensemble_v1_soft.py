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


def apply_step_cap(target: pd.Series, max_dw: float, w_max: float) -> pd.Series:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, x in enumerate(pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=float)):
        x = float(np.clip(x, -w_max, w_max))
        lo, hi = prev - max_dw, prev + max_dw
        v = min(max(x, lo), hi)
        out[i] = v
        prev = v
    return pd.Series(out, index=target.index)


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bpy = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bpy / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bpy)),
        "sharpe": float((np.mean(x) * bpy) / (np.std(x) * np.sqrt(bpy) + 1e-12)),
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
        "trend_gap",
        "basis_z",
        "funding_rate_z",
        "oi_chg_z",
    ] + feat_cols
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna().reset_index(drop=True)
    return df, feat_cols


def walk_forward_event_probs(
    df: pd.DataFrame,
    feat_cols: list[str],
    train_days: int,
    test_days: int,
    step_days: int,
) -> pd.DataFrame:
    rows = []
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise RuntimeError("No valid timestamps for walk-forward")
    fold = 0
    max_folds = 10000
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
        if len(tr_all) < 5000 or len(te_all) < 1000 or len(tr) < 500 or len(te_event) < 50:
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
        p_ev = model.predict_proba(te_event[feat_cols].to_numpy())[:, 1]
        te_all.loc[te_event.index, "p_up"] = p_ev
        rows.append(te_all)
        fold += 1

    if not rows:
        raise RuntimeError("No walk-forward folds produced event predictions")
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def run_ensemble(
    pred: pd.DataFrame,
    event_w: float,
    funding_w: float,
    perp_w: float,
    max_weight: float,
    max_dw_per_bar: float,
    trade_cost_bps: float,
    long_only: bool,
    bull_up_th: float,
    bull_min_weight: float,
    deadband: float,
    hold_bars: int,
) -> pd.DataFrame:
    out = pred.copy()
    event_conf = ((out["p_up"] - 0.5) / 0.5).clip(-1.0, 1.0).fillna(0.0)

    if "basis_z" in out.columns and "funding_rate_z" in out.columns:
        fb_raw = -(0.6 * out["basis_z"] + 0.4 * out["funding_rate_z"])
        funding_conf = (fb_raw / 2.5).clip(-1.0, 1.0).fillna(0.0)
    else:
        funding_conf = pd.Series(0.0, index=out.index)

    if all(c in out.columns for c in ["basis_z", "funding_rate_z", "oi_chg_z"]):
        crowd = out["basis_z"] + out["funding_rate_z"]
        perp_pos = (crowd * out["oi_chg_z"]).ewm(span=12, adjust=False).mean()
        perp_conf = (perp_pos / 3.0).clip(-1.0, 1.0).fillna(0.0)
    else:
        perp_conf = pd.Series(0.0, index=out.index)

    gate = (out["event"] == 1) & out["p_up"].notna()
    funding_conf = funding_conf.where(gate, 0.0)
    perp_conf = perp_conf.where(gate, 0.0)

    w_sum = abs(event_w) + abs(funding_w) + abs(perp_w)
    if w_sum <= 0:
        raise ValueError("Invalid ensemble weights")
    score_raw = (event_w * event_conf + funding_w * funding_conf + perp_w * perp_conf) / w_sum
    score = score_raw.ewm(span=12, adjust=False).mean()
    raw_target = score * float(max_weight)

    if long_only:
        raw_target = raw_target.clip(lower=0.0)

    bull_on = (out["p_up"] >= float(bull_up_th)) & (out.get("trend_gap", pd.Series(0.0, index=out.index)) > 0.0)
    raw_target = raw_target.where(~bull_on, np.maximum(raw_target, float(bull_min_weight)))
    raw_target = raw_target.where(raw_target.abs() >= float(deadband), 0.0)

    # Hold target for a short window after each event so we do not rebalance every bar.
    held_target = np.zeros(len(out), dtype=float)
    hold = 0
    cur = 0.0
    for i in range(len(out)):
        if bool(gate.iat[i]):
            cur = float(raw_target.iat[i])
            hold = int(hold_bars)
        if hold > 0:
            held_target[i] = cur
            hold -= 1
        else:
            held_target[i] = 0.0
            cur = 0.0
    raw_target = pd.Series(held_target, index=out.index)

    out["event_conf"] = event_conf
    out["funding_conf"] = funding_conf
    out["perp_conf"] = perp_conf
    out["score"] = score
    out["weight_target"] = raw_target
    out["weight"] = apply_step_cap(out["weight_target"], max_dw=float(max_dw_per_bar), w_max=float(max_weight))
    out["turnover"] = out["weight"].diff().abs().fillna(0.0)
    out["cost_r"] = -out["turnover"] * (float(trade_cost_bps) / 10000.0)
    out["strat_r"] = out["weight"].shift(1).fillna(0.0) * out["r"] + out["cost_r"]
    out["spot_r"] = out["r"]
    out["eq"] = np.exp(np.cumsum(out["strat_r"].to_numpy()))
    out["spot_eq"] = np.exp(np.cumsum(out["spot_r"].to_numpy()))
    out["active"] = (out["weight"].abs() > 1e-12).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Soft ensemble v1 (event + perp features)")
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
    ap.add_argument("--event-w", type=float, default=0.5)
    ap.add_argument("--funding-w", type=float, default=0.3)
    ap.add_argument("--perp-w", type=float, default=0.2)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--bull-up-th", type=float, default=0.60)
    ap.add_argument("--bull-min-weight", type=float, default=0.15)
    ap.add_argument("--deadband", type=float, default=0.03)
    ap.add_argument("--hold-bars", type=int, default=12)
    ap.add_argument("--out-preds-csv", default="artifacts/backtest/ensemble_v1_preds.csv")
    ap.add_argument("--out-csv", default="artifacts/backtest/ensemble_v1.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/ensemble_v1.html")
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
    pred = walk_forward_event_probs(
        df=df,
        feat_cols=feat_cols,
        train_days=int(args.wf_train_days),
        test_days=int(args.wf_test_days),
        step_days=int(args.wf_step_days),
    )
    sim = run_ensemble(
        pred=pred,
        event_w=float(args.event_w),
        funding_w=float(args.funding_w),
        perp_w=float(args.perp_w),
        max_weight=float(args.max_weight),
        max_dw_per_bar=float(args.max_dw_per_bar),
        trade_cost_bps=float(args.trade_cost_bps),
        long_only=bool(args.long_only),
        bull_up_th=float(args.bull_up_th),
        bull_min_weight=float(args.bull_min_weight),
        deadband=float(args.deadband),
        hold_bars=int(args.hold_bars),
    )

    ev_pred = pred[pred["p_up"].notna()].copy()
    auc_ev = float(roc_auc_score(ev_pred["y_up"].to_numpy(dtype=int), ev_pred["p_up"].to_numpy(dtype=float)))
    acc_ev = float(accuracy_score(ev_pred["y_up"].to_numpy(dtype=int), (ev_pred["p_up"] >= 0.5).astype(int)))
    m = perf(sim["strat_r"], 5)
    ms = perf(sim["spot_r"], 5)

    Path(args.out_preds_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_preds_csv, index=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(args.out_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Equity", "Model Prob + Ensemble Score", "Weight", "Signal Components", "Price"],
    )
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["p_up"], name="P(up)", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["score"], name="Ensemble Score", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_hline(y=float(args.bull_up_th), line_dash="dash", line_color="#9467bd", row=2, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["weight"], name="Weight", line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["event_conf"], name="Event Conf", line=dict(color="#1f77b4")), row=4, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["funding_conf"], name="Funding Conf", line=dict(color="#8c564b")), row=4, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["perp_conf"], name="Perp Conf", line=dict(color="#17becf")), row=4, col=1)
    fig.add_trace(go.Scatter(x=sim["timestamp"], y=sim["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=5, col=1)
    fig.update_layout(height=1450, title="Soft Ensemble v1")

    summary = pd.DataFrame(
        [
            {
                "rows_oos": int(len(sim)),
                "event_rate_pct": float((sim["event"] == 1).mean() * 100.0),
                "oos_auc_event": auc_ev,
                "oos_acc_event": acc_ev,
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "excess_vs_spot": m["ret"] - ms["ret"],
                "time_in_market_pct": float(sim["active"].mean() * 100.0),
                "avg_abs_weight": float(sim["weight"].abs().mean()),
                "turnover": float(sim["turnover"].sum()),
                "trade_cost_bps": float(args.trade_cost_bps),
                "event_w": float(args.event_w),
                "funding_w": float(args.funding_w),
                "perp_w": float(args.perp_w),
            }
        ]
    )
    html = (
        "<html><head><meta charset='utf-8'><title>Soft Ensemble v1</title></head><body>"
        "<h3>Soft Ensemble v1</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {args.out_preds_csv}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_html}")
    print(
        "oos_auc_event={:.4f} oos_acc_event={:.4f} ret={:.4f} sharpe={:.4f} max_dd={:.4f} tim={:.2f}% turnover={:.2f}".format(
            auc_ev,
            acc_ev,
            m["ret"],
            m["sharpe"],
            m["max_dd"],
            float(sim["active"].mean() * 100.0),
            float(sim["turnover"].sum()),
        )
    )


if __name__ == "__main__":
    main()
