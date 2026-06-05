from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


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
) -> tuple[pd.DataFrame, list[str]]:
    eth = load_price(price_csv).rename(columns={"close": "eth_close"})
    end = eth["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=window_days)
    eth = eth[(eth["timestamp"] >= start) & (eth["timestamp"] <= end)].copy()
    eth["r"] = np.log(eth["eth_close"] / eth["eth_close"].shift(1))

    df = eth[["timestamp", "eth_close", "r"]].copy()
    df["ret_1"] = df["r"]
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
    df["accel"] = df["ret_12"] - df["ret_36"] / 3.0

    feat_cols = [
        "ret_1",
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
        "accel",
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
                    zc = f"{c}_z"
                    df[zc] = rolling_z(df[c], 288)
                    feat_cols.append(zc)

    # Interaction features
    df["mom_x_trend"] = df["ret_12"] * df["trend_gap"]
    df["mom_x_vol"] = df["ret_12"] / (df["vol_36"] + 1e-12)
    feat_cols += ["mom_x_trend", "mom_x_vol"]

    h = max(1, int(horizon_bars))
    df["fwd_r"] = np.log(df["eth_close"].shift(-h) / df["eth_close"])
    df["y_up"] = (df["fwd_r"] > 0).astype(int)
    df = df[["timestamp", "eth_close", "r", "fwd_r", "y_up"] + feat_cols].dropna().reset_index(drop=True)
    return df, feat_cols


def wf_predict(
    df: pd.DataFrame,
    feat_cols: list[str],
    train_days: int,
    test_days: int,
    step_days: int,
) -> pd.DataFrame:
    rows = []
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    fold = 0
    while True:
        tr_start = t_min + pd.Timedelta(days=fold * step_days)
        tr_end = tr_start + pd.Timedelta(days=train_days)
        te_end = tr_end + pd.Timedelta(days=test_days)
        if te_end > t_max:
            break
        tr = df[(df["timestamp"] >= tr_start) & (df["timestamp"] < tr_end)]
        te = df[(df["timestamp"] >= tr_end) & (df["timestamp"] < te_end)]
        if len(tr) < 5000 or len(te) < 1000:
            fold += 1
            continue

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(tr[feat_cols].to_numpy(), tr["y_up"].to_numpy(dtype=int))
        p_up = model.predict_proba(te[feat_cols].to_numpy())[:, 1]
        part = te[["timestamp", "eth_close", "r", "fwd_r", "y_up"]].copy()
        part["p_up"] = p_up
        part["fold"] = fold
        rows.append(part)
        fold += 1

    if not rows:
        raise RuntimeError("No walk-forward folds produced outputs")
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def simulate(
    pred: pd.DataFrame,
    long_th: float,
    short_th: float,
    max_weight: float,
    max_dw: float,
    trade_cost_bps: float,
) -> pd.DataFrame:
    p = pred["p_up"]
    sig = pd.Series(0.0, index=pred.index)
    long_m = p >= long_th
    short_m = p <= short_th
    sig[long_m] = (p[long_m] - long_th) / (1.0 - long_th + 1e-12)
    sig[short_m] = -(short_th - p[short_m]) / (short_th + 1e-12)
    sig = sig.clip(-1.0, 1.0)
    w_tgt = sig * float(max_weight)
    w = apply_step_cap(w_tgt, max_dw=float(max_dw), w_max=float(max_weight))
    turnover = w.diff().abs().fillna(0.0)
    cost = turnover * (float(trade_cost_bps) / 10000.0)
    out = pred.copy()
    out["weight_target"] = w_tgt
    out["weight"] = w
    out["turnover"] = turnover
    out["cost_r"] = -cost
    out["strat_r"] = w.shift(1).fillna(0.0) * out["r"] - cost
    out["spot_r"] = out["r"]
    out["eq"] = np.exp(np.cumsum(out["strat_r"].to_numpy()))
    out["spot_eq"] = np.exp(np.cumsum(out["spot_r"].to_numpy()))
    out["active"] = (out["weight"].abs() > 1e-12).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Direction model v2 (GBDT + walk-forward + threshold sweep)")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--btc-csv", default="data/BTCUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--horizon-bars", type=int, default=12)
    ap.add_argument("--wf-train-days", type=int, default=240)
    ap.add_argument("--wf-test-days", type=int, default=30)
    ap.add_argument("--wf-step-days", type=int, default=30)
    ap.add_argument("--long-thresholds", default="0.53,0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--short-thresholds", default="0.47,0.45,0.42,0.40,0.38,0.35")
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--max-dw-per-bar", type=float, default=0.10)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--min-time-in-market-pct", type=float, default=5.0)
    ap.add_argument("--min-avg-abs-weight", type=float, default=0.01)
    ap.add_argument("--out-preds-csv", default="artifacts/backtest/direction_model_v2_preds.csv")
    ap.add_argument("--out-sweep-csv", default="artifacts/backtest/direction_model_v2_sweep.csv")
    ap.add_argument("--out-best-csv", default="artifacts/backtest/direction_model_v2.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/direction_model_v2.html")
    args = ap.parse_args()

    btc_path = Path(args.btc_csv)
    perp_path = Path(args.perp_csv)
    df, feat_cols = build_frame(
        price_csv=Path(args.price_csv),
        btc_csv=btc_path if btc_path.exists() else None,
        perp_csv=perp_path if perp_path.exists() else None,
        window_days=int(args.window_days),
        horizon_bars=int(args.horizon_bars),
    )
    pred = wf_predict(
        df=df,
        feat_cols=feat_cols,
        train_days=int(args.wf_train_days),
        test_days=int(args.wf_test_days),
        step_days=int(args.wf_step_days),
    )

    auc = float(roc_auc_score(pred["y_up"].to_numpy(dtype=int), pred["p_up"].to_numpy(dtype=float)))
    acc = float(accuracy_score(pred["y_up"].to_numpy(dtype=int), (pred["p_up"] >= 0.5).astype(int)))

    longs = sorted(set(parse_float_list(args.long_thresholds)))
    shorts = sorted(set(parse_float_list(args.short_thresholds)), reverse=True)
    sweep_rows = []
    best_score = -1e18
    best_cfg = None
    best_sim: pd.DataFrame | None = None

    for lth in longs:
        for sth in shorts:
            if sth >= lth:
                continue
            sim = simulate(
                pred=pred,
                long_th=lth,
                short_th=sth,
                max_weight=float(args.max_weight),
                max_dw=float(args.max_dw_per_bar),
                trade_cost_bps=float(args.trade_cost_bps),
            )
            m = perf(sim["strat_r"], 5)
            ms = perf(sim["spot_r"], 5)
            tim = float(sim["active"].mean() * 100.0)
            avg_abs_w = float(sim["weight"].abs().mean())
            turnover = float(sim["turnover"].sum())
            active = sim["active"].astype(bool)
            hit_active = float((np.sign(sim.loc[active, "weight"]) == np.sign(sim.loc[active, "fwd_r"])).mean()) if active.any() else np.nan
            row = {
                "long_th": lth,
                "short_th": sth,
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "excess_vs_spot": m["ret"] - ms["ret"],
                "time_in_market_pct": tim,
                "avg_abs_weight": avg_abs_w,
                "turnover": turnover,
                "hit_active": hit_active,
            }
            sweep_rows.append(row)

            # Penalize fake "flat winners"
            score = (0.0 if np.isnan(row["sharpe"]) else row["sharpe"]) + 1.5 * row["ret"] - 0.0002 * turnover
            if tim < float(args.min_time_in_market_pct):
                score -= 4.0 * (float(args.min_time_in_market_pct) - tim) / max(1.0, float(args.min_time_in_market_pct))
            if avg_abs_w < float(args.min_avg_abs_weight):
                score -= 2.0 * (float(args.min_avg_abs_weight) - avg_abs_w) / max(1e-6, float(args.min_avg_abs_weight))
            if score > best_score:
                best_score = score
                best_cfg = (lth, sth)
                best_sim = sim

    if best_sim is None or best_cfg is None:
        raise RuntimeError("No simulation output")

    sweep = pd.DataFrame(sweep_rows).sort_values(["sharpe", "ret"], ascending=[False, False]).reset_index(drop=True)
    best_m = perf(best_sim["strat_r"], 5)
    spot_m = perf(best_sim["spot_r"], 5)

    Path(args.out_preds_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_preds_csv, index=False)
    Path(args.out_sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out_sweep_csv, index=False)
    Path(args.out_best_csv).parent.mkdir(parents=True, exist_ok=True)
    best_sim.to_csv(args.out_best_csv, index=False)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity", "P(up)", "Weight", "Price"],
    )
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eq"], name="Strategy Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["p_up"], name="P(up)", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_hline(y=float(best_cfg[0]), line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.add_hline(y=float(best_cfg[1]), line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["weight"], name="Weight", line=dict(color="#9467bd")), row=3, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eth_close"], name="ETH Close", line=dict(color="#17becf")), row=4, col=1)
    fig.update_layout(height=1200, title="Direction Model v2")

    summary = pd.DataFrame(
        [
            {
                "rows_oos": int(len(pred)),
                "oos_auc": auc,
                "oos_acc": acc,
                "up_rate_oos": float(pred["y_up"].mean()),
                "best_long_th": float(best_cfg[0]),
                "best_short_th": float(best_cfg[1]),
                "ret": best_m["ret"],
                "cagr": best_m["cagr"],
                "ann_vol": best_m["ann_vol"],
                "sharpe": best_m["sharpe"],
                "max_dd": best_m["max_dd"],
                "spot_ret": spot_m["ret"],
                "excess_vs_spot": best_m["ret"] - spot_m["ret"],
                "time_in_market_pct": float(best_sim["active"].mean() * 100.0),
                "avg_abs_weight": float(best_sim["weight"].abs().mean()),
                "turnover": float(best_sim["turnover"].sum()),
                "trade_cost_bps": float(args.trade_cost_bps),
            }
        ]
    )

    html = (
        "<html><head><meta charset='utf-8'><title>Direction Model v2</title></head><body>"
        "<h3>Direction Model v2 (GBDT + walk-forward)</h3>"
        f"<p>price_csv={args.price_csv} | btc_used={btc_path.exists()} | perp_used={perp_path.exists()}</p>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        "<h4>Top 10 Threshold Configs</h4>"
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
        "oos_auc={:.4f} oos_acc={:.4f} best_long={:.2f} best_short={:.2f} ret={:.4f} sharpe={:.4f} max_dd={:.4f} tim={:.2f}%".format(
            auc,
            acc,
            float(best_cfg[0]),
            float(best_cfg[1]),
            best_m["ret"],
            best_m["sharpe"],
            best_m["max_dd"],
            float(best_sim["active"].mean() * 100.0),
        )
    )


if __name__ == "__main__":
    main()
