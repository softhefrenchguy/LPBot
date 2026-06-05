from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _resampled_close_no_lookahead(ts: pd.Series, close: pd.Series, tf: str) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True, errors="coerce"))
    tmp = pd.DataFrame({"close": pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)}, index=idx)
    if tf == "5m":
        return tmp["close"]
    return tmp["close"].resample(tf).last().shift(1).ffill()


def _metrics(log_r: pd.Series, bar_minutes: int = 5) -> dict:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy()
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    ret = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (bars_per_year / len(x)) - 1.0)
    ann_vol = float(np.std(x) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x) * bars_per_year) / (ann_vol + 1e-12))
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"ret": ret, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd}


def _load_series(path: Path, ts_col: str, r_col: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if ts_col not in df.columns or r_col not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
            "ret": pd.to_numeric(df[r_col], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out["ret"] = out["ret"].fillna(0.0)
    return out


def _eq_from_ret(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp")
    d[name] = np.exp(np.cumsum(pd.to_numeric(d["ret"], errors="coerce").fillna(0.0).to_numpy()))
    return d[["timestamp", name]]


def _weight_from_prob(p: np.ndarray, bear_th: float, bull_th: float, w_bear: float, w_neutral: float, w_bull: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    w = np.full_like(p, w_neutral)
    w = np.where(p <= bear_th, w_bear, w)
    w = np.where(p >= bull_th, w_bull, w)
    mid = (p > bear_th) & (p < bull_th)
    if bull_th > bear_th:
        t = (p[mid] - bear_th) / (bull_th - bear_th)
        w[mid] = w_neutral + t * (w_bull - w_neutral)
    return np.clip(w, 0.0, 1.0)


def _apply_dd_guard(weight: np.ndarray, r: np.ndarray, dd_kill: float, dd_recover: float, kill_weight: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weight, dtype=float).copy()
    rr = np.asarray(r, dtype=float)
    n = len(w)
    kill_on = np.zeros(n, dtype=bool)
    eq = 1.0
    peak = 1.0
    kill = False
    for i in range(n):
        ret_i = (w[i - 1] if i > 0 else 0.0) * rr[i]
        eq *= float(np.exp(ret_i))
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0
        if not kill and dd <= dd_kill:
            kill = True
        elif kill and dd >= dd_recover:
            kill = False
        if kill:
            w[i] = min(w[i], kill_weight)
        kill_on[i] = kill
    return w, kill_on


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple bull regime classifier + strategy report.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--horizon-bars", type=int, default=288, help="Future bars for bull target (288=1d on 5m).")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--bull-threshold", type=float, default=0.60)
    ap.add_argument("--bear-threshold", type=float, default=0.40)
    ap.add_argument("--w-bear", type=float, default=0.05)
    ap.add_argument("--w-neutral", type=float, default=0.25)
    ap.add_argument("--w-bull", type=float, default=0.80)
    ap.add_argument("--dd-kill", type=float, default=-0.20)
    ap.add_argument("--dd-recover", type=float, default=-0.08)
    ap.add_argument("--kill-weight", type=float, default=0.05)
    ap.add_argument("--wf-train-days", type=int, default=365)
    ap.add_argument("--wf-test-days", type=int, default=90)
    ap.add_argument("--wf-step-days", type=int, default=90)
    ap.add_argument("--out-wf-csv", default="artifacts/paper/bull_model_walkforward.csv")
    ap.add_argument("--out-preds-csv", default="artifacts/paper/bull_model_preds.csv")
    ap.add_argument("--out-report-html", default="artifacts/paper/bull_model_report.html")
    ap.add_argument("--out-strat-csv", default="artifacts/paper/bull_model_strategy.csv")
    ap.add_argument("--compare-v5-csv", default="artifacts/paper/hmm_enet_scaled_v5.csv")
    ap.add_argument("--compare-breakout-csv", default="artifacts/paper/breakout_paper.csv")
    args = ap.parse_args()

    price = pd.read_csv(args.price_csv)
    if "timestamp" not in price.columns or "close" not in price.columns:
        raise ValueError("price csv must contain timestamp, close")
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, errors="coerce")
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    ts = price["timestamp"]
    idx = pd.DatetimeIndex(ts.to_numpy())
    close = price["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    c_1h = _resampled_close_no_lookahead(ts, close, "1h").reindex(idx, method="ffill").ffill()
    c_4h = _resampled_close_no_lookahead(ts, close, "4h").reindex(idx, method="ffill").ffill()
    ema50_1h = c_1h.ewm(span=50, adjust=False).mean()
    ema200_1h = c_1h.ewm(span=200, adjust=False).mean()
    ema200_4h = c_4h.ewm(span=200, adjust=False).mean()

    feat = pd.DataFrame(
        {
            "timestamp": ts.to_numpy(),
            "close": close.to_numpy(),
            "r": r.to_numpy(),
            "f_ema_gap_1h": ((ema50_1h - ema200_1h) / (ema200_1h.abs() + 1e-12)).to_numpy(),
            "f_price_gap_1h": ((c_1h - ema200_1h) / (ema200_1h.abs() + 1e-12)).to_numpy(),
            "f_price_gap_4h": ((c_4h - ema200_4h) / (ema200_4h.abs() + 1e-12)).to_numpy(),
            "f_mom_24h": np.log(close / close.shift(288)).to_numpy(),
            "f_vol_24h": r.rolling(288, min_periods=288).std().to_numpy(),
        }
    )

    h = max(1, int(args.horizon_bars))
    feat["fwd_r"] = np.log(close.shift(-h) / close)
    feat["bull_target"] = (feat["fwd_r"] > 0).astype(int)
    feat = feat.dropna().reset_index(drop=True)

    feature_cols = ["f_ema_gap_1h", "f_price_gap_1h", "f_price_gap_4h", "f_mom_24h", "f_vol_24h"]
    cut = int(len(feat) * float(args.train_ratio))
    cut = min(max(cut, 2000), len(feat) - 1000)

    train = feat.iloc[:cut].copy()
    test = feat.iloc[cut:].copy()

    X_train = train[feature_cols].to_numpy()
    y_train = train["bull_target"].to_numpy()
    X_all = feat[feature_cols].to_numpy()
    y_all = feat["bull_target"].to_numpy()

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    feat["p_bull"] = clf.predict_proba(X_all)[:, 1]
    feat["pred_bull"] = (feat["p_bull"] >= 0.5).astype(int)
    feat["correct"] = (feat["pred_bull"] == feat["bull_target"]).astype(int)

    bull_th = float(args.bull_threshold)
    bear_th = float(args.bear_threshold)
    if bear_th >= bull_th:
        raise ValueError("bear_threshold must be lower than bull_threshold")

    p = feat["p_bull"].to_numpy()
    w_raw = _weight_from_prob(
        p,
        bear_th=bear_th,
        bull_th=bull_th,
        w_bear=float(args.w_bear),
        w_neutral=float(args.w_neutral),
        w_bull=float(args.w_bull),
    )
    w, kill_on = _apply_dd_guard(
        w_raw,
        feat["r"].to_numpy(),
        dd_kill=float(args.dd_kill),
        dd_recover=float(args.dd_recover),
        kill_weight=float(args.kill_weight),
    )
    feat["weight_raw"] = w_raw
    feat["kill_on"] = kill_on
    feat["weight"] = w
    feat["strat_r"] = feat["weight"].shift(1).fillna(0.0) * feat["r"]
    feat["eq"] = np.exp(np.cumsum(feat["strat_r"].to_numpy()))
    feat["spot_eq"] = np.exp(np.cumsum(feat["r"].to_numpy()))

    # Classifier stats
    y_test = test["bull_target"].to_numpy()
    p_test = feat.iloc[cut:]["p_bull"].to_numpy()
    pred_test = (p_test >= 0.5).astype(int)
    cls = {
        "rows_train": len(train),
        "rows_test": len(test),
        "target_up_rate_test": float(y_test.mean()),
        "accuracy_test": float(accuracy_score(y_test, pred_test)),
        "precision_bull_test": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall_bull_test": float(recall_score(y_test, pred_test, zero_division=0)),
        "auc_test": float(roc_auc_score(y_test, p_test)),
    }

    # Walk-forward OOS evaluation
    wf_rows = []
    wf_preds = []
    t0 = feat["timestamp"].min()
    tmax = feat["timestamp"].max()
    fold = 0
    while True:
        train_start = t0 + pd.Timedelta(days=fold * int(args.wf_step_days))
        train_end = train_start + pd.Timedelta(days=int(args.wf_train_days))
        test_end = train_end + pd.Timedelta(days=int(args.wf_test_days))
        if test_end > tmax:
            break
        tr = feat[(feat["timestamp"] >= train_start) & (feat["timestamp"] < train_end)]
        te = feat[(feat["timestamp"] >= train_end) & (feat["timestamp"] < test_end)]
        if len(tr) < 5000 or len(te) < 1000:
            fold += 1
            continue

        clf_wf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf_wf.fit(tr[feature_cols].to_numpy(), tr["bull_target"].to_numpy())
        p_te = clf_wf.predict_proba(te[feature_cols].to_numpy())[:, 1]
        pred_te = (p_te >= 0.5).astype(int)

        w_te_raw = _weight_from_prob(
            p_te,
            bear_th=bear_th,
            bull_th=bull_th,
            w_bear=float(args.w_bear),
            w_neutral=float(args.w_neutral),
            w_bull=float(args.w_bull),
        )
        w_te, _ = _apply_dd_guard(
            w_te_raw,
            te["r"].to_numpy(),
            dd_kill=float(args.dd_kill),
            dd_recover=float(args.dd_recover),
            kill_weight=float(args.kill_weight),
        )
        strat_r_te = pd.Series(w_te).shift(1).fillna(0.0).to_numpy() * te["r"].to_numpy()
        m_te = _metrics(pd.Series(strat_r_te), 5)
        spot_te = _metrics(te["r"], 5)
        wf_rows.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_end": test_end,
                "rows_train": len(tr),
                "rows_test": len(te),
                "acc_test": float(accuracy_score(te["bull_target"].to_numpy(), pred_te)),
                "auc_test": float(roc_auc_score(te["bull_target"].to_numpy(), p_te)),
                "ret_test": m_te["ret"],
                "spot_ret_test": spot_te["ret"],
                "excess_test": m_te["ret"] - spot_te["ret"],
                "sharpe_test": m_te["sharpe"],
                "max_dd_test": m_te["max_dd"],
                "avg_w_test": float(np.mean(w_te)),
            }
        )
        wf_part = pd.DataFrame(
            {
                "timestamp": te["timestamp"].to_numpy(),
                "r": te["r"].to_numpy(),
                "strat_r": strat_r_te,
                "p_bull": p_te,
                "pred_bull": pred_te,
                "bull_target": te["bull_target"].to_numpy(),
                "fold": fold,
            }
        )
        wf_preds.append(wf_part)
        fold += 1

    wf_df = pd.DataFrame(wf_rows)
    if not wf_df.empty:
        wf_preds_df = pd.concat(wf_preds, ignore_index=True).sort_values("timestamp")
        wf_preds_df = wf_preds_df.drop_duplicates(subset=["timestamp"], keep="first")
        wf_oos = {
            "wf_folds": int(len(wf_df)),
            "wf_acc_mean": float(wf_df["acc_test"].mean()),
            "wf_auc_mean": float(wf_df["auc_test"].mean()),
            "wf_excess_mean": float(wf_df["excess_test"].mean()),
            "wf_sharpe_mean": float(wf_df["sharpe_test"].mean()),
            "wf_max_dd_worst": float(wf_df["max_dd_test"].min()),
        }
    else:
        wf_preds_df = pd.DataFrame(columns=["timestamp", "r", "strat_r", "p_bull", "pred_bull", "bull_target", "fold"])
        wf_oos = {
            "wf_folds": 0,
            "wf_acc_mean": np.nan,
            "wf_auc_mean": np.nan,
            "wf_excess_mean": np.nan,
            "wf_sharpe_mean": np.nan,
            "wf_max_dd_worst": np.nan,
        }

    # Strategy stats
    full_m = _metrics(feat["strat_r"], 5)
    test_m = _metrics(feat.iloc[cut:]["strat_r"], 5)
    full_spot = _metrics(feat["r"], 5)
    test_spot = _metrics(feat.iloc[cut:]["r"], 5)

    strat = {
        "ret_full": full_m["ret"],
        "ret_test": test_m["ret"],
        "spot_ret_full": full_spot["ret"],
        "spot_ret_test": test_spot["ret"],
        "excess_full": full_m["ret"] - full_spot["ret"],
        "excess_test": test_m["ret"] - test_spot["ret"],
        "cagr_full": full_m["cagr"],
        "cagr_test": test_m["cagr"],
        "sharpe_full": full_m["sharpe"],
        "sharpe_test": test_m["sharpe"],
        "max_dd_full": full_m["max_dd"],
        "max_dd_test": test_m["max_dd"],
        "avg_weight_full": float(feat["weight"].mean()),
        "avg_weight_test": float(feat.iloc[cut:]["weight"].mean()),
        **wf_oos,
    }

    out_preds = Path(args.out_preds_csv)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    feat[
        [
            "timestamp",
            "close",
            "r",
            "fwd_r",
            "bull_target",
            "p_bull",
            "pred_bull",
            "correct",
            "weight",
            "strat_r",
            "eq",
            "spot_eq",
        ]
    ].to_csv(out_preds, index=False)

    out_strat = Path(args.out_strat_csv)
    feat[["timestamp", "r", "strat_r", "weight", "eq", "spot_eq", "p_bull", "bull_target"]].to_csv(out_strat, index=False)
    out_wf = Path(args.out_wf_csv)
    out_wf.parent.mkdir(parents=True, exist_ok=True)
    wf_df.to_csv(out_wf, index=False)

    # Build comparison series
    cmp = []
    model_eq = _eq_from_ret(pd.DataFrame({"timestamp": feat["timestamp"], "ret": feat["strat_r"]}), "BullModel Eq")
    spot_eq = _eq_from_ret(pd.DataFrame({"timestamp": feat["timestamp"], "ret": feat["r"]}), "Spot Eq")
    cmp.append(model_eq)
    cmp.append(spot_eq)

    v5 = _load_series(Path(args.compare_v5_csv), "timestamp", "strat_r")
    if v5 is not None:
        cmp.append(_eq_from_ret(v5, "V5 Eq"))
    br = _load_series(Path(args.compare_breakout_csv), "timestamp", "core_r")
    if br is not None:
        cmp.append(_eq_from_ret(br, "Breakout Eq"))

    # common aligned range for fair overlay
    common = cmp[0]
    for s in cmp[1:]:
        common = common.merge(s, on="timestamp", how="inner")
    common = common.sort_values("timestamp")
    for c in common.columns:
        if c != "timestamp":
            common[c] = common[c] / float(common[c].iloc[0])

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Price + p_bull", "Weight", "Equity Comparison (normalized)", "Rolling Accuracy (test window)"],
    )
    fig.add_trace(go.Scatter(x=feat["timestamp"], y=feat["close"], name="Price", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["timestamp"], y=feat["p_bull"], name="p_bull", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["timestamp"], y=feat["weight"], name="Weight", line=dict(color="#2ca02c")), row=2, col=1)
    for c in common.columns:
        if c != "timestamp":
            fig.add_trace(go.Scatter(x=common["timestamp"], y=common[c], name=c), row=3, col=1)

    test_df = feat.iloc[cut:].copy()
    test_df["rolling_acc"] = test_df["correct"].rolling(200).mean()
    fig.add_trace(go.Scatter(x=test_df["timestamp"], y=test_df["rolling_acc"], name="Rolling Acc (test)", line=dict(color="#d62728")), row=4, col=1)
    fig.update_yaxes(range=[0, 1], row=4, col=1)
    fig.update_layout(height=1250, title="Bull Classifier + Strategy Report")

    cls_df = pd.DataFrame([cls]).round(6)
    strat_df = pd.DataFrame([strat]).round(6)
    body = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = (
        "<html><head><meta charset='utf-8'><title>Bull Model Report</title></head><body>"
        "<h3>Bull Classifier Summary (logistic)</h3>"
        f"<p>price={args.price_csv} | horizon_bars={h} | train_ratio={args.train_ratio}</p>"
        "<h4>Classifier (test)</h4>"
        f"{cls_df.to_html(index=False, border=0)}"
        "<h4>Strategy</h4>"
        f"{strat_df.to_html(index=False, border=0)}"
        f"{body}"
        "</body></html>"
    )
    out_html = Path(args.out_report_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print("wrote", out_preds)
    print("wrote", out_strat)
    print("wrote", out_wf)
    print("wrote", out_html)
    print(cls_df.to_string(index=False))
    print(strat_df.to_string(index=False))


if __name__ == "__main__":
    main()
