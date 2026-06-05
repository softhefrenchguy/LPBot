import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _parse_list(raw: str, cast=float):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _metrics(df: pd.DataFrame, days: int) -> dict:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    end = d["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=days)
    d = d[d["timestamp"] >= start].copy()
    if d.empty:
        return {}

    r = pd.to_numeric(d["r"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(d["strat_r"], errors="coerce").fillna(0.0)
    w = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    eq = np.exp(np.cumsum(s.to_numpy()))
    spot_eq = np.exp(np.cumsum(r.to_numpy()))
    bars_per_year = 365 * 24 * (60 / 5)
    ann_vol = float(np.std(s.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(s.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ret = float(eq[-1] - 1.0)
    spot_ret = float(spot_eq[-1] - 1.0)
    return {
        f"ret_{days}d": ret,
        f"spot_ret_{days}d": spot_ret,
        f"excess_{days}d": ret - spot_ret,
        f"sharpe_{days}d": sharpe,
        f"mdd_{days}d": mdd,
        f"avg_w_{days}d": float(w.mean()),
    }


def _bull_capture(df: pd.DataFrame, days: int = 90) -> dict:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    end = d["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=days)
    d = d[d["timestamp"] >= start].copy()
    if d.empty:
        return {}

    d = d.set_index("timestamp")
    close = pd.to_numeric(d["close"], errors="coerce").ffill()
    ema = close.resample("1D").last().ffill().ewm(span=200, adjust=False).mean()
    bull = ((close.resample("1D").last().ffill() > ema) & (ema.diff() > 0)).reindex(d.index, method="ffill").fillna(False)
    r = pd.to_numeric(d["r"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(d["strat_r"], errors="coerce").fillna(0.0)

    up = bull & (r > 0)
    dn = bull & (r < 0)
    up_spot = float(r[up].sum())
    up_strat = float(s[up].sum())
    dn_spot = float((-r[dn]).sum())
    dn_strat = float((-s[dn]).sum())
    up_cap = up_strat / up_spot if abs(up_spot) > 1e-12 else np.nan
    dn_cap = dn_strat / dn_spot if abs(dn_spot) > 1e-12 else np.nan
    return {"bull_up_cap_90d": up_cap, "bull_dn_cap_90d": dn_cap, "bull_cap_diff_90d": up_cap - dn_cap}


def _score(row: pd.Series) -> float:
    # Economic objective with light risk/capture constraints.
    v = 0.0
    v += float(row.get("excess_90d", 0.0)) * 1.0
    v += float(row.get("excess_365d", 0.0)) * 0.5
    v += float(row.get("sharpe_90d", 0.0)) * 0.01
    v += float(row.get("bull_cap_diff_90d", 0.0)) * 0.1
    # penalty for deep DD
    dd = abs(float(row.get("mdd_90d", 0.0)))
    if dd > 0.30:
        v -= (dd - 0.30) * 0.5
    return v


def _run_one(args, combo: dict, out_csv: Path) -> pd.DataFrame:
    cmd = [
        sys.executable,
        "-m",
        "lpbot.hmm.hmm_backtest_v1.cli_hmm_elasticnet_gate",
        "--price-csv",
        args.price_csv,
        "--exposure-csv",
        args.exposure_csv,
        "--regime-csv",
        args.regime_csv,
        "--mode",
        "scale",
        "--riskon-values",
        args.riskon_values,
        "--riskoff-values",
        args.riskoff_values,
        "--neutral-scale",
        str(args.neutral_scale),
        "--riskoff-scale",
        str(args.riskoff_scale),
        "--bull-override",
        "--bull-timeframe",
        args.bull_timeframe,
        "--bull-ema",
        str(args.bull_ema),
        "--bull-min-weight",
        str(args.bull_min_weight),
        "--bull-fast-derisk",
        "--bull-derisk-timeframe",
        args.bull_derisk_timeframe,
        "--bull-derisk-ema-fast",
        str(args.bull_derisk_ema_fast),
        "--bull-derisk-ema-slow",
        str(args.bull_derisk_ema_slow),
        "--bull-derisk-max-weight",
        str(combo["bull_derisk_max_weight"]),
        "--vote-override",
        "--vote-tf1",
        args.vote_tf1,
        "--vote-tf2",
        args.vote_tf2,
        "--vote-ema-fast",
        str(args.vote_ema_fast),
        "--vote-ema-slow",
        str(args.vote_ema_slow),
        "--vote-neutral-ema-gap",
        str(args.vote_neutral_ema_gap),
        "--vote-neutral-price-gap",
        str(args.vote_neutral_price_gap),
        "--vote-neutral-cap",
        str(args.vote_neutral_cap),
        "--vote-bear-cap",
        str(args.vote_bear_cap),
        "--prob-override",
        "--prob-tf1",
        args.prob_tf1,
        "--prob-tf2",
        args.prob_tf2,
        "--prob-ema-fast",
        str(args.prob_ema_fast),
        "--prob-ema-slow",
        str(args.prob_ema_slow),
        "--prob-logit-k",
        str(args.prob_logit_k),
        "--prob-w-vote",
        str(args.prob_w_vote),
        "--prob-w-ema-gap",
        str(args.prob_w_ema_gap),
        "--prob-w-price1",
        str(args.prob_w_price1),
        "--prob-w-price2",
        str(args.prob_w_price2),
        "--prob-w-regime",
        str(args.prob_w_regime),
        "--prob-scale-ema-gap",
        str(args.prob_scale_ema_gap),
        "--prob-scale-price-gap",
        str(args.prob_scale_price_gap),
        "--prob-bull-threshold",
        str(combo["prob_bull_threshold"]),
        "--prob-bear-threshold",
        str(combo["prob_bear_threshold"]),
        "--prob-neutral-cap",
        str(combo["prob_neutral_cap"]),
        "--prob-bear-cap",
        str(combo["prob_bear_cap"]),
        "--prob-bull-min",
        str(args.prob_bull_min),
        "--out",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pd.read_csv(out_csv)


def _write_top_html(df: pd.DataFrame, out_html: Path) -> None:
    cols = [
        "score",
        "ret_90d",
        "excess_90d",
        "sharpe_90d",
        "mdd_90d",
        "ret_365d",
        "excess_365d",
        "sharpe_365d",
        "bull_up_cap_90d",
        "bull_dn_cap_90d",
        "bull_cap_diff_90d",
        "prob_bull_threshold",
        "prob_bear_threshold",
        "prob_neutral_cap",
        "prob_bear_cap",
        "bull_derisk_max_weight",
    ]
    table = df[cols].head(10).round(6).to_html(index=False, border=0)
    html = f"<html><head><meta charset='utf-8'><title>Prob Sweep Top10</title></head><body><h3>Top 10 Candidates</h3>{table}</body></html>"
    out_html.write_text(html, encoding="utf-8")


def _write_comparison_html(best_df: pd.DataFrame, baseline_df: pd.DataFrame, out_html: Path) -> None:
    def prep(d: pd.DataFrame) -> pd.DataFrame:
        x = d.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
        x = x.dropna(subset=["timestamp"]).sort_values("timestamp")
        x["r"] = pd.to_numeric(x["r"], errors="coerce").fillna(0.0)
        x["strat_r"] = pd.to_numeric(x["strat_r"], errors="coerce").fillna(0.0)
        x["eq"] = np.exp(np.cumsum(x["strat_r"].to_numpy()))
        x["spot_eq"] = np.exp(np.cumsum(x["r"].to_numpy()))
        return x

    b = prep(best_df)
    v = prep(baseline_df)
    # align to common range
    start = max(b["timestamp"].iloc[0], v["timestamp"].iloc[0])
    end = min(b["timestamp"].iloc[-1], v["timestamp"].iloc[-1])
    b = b[(b["timestamp"] >= start) & (b["timestamp"] <= end)]
    v = v[(v["timestamp"] >= start) & (v["timestamp"] <= end)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Full Range", "Last 90 Days"], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=b["timestamp"], y=b["eq"], name="Best Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v["timestamp"], y=v["eq"], name="Baseline Eq (v5)", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=b["timestamp"], y=b["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)

    end2 = b["timestamp"].iloc[-1]
    start2 = end2 - pd.Timedelta(days=90)
    b90 = b[b["timestamp"] >= start2]
    v90 = v[v["timestamp"] >= start2]
    fig.add_trace(go.Scatter(x=b90["timestamp"], y=b90["eq"], name="Best Eq (90d)", line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=v90["timestamp"], y=v90["eq"], name="Baseline Eq (90d)", line=dict(color="#ff7f0e"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=b90["timestamp"], y=b90["spot_eq"], name="Spot Eq (90d)", line=dict(color="#7f7f7f"), showlegend=False), row=2, col=1)
    fig.update_layout(height=900, title="Best Candidate vs Baseline (v5)")
    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Constrained sweep for probabilistic HMM+ENet overlay.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--exposure-csv", default="artifacts/paper/enet_base_exposure.csv")
    ap.add_argument("--regime-csv", default="data/regimes/8h/macro/ETHUSDC/regimes_8h.csv")
    ap.add_argument("--riskon-values", default="1")
    ap.add_argument("--riskoff-values", default="0")
    ap.add_argument("--neutral-scale", type=float, default=0.6)
    ap.add_argument("--riskoff-scale", type=float, default=0.3)
    ap.add_argument("--bull-timeframe", default="1h")
    ap.add_argument("--bull-ema", type=int, default=200)
    ap.add_argument("--bull-min-weight", type=float, default=0.35)
    ap.add_argument("--bull-derisk-timeframe", default="same")
    ap.add_argument("--bull-derisk-ema-fast", type=int, default=12)
    ap.add_argument("--bull-derisk-ema-slow", type=int, default=36)
    ap.add_argument("--vote-tf1", default="1h")
    ap.add_argument("--vote-tf2", default="4h")
    ap.add_argument("--vote-ema-fast", type=int, default=50)
    ap.add_argument("--vote-ema-slow", type=int, default=200)
    ap.add_argument("--vote-neutral-ema-gap", type=float, default=0.003)
    ap.add_argument("--vote-neutral-price-gap", type=float, default=0.002)
    ap.add_argument("--vote-neutral-cap", type=float, default=0.30)
    ap.add_argument("--vote-bear-cap", type=float, default=0.12)
    ap.add_argument("--prob-tf1", default="1h")
    ap.add_argument("--prob-tf2", default="4h")
    ap.add_argument("--prob-ema-fast", type=int, default=50)
    ap.add_argument("--prob-ema-slow", type=int, default=200)
    ap.add_argument("--prob-logit-k", type=float, default=3.0)
    ap.add_argument("--prob-w-vote", type=float, default=1.0)
    ap.add_argument("--prob-w-ema-gap", type=float, default=1.0)
    ap.add_argument("--prob-w-price1", type=float, default=1.0)
    ap.add_argument("--prob-w-price2", type=float, default=1.0)
    ap.add_argument("--prob-w-regime", type=float, default=0.5)
    ap.add_argument("--prob-scale-ema-gap", type=float, default=0.003)
    ap.add_argument("--prob-scale-price-gap", type=float, default=0.004)
    ap.add_argument("--prob-bull-min", type=float, default=0.0)
    ap.add_argument("--bull-th-list", default="0.58,0.62,0.66")
    ap.add_argument("--bear-th-list", default="0.34,0.38,0.42")
    ap.add_argument("--neutral-cap-list", default="0.25,0.30,0.35")
    ap.add_argument("--bear-cap-list", default="0.08,0.12,0.16")
    ap.add_argument("--derisk-max-list", default="0.25,0.35")
    ap.add_argument("--out-csv", default="artifacts/paper/hmm_enet_prob_sweep.csv")
    ap.add_argument("--out-top-html", default="artifacts/paper/hmm_enet_prob_top10.html")
    ap.add_argument("--out-comparison-html", default="artifacts/paper/hmm_enet_prob_comparison.html")
    ap.add_argument("--out-best-csv", default="artifacts/paper/hmm_enet_prob_best.csv")
    ap.add_argument("--baseline-csv", default="artifacts/paper/hmm_enet_scaled_v5.csv")
    args = ap.parse_args()

    bull_th_list = _parse_list(args.bull_th_list, float)
    bear_th_list = _parse_list(args.bear_th_list, float)
    neutral_cap_list = _parse_list(args.neutral_cap_list, float)
    bear_cap_list = _parse_list(args.bear_cap_list, float)
    derisk_max_list = _parse_list(args.derisk_max_list, float)

    combos = []
    for bth, bbh, ncap, bcap, dmx in itertools.product(
        bull_th_list, bear_th_list, neutral_cap_list, bear_cap_list, derisk_max_list
    ):
        if bbh >= bth:
            continue
        if bcap > ncap:
            continue
        combos.append(
            {
                "prob_bull_threshold": bth,
                "prob_bear_threshold": bbh,
                "prob_neutral_cap": ncap,
                "prob_bear_cap": bcap,
                "bull_derisk_max_weight": dmx,
            }
        )

    tmp_out = Path("artifacts/paper/_tmp_hmm_prob_sweep.csv")
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"running {len(combos)} combos...")
    for i, combo in enumerate(combos, 1):
        try:
            d = _run_one(args, combo, tmp_out)
            r = {}
            r.update(combo)
            r.update(_metrics(d, 90))
            r.update(_metrics(d, 365))
            r.update(_bull_capture(d, 90))
            rows.append(r)
            if i % 10 == 0 or i == len(combos):
                print(f"done {i}/{len(combos)}")
        except Exception:
            continue

    if not rows:
        raise RuntimeError("no successful runs")

    res = pd.DataFrame(rows)
    res["score"] = res.apply(_score, axis=1)
    res = res.sort_values("score", ascending=False).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print("wrote", out_csv)

    out_top_html = Path(args.out_top_html)
    out_top_html.parent.mkdir(parents=True, exist_ok=True)
    _write_top_html(res, out_top_html)
    print("wrote", out_top_html)

    best = res.iloc[0].to_dict()
    best_combo = {
        "prob_bull_threshold": float(best["prob_bull_threshold"]),
        "prob_bear_threshold": float(best["prob_bear_threshold"]),
        "prob_neutral_cap": float(best["prob_neutral_cap"]),
        "prob_bear_cap": float(best["prob_bear_cap"]),
        "bull_derisk_max_weight": float(best["bull_derisk_max_weight"]),
    }
    best_df = _run_one(args, best_combo, Path(args.out_best_csv))
    print("wrote", args.out_best_csv)

    baseline_df = pd.read_csv(args.baseline_csv)
    out_cmp = Path(args.out_comparison_html)
    out_cmp.parent.mkdir(parents=True, exist_ok=True)
    _write_comparison_html(best_df, baseline_df, out_cmp)
    print("wrote", out_cmp)


if __name__ == "__main__":
    main()
