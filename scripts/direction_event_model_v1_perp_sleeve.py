from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, roc_auc_score

from direction_event_model_v1 import (
    build_frame,
    perf,
    run_event_strategy,
    walk_forward_event_probs,
)


def apply_step_cap(target: pd.Series, max_dw: float, w_max: float) -> pd.Series:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    arr = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for i, x in enumerate(arr):
        x = float(np.clip(x, 0.0, w_max))
        lo, hi = prev - max_dw, prev + max_dw
        v = min(max(x, lo), hi)
        out[i] = v
        prev = v
    return pd.Series(out, index=target.index)


def add_perp_sleeve(
    sim: pd.DataFrame,
    source_df: pd.DataFrame,
    sleeve_scale: float,
    sleeve_max: float,
    sleeve_score_cap: float,
    sleeve_ema_span: int,
    sleeve_min_pup: float,
    sleeve_require_trend_pos: bool,
    sleeve_min_crowd_sum: float,
    max_weight_total: float,
    max_dw_total: float,
    trade_cost_bps: float,
) -> pd.DataFrame:
    out = sim.copy()
    if not all(c in source_df.columns for c in ["basis_z", "funding_rate_z", "oi_chg_z"]):
        out["sleeve_score"] = 0.0
        out["sleeve_weight"] = 0.0
        out["weight_total"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(
            lower=-float(max_weight_total), upper=float(max_weight_total)
        )
    else:
        extra = source_df[["timestamp", "basis_z", "funding_rate_z", "oi_chg_z"]].copy()
        out = out.merge(extra, on="timestamp", how="left")
        for c in ["basis_z", "funding_rate_z", "oi_chg_z"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        # Inverted perp-positioning signal from diagnostics.
        out["sleeve_score_raw"] = (out["basis_z"] + out["funding_rate_z"]) * out["oi_chg_z"]
        out["sleeve_score"] = out["sleeve_score_raw"].ewm(span=int(sleeve_ema_span), adjust=False).mean()
        conf = (out["sleeve_score"] / float(sleeve_score_cap)).clip(lower=0.0, upper=1.0)
        sleeve_tgt = (conf * float(sleeve_scale)).clip(upper=float(sleeve_max))
        gate = (pd.to_numeric(out["weight"], errors="coerce").fillna(0.0) > 0.0) & (
            pd.to_numeric(out["p_up"], errors="coerce").fillna(0.0) >= float(sleeve_min_pup)
        )
        crowd_sum = out["basis_z"] + out["funding_rate_z"]
        gate = gate & (crowd_sum >= float(sleeve_min_crowd_sum))
        if sleeve_require_trend_pos and "trend_gap" in out.columns:
            gate = gate & (pd.to_numeric(out["trend_gap"], errors="coerce").fillna(0.0) > 0.0)
        out["sleeve_weight"] = sleeve_tgt.where(gate, 0.0)
        out["weight_total_tgt"] = (
            pd.to_numeric(out["weight"], errors="coerce").fillna(0.0) + out["sleeve_weight"]
        ).clip(lower=0.0, upper=float(max_weight_total))
        out["weight_total"] = apply_step_cap(out["weight_total_tgt"], max_dw=float(max_dw_total), w_max=float(max_weight_total))

    out["turnover_total"] = out["weight_total"].diff().abs().fillna(0.0)
    cost = out["turnover_total"] * (float(trade_cost_bps) / 10000.0)
    out["strat_r_total"] = out["weight_total"].shift(1).fillna(0.0) * out["r"] - cost
    out["eq_total"] = np.exp(np.cumsum(out["strat_r_total"].to_numpy()))
    out["active_total"] = (out["weight_total"].abs() > 1e-12).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Event model v1 + capped perp sleeve")
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
    ap.add_argument("--long-thresholds", default="0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--short-thresholds", default="0.45,0.42,0.40,0.38,0.35")
    ap.add_argument("--hold-bars-list", default="6,12,24")
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--min-time-in-market-pct", type=float, default=2.0)
    ap.add_argument("--sleeve-scale", type=float, default=0.25)
    ap.add_argument("--sleeve-max", type=float, default=0.30)
    ap.add_argument("--sleeve-score-cap", type=float, default=3.0)
    ap.add_argument("--sleeve-ema-span", type=int, default=12)
    ap.add_argument("--sleeve-min-pup", type=float, default=0.55)
    ap.add_argument("--sleeve-require-trend-pos", action="store_true")
    ap.add_argument("--sleeve-min-crowd-sum", type=float, default=-1e9)
    ap.add_argument("--max-weight-total", type=float, default=1.0)
    ap.add_argument("--max-dw-total", type=float, default=0.10)
    ap.add_argument("--out-preds-csv", default="artifacts/backtest/direction_event_model_v1_perp_preds.csv")
    ap.add_argument("--out-sweep-csv", default="artifacts/backtest/direction_event_model_v1_perp_sweep.csv")
    ap.add_argument("--out-csv", default="artifacts/backtest/direction_event_model_v1_perp.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/direction_event_model_v1_perp.html")
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

    ev_pred = pred[pred["p_up"].notna()].copy()
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
                sim_base = run_event_strategy(
                    pred=pred,
                    long_th=lth,
                    short_th=sth,
                    hold_bars=int(h),
                    max_weight=float(args.max_weight),
                    trade_cost_bps=float(args.trade_cost_bps),
                )
                sim = add_perp_sleeve(
                    sim=sim_base,
                    source_df=df,
                    sleeve_scale=float(args.sleeve_scale),
                    sleeve_max=float(args.sleeve_max),
                    sleeve_score_cap=float(args.sleeve_score_cap),
                    sleeve_ema_span=int(args.sleeve_ema_span),
                    sleeve_min_pup=float(args.sleeve_min_pup),
                    sleeve_require_trend_pos=bool(args.sleeve_require_trend_pos),
                    sleeve_min_crowd_sum=float(args.sleeve_min_crowd_sum),
                    max_weight_total=float(args.max_weight_total),
                    max_dw_total=float(args.max_dw_total),
                    trade_cost_bps=float(args.trade_cost_bps),
                )
                m = perf(sim["strat_r_total"], 5)
                m_base = perf(sim["strat_r"], 5)
                ms = perf(sim["spot_r"], 5)
                tim = float(sim["active_total"].mean() * 100.0)
                row = {
                    "hold_bars": int(h),
                    "long_th": float(lth),
                    "short_th": float(sth),
                    "ret_total": m["ret"],
                    "ret_base": m_base["ret"],
                    "spot_ret": ms["ret"],
                    "excess_vs_spot": m["ret"] - ms["ret"],
                    "lift_vs_base": m["ret"] - m_base["ret"],
                    "sharpe_total": m["sharpe"],
                    "sharpe_base": m_base["sharpe"],
                    "max_dd_total": m["max_dd"],
                    "max_dd_base": m_base["max_dd"],
                    "time_in_market_total_pct": tim,
                    "avg_abs_weight_total": float(sim["weight_total"].abs().mean()),
                    "turnover_total": float(sim["turnover_total"].sum()),
                }
                sweep_rows.append(row)
                score = (0.0 if np.isnan(row["sharpe_total"]) else row["sharpe_total"]) + 1.2 * row["ret_total"] - 0.0002 * row["turnover_total"]
                if tim < float(args.min_time_in_market_pct):
                    score -= 3.0
                if score > best_score:
                    best_score = score
                    best_sim = sim
                    best_cfg = (h, lth, sth)

    if best_sim is None or best_cfg is None:
        raise RuntimeError("No valid strategy configuration found")

    sweep = pd.DataFrame(sweep_rows).sort_values(["sharpe_total", "ret_total"], ascending=[False, False]).reset_index(drop=True)
    m_total = perf(best_sim["strat_r_total"], 5)
    m_base = perf(best_sim["strat_r"], 5)
    m_spot = perf(best_sim["spot_r"], 5)

    Path(args.out_preds_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_preds_csv, index=False)
    Path(args.out_sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out_sweep_csv, index=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    best_sim.to_csv(args.out_csv, index=False)

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Equity (Total/Base/Spot)", "P(up) + Sleeve Score", "Weights", "Sleeve Components", "Events", "Price"],
    )
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eq_total"], name="Total Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eq"], name="Base Eq", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["p_up"], name="P(up)", line=dict(color="#2ca02c")), row=2, col=1)
    if "sleeve_score" in best_sim.columns:
        fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["sleeve_score"], name="Sleeve Score", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["weight"], name="Base Weight", line=dict(color="#ff7f0e")), row=3, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["weight_total"], name="Total Weight", line=dict(color="#1f77b4")), row=3, col=1)
    if "sleeve_weight" in best_sim.columns:
        fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["sleeve_weight"], name="Sleeve Weight", line=dict(color="#9467bd")), row=4, col=1)
    for c, nm, clr in [("basis_z", "Basis Z", "#17becf"), ("funding_rate_z", "Funding Z", "#8c564b"), ("oi_chg_z", "OI Z", "#bcbd22")]:
        if c in best_sim.columns:
            fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim[c], name=nm, line=dict(color=clr)), row=4, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_breakout"], name="Breakout Event", line=dict(color="#17becf")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_crowd"], name="Crowd Event", line=dict(color="#8c564b")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["event_vol"], name="Vol Event", line=dict(color="#bcbd22")), row=5, col=1)
    fig.add_trace(go.Scatter(x=best_sim["timestamp"], y=best_sim["eth_close"], name="ETH Close", line=dict(color="#2ca02c")), row=6, col=1)
    fig.update_layout(height=1700, title="Event Model v1 + Perp Sleeve")

    summary = pd.DataFrame(
        [
            {
                "rows_oos": int(len(best_sim)),
                "oos_auc_event": auc_ev,
                "oos_acc_event": acc_ev,
                "best_hold_bars": int(best_cfg[0]),
                "best_long_th": float(best_cfg[1]),
                "best_short_th": float(best_cfg[2]),
                "ret_total": m_total["ret"],
                "ret_base": m_base["ret"],
                "ret_spot": m_spot["ret"],
                "lift_vs_base": m_total["ret"] - m_base["ret"],
                "excess_vs_spot": m_total["ret"] - m_spot["ret"],
                "sharpe_total": m_total["sharpe"],
                "sharpe_base": m_base["sharpe"],
                "max_dd_total": m_total["max_dd"],
                "max_dd_base": m_base["max_dd"],
                "time_in_market_total_pct": float(best_sim["active_total"].mean() * 100.0),
                "avg_abs_weight_total": float(best_sim["weight_total"].abs().mean()),
                "turnover_total": float(best_sim["turnover_total"].sum()),
                "sleeve_scale": float(args.sleeve_scale),
                "sleeve_max": float(args.sleeve_max),
                "sleeve_min_pup": float(args.sleeve_min_pup),
                "sleeve_require_trend_pos": bool(args.sleeve_require_trend_pos),
                "sleeve_min_crowd_sum": float(args.sleeve_min_crowd_sum),
            }
        ]
    )
    html = (
        "<html><head><meta charset='utf-8'><title>Event v1 + Perp Sleeve</title></head><body>"
        "<h3>Event v1 + Perp Sleeve</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        "<h4>Top 10 configs</h4>"
        f"{sweep.head(10).round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {args.out_preds_csv}")
    print(f"wrote {args.out_sweep_csv}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_html}")
    print(
        "oos_auc_event={:.4f} oos_acc_event={:.4f} best_hold={} best_long={:.2f} best_short={:.2f} ret_total={:.4f} ret_base={:.4f} lift={:.4f} sharpe_total={:.4f}".format(
            auc_ev,
            acc_ev,
            int(best_cfg[0]),
            float(best_cfg[1]),
            float(best_cfg[2]),
            m_total["ret"],
            m_base["ret"],
            m_total["ret"] - m_base["ret"],
            m_total["sharpe"],
        )
    )


if __name__ == "__main__":
    main()
