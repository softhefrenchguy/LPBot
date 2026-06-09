from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Offense sleeve: momentum + funding/basis agreement (daily logic, 5m execution).")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--ema-days", type=int, default=20)
    ap.add_argument("--mom-days", type=int, default=5)
    ap.add_argument("--basis-th", type=float, default=0.0)
    ap.add_argument("--funding-th", type=float, default=0.0)
    ap.add_argument("--vol-quantile-max", type=float, default=0.85)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="artifacts/backtest/offense_funding_momentum_v1.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/offense_funding_momentum_v1.html")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/offense_funding_momentum_v1_summary.csv")
    args = ap.parse_args()

    p = pd.read_csv(args.price_csv)
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    p["r"] = np.log(p["close"] / p["close"].shift(1)).fillna(0.0)

    daily = p.set_index("timestamp").resample("1D").agg(close=("close", "last"), r=("r", "sum"))
    daily = daily.dropna(subset=["close"]).copy()
    daily["ema"] = daily["close"].ewm(span=int(args.ema_days), adjust=False).mean()
    daily["mom"] = daily["close"].pct_change(int(args.mom_days))
    daily["vol20"] = daily["r"].rolling(20, min_periods=10).std() * np.sqrt(365)

    pf = pd.read_csv(args.perp_csv, low_memory=False) if Path(args.perp_csv).exists() else pd.DataFrame()
    if len(pf) > 0 and "timestamp" in pf.columns:
        pf["timestamp"] = pd.to_datetime(pf["timestamp"], utc=True, errors="coerce")
        for c in ["basis", "funding_rate", "oi_chg_event"]:
            if c in pf.columns:
                pf[c] = pd.to_numeric(pf[c], errors="coerce")
        agg = {"basis": "mean", "funding_rate": "mean"}
        if "oi_chg_event" in pf.columns:
            agg["oi_chg_event"] = "max"
        pfd = pf.dropna(subset=["timestamp"]).set_index("timestamp").resample("1D").agg(agg)
        daily = daily.join(pfd, how="left")
    else:
        daily["basis"] = np.nan
        daily["funding_rate"] = np.nan
        daily["oi_chg_event"] = 0.0
    daily["basis"] = daily["basis"].ffill().fillna(0.0)
    daily["funding_rate"] = daily["funding_rate"].ffill().fillna(0.0)
    daily["oi_chg_event"] = pd.to_numeric(daily.get("oi_chg_event", 0.0), errors="coerce").fillna(0.0)

    daily["funding_5d_chg"] = daily["funding_rate"].diff(5)
    daily["basis_3d_chg"] = daily["basis"].diff(3)

    vol_thr = float(daily["vol20"].quantile(float(args.vol_quantile_max)))
    trend_ok = (daily["close"] > daily["ema"]) & (daily["mom"] > 0.0)
    perp_ok = (
        (daily["funding_rate"] > float(args.funding_th))
        & (daily["funding_5d_chg"] > 0.0)
        & (daily["basis"] > float(args.basis_th))
        & (daily["basis_3d_chg"] >= 0.0)
    )
    vol_ok = daily["vol20"] <= vol_thr
    oi_boost = daily["oi_chg_event"] > 0.0

    # offense on when trend agrees with perp and vol is not extreme; OI event can also unlock if trend/perp already true
    on = trend_ok & perp_ok & (vol_ok | oi_boost)
    daily["offense_on"] = on.astype(int)
    daily["weight_daily"] = daily["offense_on"].astype(float)
    daily["weight_exec"] = daily["weight_daily"].shift(1).fillna(0.0)

    intraday = p[["timestamp", "close", "r"]].copy()
    intraday["day"] = intraday["timestamp"].dt.floor("D")
    d2 = daily.reset_index().rename(columns={"timestamp": "day"})
    intraday = intraday.merge(
        d2[
            [
                "day",
                "ema",
                "mom",
                "basis",
                "funding_rate",
                "funding_5d_chg",
                "basis_3d_chg",
                "vol20",
                "offense_on",
                "weight_exec",
            ]
        ],
        on="day",
        how="left",
    )
    intraday["weight"] = intraday["weight_exec"].ffill().fillna(0.0)
    intraday["turnover"] = intraday["weight"].diff().abs().fillna(0.0)
    cost = intraday["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    intraday["cost_r"] = -cost
    intraday["strat_r"] = intraday["weight"].shift(1).fillna(0.0) * intraday["r"] - cost
    intraday["spot_r"] = intraday["r"]
    intraday["eq"] = np.exp(np.cumsum(intraday["strat_r"].to_numpy()))
    intraday["spot_eq"] = np.exp(np.cumsum(intraday["spot_r"].to_numpy()))

    m = perf(intraday["strat_r"], 5)
    s = perf(intraday["spot_r"], 5)
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(intraday)),
                "start": str(intraday["timestamp"].iloc[0]),
                "end": str(intraday["timestamp"].iloc[-1]),
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": s["ret"],
                "excess_vs_spot": m["ret"] - s["ret"],
                "time_in_market_pct": float((intraday["weight"] > 0).mean() * 100.0),
                "avg_weight": float(intraday["weight"].mean()),
                "turnover": float(intraday["turnover"].sum()),
                "offense_days_pct": float(daily["offense_on"].mean() * 100.0),
                "vol20_threshold": vol_thr,
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    intraday.to_csv(out_csv, index=False)
    out_sum = Path(args.out_summary_csv)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Equity", "Weight", "Price", "Daily Features"])
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["eq"], name="Offense Eq", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["spot_eq"], name="Spot Eq", line=dict(color="#7f7f7f")), row=1, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["weight"], name="Weight", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=intraday["timestamp"], y=intraday["close"], name="ETH Close", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["mom"], name="mom_5d", line=dict(color="#9467bd")), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["funding_rate"], name="funding_rate", line=dict(color="#17becf")), row=4, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["basis"], name="basis", line=dict(color="#8c564b")), row=4, col=1)
    fig.update_layout(height=1200, title="Offense Funding+Momentum v1")

    html = (
        "<html><head><meta charset='utf-8'><title>Offense Funding+Momentum v1</title></head><body>"
        "<h3>Offense Funding+Momentum v1</h3>"
        + summary.round(6).to_html(index=False, border=0)
        + fig.to_html(full_html=False, include_plotlyjs="cdn")
        + "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_sum}")
    print(f"wrote {out_html}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
