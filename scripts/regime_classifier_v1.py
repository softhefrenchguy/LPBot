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


def load_price(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns or "close" not in d.columns:
        raise ValueError(f"{path} must contain timestamp and close")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    d["r"] = np.log(d["close"] / d["close"].shift(1)).fillna(0.0)
    return d.reset_index(drop=True)


def load_perp_daily(path: Path, day_index: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=day_index)
    d = pd.read_csv(path, low_memory=False)
    if "timestamp" not in d.columns:
        return pd.DataFrame(index=day_index)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = d.set_index("timestamp").resample("1D").agg(
        basis=("basis", "mean") if "basis" in d.columns else ("timestamp", "size"),
        funding_rate=("funding_rate", "mean") if "funding_rate" in d.columns else ("timestamp", "size"),
    )
    for c in ["basis", "funding_rate"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.reindex(day_index).ffill()
    return out


def classify_regime(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["vol_14d"] = out["r"].rolling(14, min_periods=7).std() * np.sqrt(365)
    out["ath"] = out["close"].cummax()
    out["dd_from_ath"] = out["close"] / out["ath"] - 1.0

    if "funding_rate" in out.columns:
        out["funding_5d_chg"] = out["funding_rate"].diff(5)
    else:
        out["funding_5d_chg"] = 0.0
    if "basis" in out.columns:
        out["basis_5d_chg"] = out["basis"].diff(5)
    else:
        out["basis_5d_chg"] = 0.0

    bull = (
        (out["close"] > out["ema20"])
        & (out["ema20"] > out["ema50"])
        & (out["ret_5d"] > 0.0)
        & (out["dd_from_ath"] > -0.25)
    )
    bear = (
        (out["close"] < out["ema50"])
        & (out["ret_10d"] < -0.01)
        & (out["dd_from_ath"] < -0.12)
    )

    regime = np.where(bull, "BULL", np.where(bear, "BEAR", "CHOP"))
    out["regime_raw"] = regime
    # one-day lag to avoid lookahead in downstream routing
    out["regime"] = pd.Series(out["regime_raw"], index=out.index).shift(1).fillna("CHOP")
    return out


def regime_perf(df: pd.DataFrame, strat_col: str, spot_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for rg in ["BULL", "BEAR", "CHOP"]:
        d = df[df["regime"] == rg].copy()
        if len(d) == 0:
            continue
        m = perf(d[strat_col], 5)
        s = perf(d[spot_col], 5)
        rows.append(
            {
                "regime": rg,
                "rows": int(len(d)),
                "bar_pct": float(len(d) / len(df) * 100.0),
                "strat_ret": m["ret"],
                "strat_sharpe": m["sharpe"],
                "strat_max_dd": m["max_dd"],
                "spot_ret": s["ret"],
                "excess_vs_spot": m["ret"] - s["ret"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-based BULL/BEAR/CHOP classifier + optional strategy split report.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--perp-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--strategy-csv", default="artifacts/backtest/direction_event_model_v1_5y_1d_hiconv.csv")
    ap.add_argument("--strategy-ret-col", default="strat_r")
    ap.add_argument("--strategy-spot-col", default="spot_r")
    ap.add_argument("--out-regime-daily-csv", default="artifacts/backtest/regime_classifier_v1_daily.csv")
    ap.add_argument("--out-merged-csv", default="artifacts/backtest/regime_classifier_v1_strategy_merged.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/regime_classifier_v1_summary.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_classifier_v1_report.html")
    args = ap.parse_args()

    p = load_price(Path(args.price_csv))
    daily = p.set_index("timestamp").resample("1D").agg(close=("close", "last"), r=("r", "sum"))
    daily = daily.dropna(subset=["close"]).copy()
    perp_d = load_perp_daily(Path(args.perp_csv), daily.index)
    daily = daily.join(perp_d, how="left")
    daily = classify_regime(daily).reset_index().rename(columns={"index": "day", "timestamp": "day"})
    daily["day"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")

    out_daily = Path(args.out_regime_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_daily, index=False)

    summary_rows: list[dict[str, float | str | int]] = []
    merged = None
    strat_path = Path(args.strategy_csv)
    if strat_path.exists():
        s = pd.read_csv(strat_path)
        if "timestamp" in s.columns and args.strategy_ret_col in s.columns and args.strategy_spot_col in s.columns:
            s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
            s[args.strategy_ret_col] = pd.to_numeric(s[args.strategy_ret_col], errors="coerce")
            s[args.strategy_spot_col] = pd.to_numeric(s[args.strategy_spot_col], errors="coerce")
            s = s.dropna(subset=["timestamp", args.strategy_ret_col, args.strategy_spot_col]).sort_values("timestamp")
            s["day"] = s["timestamp"].dt.floor("D")
            m = s.merge(daily[["day", "regime"]], on="day", how="left")
            m["regime"] = m["regime"].fillna("CHOP")
            merged = m
            rp = regime_perf(m, args.strategy_ret_col, args.strategy_spot_col)
            for rec in rp.to_dict(orient="records"):
                rec["source"] = "strategy_5m"
                summary_rows.append(rec)

            out_merged = Path(args.out_merged_csv)
            out_merged.parent.mkdir(parents=True, exist_ok=True)
            m.to_csv(out_merged, index=False)

    regime_counts = daily["regime"].value_counts(normalize=True).rename("pct").mul(100).reset_index()
    regime_counts.columns = ["regime", "pct"]
    for _, r in regime_counts.iterrows():
        summary_rows.append(
            {
                "source": "daily_distribution",
                "regime": r["regime"],
                "rows": int((daily["regime"] == r["regime"]).sum()),
                "bar_pct": float(r["pct"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Price + Regime", "Feature Panel"])
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["close"], name="ETH Close", line=dict(color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema20"], name="EMA20", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema50"], name="EMA50", line=dict(color="#ff7f0e")), row=1, col=1)

    y0, y1 = float(daily["close"].min()), float(daily["close"].max())
    for rg, color in [("BULL", "rgba(46,204,113,0.10)"), ("BEAR", "rgba(231,76,60,0.10)"), ("CHOP", "rgba(149,165,166,0.08)")]:
        mask = daily["regime"] == rg
        starts = daily.loc[mask & ~mask.shift(1, fill_value=False), "day"]
        ends = daily.loc[mask & ~mask.shift(-1, fill_value=False), "day"]
        for sdt, edt in zip(starts, ends):
            fig.add_vrect(x0=sdt, x1=edt, fillcolor=color, line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ret_5d"], name="ret_5d", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["dd_from_ath"], name="dd_from_ath", line=dict(color="#8c564b")), row=2, col=1)
    fig.update_layout(height=900, title="Regime Classifier v1")

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Regime Classifier v1</title></head><body>",
        "<h3>Regime Classifier v1</h3>",
        f"<p>price={args.price_csv} | perp={args.perp_csv} | strategy={args.strategy_csv}</p>",
        "<h4>Regime Distribution + Strategy Split</h4>",
        summary.round(6).to_html(index=False, border=0),
        fig.to_html(full_html=False, include_plotlyjs='cdn'),
        "</body></html>",
    ]
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"wrote {out_daily}")
    if merged is not None:
        print(f"wrote {args.out_merged_csv}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_html}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
