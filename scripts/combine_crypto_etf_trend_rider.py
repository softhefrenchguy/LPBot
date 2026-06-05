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
    ts_col = next((cols[c] for c in ("timestamp", "date", "datetime") if c in cols), None)
    px_col = next((cols[c] for c in ("close", "adj close", "adj_close") if c in cols), None)
    if ts_col is None or px_col is None:
        raise ValueError(f"{path} missing date/close columns")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
            "close": pd.to_numeric(df[px_col], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.set_index("timestamp")["close"]


def _perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Crypto+ETF combo with long-trend crypto floor.")
    ap.add_argument("--crypto-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--crypto-r-col", default="r")
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--etf-dir", default="data/etf")
    ap.add_argument("--etf-symbols", default="GLD,DBC,DBA,DBB")
    ap.add_argument("--etf-weights", default="0.50,0.1666666667,0.1666666667,0.1666666666")
    ap.add_argument("--window-days", type=int, default=1826)
    ap.add_argument("--switch-cost-bps", type=float, default=2.0)
    ap.add_argument("--trend-ema-days", type=int, default=200)
    ap.add_argument("--bull-floor", type=float, default=0.35)
    ap.add_argument("--bear-cap", type=float, default=0.10)
    ap.add_argument("--out-csv", default="artifacts/backtest/crypto_etf_combo_trend_rider_5y.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/crypto_etf_combo_trend_rider_5y.html")
    ap.add_argument("--summary-csv", default="artifacts/backtest/crypto_etf_combo_trend_rider_5y_summary.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.etf_symbols.split(",") if x.strip()]
    weights = np.array([float(x) for x in args.etf_weights.split(",")], dtype=float)
    if len(symbols) != len(weights):
        raise ValueError("etf-symbols and etf-weights lengths do not match")
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("ETF weights must sum to 1.0")

    cdf = pd.read_csv(args.crypto_csv)
    for c in (args.timestamp_col, args.close_col, args.weight_col):
        if c not in cdf.columns:
            raise ValueError(f"{args.crypto_csv} missing {c}")
    cdf["timestamp"] = pd.to_datetime(cdf[args.timestamp_col], utc=True, errors="coerce")
    cdf["close"] = pd.to_numeric(cdf[args.close_col], errors="coerce")
    cdf["w_sig"] = pd.to_numeric(cdf[args.weight_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if args.crypto_r_col in cdf.columns:
        cdf["r"] = pd.to_numeric(cdf[args.crypto_r_col], errors="coerce")
    else:
        cdf["r"] = np.log(cdf["close"] / cdf["close"].shift(1))
    cdf = cdf.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")
    cdf["day"] = cdf["timestamp"].dt.floor("D")

    daily = (
        cdf.groupby("day", as_index=False)
        .agg(crypto_r_daily=("r", "sum"), w_sig=("w_sig", "mean"), eth_close=("close", "last"))
        .sort_values("day")
    )

    # ETF daily basket return.
    etf_ret_parts: list[pd.Series] = []
    for sym, w in zip(symbols, weights):
        p = Path(args.etf_dir) / f"{sym}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        s = _load_etf_close(p)
        etf_ret_parts.append(np.log(s / s.shift(1)) * float(w))
    etf_r = pd.concat(etf_ret_parts, axis=1).sum(axis=1, min_count=1).rename("basket_r_trading")
    daily = daily.merge(etf_r.reset_index().rename(columns={"timestamp": "day"}), on="day", how="left")
    daily["basket_r_daily"] = pd.to_numeric(daily["basket_r_trading"], errors="coerce").fillna(0.0)

    # Window.
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()

    # Trend state from daily close (one-day lagged for execution).
    ema = daily["eth_close"].ewm(span=int(args.trend_ema_days), adjust=False).mean()
    daily["bull_on"] = (daily["eth_close"] > ema).astype(int)
    daily["bull_on_exec"] = daily["bull_on"].shift(1).fillna(0).astype(int)

    # Baseline and trend-rider weights.
    w_base = daily["w_sig"].shift(1).fillna(0.0).clip(0.0, 1.0)
    w_rider = w_base.copy()
    w_rider = np.where(daily["bull_on_exec"] == 1, np.maximum(w_rider, float(args.bull_floor)), w_rider)
    w_rider = np.where(daily["bull_on_exec"] == 0, np.minimum(w_rider, float(args.bear_cap)), w_rider)
    daily["w_crypto_base"] = w_base
    daily["w_crypto_rider"] = pd.Series(w_rider, index=daily.index).clip(0.0, 1.0)

    cost_k = float(args.switch_cost_bps) / 10000.0
    daily["turnover_base"] = daily["w_crypto_base"].diff().abs().fillna(0.0)
    daily["turnover_rider"] = daily["w_crypto_rider"].diff().abs().fillna(0.0)
    daily["cost_base"] = daily["turnover_base"] * cost_k
    daily["cost_rider"] = daily["turnover_rider"] * cost_k

    daily["combo_base_r"] = (
        daily["w_crypto_base"] * daily["crypto_r_daily"]
        + (1.0 - daily["w_crypto_base"]) * daily["basket_r_daily"]
        - daily["cost_base"]
    )
    daily["combo_rider_r"] = (
        daily["w_crypto_rider"] * daily["crypto_r_daily"]
        + (1.0 - daily["w_crypto_rider"]) * daily["basket_r_daily"]
        - daily["cost_rider"]
    )

    daily["combo_base_eq"] = np.exp(daily["combo_base_r"].cumsum())
    daily["combo_rider_eq"] = np.exp(daily["combo_rider_r"].cumsum())
    daily["eth_eq"] = np.exp(daily["crypto_r_daily"].cumsum())
    daily["basket_eq"] = np.exp(daily["basket_r_daily"].cumsum())
    for c in ("combo_base_eq", "combo_rider_eq", "eth_eq", "basket_eq"):
        daily[c] = daily[c] / float(daily[c].iloc[0])

    s_base = _perf(daily["combo_base_r"])
    s_rider = _perf(daily["combo_rider_r"])
    s_eth = _perf(daily["crypto_r_daily"])
    s_basket = _perf(daily["basket_r_daily"])
    summary = pd.DataFrame(
        [
            {
                "model": "combo_base",
                **s_base,
                "avg_w_crypto": float(daily["w_crypto_base"].mean()),
                "crypto_time_in_mkt": float((daily["w_crypto_base"] > 1e-12).mean()),
                "switch_cost_total": float(daily["cost_base"].sum()),
            },
            {
                "model": "combo_trend_rider",
                **s_rider,
                "avg_w_crypto": float(daily["w_crypto_rider"].mean()),
                "crypto_time_in_mkt": float((daily["w_crypto_rider"] > 1e-12).mean()),
                "switch_cost_total": float(daily["cost_rider"].sum()),
            },
            {
                "model": "eth_spot",
                **s_eth,
                "avg_w_crypto": 1.0,
                "crypto_time_in_mkt": 1.0,
                "switch_cost_total": 0.0,
            },
            {
                "model": "etf_basket",
                **s_basket,
                "avg_w_crypto": 0.0,
                "crypto_time_in_mkt": 0.0,
                "switch_cost_total": 0.0,
            },
        ]
    )
    summary["window_start"] = str(daily["day"].iloc[0])
    summary["window_end"] = str(daily["day"].iloc[-1])
    summary["rows"] = len(daily)
    summary["trend_ema_days"] = int(args.trend_ema_days)
    summary["bull_floor"] = float(args.bull_floor)
    summary["bear_cap"] = float(args.bear_cap)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    Path(args.summary_csv).write_text(summary.to_csv(index=False), encoding="utf-8")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity (Normalized)", "Crypto Weights", "Trend State / ETH"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["combo_base_eq"], name="Combo Base Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["combo_rider_eq"], name="Combo Trend Rider Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eth_eq"], name="ETH Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["basket_eq"], name="ETF Basket Eq"), row=1, col=1)

    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_crypto_base"], name="w_crypto_base"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_crypto_rider"], name="w_crypto_rider"), row=2, col=1)

    fig.add_trace(go.Scatter(x=daily["day"], y=daily["bull_on_exec"], name="bull_on_exec"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eth_eq"], name="ETH Eq (ref)", opacity=0.5), row=3, col=1)
    fig.update_layout(height=1200, title="Crypto+ETF Combo: Trend Rider vs Base")

    out_html = Path(args.out_html)
    html = (
        "<html><head><meta charset='utf-8'><title>Trend Rider Combo</title></head><body>"
        "<h3>Crypto+ETF Combo: Trend Rider vs Base (5Y)</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html.write_text(html, encoding="utf-8")

    print("wrote", out_csv)
    print("wrote", args.summary_csv)
    print("wrote", out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
