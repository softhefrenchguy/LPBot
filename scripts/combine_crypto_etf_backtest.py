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
    ts_col = None
    for c in ("timestamp", "date", "datetime"):
        if c in cols:
            ts_col = cols[c]
            break
    if ts_col is None:
        raise ValueError(f"{path} missing timestamp/date column")

    px_col = None
    for c in ("close", "adj close", "adj_close"):
        if c in cols:
            px_col = cols[c]
            break
    if px_col is None:
        raise ValueError(f"{path} missing close column")

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
    ap = argparse.ArgumentParser(description="Combine crypto sleeve with ETF base basket on daily calendar.")
    ap.add_argument("--crypto-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--crypto-r-col", default="r", help="If missing, will be computed from close.")
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--etf-dir", default="data/etf")
    ap.add_argument("--etf-symbols", default="GLD,DBC,DBA,DBB")
    ap.add_argument("--etf-weights", default="0.50,0.1666666667,0.1666666667,0.1666666666")
    ap.add_argument("--window-days", type=int, default=1826, help="5 years ~= 1826 days")
    ap.add_argument("--switch-cost-bps", type=float, default=2.0, help="Applied on abs(delta daily crypto weight).")
    ap.add_argument("--out-csv", default="artifacts/backtest/crypto_etf_combo_5y.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/crypto_etf_combo_5y.html")
    ap.add_argument("--summary-csv", default="artifacts/backtest/crypto_etf_combo_5y_summary.csv")
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.etf_symbols.split(",") if x.strip()]
    weights = np.array([float(x) for x in args.etf_weights.split(",")], dtype=float)
    if len(symbols) != len(weights):
        raise ValueError("etf-symbols and etf-weights must have same length")
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("ETF weights must sum to 1.0")

    cdf = pd.read_csv(args.crypto_csv)
    if args.timestamp_col not in cdf.columns:
        raise ValueError(f"{args.crypto_csv} missing {args.timestamp_col}")
    if args.weight_col not in cdf.columns:
        raise ValueError(f"{args.crypto_csv} missing {args.weight_col}")
    if args.close_col not in cdf.columns:
        raise ValueError(f"{args.crypto_csv} missing {args.close_col}")

    cdf["timestamp"] = pd.to_datetime(cdf[args.timestamp_col], utc=True, errors="coerce")
    cdf["close"] = pd.to_numeric(cdf[args.close_col], errors="coerce")
    cdf["weight"] = pd.to_numeric(cdf[args.weight_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if args.crypto_r_col in cdf.columns:
        cdf["r"] = pd.to_numeric(cdf[args.crypto_r_col], errors="coerce")
    else:
        cdf["r"] = np.log(cdf["close"] / cdf["close"].shift(1))
    cdf = cdf.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")

    # Daily crypto aggregates.
    cdf["day"] = cdf["timestamp"].dt.floor("D")
    daily_crypto = (
        cdf.groupby("day", as_index=False)
        .agg(
            crypto_r_daily=("r", "sum"),
            w_crypto_signal=("weight", "mean"),
            eth_close=("close", "last"),
        )
        .sort_values("day")
    )

    # ETF basket daily return on trading days.
    etf_ret_parts: list[pd.Series] = []
    etf_close_parts: list[pd.Series] = []
    for sym, w in zip(symbols, weights):
        p = Path(args.etf_dir) / f"{sym}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        s = _load_etf_close(p).rename(sym)
        r = np.log(s / s.shift(1)).rename(sym)
        etf_ret_parts.append(r * float(w))
        etf_close_parts.append(s)

    etf_r_daily = pd.concat(etf_ret_parts, axis=1).sum(axis=1, min_count=1).rename("basket_r_trading")
    etf_r_daily = etf_r_daily.reset_index().rename(columns={"timestamp": "day"})

    # Merge on full daily calendar. ETF return is 0 on non-trading days.
    daily = daily_crypto.merge(etf_r_daily, on="day", how="left").sort_values("day")
    daily["basket_r_daily"] = pd.to_numeric(daily["basket_r_trading"], errors="coerce").fillna(0.0)

    # Restrict to last window days.
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()
    if daily.empty:
        raise ValueError("No data left after window filter")

    # Execute next day to avoid lookahead.
    daily["w_crypto_exec"] = daily["w_crypto_signal"].shift(1).fillna(0.0).clip(0.0, 1.0)
    daily["turnover"] = daily["w_crypto_exec"].diff().abs().fillna(0.0)
    cost_k = float(args.switch_cost_bps) / 10000.0
    daily["switch_cost"] = daily["turnover"] * cost_k

    daily["crypto_sleeve_r"] = daily["w_crypto_exec"] * daily["crypto_r_daily"]
    daily["etf_sleeve_r"] = (1.0 - daily["w_crypto_exec"]) * daily["basket_r_daily"]
    daily["combo_r_daily"] = daily["crypto_sleeve_r"] + daily["etf_sleeve_r"] - daily["switch_cost"]

    daily["combo_eq"] = np.exp(daily["combo_r_daily"].cumsum())
    daily["eth_eq"] = np.exp(daily["crypto_r_daily"].cumsum())
    daily["basket_eq"] = np.exp(daily["basket_r_daily"].cumsum())
    daily["crypto_sleeve_eq"] = np.exp(daily["crypto_sleeve_r"].cumsum())
    daily["etf_sleeve_eq"] = np.exp(daily["etf_sleeve_r"].cumsum())

    # Normalize to 1 at start.
    for col in ("combo_eq", "eth_eq", "basket_eq", "crypto_sleeve_eq", "etf_sleeve_eq"):
        daily[col] = daily[col] / float(daily[col].iloc[0])

    combo_m = _perf(daily["combo_r_daily"])
    eth_m = _perf(daily["crypto_r_daily"])
    basket_m = _perf(daily["basket_r_daily"])
    summary = pd.DataFrame(
        [
            {
                "window_start": str(daily["day"].iloc[0]),
                "window_end": str(daily["day"].iloc[-1]),
                "rows": int(len(daily)),
                "model": "combo",
                **combo_m,
                "avg_w_crypto": float(daily["w_crypto_exec"].mean()),
                "crypto_time_in_mkt": float((daily["w_crypto_exec"] > 1e-12).mean()),
                "etf_time_in_mkt": float((daily["w_crypto_exec"] < 1.0 - 1e-12).mean()),
                "total_switch_cost": float(daily["switch_cost"].sum()),
                "etf_symbols": ",".join(symbols),
                "etf_weights": ",".join(f"{w:.6f}" for w in weights),
            },
            {
                "window_start": str(daily["day"].iloc[0]),
                "window_end": str(daily["day"].iloc[-1]),
                "rows": int(len(daily)),
                "model": "eth_spot",
                **eth_m,
                "avg_w_crypto": 1.0,
                "crypto_time_in_mkt": 1.0,
                "etf_time_in_mkt": 0.0,
                "total_switch_cost": 0.0,
                "etf_symbols": "",
                "etf_weights": "",
            },
            {
                "window_start": str(daily["day"].iloc[0]),
                "window_end": str(daily["day"].iloc[-1]),
                "rows": int(len(daily)),
                "model": "etf_basket",
                **basket_m,
                "avg_w_crypto": 0.0,
                "crypto_time_in_mkt": 0.0,
                "etf_time_in_mkt": 1.0,
                "total_switch_cost": 0.0,
                "etf_symbols": ",".join(symbols),
                "etf_weights": ",".join(f"{w:.6f}" for w in weights),
            },
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    out_summary = Path(args.summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Equity (Normalized)", "Crypto vs ETF Sleeve Contribution", "Daily Weights / Turnover"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["combo_eq"], name="Combo Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eth_eq"], name="ETH Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["basket_eq"], name="ETF Basket Eq"), row=1, col=1)

    fig.add_trace(go.Scatter(x=daily["day"], y=daily["crypto_sleeve_eq"], name="Crypto Sleeve Eq"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["etf_sleeve_eq"], name="ETF Sleeve Eq"), row=2, col=1)

    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_crypto_exec"], name="w_crypto_exec"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=1.0 - daily["w_crypto_exec"], name="w_etf_exec"), row=3, col=1)
    fig.add_trace(go.Bar(x=daily["day"], y=daily["turnover"], name="turnover", opacity=0.3), row=3, col=1)
    fig.update_layout(height=1200, title="Crypto + ETF Base Basket (5Y Calendar-Aligned)")

    html = (
        "<html><head><meta charset='utf-8'><title>Crypto ETF Combo</title></head><body>"
        "<h3>Crypto + ETF Base Basket (5Y)</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print("wrote", out_csv)
    print("wrote", out_summary)
    print("wrote", out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
