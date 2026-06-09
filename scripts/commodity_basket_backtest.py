from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_price_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    ts_col = None
    for c in ("timestamp", "date", "datetime"):
        if c in cols:
            ts_col = cols[c]
            break
    if ts_col is None:
        raise ValueError(f"{path} missing timestamp/date column")

    close_col = None
    for c in ("close", "adj close", "adj_close"):
        if c in cols:
            close_col = cols[c]
            break
    if close_col is None:
        raise ValueError(f"{path} missing close/adj close column")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    out = pd.DataFrame({"timestamp": ts, "close": close}).dropna().sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.set_index("timestamp")["close"]


def _annual_metrics(log_r: pd.Series, bars_per_year: float = 365.0) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x) * np.sqrt(bars_per_year))
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(eq)) - 1.0),
        "ann_vol": ann_vol,
        "sharpe": float((np.mean(x) * bars_per_year) / (ann_vol + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


@dataclass
class BasketResult:
    frame: pd.DataFrame
    summary: dict[str, float]


def _rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "none":
        m = np.zeros(len(index), dtype=bool)
        if len(m):
            m[0] = True
        return m
    if frequency == "monthly":
        p = index.to_period("M")
    elif frequency == "quarterly":
        p = index.to_period("Q")
    else:
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
    changes = np.r_[True, p[1:] != p[:-1]]
    return changes


def run_basket(
    closes: pd.DataFrame,
    target_weights: np.ndarray,
    rebalance: str,
    fee_bps: float,
) -> BasketResult:
    rets = np.log(closes / closes.shift(1)).fillna(0.0)
    simple = np.exp(rets) - 1.0
    idx = closes.index

    reb = _rebalance_mask(idx, rebalance)
    w_prev = np.zeros(closes.shape[1], dtype=float)
    nav = 1.0

    nav_series = np.zeros(len(idx), dtype=float)
    turnover_series = np.zeros(len(idx), dtype=float)
    cost_series = np.zeros(len(idx), dtype=float)
    port_log_r = np.zeros(len(idx), dtype=float)
    w_store = np.zeros((len(idx), closes.shape[1]), dtype=float)

    for i in range(len(idx)):
        r_t = simple.iloc[i].to_numpy(float)

        if reb[i]:
            desired = target_weights.copy()
            if w_prev.sum() > 0:
                current = w_prev / w_prev.sum()
            else:
                current = np.zeros_like(desired)
            turnover = float(np.abs(desired - current).sum())
            cost = turnover * (fee_bps / 10000.0)
            nav *= 1.0 - cost
            w_prev = desired.copy()
            turnover_series[i] = turnover
            cost_series[i] = cost

        # Apply daily asset move on currently held weights.
        gross = 1.0 + float(np.dot(w_prev, r_t))
        gross = max(gross, 1e-12)
        nav *= gross
        port_log_r[i] = np.log(gross) + np.log(max(1.0 - cost_series[i], 1e-12))
        w_prev = w_prev * (1.0 + r_t)
        w_store[i] = w_prev
        nav_series[i] = nav

    out = pd.DataFrame(index=idx)
    out["nav"] = nav_series
    out["port_log_r"] = port_log_r
    out["turnover"] = turnover_series
    out["cost"] = cost_series
    for j, c in enumerate(closes.columns):
        out[f"w_{c}"] = w_store[:, j]

    m = _annual_metrics(out["port_log_r"], bars_per_year=365.0)
    m["avg_turnover_at_rebalance"] = float(out.loc[out["turnover"] > 0, "turnover"].mean()) if (out["turnover"] > 0).any() else 0.0
    m["total_cost"] = float(out["cost"].sum())
    return BasketResult(frame=out, summary=m)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest a commodity ETF basket from local CSV files.")
    ap.add_argument("--prices-dir", default="data/etf")
    ap.add_argument("--symbols", default="IAUM,PDBC,PDBA,DBB")
    ap.add_argument("--weights", default="0.40,0.30,0.20,0.10")
    ap.add_argument("--rebalance", choices=["monthly", "quarterly", "none"], default="quarterly")
    ap.add_argument("--fee-bps", type=float, default=2.0, help="Applied on weight turnover at rebalance.")
    ap.add_argument("--eth-csv", default="data/ETHUSDC_1h.csv")
    ap.add_argument("--start", default="", help="Optional UTC start, e.g. 2025-01-01")
    ap.add_argument("--end", default="", help="Optional UTC end, e.g. 2026-03-01")
    ap.add_argument("--out-csv", default="artifacts/backtest/commodity_basket_backtest.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/commodity_basket_backtest.html")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    weights = np.array([float(x) for x in args.weights.split(",")], dtype=float)
    if len(symbols) != len(weights):
        raise ValueError("symbols and weights must have same length")
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("weights must sum to 1.0")

    prices_dir = Path(args.prices_dir)
    closes = []
    for s in symbols:
        p = prices_dir / f"{s}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}. Add CSV for {s} with timestamp/date + close")
        ser = _load_price_csv(p).rename(s)
        closes.append(ser)

    basket_px = pd.concat(closes, axis=1).sort_index().ffill().dropna()

    if args.start:
        basket_px = basket_px[basket_px.index >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        basket_px = basket_px[basket_px.index <= pd.Timestamp(args.end, tz="UTC")]
    if basket_px.empty:
        raise ValueError("No overlapping ETF data in selected window.")

    result = run_basket(basket_px, weights, args.rebalance, args.fee_bps)
    out = result.frame.copy()
    out["timestamp"] = out.index
    out["basket_eq"] = out["nav"] / float(out["nav"].iloc[0])

    # ETH benchmark on daily close from local 1h file.
    eth = _load_price_csv(Path(args.eth_csv))
    eth_d = eth.resample("1D").last().dropna()
    common = out.set_index("timestamp").resample("1D").last().dropna().join(eth_d.rename("eth_close"), how="inner")
    common["eth_r"] = np.log(common["eth_close"] / common["eth_close"].shift(1)).fillna(0.0)
    common["eth_eq"] = np.exp(common["eth_r"].cumsum())
    common["eth_eq"] = common["eth_eq"] / float(common["eth_eq"].iloc[0])

    basket_daily_r = np.log(common["basket_eq"] / common["basket_eq"].shift(1)).fillna(0.0)
    m_b = _annual_metrics(basket_daily_r, bars_per_year=365.0)
    m_e = _annual_metrics(common["eth_r"], bars_per_year=365.0)

    summary = pd.DataFrame(
        [
            {
                "rows_daily": int(len(common)),
                "rebalance": args.rebalance,
                "fee_bps": float(args.fee_bps),
                "symbols": ",".join(symbols),
                "weights": ",".join(f"{w:.2f}" for w in weights),
                "basket_ret": m_b["ret"],
                "basket_cagr": m_b["cagr"],
                "basket_sharpe": m_b["sharpe"],
                "basket_max_dd": m_b["max_dd"],
                "eth_ret": m_e["ret"],
                "eth_cagr": m_e["cagr"],
                "eth_sharpe": m_e["sharpe"],
                "eth_max_dd": m_e["max_dd"],
                "basket_excess_vs_eth": m_b["ret"] - m_e["ret"],
                "total_rebalance_cost": result.summary["total_cost"],
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=["Equity", "Weights (drifted)", "Turnover/Cost"])
    fig.add_trace(go.Scatter(x=common.index, y=common["basket_eq"], name="Basket Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=common.index, y=common["eth_eq"], name="ETH Eq"), row=1, col=1)
    for s in symbols:
        c = f"w_{s}"
        if c in out.columns:
            fig.add_trace(go.Scatter(x=out["timestamp"], y=out[c], name=c), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["turnover"], name="Turnover"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["cost"], name="Cost"), row=3, col=1)
    fig.update_layout(height=1200, title="Commodity Basket Backtest")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<html><head><meta charset='utf-8'><title>Commodity Basket Backtest</title></head><body>"
        "<h3>Commodity Basket Backtest</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
