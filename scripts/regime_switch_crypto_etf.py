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
    dcol = next((cols[c] for c in ("timestamp", "date", "datetime") if c in cols), None)
    pcol = next((cols[c] for c in ("close", "adj close", "adj_close") if c in cols), None)
    if dcol is None or pcol is None:
        raise ValueError(f"{path} missing date/close columns")
    out = pd.DataFrame(
        {"day": pd.to_datetime(df[dcol], utc=True, errors="coerce"), "close": pd.to_numeric(df[pcol], errors="coerce")}
    ).dropna()
    out = out.sort_values("day").drop_duplicates("day", keep="last")
    return out.set_index("day")["close"]


def _perf(log_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
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
    ap = argparse.ArgumentParser(description="Hard regime switch between crypto and ETF basket.")
    ap.add_argument("--crypto-csv", default="artifacts/backtest/breakout_current_like_full.csv")
    ap.add_argument("--window-days", type=int, default=1826)
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--slope-lookback", type=int, default=20)
    ap.add_argument("--persist-on", type=int, default=3)
    ap.add_argument("--persist-off", type=int, default=3)
    ap.add_argument("--bull-w", type=float, default=0.80)
    ap.add_argument("--bear-w", type=float, default=0.10)
    ap.add_argument("--switch-cost-bps", type=float, default=2.0)
    ap.add_argument("--etf-dir", default="data/etf")
    ap.add_argument("--etf-symbols", default="GLD,DBC,DBA,DBB")
    ap.add_argument("--etf-weights", default="0.50,0.1666666667,0.1666666667,0.1666666666")
    ap.add_argument("--out-csv", default="artifacts/backtest/regime_switch_crypto_etf_5y.csv")
    ap.add_argument("--summary-csv", default="artifacts/backtest/regime_switch_crypto_etf_5y_summary.csv")
    ap.add_argument("--out-html", default="artifacts/backtest/regime_switch_crypto_etf_5y.html")
    args = ap.parse_args()

    # Crypto daily
    c = pd.read_csv(args.crypto_csv)
    need = {"timestamp", "close", "r"}
    if not need.issubset(c.columns):
        raise ValueError(f"{args.crypto_csv} missing columns: {sorted(need)}")
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True, errors="coerce")
    c["close"] = pd.to_numeric(c["close"], errors="coerce")
    c["r"] = pd.to_numeric(c["r"], errors="coerce")
    c = c.dropna(subset=["timestamp", "close", "r"]).sort_values("timestamp")
    c["day"] = c["timestamp"].dt.floor("D")
    daily = c.groupby("day", as_index=False).agg(close=("close", "last"), crypto_r=("r", "sum")).sort_values("day")

    # ETF basket daily
    syms = [x.strip().upper() for x in args.etf_symbols.split(",") if x.strip()]
    ws = np.array([float(x) for x in args.etf_weights.split(",")], dtype=float)
    if len(syms) != len(ws):
        raise ValueError("etf-symbols and etf-weights must have same length")
    if not np.isclose(ws.sum(), 1.0, atol=1e-9):
        raise ValueError("etf-weights must sum to 1.0")

    etf_parts: list[pd.Series] = []
    for s, w in zip(syms, ws):
        ser = _load_etf_close(Path(args.etf_dir) / f"{s}.csv")
        etf_parts.append(np.log(ser / ser.shift(1)) * float(w))
    etf_r = pd.concat(etf_parts, axis=1).sum(axis=1, min_count=1).rename("etf_r").reset_index()

    daily = daily.merge(etf_r, on="day", how="left").sort_values("day")
    daily["etf_r"] = pd.to_numeric(daily["etf_r"], errors="coerce").fillna(0.0)

    # Window
    end = daily["day"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    daily = daily[daily["day"] >= start].copy()

    # Regime
    daily["ema_fast"] = daily["close"].ewm(span=int(args.ema_fast), adjust=False).mean()
    daily["ema_slow"] = daily["close"].ewm(span=int(args.ema_slow), adjust=False).mean()
    daily["slope_slow"] = daily["ema_slow"].pct_change(int(args.slope_lookback))
    bull_raw = (daily["close"] > daily["ema_slow"]) & (daily["ema_fast"] > daily["ema_slow"]) & (daily["slope_slow"] > 0)
    bear_raw = (daily["close"] < daily["ema_slow"]) & (daily["ema_fast"] < daily["ema_slow"]) & (daily["slope_slow"] < 0)

    state = np.zeros(len(daily), dtype=int)
    on = False
    on_streak = 0
    off_streak = 0
    for i in range(len(daily)):
        on_streak = on_streak + 1 if bool(bull_raw.iat[i]) else 0
        off_streak = off_streak + 1 if bool(bear_raw.iat[i]) else 0
        if not on and on_streak >= int(args.persist_on):
            on = True
            off_streak = 0
        elif on and off_streak >= int(args.persist_off):
            on = False
            on_streak = 0
        state[i] = 1 if on else 0
    daily["bull_state"] = state
    daily["bull_exec"] = daily["bull_state"].shift(1).fillna(0).astype(int)

    # Weights and returns
    daily["w_crypto"] = np.where(daily["bull_exec"] > 0, float(args.bull_w), float(args.bear_w))
    daily["turnover"] = daily["w_crypto"].diff().abs().fillna(0.0)
    cost_k = float(args.switch_cost_bps) / 10000.0
    daily["cost"] = daily["turnover"] * cost_k
    daily["combo_r"] = daily["w_crypto"] * daily["crypto_r"] + (1.0 - daily["w_crypto"]) * daily["etf_r"] - daily["cost"]

    # Diagnostics
    up = daily["crypto_r"] > 0
    dn = daily["crypto_r"] < 0
    up_capture = float(daily.loc[up, "combo_r"].sum() / (daily.loc[up, "crypto_r"].sum() + 1e-12))
    down_capture = float(daily.loc[dn, "combo_r"].sum() / (daily.loc[dn, "crypto_r"].sum() - 1e-12))

    # Equity
    daily["eq_combo"] = np.exp(daily["combo_r"].cumsum())
    daily["eq_spot"] = np.exp(daily["crypto_r"].cumsum())
    daily["eq_etf"] = np.exp(daily["etf_r"].cumsum())
    for ccol in ("eq_combo", "eq_spot", "eq_etf"):
        daily[ccol] = daily[ccol] / float(daily[ccol].iloc[0])

    m_combo = _perf(daily["combo_r"])
    m_spot = _perf(daily["crypto_r"])
    m_etf = _perf(daily["etf_r"])
    summary = pd.DataFrame(
        [
            {
                "model": "regime_switch_combo",
                **m_combo,
                "avg_w_crypto": float(daily["w_crypto"].mean()),
                "bull_state_pct": float((daily["bull_exec"] > 0).mean()),
                "up_capture": up_capture,
                "down_capture": down_capture,
                "turnover_total": float(daily["turnover"].sum()),
            },
            {"model": "eth_spot", **m_spot, "avg_w_crypto": 1.0, "bull_state_pct": 1.0, "up_capture": 1.0, "down_capture": 1.0, "turnover_total": 0.0},
            {"model": "etf_basket", **m_etf, "avg_w_crypto": 0.0, "bull_state_pct": 0.0, "up_capture": 0.0, "down_capture": 0.0, "turnover_total": 0.0},
        ]
    )
    summary["window_start"] = str(daily["day"].iloc[0])
    summary["window_end"] = str(daily["day"].iloc[-1])
    summary["rows"] = len(daily)
    summary["bull_w"] = float(args.bull_w)
    summary["bear_w"] = float(args.bear_w)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Equity", "Price + EMAs", "Bull State", "Crypto Weight"],
    )
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_combo"], name="Combo Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_spot"], name="ETH Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["eq_etf"], name="ETF Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["close"], name="ETH close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema_fast"], name=f"EMA{args.ema_fast}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["ema_slow"], name=f"EMA{args.ema_slow}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["bull_exec"], name="bull_exec"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["w_crypto"], name="w_crypto"), row=4, col=1)
    fig.update_layout(height=1400, title="Regime Switch Crypto+ETF")

    html = (
        "<html><head><meta charset='utf-8'><title>Regime Switch Crypto ETF</title></head><body>"
        "<h3>Regime Switch Crypto + ETF</h3>"
        f"{summary.round(6).to_html(index=False, border=0)}"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "</body></html>"
    )
    Path(args.out_html).write_text(html, encoding="utf-8")

    print("wrote", args.out_csv)
    print("wrote", args.summary_csv)
    print("wrote", args.out_html)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
