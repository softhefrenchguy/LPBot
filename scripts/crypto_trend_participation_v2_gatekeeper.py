from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must include timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _metrics(log_r: pd.Series, bars_per_year: float) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x) * np.sqrt(bars_per_year))
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(x)) - 1.0),
        "ann_vol": ann_vol,
        "sharpe": float((np.mean(x) * bars_per_year) / (ann_vol + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def _apply_rebalance_controls(
    target: np.ndarray,
    decision_step_bars: int,
    min_rebalance_delta: float,
    max_dw_per_bar: float,
) -> np.ndarray:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    step = max(int(decision_step_bars), 1)
    for i, t in enumerate(target):
        if i % step != 0:
            out[i] = prev
            continue
        if abs(float(t) - prev) < min_rebalance_delta:
            out[i] = prev
            continue
        if max_dw_per_bar > 0:
            prev = prev + float(np.clip(float(t) - prev, -max_dw_per_bar, max_dw_per_bar))
        else:
            prev = float(t)
        out[i] = prev
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Crypto trend participation v2 + gatekeeper report.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-fast", type=int, default=72)
    p.add_argument("--ema-slow", type=int, default=288)
    p.add_argument("--hyst-on-pct", type=float, default=0.001)
    p.add_argument("--hyst-off-pct", type=float, default=0.001)
    p.add_argument("--cooldown-bars", type=int, default=24)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--vol-window", type=int, default=72)
    p.add_argument("--vol-smooth-window", type=int, default=72)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--decision-step-bars", type=int, default=3)
    p.add_argument("--min-rebalance-delta", type=float, default=0.02)
    p.add_argument("--max-dw-per-bar", type=float, default=0.05)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--last-days", type=int, default=365)

    # Gatekeeper thresholds
    p.add_argument("--min-sharpe-pass", type=float, default=0.7)
    p.add_argument("--max-dd-pass", type=float, default=-0.25, help="Maximum allowed drawdown (negative).")
    p.add_argument("--min-ret-vs-spot-pass", type=float, default=0.0, help="Required strategy ret - spot ret.")
    p.add_argument("--max-avg-turnover-pass", type=float, default=0.02)

    p.add_argument("--out-csv", default="artifacts/backtest/crypto_trend_participation_v2.csv")
    p.add_argument("--out-html", default="artifacts/backtest/crypto_trend_participation_v2.html")
    p.add_argument("--out-summary-csv", default="artifacts/backtest/crypto_trend_participation_v2_summary.csv")
    args = p.parse_args()

    df = _load_price(Path(args.price_5m))
    ts = df["timestamp"]
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    bars_per_year = 365 * 24 * (60 / args.bar_minutes)

    ema_fast = close.ewm(span=args.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=args.ema_slow, adjust=False).mean()
    trend_on_raw = (ema_fast > ema_slow) & (close > ema_fast * (1.0 + args.hyst_on_pct))
    trend_off_raw = (ema_fast < ema_slow) | (close < ema_fast * (1.0 - args.hyst_off_pct))

    state = np.zeros(len(df), dtype=float)
    cooldown = 0
    entries = np.zeros(len(df), dtype=int)
    exits = np.zeros(len(df), dtype=int)
    pos = 0.0
    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1
        if pos > 0 and bool(trend_off_raw.iat[i]):
            pos = 0.0
            exits[i] = 1
            cooldown = int(args.cooldown_bars)
        if pos == 0 and cooldown == 0 and bool(trend_on_raw.iat[i]):
            pos = 1.0
            entries[i] = 1
        state[i] = pos

    sigma = r.rolling(args.vol_window, min_periods=args.vol_window).std() * np.sqrt(bars_per_year)
    sigma_s = sigma.rolling(args.vol_smooth_window, min_periods=max(5, args.vol_smooth_window // 3)).mean()
    weight_raw = (args.target_vol / sigma_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, args.w_max)
    target = (weight_raw * state).to_numpy(float)
    w = _apply_rebalance_controls(
        target=target,
        decision_step_bars=args.decision_step_bars,
        min_rebalance_delta=args.min_rebalance_delta,
        max_dw_per_bar=args.max_dw_per_bar,
    )
    w = np.clip(w, 0.0, args.w_max)
    w_ser = pd.Series(w, index=df.index)
    w_prev = w_ser.shift(1).fillna(0.0)
    turnover = (w_ser - w_prev).abs()
    trade_cost_r = -turnover * (args.trade_cost_bps / 10000.0)
    strat_r = w_prev * r + trade_cost_r
    eq = np.exp(np.cumsum(strat_r.to_numpy()))
    spot_eq = np.exp(np.cumsum(r.to_numpy()))

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "r": r,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend_on_raw": trend_on_raw.astype(int),
            "trend_off_raw": trend_off_raw.astype(int),
            "state": state,
            "entry_flag": entries,
            "exit_flag": exits,
            "sigma_ann": sigma,
            "sigma_ann_smooth": sigma_s,
            "weight_raw": weight_raw,
            "weight": w_ser,
            "turnover": turnover,
            "trade_cost_r": trade_cost_r,
            "strat_r": strat_r,
            "eq": eq,
            "spot_eq": spot_eq,
        }
    )

    if args.last_days and args.last_days > 0:
        end_ts = out["timestamp"].iloc[-1]
        start_ts = end_ts - pd.Timedelta(days=args.last_days)
        out = out[out["timestamp"] >= start_ts].copy()

    m = _metrics(out["strat_r"], bars_per_year)
    ms = _metrics(out["r"], bars_per_year)

    avg_turnover = float(out["turnover"].mean())
    pass_sharpe = bool(m["sharpe"] >= args.min_sharpe_pass)
    pass_dd = bool(m["max_dd"] >= args.max_dd_pass)
    pass_ret_vs_spot = bool((m["ret"] - ms["ret"]) >= args.min_ret_vs_spot_pass)
    pass_turnover = bool(avg_turnover <= args.max_avg_turnover_pass)
    pass_all = pass_sharpe and pass_dd and pass_ret_vs_spot and pass_turnover

    gatekeeper = pd.DataFrame(
        [
            {"criterion": f"sharpe >= {args.min_sharpe_pass}", "value": m["sharpe"], "pass": pass_sharpe},
            {"criterion": f"max_dd >= {args.max_dd_pass}", "value": m["max_dd"], "pass": pass_dd},
            {"criterion": f"ret-spot >= {args.min_ret_vs_spot_pass}", "value": m["ret"] - ms["ret"], "pass": pass_ret_vs_spot},
            {"criterion": f"avg_turnover <= {args.max_avg_turnover_pass}", "value": avg_turnover, "pass": pass_turnover},
            {"criterion": "OVERALL_PASS", "value": float(pass_all), "pass": pass_all},
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(out)),
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "spot_sharpe": ms["sharpe"],
                "spot_max_dd": ms["max_dd"],
                "excess_vs_spot": m["ret"] - ms["ret"],
                "entries": int(out["entry_flag"].sum()),
                "exits": int(out["exit_flag"].sum()),
                "time_in_market_pct": float((out["weight"] > 1e-12).mean() * 100.0),
                "avg_weight": float(out["weight"].mean()),
                "avg_turnover": avg_turnover,
                "overall_pass": pass_all,
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Equity", "Price + EMAs", "Weights", "Signals", "Returns"),
    )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["eq"], name="Strategy Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["spot_eq"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["close"], name="Close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_fast"], name="EMA Fast"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_slow"], name="EMA Slow"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight"], name="Weight"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight_raw"], name="Weight Raw"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["state"], name="State"), row=4, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["entry_flag"], name="Entry"), row=4, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["exit_flag"], name="Exit"), row=4, col=1)
    fig.add_trace(go.Bar(x=out["timestamp"], y=out["strat_r"], name="Strat LogRet"), row=5, col=1)
    fig.update_layout(height=1650, title="Crypto Trend Participation v2 (Gatekeeper)")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = "<html><head><meta charset='utf-8'><title>Crypto Trend Participation v2</title></head><body>"
    html += "<h3>Crypto Trend Participation v2</h3>"
    html += "<h4>Summary</h4>" + summary.round(6).to_html(index=False, border=0)
    html += "<h4>Gatekeeper</h4>" + gatekeeper.round(6).to_html(index=False, border=0)
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "</body></html>"
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {args.out_summary_csv}")
    print(f"wrote {out_html}")
    print(summary.to_string(index=False))
    print(gatekeeper.to_string(index=False))


if __name__ == "__main__":
    main()
