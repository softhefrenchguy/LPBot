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
        raise ValueError("price csv must include timestamp and close columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def _metrics(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bpy = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x) * np.sqrt(bpy))
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bpy / len(eq)) - 1.0),
        "ann_vol": ann_vol,
        "sharpe": float((np.mean(x) * bpy) / (ann_vol + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Long-only limit-exit model with EMA+time re-entry.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-span", type=int, default=200)
    p.add_argument("--hyst-on-pct", type=float, default=0.001)
    p.add_argument("--hyst-off-pct", type=float, default=0.001)
    p.add_argument("--tp-pct", type=float, default=0.06, help="Take-profit from entry price, e.g. 0.06 = +6%")
    p.add_argument("--sl-pct", type=float, default=0.03, help="Stop-loss from entry price, e.g. 0.03 = -3%")
    p.add_argument("--max-hold-bars", type=int, default=0, help="0 disables time exit")
    p.add_argument("--reentry-cooldown-bars", type=int, default=24, help="Bars to wait after an exit")
    p.add_argument("--min-above-ema-bars", type=int, default=1, help="Consecutive bars above EMA required before entry")
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--last-days", type=int, default=365)
    p.add_argument("--out-csv", default="artifacts/backtest/limit_exit_ema_reentry.csv")
    p.add_argument("--out-html", default="artifacts/backtest/limit_exit_ema_reentry.html")
    args = p.parse_args()

    df = _load_price(Path(args.price_5m))
    close = df["close"]
    ts = df["timestamp"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    ema = close.ewm(span=args.ema_span, adjust=False).mean()

    in_pos = np.zeros(len(df), dtype=float)
    entry_px = np.full(len(df), np.nan)
    tp_level = np.full(len(df), np.nan)
    sl_level = np.full(len(df), np.nan)
    exit_reason = np.array([""] * len(df), dtype=object)
    entry_flag = np.zeros(len(df), dtype=int)
    exit_flag = np.zeros(len(df), dtype=int)

    pos = 0.0
    cur_entry = np.nan
    bars_in_trade = 0
    cooldown = 0
    above_ema_count = 0
    min_above = max(int(args.min_above_ema_bars), 1)

    for i in range(len(df)):
        px = float(close.iat[i])
        em = float(ema.iat[i])
        above = px > em * (1.0 + float(args.hyst_on_pct))
        below = px < em * (1.0 - float(args.hyst_off_pct))
        above_ema_count = above_ema_count + 1 if above else 0

        if cooldown > 0:
            cooldown -= 1

        if pos > 0:
            bars_in_trade += 1
            tp_hit = px >= cur_entry * (1.0 + float(args.tp_pct))
            sl_hit = px <= cur_entry * (1.0 - float(args.sl_pct))
            ema_exit = below
            time_exit = int(args.max_hold_bars) > 0 and bars_in_trade >= int(args.max_hold_bars)

            if sl_hit:
                pos = 0.0
                exit_flag[i] = 1
                exit_reason[i] = "sl"
                cooldown = int(args.reentry_cooldown_bars)
            elif tp_hit:
                pos = 0.0
                exit_flag[i] = 1
                exit_reason[i] = "tp"
                cooldown = int(args.reentry_cooldown_bars)
            elif ema_exit:
                pos = 0.0
                exit_flag[i] = 1
                exit_reason[i] = "ema"
                cooldown = int(args.reentry_cooldown_bars)
            elif time_exit:
                pos = 0.0
                exit_flag[i] = 1
                exit_reason[i] = "time"
                cooldown = int(args.reentry_cooldown_bars)

        if pos == 0.0 and cooldown == 0 and above_ema_count >= min_above:
            pos = float(args.position_size)
            cur_entry = px
            bars_in_trade = 0
            entry_flag[i] = 1

        in_pos[i] = pos
        if pos > 0:
            entry_px[i] = cur_entry
            tp_level[i] = cur_entry * (1.0 + float(args.tp_pct))
            sl_level[i] = cur_entry * (1.0 - float(args.sl_pct))

    w = pd.Series(in_pos, index=df.index).clip(lower=0.0, upper=1.0)
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs()
    trade_cost_r = -turnover * (float(args.trade_cost_bps) / 10000.0)
    strat_r = w_prev * r + trade_cost_r
    eq = np.exp(strat_r.cumsum())
    spot_eq = np.exp(r.cumsum())

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "r": r,
            "ema": ema,
            "weight": w,
            "entry_px": entry_px,
            "tp_level": tp_level,
            "sl_level": sl_level,
            "entry_flag": entry_flag,
            "exit_flag": exit_flag,
            "exit_reason": exit_reason,
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

    m = _metrics(out["strat_r"], args.bar_minutes)
    ms = _metrics(out["r"], args.bar_minutes)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity", "Price / EMA / TP-SL", "Weight", "Returns"),
    )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["eq"], name="Strategy Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["spot_eq"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["close"], name="Close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema"], name="EMA"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["tp_level"], name="TP Level"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["sl_level"], name="SL Level"), row=2, col=1)
    if out["entry_flag"].sum() > 0:
        entries = out[out["entry_flag"] == 1]
        fig.add_trace(
            go.Scatter(x=entries["timestamp"], y=entries["close"], mode="markers", name="Entries", marker=dict(symbol="triangle-up", size=8)),
            row=2,
            col=1,
        )
    if out["exit_flag"].sum() > 0:
        exits = out[out["exit_flag"] == 1]
        fig.add_trace(
            go.Scatter(x=exits["timestamp"], y=exits["close"], mode="markers", name="Exits", marker=dict(symbol="triangle-down", size=8)),
            row=2,
            col=1,
        )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight"], name="Weight"), row=3, col=1)
    fig.add_trace(go.Bar(x=out["timestamp"], y=out["strat_r"], name="Strat LogRet"), row=4, col=1)
    fig.update_layout(height=1300, title="Limit Exit + EMA/Time Re-entry Backtest")

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(out)),
                "entries": int(out["entry_flag"].sum()),
                "exits": int(out["exit_flag"].sum()),
                "time_in_market_pct": float((out["weight"] > 1e-12).mean() * 100.0),
                "avg_weight": float(out["weight"].mean()),
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "excess_vs_spot": m["ret"] - ms["ret"],
                "tp_pct": float(args.tp_pct),
                "sl_pct": float(args.sl_pct),
                "ema_span": int(args.ema_span),
                "cooldown_bars": int(args.reentry_cooldown_bars),
            }
        ]
    )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<html><head><meta charset='utf-8'><title>Limit Exit EMA Re-entry</title></head><body>"
        "<h3>Limit Exit + EMA/Time Re-entry</h3>"
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
