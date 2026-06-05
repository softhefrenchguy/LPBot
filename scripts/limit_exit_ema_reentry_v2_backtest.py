from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "close", "high", "low"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"price csv missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("close", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "high", "low"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _metrics(log_r: pd.Series, bar_minutes: int) -> dict[str, float]:
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


def _trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "trades": 0,
                    "win_rate": np.nan,
                    "avg_trade_ret": np.nan,
                    "median_trade_ret": np.nan,
                    "avg_win": np.nan,
                    "avg_loss": np.nan,
                    "profit_factor": np.nan,
                    "avg_hold_bars": np.nan,
                    "median_hold_bars": np.nan,
                }
            ]
        )
    wins = trades[trades["trade_ret"] > 0]["trade_ret"]
    losses = trades[trades["trade_ret"] <= 0]["trade_ret"]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 1e-12 else np.nan
    return pd.DataFrame(
        [
            {
                "trades": int(len(trades)),
                "win_rate": float((trades["trade_ret"] > 0).mean()),
                "avg_trade_ret": float(trades["trade_ret"].mean()),
                "median_trade_ret": float(trades["trade_ret"].median()),
                "avg_win": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss": float(losses.mean()) if len(losses) else np.nan,
                "profit_factor": profit_factor,
                "avg_hold_bars": float(trades["hold_bars"].mean()),
                "median_hold_bars": float(trades["hold_bars"].median()),
            }
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Limit exit model v2: trend gate + ATR exits + EMA no-trade zone + diagnostics.")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-fast", type=int, default=100)
    p.add_argument("--ema-slow", type=int, default=300)
    p.add_argument("--hyst-on-pct", type=float, default=0.001)
    p.add_argument("--hyst-off-pct", type=float, default=0.001)
    p.add_argument("--no-trade-band-pct", type=float, default=0.002)
    p.add_argument("--min-above-ema-bars", type=int, default=3)
    p.add_argument("--atr-window", type=int, default=72)
    p.add_argument("--tp-atr-mult", type=float, default=2.5)
    p.add_argument("--sl-atr-mult", type=float, default=1.5)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--reentry-cooldown-bars", type=int, default=48)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--last-days", type=int, default=365)
    p.add_argument("--out-csv", default="artifacts/backtest/limit_exit_ema_reentry_v2.csv")
    p.add_argument("--out-html", default="artifacts/backtest/limit_exit_ema_reentry_v2.html")
    args = p.parse_args()

    df = _load_price(Path(args.price_5m))
    ts = df["timestamp"]
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)
    ema_fast = close.ewm(span=args.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=args.ema_slow, adjust=False).mean()
    atr = _atr(df, args.atr_window)

    trend_ok = (ema_fast > ema_slow) & (close > ema_fast)
    no_trade_zone = ((close - ema_fast).abs() / ema_fast) <= float(args.no_trade_band_pct)
    above_ema = close > ema_fast * (1.0 + float(args.hyst_on_pct))
    below_ema = close < ema_fast * (1.0 - float(args.hyst_off_pct))

    w = np.zeros(len(df), dtype=float)
    entry_px = np.full(len(df), np.nan)
    tp_level = np.full(len(df), np.nan)
    sl_level = np.full(len(df), np.nan)
    entry_flag = np.zeros(len(df), dtype=int)
    exit_flag = np.zeros(len(df), dtype=int)
    exit_reason = np.array([""] * len(df), dtype=object)
    cooldown_arr = np.zeros(len(df), dtype=int)

    pos = 0.0
    entry_price = np.nan
    entry_atr = np.nan
    entry_idx = -1
    bars_in_trade = 0
    cooldown = 0
    above_count = 0
    trades: list[dict[str, object]] = []

    min_above = max(int(args.min_above_ema_bars), 1)
    for i in range(len(df)):
        px = float(close.iat[i])
        cooldown = max(cooldown - 1, 0)
        cooldown_arr[i] = cooldown
        above_count = above_count + 1 if bool(above_ema.iat[i]) else 0

        if pos > 0:
            bars_in_trade += 1
            tp = entry_price + float(args.tp_atr_mult) * entry_atr
            sl = entry_price - float(args.sl_atr_mult) * entry_atr
            tp_level[i] = tp
            sl_level[i] = sl
            entry_px[i] = entry_price

            sl_hit = px <= sl
            tp_hit = px >= tp
            ema_exit = bool(below_ema.iat[i])
            time_exit = int(args.max_hold_bars) > 0 and bars_in_trade >= int(args.max_hold_bars)

            reason = ""
            if sl_hit:
                reason = "sl"
            elif tp_hit:
                reason = "tp"
            elif ema_exit:
                reason = "ema"
            elif time_exit:
                reason = "time"

            if reason:
                pos = 0.0
                exit_flag[i] = 1
                exit_reason[i] = reason
                cooldown = int(args.reentry_cooldown_bars)
                cooldown_arr[i] = cooldown
                trade_ret = float(px / entry_price - 1.0)
                trades.append(
                    {
                        "entry_ts": ts.iat[entry_idx],
                        "exit_ts": ts.iat[i],
                        "entry_px": float(entry_price),
                        "exit_px": float(px),
                        "trade_ret": trade_ret,
                        "hold_bars": int(bars_in_trade),
                        "exit_reason": reason,
                    }
                )

        if pos == 0.0:
            can_enter = (
                cooldown == 0
                and bool(trend_ok.iat[i])
                and not bool(no_trade_zone.iat[i])
                and above_count >= min_above
                and pd.notna(atr.iat[i])
                and float(atr.iat[i]) > 0
            )
            if can_enter:
                pos = float(args.position_size)
                entry_price = px
                entry_atr = float(atr.iat[i])
                entry_idx = i
                bars_in_trade = 0
                entry_flag[i] = 1
                entry_px[i] = entry_price
                tp_level[i] = entry_price + float(args.tp_atr_mult) * entry_atr
                sl_level[i] = entry_price - float(args.sl_atr_mult) * entry_atr

        w[i] = pos
        if pos > 0:
            entry_px[i] = entry_price

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "high": df["high"],
            "low": df["low"],
            "r": r,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr,
            "trend_ok": trend_ok.astype(int),
            "no_trade_zone": no_trade_zone.astype(int),
            "above_ema": above_ema.astype(int),
            "below_ema": below_ema.astype(int),
            "cooldown": cooldown_arr,
            "weight": w,
            "entry_px": entry_px,
            "tp_level": tp_level,
            "sl_level": sl_level,
            "entry_flag": entry_flag,
            "exit_flag": exit_flag,
            "exit_reason": exit_reason,
        }
    )

    w_ser = pd.Series(w, index=out.index)
    w_prev = w_ser.shift(1).fillna(0.0)
    turnover = (w_ser - w_prev).abs()
    out["trade_cost_r"] = -turnover * (float(args.trade_cost_bps) / 10000.0)
    out["strat_r"] = w_prev * out["r"] + out["trade_cost_r"]
    out["spot_r"] = out["r"]
    out["eq"] = np.exp(out["strat_r"].cumsum())
    out["spot_eq"] = np.exp(out["spot_r"].cumsum())
    out["turnover"] = turnover

    if args.last_days and args.last_days > 0:
        end_ts = out["timestamp"].iloc[-1]
        start_ts = end_ts - pd.Timedelta(days=args.last_days)
        out = out[out["timestamp"] >= start_ts].copy()

    # Rebuild trades in chosen window for accurate diagnostics.
    start_bound = out["timestamp"].iloc[0]
    end_bound = out["timestamp"].iloc[-1]
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df[(trades_df["entry_ts"] >= start_bound) & (trades_df["exit_ts"] <= end_bound)].copy()
    exit_counts = out[out["exit_flag"] == 1]["exit_reason"].value_counts(dropna=False).rename_axis("exit_reason").to_frame("count")
    if not exit_counts.empty:
        exit_counts["pct"] = exit_counts["count"] / exit_counts["count"].sum()

    m = _metrics(out["strat_r"], args.bar_minutes)
    ms = _metrics(out["spot_r"], args.bar_minutes)
    trade_stats = _trade_stats(trades_df)

    signal_stats = pd.DataFrame(
        [
            {
                "rows": int(len(out)),
                "entries": int(out["entry_flag"].sum()),
                "exits": int(out["exit_flag"].sum()),
                "trend_ok_pct": float(out["trend_ok"].mean() * 100.0),
                "no_trade_zone_pct": float(out["no_trade_zone"].mean() * 100.0),
                "time_in_market_pct": float((out["weight"] > 1e-12).mean() * 100.0),
                "avg_weight": float(out["weight"].mean()),
                "avg_daily_turnover": float(out["turnover"].mean()),
                "total_turnover": float(out["turnover"].sum()),
            }
        ]
    )

    perf_df = pd.DataFrame(
        [
            {
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": ms["ret"],
                "spot_sharpe": ms["sharpe"],
                "spot_max_dd": ms["max_dd"],
                "excess_vs_spot": m["ret"] - ms["ret"],
            }
        ]
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Equity", "Price / EMAs", "Entry Levels", "Weight", "Signals"),
    )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["eq"], name="Strategy Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["spot_eq"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["close"], name="Close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_fast"], name="EMA Fast"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_slow"], name="EMA Slow"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["tp_level"], name="TP Level"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["sl_level"], name="SL Level"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["entry_px"], name="Entry Px"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight"], name="Weight"), row=4, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["trend_ok"], name="Trend OK"), row=5, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["no_trade_zone"], name="No-Trade Zone"), row=5, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["entry_flag"], name="Entry Flag"), row=5, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["exit_flag"], name="Exit Flag"), row=5, col=1)
    fig.update_layout(height=1600, title="Limit Exit v2: Trend Gate + ATR Exits + EMA No-Trade Zone")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = "<html><head><meta charset='utf-8'><title>Limit Exit EMA Reentry v2</title></head><body>"
    html += "<h3>Limit Exit v2 Backtest</h3>"
    html += "<p>Added: trend gate (EMA fast/slow), ATR exits, EMA no-trade zone, diagnostics.</p>"
    html += "<h4>Performance</h4>" + perf_df.round(6).to_html(index=False, border=0)
    html += "<h4>Signal/Participation Stats</h4>" + signal_stats.round(6).to_html(index=False, border=0)
    html += "<h4>Trade Stats</h4>" + trade_stats.round(6).to_html(index=False, border=0)
    if not exit_counts.empty:
        html += "<h4>Exit Reason Breakdown</h4>" + exit_counts.round(6).to_html(border=0)
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "</body></html>"
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(perf_df.to_string(index=False))
    print(signal_stats.to_string(index=False))
    print(trade_stats.to_string(index=False))


if __name__ == "__main__":
    main()
