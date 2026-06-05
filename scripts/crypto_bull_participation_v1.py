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


def _step_cap(target: np.ndarray, max_dw: float) -> np.ndarray:
    out = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, t in enumerate(target):
        if max_dw > 0:
            prev = prev + float(np.clip(t - prev, -max_dw, max_dw))
        else:
            prev = float(t)
        out[i] = prev
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Crypto bull-participation v1 (slow trend + fast re-entry)")
    p.add_argument("--price-5m", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--ema-slow", type=int, default=300, help="slow bull regime filter")
    p.add_argument("--ema-fast", type=int, default=72, help="fast re-entry filter")
    p.add_argument("--hyst-on-pct", type=float, default=0.001)
    p.add_argument("--hyst-off-pct", type=float, default=0.001)
    p.add_argument("--min-above-fast-bars", type=int, default=2)
    p.add_argument("--cooldown-bars", type=int, default=12)
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--vol-window", type=int, default=36)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--max-dw-per-bar", type=float, default=0.10)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--last-days", type=int, default=365)
    p.add_argument("--out-csv", default="artifacts/backtest/crypto_bull_participation_v1.csv")
    p.add_argument("--out-html", default="artifacts/backtest/crypto_bull_participation_v1.html")
    args = p.parse_args()

    df = _load_price(Path(args.price_5m))
    ts = df["timestamp"]
    close = df["close"]
    r = np.log(close / close.shift(1)).fillna(0.0)

    bars_per_year = 365 * 24 * (60 / args.bar_minutes)

    ema_slow = close.ewm(span=args.ema_slow, adjust=False).mean()
    ema_fast = close.ewm(span=args.ema_fast, adjust=False).mean()
    bull_regime = close > ema_slow
    fast_above = close > ema_fast * (1.0 + args.hyst_on_pct)
    fast_below = close < ema_fast * (1.0 - args.hyst_off_pct)

    sigma = r.rolling(args.vol_window, min_periods=args.vol_window).std() * np.sqrt(bars_per_year)
    weight_raw = (args.target_vol / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, args.w_max)

    state = np.zeros(len(df), dtype=float)
    above_count = 0
    cooldown = 0
    entries = np.zeros(len(df), dtype=int)
    exits = np.zeros(len(df), dtype=int)
    exit_reason = np.array([""] * len(df), dtype=object)

    pos = 0.0
    min_above = max(int(args.min_above_fast_bars), 1)
    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1
        above_count = above_count + 1 if bool(fast_above.iat[i]) else 0

        if pos > 0:
            if not bool(bull_regime.iat[i]):
                pos = 0.0
                exits[i] = 1
                exit_reason[i] = "slow_regime_off"
                cooldown = int(args.cooldown_bars)
            elif bool(fast_below.iat[i]):
                pos = 0.0
                exits[i] = 1
                exit_reason[i] = "fast_exit"
                cooldown = int(args.cooldown_bars)

        if pos == 0.0 and cooldown == 0:
            if bool(bull_regime.iat[i]) and above_count >= min_above:
                pos = 1.0
                entries[i] = 1

        state[i] = pos

    target_w = weight_raw.to_numpy() * state
    w = _step_cap(target_w, float(args.max_dw_per_bar))
    w = np.clip(w, 0.0, float(args.w_max))

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
            "bull_regime": bull_regime.astype(int),
            "fast_above": fast_above.astype(int),
            "fast_below": fast_below.astype(int),
            "entry_flag": entries,
            "exit_flag": exits,
            "exit_reason": exit_reason,
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

    # Overall metrics
    m = _metrics(out["strat_r"], bars_per_year)
    ms = _metrics(out["r"], bars_per_year)

    # Bull-window contribution stats (avoid misleading compounding on non-contiguous subsets).
    bull_mask = out["bull_regime"] == 1
    bear_mask = ~bull_mask
    bull_strat_log = float(out.loc[bull_mask, "strat_r"].sum())
    bull_spot_log = float(out.loc[bull_mask, "r"].sum())
    bear_strat_log = float(out.loc[bear_mask, "strat_r"].sum())
    bear_spot_log = float(out.loc[bear_mask, "r"].sum())

    signal_stats = pd.DataFrame(
        [
            {
                "rows": int(len(out)),
                "entries": int(out["entry_flag"].sum()),
                "exits": int(out["exit_flag"].sum()),
                "bull_regime_pct": float(out["bull_regime"].mean() * 100.0),
                "time_in_market_pct": float((out["weight"] > 1e-12).mean() * 100.0),
                "avg_weight": float(out["weight"].mean()),
                "avg_turnover": float(out["turnover"].mean()),
                "total_turnover": float(out["turnover"].sum()),
            }
        ]
    )
    perf_overall = pd.DataFrame(
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
    perf_regime = pd.DataFrame(
        [
            {
                "bull_bars": int(bull_mask.sum()),
                "bull_bar_pct": float(bull_mask.mean() * 100.0),
                "strat_ret_contrib_bull": float(np.exp(bull_strat_log) - 1.0),
                "spot_ret_contrib_bull": float(np.exp(bull_spot_log) - 1.0),
                "bull_capture_log_ratio": float(bull_strat_log / bull_spot_log) if abs(bull_spot_log) > 1e-12 else np.nan,
                "strat_ret_contrib_bear": float(np.exp(bear_strat_log) - 1.0),
                "spot_ret_contrib_bear": float(np.exp(bear_spot_log) - 1.0),
                "bear_capture_log_ratio": float(bear_strat_log / bear_spot_log) if abs(bear_spot_log) > 1e-12 else np.nan,
            }
        ]
    )

    exit_counts = out[out["exit_flag"] == 1]["exit_reason"].value_counts(dropna=False).rename_axis("exit_reason").to_frame("count")
    if not exit_counts.empty:
        exit_counts["pct"] = exit_counts["count"] / exit_counts["count"].sum()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Equity", "Price + EMAs", "Weight", "Signals", "Returns"),
    )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["eq"], name="Strategy Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["spot_eq"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["close"], name="Close"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_fast"], name="EMA Fast"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_slow"], name="EMA Slow"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight"], name="Weight"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["bull_regime"], name="Bull Regime"), row=4, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["entry_flag"], name="Entry"), row=4, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["exit_flag"], name="Exit"), row=4, col=1)
    fig.add_trace(go.Bar(x=out["timestamp"], y=out["strat_r"], name="Strat LogRet"), row=5, col=1)
    fig.update_layout(height=1600, title="Crypto Bull Participation v1")

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = "<html><head><meta charset='utf-8'><title>Crypto Bull Participation v1</title></head><body>"
    html += "<h3>Crypto Bull Participation v1 (Slow Trend + Fast Re-entry)</h3>"
    html += "<h4>Overall Performance</h4>" + perf_overall.round(6).to_html(index=False, border=0)
    html += "<h4>Bull/Bear Regime Contribution</h4>" + perf_regime.round(6).to_html(index=False, border=0)
    html += "<h4>Signal Stats</h4>" + signal_stats.round(6).to_html(index=False, border=0)
    if not exit_counts.empty:
        html += "<h4>Exit Reason Breakdown</h4>" + exit_counts.round(6).to_html(border=0)
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "</body></html>"
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(perf_overall.to_string(index=False))
    print(perf_regime.to_string(index=False))
    print(signal_stats.to_string(index=False))


if __name__ == "__main__":
    main()
