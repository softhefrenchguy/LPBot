from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _perf(log_r: np.ndarray, bar_minutes: int = 5) -> dict[str, float]:
    x = np.asarray(log_r, dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    ret = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (bars_per_year / len(x)) - 1.0)
    ann_vol = float(np.std(x) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x) * bars_per_year) / (ann_vol + 1e-12))
    max_dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"ret": ret, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def _ramp_weight(target: np.ndarray, max_dw: float, decision_step_bars: int, min_rebalance_delta: float) -> np.ndarray:
    out = np.zeros_like(target, dtype=float)
    if len(target) == 0:
        return out
    out[0] = float(target[0])
    step = float(max(0.0, max_dw))
    dec_step = int(max(1, decision_step_bars))
    min_delta = float(max(0.0, min_rebalance_delta))
    for i in range(1, len(target)):
        prev = out[i - 1]
        if i % dec_step != 0:
            out[i] = prev
            continue
        t = float(target[i])
        if abs(t - prev) < min_delta:
            # Do not leave residual dust when target is flat.
            out[i] = 0.0 if t <= 0.0 else prev
            continue
        if t > prev:
            out[i] = min(prev + step, t)
        else:
            out[i] = max(prev - step, t)
    return out


def _hysteresis_bool(on_cond: np.ndarray, off_cond: np.ndarray, persist_on_bars: int, persist_off_bars: int) -> np.ndarray:
    n = len(on_cond)
    out = np.zeros(n, dtype=bool)
    state = False
    on_count = 0
    off_count = 0
    p_on = int(max(1, persist_on_bars))
    p_off = int(max(1, persist_off_bars))
    for i in range(n):
        if on_cond[i]:
            on_count += 1
        else:
            on_count = 0
        if off_cond[i]:
            off_count += 1
        else:
            off_count = 0

        if (not state) and on_count >= p_on:
            state = True
            off_count = 0
        elif state and off_count >= p_off:
            state = False
            on_count = 0
        out[i] = state
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay bull signal on top of breakout base weight.")
    p.add_argument("--base-csv", default="artifacts/backtest/breakout_base.csv")
    p.add_argument("--ema-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    p.add_argument("--ema-span", type=int, default=200)
    p.add_argument("--score-span", type=int, default=144)
    p.add_argument("--score-std-window", type=int, default=288)
    p.add_argument("--score-quantile", type=float, default=0.9)
    p.add_argument("--score-quantile-window", type=int, default=20000)
    p.add_argument("--bull-floor", type=float, default=0.35)
    p.add_argument("--bull-persist-on-bars", type=int, default=6)
    p.add_argument("--bull-persist-off-bars", type=int, default=12)
    p.add_argument("--weight-step", type=float, default=0.05, help="Quantize target weight to reduce churn; <=0 disables")
    p.add_argument("--max-dw-per-bar", type=float, default=0.02)
    p.add_argument("--decision-step-bars", type=int, default=1, help="Only rebalance every N bars")
    p.add_argument("--min-rebalance-delta", type=float, default=0.0, help="Ignore target changes smaller than this")
    p.add_argument("--cost-bps-per-turn", type=float, default=0.0)
    p.add_argument("--out-csv", default="artifacts/backtest/breakout_bullsignal.csv")
    p.add_argument("--out-html", default="artifacts/backtest/breakout_bullsignal.html")
    args = p.parse_args()

    d = pd.read_csv(args.base_csv)
    req = {"timestamp", "close", "r", "weight"}
    if not req.issubset(set(d.columns)):
        raise ValueError(f"{args.base_csv} must contain columns: {sorted(req)}")

    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["r"] = pd.to_numeric(d["r"], errors="coerce").fillna(0.0)
    d["weight_base"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

    # EMA bull signal (no lookahead on higher timeframe via shift(1))
    tmp = d.set_index("timestamp")["close"]
    tf_map = {"same": "5min", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = tf_map[args.ema_timeframe]
    if tf == "5min":
        close_tf = tmp
        ema_tf = close_tf.ewm(span=int(args.ema_span), adjust=False).mean()
        ema_on = (close_tf > ema_tf).fillna(False).to_numpy(dtype=bool)
    else:
        close_tf = tmp.resample(tf).last().shift(1).ffill()
        ema_tf = close_tf.ewm(span=int(args.ema_span), adjust=False).mean()
        ema_on = (close_tf > ema_tf).reindex(tmp.index, method="ffill").fillna(False).to_numpy(dtype=bool)

    # velocity + acceleration score
    y = np.log(d["close"])
    e = y.ewm(span=int(args.score_span), adjust=False).mean()
    v = e.diff()
    a = v.diff()
    sigma = v.rolling(int(args.score_std_window), min_periods=max(50, int(args.score_std_window // 4))).std()
    score = (v / (sigma + 1e-12)) + 0.5 * (a / (sigma + 1e-12))
    score = score.replace([np.inf, -np.inf], np.nan)

    qwin = int(args.score_quantile_window)
    score_q = score.rolling(qwin, min_periods=max(1000, qwin // 10)).quantile(float(args.score_quantile)).shift(1)
    score_on = (score >= score_q).fillna(False).to_numpy(dtype=bool)

    bull_signal_raw = ema_on & score_on
    bull_signal = _hysteresis_bool(
        on_cond=bull_signal_raw,
        off_cond=~bull_signal_raw,
        persist_on_bars=int(args.bull_persist_on_bars),
        persist_off_bars=int(args.bull_persist_off_bars),
    )

    target = d["weight_base"].to_numpy(copy=True)
    floor = float(args.bull_floor)
    target[bull_signal] = np.maximum(target[bull_signal], floor)
    target = np.clip(target, 0.0, 1.0)
    if float(args.weight_step) > 0:
        step = float(args.weight_step)
        target = np.clip(np.round(target / step) * step, 0.0, 1.0)

    weight = _ramp_weight(
        target,
        max_dw=float(args.max_dw_per_bar),
        decision_step_bars=int(args.decision_step_bars),
        min_rebalance_delta=float(args.min_rebalance_delta),
    )
    turnover = np.abs(np.diff(np.r_[weight[0], weight]))
    cost = (float(args.cost_bps_per_turn) / 10000.0) * turnover

    strat_r = np.r_[0.0, weight[:-1] * d["r"].to_numpy()[1:]] - cost
    base_r = np.r_[0.0, d["weight_base"].to_numpy()[:-1] * d["r"].to_numpy()[1:]]
    spot_r = d["r"].to_numpy()

    out = d[["timestamp", "close", "r"]].copy()
    out["ema_on"] = ema_on.astype(int)
    out["score"] = score
    out["score_q"] = score_q
    out["score_on"] = score_on.astype(int)
    out["bull_signal_raw"] = bull_signal_raw.astype(int)
    out["bull_signal"] = bull_signal.astype(int)
    out["weight_base"] = d["weight_base"]
    out["weight_target"] = target
    out["weight"] = weight
    out["turnover"] = turnover
    out["cost_r"] = cost
    out["base_r"] = base_r
    out["strat_r"] = strat_r
    out["spot_r"] = spot_r
    out["base_eq"] = np.exp(np.cumsum(base_r))
    out["strat_eq"] = np.exp(np.cumsum(strat_r))
    out["spot_eq"] = np.exp(np.cumsum(spot_r))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # report
    perf_base = _perf(base_r)
    perf_new = _perf(strat_r)
    perf_spot = _perf(spot_r)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Equity", "Weight", "Bull Signal"),
    )
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["strat_eq"], name="BullSignal Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["base_eq"], name="Base Breakout Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["spot_eq"], name="Spot Eq"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight"], name="Weight New"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["weight_base"], name="Weight Base"), row=2, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["bull_signal"], name="Bull Signal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["ema_on"], name="EMA On"), row=3, col=1)
    fig.add_trace(go.Scatter(x=out["timestamp"], y=out["score_on"], name="Score On"), row=3, col=1)
    fig.update_layout(height=980, title="Breakout + Bull Signal Overlay")

    hdr = (
        "<html><head><meta charset='utf-8'><title>Breakout Bull Signal</title></head><body>"
        "<h3>Breakout + Bull Signal Overlay</h3>"
        f"<p>ema={args.ema_timeframe}:{args.ema_span} | score_span={args.score_span} | "
        f"score_q={args.score_quantile} (roll={args.score_quantile_window}) | "
        f"bull_floor={args.bull_floor} | max_dw={args.max_dw_per_bar} | "
        f"bull_persist_on={args.bull_persist_on_bars} | bull_persist_off={args.bull_persist_off_bars} | "
        f"weight_step={args.weight_step} | "
        f"decision_step={args.decision_step_bars} | min_delta={args.min_rebalance_delta} | "
        f"cost_bps={args.cost_bps_per_turn}</p>"
    )
    tbl = pd.DataFrame(
        [
            {"model": "bullsignal", **perf_new},
            {"model": "base_breakout", **perf_base},
            {"model": "spot", **perf_spot},
            {
                "model": "coverage",
                "ret": float(out["bull_signal"].mean()),
                "cagr": float(out["ema_on"].mean()),
                "ann_vol": float(out["score_on"].mean()),
                "sharpe": float(np.mean(turnover)),
                "max_dd": float(np.mean(out["weight"] > 0)),
            },
        ]
    )
    html = hdr + tbl.round(6).to_html(index=False, border=0) + fig.to_html(full_html=False, include_plotlyjs="cdn") + "</body></html>"

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print("wrote", out_csv)
    print("wrote", out_html)
    print(tbl.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
