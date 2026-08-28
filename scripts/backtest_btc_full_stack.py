from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


BINANCE_BASE = "https://api.binance.com/api/v3/klines"


def _to_ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _fill_symbol_gaps(d: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Crypto trades every calendar day, so any missing daily bar inside [start, end] is a genuine
    Binance data gap, not just a quiet weekend/holiday. Confirmed historically for ETHUSDC and BTCUSDC:
    both have zero klines from 2022-09-30 to 2023-03-11 (Binance suspended/delisted these USDC pairs for
    that window; verified directly against the raw klines endpoint, tvlUSD-style history is unaffected).
    Backfill any such gap from the equivalent *USDT symbol, which traded continuously throughout."""
    if d.empty or not symbol.upper().endswith("USDC"):
        return d
    full_days = pd.date_range(pd.Timestamp(start, tz="UTC").floor("D"), pd.Timestamp(end, tz="UTC").floor("D"), freq="D")
    have_days = set(d["timestamp"].dt.floor("D"))
    missing_days = [ts for ts in full_days if ts not in have_days]
    if not missing_days:
        return d
    proxy_symbol = symbol.upper()[:-4] + "USDT"
    gap_start = (min(missing_days) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    gap_end = (max(missing_days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fill = _download_binance_daily(proxy_symbol, gap_start, gap_end)
    if fill.empty:
        return d
    fill = fill[fill["timestamp"].dt.floor("D").isin(missing_days)]
    if fill.empty:
        return d
    print(
        f"[_download_binance_daily] {symbol}: filled {len(fill)} missing day(s) "
        f"({fill['timestamp'].min().date()} to {fill['timestamp'].max().date()}) using {proxy_symbol} as a proxy "
        f"(Binance has no {symbol} klines for this window)."
    )
    combined = pd.concat([d, fill], ignore_index=True).sort_values("timestamp")
    return combined.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)


def _download_binance_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    out: list[list[Any]] = []
    cur = start_ms
    while True:
        qs = urlencode(
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cur,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        url = f"{BINANCE_BASE}?{qs}"
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            break
        out.extend(data)
        last_open_ms = int(data[-1][0])
        if last_open_ms >= end_ms:
            break
        cur = last_open_ms + 1
        if len(data) < 1000:
            break

    if not out:
        return pd.DataFrame()
    d = pd.DataFrame(
        out,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "n_trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    d["timestamp"] = pd.to_datetime(pd.to_numeric(d["open_time"], errors="coerce"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "close"]).sort_values("timestamp")
    d = d[(d["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (d["timestamp"] <= pd.Timestamp(end, tz="UTC"))]
    d = d[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return _fill_symbol_gaps(d, symbol, start, end)


def smooth_short_islands(labels: pd.Series, min_persistence: int) -> pd.Series:
    x = labels.astype(str).copy().reset_index(drop=True)
    n = len(x)
    if n == 0 or min_persistence <= 1:
        return x
    changed = True
    while changed:
        changed = False
        run_id = (x != x.shift(1)).cumsum()
        runs = (
            pd.DataFrame({"idx": np.arange(n), "label": x, "run": run_id})
            .groupby("run")
            .agg(start=("idx", "min"), end=("idx", "max"), length=("idx", "size"), label=("label", "first"))
            .reset_index(drop=True)
        )
        for i in range(len(runs)):
            r = runs.iloc[i]
            if int(r["length"]) >= int(min_persistence):
                continue
            if i == 0 or i == len(runs) - 1:
                continue
            prev_label = runs.iloc[i - 1]["label"]
            next_label = runs.iloc[i + 1]["label"]
            if prev_label == next_label:
                s, e = int(r["start"]), int(r["end"])
                x.iloc[s : e + 1] = prev_label
                changed = True
    return x


def classify_regime_v2_on_btc(
    daily: pd.DataFrame,
    dd_window: int = 20,
    dd_bear_th: float = -0.15,
    min_persistence: int = 5,
) -> pd.DataFrame:
    d = daily.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ret_5d"] = d["close"].pct_change(5)
    d["ret_10d"] = d["close"].pct_change(10)
    d["rolling_dd"] = d["close"] / d["close"].rolling(int(dd_window), min_periods=max(5, dd_window // 2)).max() - 1.0

    bull_raw = (d["close"] > d["ema20"]) & (d["ema20"] > d["ema50"]) & (d["ret_5d"] > 0.0) & (d["ret_10d"] > 0.0)
    bear_raw = (d["close"] < d["ema20"]) & (d["ema20"] < d["ema50"]) & (d["ret_5d"] < 0.0)
    d["regime_raw"] = np.where(bull_raw, "BULL", np.where(bear_raw, "BEAR", "CHOP"))
    d.loc[d["rolling_dd"] < float(dd_bear_th), "regime_raw"] = "BEAR"

    d["regime_smoothed"] = pd.Series(smooth_short_islands(d["regime_raw"], int(min_persistence)).to_numpy(), index=d.index)
    d["regime_v2"] = d["regime_smoothed"].shift(1).fillna("CHOP")
    return d


def load_eth_defensive_proxy(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if not {"timestamp", "weight"}.issubset(d.columns):
        raise ValueError(f"{path} must contain timestamp and weight")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = d.set_index("timestamp")["weight"].resample("1D").last().ffill().to_frame("def_proxy_signal")
    out.index.name = "day"
    return out.reset_index()


def perf(simple_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_r, errors="coerce").fillna(0.0)
    years = len(x) / 252.0 if len(x) else np.nan
    eq = (1.0 + x).cumprod()
    total = float(eq.iloc[-1]) if len(eq) else np.nan
    cagr = (total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(np.mean(ex) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else np.nan
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    return {"ret": total - 1.0, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def grade_btc(sharpe: float) -> str:
    if np.isfinite(sharpe) and sharpe >= 0.8:
        return "PASS"
    if np.isfinite(sharpe) and sharpe >= 0.6:
        return "MARGINAL"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC full-stack diagnostic (ETH-style stack with funding proxy)")
    ap.add_argument("--symbol", default="BTCUSDC")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--eth-defensive-csv", default="artifacts/backtest/direction_event_model_v1_flat_defensive_6y_gapfilled.csv")
    ap.add_argument("--eth-reference-summary", default="artifacts/backtest/combined_offtf_ema21_55_144_defv1_flat_routerA_6y_summary.csv")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--cost-mode", choices=["entry_exit", "weight_change"], default="weight_change")
    ap.add_argument("--confirm-days", type=int, default=3)
    ap.add_argument("--out-price-csv", default="data/btc_daily.csv")
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/btc_full_stack_daily.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/btc_full_stack_summary.csv")
    args = ap.parse_args()

    # 1) BTC daily data from Binance spot.
    btc = _download_binance_daily(symbol=args.symbol, start=args.start, end=args.end)
    if btc.empty:
        raise RuntimeError("No BTC data downloaded from Binance for requested window.")
    Path(args.out_price_csv).parent.mkdir(parents=True, exist_ok=True)
    btc.to_csv(args.out_price_csv, index=False)

    # 2) Regime (same v2 logic, applied to BTC daily).
    d = btc.copy()
    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)

    # 3) Offensive sleeve (EMA21/55/144 + 3d confirm, vol scalar).
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema55"] = d["close"].ewm(span=55, adjust=False).mean()
    d["ema144"] = d["close"].ewm(span=144, adjust=False).mean()
    d["stack_aligned"] = (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])
    conf = max(1, int(args.confirm_days))
    entry = (d["stack_aligned"].rolling(conf, min_periods=conf).min() == 1).fillna(False)
    exit_ = ((~d["stack_aligned"]).rolling(conf, min_periods=conf).min() == 1).fillna(False)
    rv = d["close"].pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    d["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(d), dtype=int)
    active = 0
    for i in range(len(d)):
        if active == 0 and bool(entry.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
        pos[i] = active
    d["off_active"] = pos
    d["off_raw_signal"] = np.where(d["off_active"] == 1, d["vol_scalar"], 0.0)

    off_scale_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    d["off_scale"] = d["regime_v2"].map(off_scale_map).fillna(0.0)
    d["off_target"] = d["off_raw_signal"] * d["off_scale"]

    # 4) Defensive proxy sleeve from ETH funding/event model.
    def_proxy = load_eth_defensive_proxy(Path(args.eth_defensive_csv))
    d = d.merge(def_proxy, on="day", how="left")
    d["def_proxy_signal"] = pd.to_numeric(d["def_proxy_signal"], errors="coerce").fillna(0.0)
    def_scale_map = {"BULL": 0.0, "CHOP": 0.4, "BEAR": 1.0}
    d["def_scale"] = d["regime_v2"].map(def_scale_map).fillna(0.0)
    d["def_target"] = d["def_proxy_signal"] * d["def_scale"]

    # 5) Router combine.
    d["weight_target"] = (d["off_target"] + d["def_target"]).clip(lower=0.0, upper=1.0)

    # 6) Lag-1 realistic execution, open-to-open returns.
    d["exec_return"] = d["open"].shift(-1) / d["open"] - 1.0
    d["weight_exec"] = d["weight_target"].shift(1).fillna(0.0)
    prev_w = d["weight_exec"].shift(1).fillna(0.0)
    if str(args.cost_mode).lower() == "entry_exit":
        crossed = ((d["weight_exec"] > 0).astype(int) != (prev_w > 0).astype(int))
        d["turnover"] = np.where(crossed, (d["weight_exec"] - prev_w).abs(), 0.0)
    else:
        d["turnover"] = (d["weight_exec"] - prev_w).abs()
    d["cost"] = d["turnover"] * (float(args.cost_bps) / 10000.0)
    d["strategy_return"] = d["weight_exec"] * d["exec_return"].fillna(0.0) - d["cost"]
    d = d.dropna(subset=["exec_return"]).copy()
    d["eq"] = (1.0 + d["strategy_return"]).cumprod()
    d["spot_eq"] = (1.0 + d["exec_return"]).cumprod()

    # 7) Metrics.
    m_btc = perf(d["strategy_return"])
    trades = int(((d["weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())
    tim = float((d["weight_exec"] > 0).mean())
    verdict = grade_btc(m_btc["sharpe"])

    # ETH reference (from project canonical summary file, note non-comparable stack/window/frequency).
    eth_ref = {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "time_in_market_pct": np.nan}
    ref_path = Path(args.eth_reference_summary)
    if ref_path.exists():
        rs = pd.read_csv(ref_path)
        if len(rs):
            eth_ref["cagr"] = float(pd.to_numeric(rs.loc[0, "cagr"], errors="coerce"))
            eth_ref["sharpe"] = float(pd.to_numeric(rs.loc[0, "sharpe"], errors="coerce"))
            eth_ref["max_dd"] = float(pd.to_numeric(rs.loc[0, "max_dd"], errors="coerce"))
            eth_ref["time_in_market_pct"] = float(pd.to_numeric(rs.loc[0, "time_in_market_pct"], errors="coerce"))

    summary = pd.DataFrame(
        [
            {
                "period_start": str(pd.to_datetime(d["timestamp"].iloc[0]).date()),
                "period_end": str(pd.to_datetime(d["timestamp"].iloc[-1]).date()),
                "btc_symbol": args.symbol,
                "confirm_days": conf,
                "cost_mode": str(args.cost_mode),
                "btc_cagr": m_btc["cagr"],
                "btc_sharpe": m_btc["sharpe"],
                "btc_max_dd": m_btc["max_dd"],
                "btc_ann_vol": m_btc["ann_vol"],
                "btc_trades": trades,
                "btc_time_in_market_pct": tim * 100.0,
                "btc_verdict": verdict,
                "eth_ref_cagr": eth_ref["cagr"],
                "eth_ref_sharpe": eth_ref["sharpe"],
                "eth_ref_max_dd": eth_ref["max_dd"],
                "eth_ref_time_in_market_pct": eth_ref["time_in_market_pct"],
                "notes": "ETH reference is 5m full-stack and different window; not strict apples-to-apples.",
            }
        ]
    )

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    d[
        [
            "timestamp",
            "day",
            "open",
            "close",
            "regime_v2",
            "ema21",
            "ema55",
            "ema144",
            "stack_aligned",
            "off_active",
            "vol_scalar",
            "off_target",
            "def_proxy_signal",
            "def_target",
            "weight_target",
            "weight_exec",
            "exec_return",
            "turnover",
            "cost",
            "strategy_return",
            "eq",
            "spot_eq",
        ]
    ].to_csv(out_daily, index=False)

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print("===============================================")
    print("BTC vs ETH FULL STACK COMPARISON")
    print("Lag-1 realistic execution (daily simplification)")
    print("===============================================")
    print("Metric            ETH_ref         BTC")
    print("-----------------------------------------------")
    print(f"CAGR              {eth_ref['cagr']*100:>6.2f}%      {m_btc['cagr']*100:>6.2f}%")
    print(f"Sharpe            {eth_ref['sharpe']:>6.3f}      {m_btc['sharpe']:>6.3f}")
    print(f"MaxDD             {eth_ref['max_dd']*100:>6.2f}%      {m_btc['max_dd']*100:>6.2f}%")
    print(f"Trades            {'n/a':>6}      {trades:>6}")
    print(f"Time in market    {eth_ref['time_in_market_pct']:>6.2f}%      {tim*100.0:>6.2f}%")
    print(f"Cost mode         {'n/a':>6}      {args.cost_mode:>6}")
    print("-----------------------------------------------")
    print(f"Verdict: BTC {verdict} (PASS>=0.8, MARGINAL>=0.6)")
    print("===============================================")
    print(f"Saved: {args.out_price_csv}")
    print(f"Saved: {args.out_daily_csv}")
    print(f"Saved: {args.out_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
