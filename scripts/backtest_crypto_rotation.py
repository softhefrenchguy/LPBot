from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency yfinance. Install with: pip install yfinance pandas numpy --break-system-packages"
    ) from exc


ASSETS = [
    {"name": "ETH", "ticker": "ETH-USD"},
    {"name": "BTC", "ticker": "BTC-USD"},
]


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _download_asset(ticker: str, start: str, end: str, yf_cache_dir: Path) -> pd.DataFrame:
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(yf_cache_dir))
    except Exception:
        pass
    d = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if d is None or len(d) == 0:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in d.columns]
    d = d.reset_index()
    if "Date" not in d.columns:
        if "index" in d.columns:
            d = d.rename(columns={"index": "Date"})
        elif "Datetime" in d.columns:
            d = d.rename(columns={"Datetime": "Date"})
        elif len(d.columns) > 0:
            d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    if "Adj Close" in d.columns:
        d["price"] = pd.to_numeric(d["Adj Close"], errors="coerce")
    elif "Close" in d.columns:
        d["price"] = pd.to_numeric(d["Close"], errors="coerce")
    else:
        return pd.DataFrame()
    if "Open" in d.columns:
        d["open"] = pd.to_numeric(d["Open"], errors="coerce")
    else:
        d["open"] = pd.to_numeric(d["price"], errors="coerce")
    d["price"] = d["price"].ffill(limit=3)
    d["open"] = d["open"].ffill(limit=3)
    d = d.dropna(subset=["price", "open"])
    return d[["Date", "price", "open"]].copy()


def _compute_regime(price: pd.Series) -> pd.Series:
    ret20 = price / price.shift(20) - 1.0
    dd20 = price / price.rolling(20, min_periods=20).max() - 1.0
    regime = pd.Series("CHOP", index=price.index, dtype=object)
    regime[(ret20 > 0.05) & (dd20 > -0.10)] = "BULL"
    regime[dd20 < -0.15] = "BEAR"
    return regime


def _build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().set_index("Date").sort_index()
    df["daily_return"] = df["price"].pct_change().fillna(0.0)
    df["exec_return"] = df["open"].shift(-1) / df["open"] - 1.0
    df["ret20"] = df["price"] / df["price"].shift(20) - 1.0
    df["ema21"] = df["price"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["price"].ewm(span=55, adjust=False).mean()
    df["ema144"] = df["price"].ewm(span=144, adjust=False).mean()
    df["stack"] = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema144"])
    grp = (~df["stack"]).cumsum()
    run = df["stack"].groupby(grp).cumcount() + 1
    df["days_aligned"] = run.where(df["stack"], 0).astype(int)
    df["regime"] = _compute_regime(df["price"])
    rv = df["daily_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["vol_scalar"] = vol_scalar.clip(lower=0.25, upper=1.0).fillna(0.25)
    return df


def _metrics(r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
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
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "ann_vol": ann_vol}


def _single_asset_strategy(df: pd.DataFrame, cost_bps: float, exec_lag_days: int) -> pd.DataFrame:
    out = df.copy()
    entry = (out["stack"].rolling(3, min_periods=3).min() == 1).fillna(False)
    exit_ = ((~out["stack"]).rolling(3, min_periods=3).min() == 1).fillna(False)
    off = out["regime"].replace({"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}).astype(float)
    pos = np.zeros(len(out), dtype=int)
    active = 0
    for i in range(len(out)):
        if active == 0 and bool(entry.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
        pos[i] = active
    out["active"] = pos
    out["scaled_weight_signal"] = np.where(out["active"] == 1, out["vol_scalar"] * off, 0.0)
    out["scaled_weight_exec"] = out["scaled_weight_signal"].shift(int(exec_lag_days)).fillna(0.0)
    prev_w = out["scaled_weight_exec"].shift(1).fillna(0.0)
    turnover = (out["scaled_weight_exec"] - prev_w).abs()
    cost = turnover * (cost_bps / 10000.0)
    out["strategy_return"] = out["scaled_weight_exec"] * out["exec_return"].fillna(0.0) - cost
    out["turnover"] = turnover
    return out


def _score_frame(eth: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    idx = eth.index.union(btc.index)
    e = eth.reindex(idx)
    b = btc.reindex(idx)
    ret20 = pd.DataFrame({"ETH": e["ret20"], "BTC": b["ret20"]}, index=idx)
    perf_pts = ret20.rank(axis=1, method="average", pct=True).fillna(0.0) * 3.0

    def _asset_score(x: pd.DataFrame, perf_col: pd.Series) -> pd.Series:
        regime_pts = x["regime"].replace({"BULL": 2.0, "CHOP": 1.0, "BEAR": 0.0}).astype(float)
        s = (
            3.0 * x["stack"].astype(float)
            + 2.0 * (x["days_aligned"] >= 3).astype(float)
            + regime_pts
            + (x["price"] > x["ema144"]).astype(float)
            + perf_col
        )
        eligible = x["stack"] & (x["days_aligned"] >= 3) & (x["regime"] != "BEAR")
        return s.where(eligible, -1.0)

    score = pd.DataFrame(index=idx)
    score["ETH"] = _asset_score(e, perf_pts["ETH"])
    score["BTC"] = _asset_score(b, perf_pts["BTC"])
    return score


def _crypto_rotation(
    eth: pd.DataFrame,
    btc: pd.DataFrame,
    alloc_total: float,
    cost_bps: float,
    min_hold_days: int,
    switch_gap: float,
    enter_score: float,
    exit_score: float,
    exec_lag_days: int,
) -> pd.DataFrame:
    idx = eth.index.union(btc.index)
    e = eth.reindex(idx).copy()
    b = btc.reindex(idx).copy()
    score = _score_frame(eth, btc).reindex(idx)
    off_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}

    held: str | None = None
    held_days = 0
    w_eth_signal = np.zeros(len(idx), dtype=float)
    w_btc_signal = np.zeros(len(idx), dtype=float)
    switches = 0

    for i, t in enumerate(idx):
        s_eth = float(score.at[t, "ETH"]) if pd.notna(score.at[t, "ETH"]) else -1e9
        s_btc = float(score.at[t, "BTC"]) if pd.notna(score.at[t, "BTC"]) else -1e9
        best = "ETH" if s_eth >= s_btc else "BTC"
        best_score = max(s_eth, s_btc)
        held_score = s_eth if held == "ETH" else s_btc if held == "BTC" else -1e9

        if held is None:
            if best_score >= enter_score:
                held = best
                held_days = 0
                switches += 1
        else:
            held_days += 1
            can_switch = held_days >= int(min_hold_days)
            if can_switch and held_score < exit_score:
                if best_score >= enter_score and best != held:
                    held = best
                    held_days = 0
                    switches += 1
                elif best_score < enter_score:
                    held = None
                    held_days = 0
                    switches += 1
            elif can_switch and best != held and best_score >= held_score + switch_gap and best_score >= enter_score:
                held = best
                held_days = 0
                switches += 1

        we = 0.0
        wb = 0.0
        if held == "ETH":
            off = float(off_map.get(str(e.at[t, "regime"]), 0.0)) if pd.notna(e.at[t, "regime"]) else 0.0
            vol = float(e.at[t, "vol_scalar"]) if pd.notna(e.at[t, "vol_scalar"]) else 0.25
            we = alloc_total * max(0.0, min(1.0, vol)) * off
        elif held == "BTC":
            off = float(off_map.get(str(b.at[t, "regime"]), 0.0)) if pd.notna(b.at[t, "regime"]) else 0.0
            vol = float(b.at[t, "vol_scalar"]) if pd.notna(b.at[t, "vol_scalar"]) else 0.25
            wb = alloc_total * max(0.0, min(1.0, vol)) * off

        w_eth_signal[i] = we
        w_btc_signal[i] = wb

    w_eth_exec = pd.Series(w_eth_signal, index=idx).shift(int(exec_lag_days)).fillna(0.0)
    w_btc_exec = pd.Series(w_btc_signal, index=idx).shift(int(exec_lag_days)).fillna(0.0)
    turnover = (w_eth_exec - w_eth_exec.shift(1).fillna(0.0)).abs() + (w_btc_exec - w_btc_exec.shift(1).fillna(0.0)).abs()
    cost = turnover * (cost_bps / 10000.0)
    port_r = (w_eth_exec * e["exec_return"].fillna(0.0)) + (w_btc_exec * b["exec_return"].fillna(0.0)) - cost

    out = pd.DataFrame(
        {
            "date": idx,
            "eth_weight_signal": w_eth_signal,
            "btc_weight_signal": w_btc_signal,
            "eth_weight_exec": w_eth_exec.values,
            "btc_weight_exec": w_btc_exec.values,
            "turnover": turnover.values,
            "portfolio_return": port_r.values,
            "top_asset_exec": np.where(w_eth_exec.values > 0, "ETH", np.where(w_btc_exec.values > 0, "BTC", "CASH")),
        }
    )
    out["equity"] = (1.0 + out["portfolio_return"]).cumprod()
    out.attrs["switches"] = switches
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="ETH vs BTC: independent vs rotation backtest")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--alloc", type=float, default=0.25, help="Total portfolio allocation for A/B/D, split in C.")
    p.add_argument("--min-hold-days", type=int, default=21)
    p.add_argument("--switch-gap", type=float, default=2.0)
    p.add_argument("--enter-score", type=float, default=7.0)
    p.add_argument("--exit-score", type=float, default=5.0)
    p.add_argument("--exec-lag-days", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/backtest/crypto_rotation")
    p.add_argument("--out-summary", default="artifacts/backtest/crypto_rotation_summary.csv")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = out_dir / ".yf_tz_cache"

    raw: dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        d = _download_asset(a["ticker"], args.start, args.end, yf_cache_dir)
        if len(d) == 0:
            print(f"Warning: empty download for {a['name']}")
            return 1
        raw[a["name"]] = d

    eth = _build_features(raw["ETH"])
    btc = _build_features(raw["BTC"])

    eth_s = _single_asset_strategy(eth, args.cost_bps, args.exec_lag_days).iloc[144:].copy()
    btc_s = _single_asset_strategy(btc, args.cost_bps, args.exec_lag_days).iloc[144:].copy()
    eth_s = eth_s.dropna(subset=["exec_return"]).copy()
    btc_s = btc_s.dropna(subset=["exec_return"]).copy()

    idx = eth_s.index.intersection(btc_s.index)
    eth_s = eth_s.reindex(idx).fillna(0.0)
    btc_s = btc_s.reindex(idx).fillna(0.0)

    # Strategy A: ETH only (alloc).
    a_ret = float(args.alloc) * eth_s["strategy_return"]
    # Strategy B: BTC only (alloc).
    b_ret = float(args.alloc) * btc_s["strategy_return"]
    # Strategy C: ETH+BTC independently (alloc split equally).
    c_ret = (float(args.alloc) / 2.0) * eth_s["strategy_return"] + (float(args.alloc) / 2.0) * btc_s["strategy_return"]
    # Strategy D: rotation.
    d_df = _crypto_rotation(
        eth=eth.iloc[144:].copy(),
        btc=btc.iloc[144:].copy(),
        alloc_total=float(args.alloc),
        cost_bps=float(args.cost_bps),
        min_hold_days=int(args.min_hold_days),
        switch_gap=float(args.switch_gap),
        enter_score=float(args.enter_score),
        exit_score=float(args.exit_score),
        exec_lag_days=int(args.exec_lag_days),
    )
    d_df = d_df[(d_df["date"] >= idx.min()) & (d_df["date"] <= idx.max())].copy()
    d_df = d_df.set_index("date").reindex(idx).fillna(
        {
            "portfolio_return": 0.0,
            "turnover": 0.0,
            "eth_weight_signal": 0.0,
            "btc_weight_signal": 0.0,
            "eth_weight_exec": 0.0,
            "btc_weight_exec": 0.0,
            "top_asset_exec": "CASH",
        }
    )
    d_ret = d_df["portfolio_return"]

    # Trade counts.
    a_trades = int((eth_s["active"].diff().fillna(eth_s["active"]).abs() > 0).sum())
    b_trades = int((btc_s["active"].diff().fillna(btc_s["active"]).abs() > 0).sum())
    c_trades = a_trades + b_trades
    d_trades = int(d_df.attrs.get("switches", int((d_df["top_asset_exec"] != d_df["top_asset_exec"].shift(1)).sum())))

    rows: list[dict[str, Any]] = []
    for label, r, t in [
        ("A_ETH_only_25pct", a_ret, a_trades),
        ("B_BTC_only_25pct", b_ret, b_trades),
        ("C_independent_12p5_each", c_ret, c_trades),
        ("D_rotate_eth_btc_25pct", d_ret, d_trades),
    ]:
        m = _metrics(r)
        rows.append({"strategy": label, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"], "ann_vol": m["ann_vol"], "trades": int(t)})

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    daily = pd.DataFrame(
        {
            "date": idx,
            "A_ETH_only_25pct": a_ret.values,
            "B_BTC_only_25pct": b_ret.values,
            "C_independent_12p5_each": c_ret.values,
            "D_rotate_eth_btc_25pct": d_ret.values,
        }
    )
    daily.to_csv(out_dir / "crypto_rotation_daily_returns.csv", index=False)
    d_df.reset_index().to_csv(out_dir / "crypto_rotation_state.csv", index=False)

    print("===================================================")
    print("ETH vs BTC ROTATION TEST (2019-2024)")
    print("===================================================")
    print("Strategy                         CAGR    Sharpe  MaxDD   Trades")
    print("---------------------------------------------------")
    for r in summary.itertuples(index=False):
        print(f"{r.strategy:<30} {r.cagr*100:>6.1f}%  {r.sharpe:>6.3f}  {r.max_dd*100:>6.1f}%  {int(r.trades):>6}")
    print("===================================================")
    print(f"Saved summary: {args.out_summary}")
    print(f"Saved detail:  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
