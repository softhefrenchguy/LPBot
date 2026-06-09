from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    {"name": "Oil/WTI", "ticker": "CL=F"},
    {"name": "Nat Gas", "ticker": "NG=F"},
    {"name": "Wheat", "ticker": "ZW=F"},
]


@dataclass
class PortfolioMetrics:
    cagr: float
    sharpe: float
    max_dd: float
    ann_vol: float
    n_rebalances: int
    time_in_market: float
    buyhold_cagr: float
    excess_cagr: float


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
    return d[["Date", "price"]].copy()


def _max_consecutive_true(mask: pd.Series) -> int:
    if len(mask) == 0:
        return 0
    grp = (~mask).cumsum()
    run = mask.groupby(grp).cumcount() + 1
    run = run.where(mask, 0)
    return int(run.max()) if len(run) else 0


def _prep_prices(d: pd.DataFrame, name: str) -> pd.DataFrame:
    raw_nan = d["price"].isna()
    d["price"] = d["price"].ffill(limit=3)
    still_nan = d["price"].isna()
    long_gap_days = _max_consecutive_true(still_nan)
    if long_gap_days > 0:
        print(
            f"Warning: Data quality issue for {name}: gaps >3 days detected (max unresolved gap {long_gap_days}d). Dropping unresolved rows."
        )
    d = d.dropna(subset=["price"]).copy()
    if raw_nan.mean() > 0.05:
        print(f"Warning: Data quality for {name} may be poor ({raw_nan.mean()*100:.1f}% raw NaN).")
    return d


def _compute_regime(price: pd.Series) -> pd.Series:
    ret20 = price / price.shift(20) - 1.0
    dd20 = price / price.rolling(20, min_periods=20).max() - 1.0
    regime = pd.Series("CHOP", index=price.index, dtype=object)
    regime[(ret20 > 0.05) & (dd20 > -0.10)] = "BULL"
    regime[dd20 < -0.15] = "BEAR"
    return regime


def _asset_features(px: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": px}).copy()
    df["ret1"] = df["price"].pct_change().fillna(0.0)
    df["ret20"] = df["price"] / df["price"].shift(20) - 1.0
    df["ema21"] = df["price"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["price"].ewm(span=55, adjust=False).mean()
    df["ema144"] = df["price"].ewm(span=144, adjust=False).mean()
    df["stack"] = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema144"])
    grp = (~df["stack"]).cumsum()
    days_aligned = df["stack"].groupby(grp).cumcount() + 1
    df["days_aligned"] = days_aligned.where(df["stack"], 0).astype(int)
    df["regime"] = _compute_regime(df["price"])
    rv = df["ret1"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["vol_scalar"] = vol_scalar.clip(lower=0.25, upper=1.0).fillna(0.25)
    return df


def _build_score_frame(asset_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = sorted(set().union(*[set(f.index) for f in asset_frames.values()]))
    index = pd.DatetimeIndex(idx, tz="UTC")

    ret1 = pd.DataFrame(index=index)
    ret20 = pd.DataFrame(index=index)
    vol = pd.DataFrame(index=index)
    regime = pd.DataFrame(index=index, dtype=object)
    stack = pd.DataFrame(index=index, dtype=float)
    aligned = pd.DataFrame(index=index, dtype=float)
    above144 = pd.DataFrame(index=index, dtype=float)

    for name, f in asset_frames.items():
        ff = f.reindex(index)
        ret1[name] = pd.to_numeric(ff["ret1"], errors="coerce").fillna(0.0)
        ret20[name] = pd.to_numeric(ff["ret20"], errors="coerce")
        vol[name] = pd.to_numeric(ff["vol_scalar"], errors="coerce")
        regime[name] = ff["regime"]
        stack[name] = ff["stack"].astype(float)
        aligned[name] = (ff["days_aligned"] >= 3).astype(float)
        above144[name] = (ff["price"] > ff["ema144"]).astype(float)

    regime_pts = regime.replace({"BULL": 2.0, "CHOP": 1.0, "BEAR": 0.0})
    perf_rank = ret20.rank(axis=1, method="average", pct=True)
    perf_pts = (perf_rank * 3.0).fillna(0.0)

    score = (3.0 * stack.fillna(0.0)) + (2.0 * aligned.fillna(0.0)) + regime_pts.fillna(0.0) + above144.fillna(0.0) + perf_pts
    # Eligibility gate: confirmed trend and not BEAR for that asset.
    eligible = (stack.fillna(0.0) > 0) & (aligned.fillna(0.0) > 0) & (regime != "BEAR")
    score = score.where(eligible, -1.0)
    return score, ret1, vol


def _candidate_weights_for_day(
    score_row: pd.Series, vol_row: pd.Series, top_k: int, min_score: float, max_gross: float
) -> tuple[pd.Series, list[str]]:
    w = pd.Series(0.0, index=score_row.index, dtype=float)
    s = pd.to_numeric(score_row, errors="coerce").dropna()
    if s.empty:
        return w, []
    picks = s[s >= min_score].sort_values(ascending=False).head(top_k)
    if picks.empty:
        return w, []
    v = pd.to_numeric(vol_row.reindex(picks.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.25)
    raw = v.clip(lower=0.25, upper=1.0)
    raw_sum = float(raw.sum())
    if raw_sum <= 0:
        return w, []
    w.loc[picks.index] = max_gross * (raw / raw_sum)
    return w, picks.index.tolist()


def _target_weights(
    score: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int,
    min_score: float,
    max_gross: float,
    min_hold_days: int,
    switch_gap: float,
    exit_score: float,
) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    current = pd.Series(0.0, index=score.columns, dtype=float)
    days_since_switch = 10_000

    for t in score.index:
        srow = score.loc[t]
        vrow = vol.loc[t]
        candidate, cand_assets = _candidate_weights_for_day(srow, vrow, top_k=top_k, min_score=min_score, max_gross=max_gross)
        held_assets = current[current > 0].index.tolist()

        if not held_assets and cand_assets:
            current = candidate.copy()
            days_since_switch = 0
            w.loc[t] = current
            continue

        if held_assets and days_since_switch < int(min_hold_days):
            w.loc[t] = current
            days_since_switch += 1
            continue

        held_scores = pd.to_numeric(srow.reindex(held_assets), errors="coerce") if held_assets else pd.Series(dtype=float)
        cand_scores = pd.to_numeric(srow.reindex(cand_assets), errors="coerce") if cand_assets else pd.Series(dtype=float)

        held_mean = float(held_scores.mean()) if len(held_scores) else -1e9
        cand_mean = float(cand_scores.mean()) if len(cand_scores) else -1e9
        held_min = float(held_scores.min()) if len(held_scores) else -1e9

        switch = False
        if held_assets and held_min < float(exit_score):
            # Current holdings are degrading; rotate to better candidate or cash.
            switch = True
        elif cand_assets and (cand_mean >= held_mean + float(switch_gap)):
            switch = True
        elif (not held_assets) and cand_assets:
            switch = True

        if switch:
            current = candidate.copy()
            days_since_switch = 0
        else:
            days_since_switch += 1

        w.loc[t] = current

    return w


def _portfolio_metrics(port_r: pd.Series, gross: pd.Series, n_rebalances: int, buyhold_r: pd.Series) -> PortfolioMetrics:
    port_r = pd.to_numeric(port_r, errors="coerce").fillna(0.0)
    years = len(port_r) / 252.0 if len(port_r) > 0 else np.nan
    eq = (1.0 + port_r).cumprod()
    total = float(eq.iloc[-1]) if len(eq) else np.nan
    cagr = (total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    daily_excess = port_r - (0.05 / 252.0)
    sd = float(daily_excess.std(ddof=0))
    sharpe = float(np.mean(daily_excess) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else np.nan
    ann_vol = float(port_r.std(ddof=0) * np.sqrt(252.0))
    tim = float((gross > 0).mean()) if len(gross) else np.nan

    buy_eq = (1.0 + pd.to_numeric(buyhold_r, errors="coerce").fillna(0.0)).cumprod()
    buy_total = float(buy_eq.iloc[-1]) if len(buy_eq) else np.nan
    buy_cagr = (buy_total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    excess = cagr - buy_cagr if np.isfinite(cagr) and np.isfinite(buy_cagr) else np.nan

    return PortfolioMetrics(
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        ann_vol=float(ann_vol),
        n_rebalances=int(n_rebalances),
        time_in_market=float(tim),
        buyhold_cagr=float(buy_cagr),
        excess_cagr=float(excess),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Cross-asset ranked rotation backtest (daily points + top-k rotation)")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--min-score", type=float, default=7.0)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--min-hold-days", type=int, default=5)
    p.add_argument("--switch-gap", type=float, default=1.0, help="Switch only if candidate score exceeds held score by this gap.")
    p.add_argument("--exit-score", type=float, default=6.0, help="Force exit/switch if held score drops below this.")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--out-dir", default="artifacts/backtest/multi_asset_rotation")
    p.add_argument("--out-summary", default="artifacts/backtest/multi_asset_rotation_summary.csv")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = out_dir / ".yf_tz_cache"

    asset_prices: dict[str, pd.Series] = {}
    asset_frames: dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        name = a["name"]
        ticker = a["ticker"]
        try:
            raw = _download_asset(ticker, args.start, args.end, yf_cache_dir)
        except Exception as exc:
            print(f"Warning: Could not download {name}: {exc}")
            continue
        if len(raw) == 0:
            print(f"Warning: Could not download {name}: empty dataset")
            continue
        prepped = _prep_prices(raw, name=name)
        if len(prepped) < 200:
            print(f"Warning: Insufficient data for {name}")
            continue
        px = prepped.set_index("Date")["price"].astype(float).sort_index()
        asset_prices[name] = px
        asset_frames[name] = _asset_features(px)

    if len(asset_frames) < 2:
        print("Need at least 2 assets with valid data.")
        return 0

    score, ret1, vol = _build_score_frame(asset_frames)
    # Warmup: EMA144 + 20d vol/ret lookback.
    score = score.iloc[164:].copy()
    ret1 = ret1.reindex(score.index).fillna(0.0)
    vol = vol.reindex(score.index).fillna(0.25)

    w_target = _target_weights(
        score,
        vol,
        top_k=int(args.top_k),
        min_score=float(args.min_score),
        max_gross=float(args.max_gross),
        min_hold_days=int(args.min_hold_days),
        switch_gap=float(args.switch_gap),
        exit_score=float(args.exit_score),
    )
    w_exec = w_target.shift(1).fillna(0.0)
    turnover = (w_exec - w_exec.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (float(args.cost_bps) / 10000.0)

    port_gross = (w_exec * ret1).sum(axis=1)
    port_ret = port_gross - cost
    eq = (1.0 + port_ret).cumprod()

    # Buy-hold comparator: equal weight across assets with full sample from same dates.
    bh_r = ret1.mean(axis=1)

    # Daily top picks text.
    def _top_label(row: pd.Series) -> str:
        s = row.sort_values(ascending=False)
        s = s[s >= float(args.min_score)].head(int(args.top_k))
        return ",".join(s.index.tolist()) if len(s) else "CASH"

    top_assets = score.apply(_top_label, axis=1)
    gross_exposure = w_exec.sum(axis=1)

    daily = pd.DataFrame(
        {
            "date": score.index,
            "portfolio_return": port_ret.values,
            "portfolio_gross_return": port_gross.values,
            "turnover": turnover.values,
            "cost": cost.values,
            "gross_exposure": gross_exposure.values,
            "top_assets": top_assets.values,
            "equity": eq.values,
            "buyhold_eq": (1.0 + bh_r).cumprod().values,
        }
    )
    for c in w_exec.columns:
        daily[f"w_{_sanitize_name(c)}"] = w_exec[c].values
    daily_path = out_dir / "ranked_rotation_daily.csv"
    daily.to_csv(daily_path, index=False)

    score_out = score.copy()
    score_out.insert(0, "date", score.index)
    score_path = out_dir / "ranked_rotation_scores.csv"
    score_out.to_csv(score_path, index=False)

    metrics = _portfolio_metrics(port_r=port_ret, gross=gross_exposure, n_rebalances=int((turnover > 0).sum()), buyhold_r=bh_r)
    summary = pd.DataFrame(
        [
            {
                "start": str(score.index.min().date()),
                "end": str(score.index.max().date()),
                "assets": ",".join(score.columns.tolist()),
                "top_k": int(args.top_k),
                "min_score": float(args.min_score),
                "max_gross": float(args.max_gross),
                "min_hold_days": int(args.min_hold_days),
                "switch_gap": float(args.switch_gap),
                "exit_score": float(args.exit_score),
                "cost_bps": float(args.cost_bps),
                "cagr": metrics.cagr,
                "sharpe": metrics.sharpe,
                "max_dd": metrics.max_dd,
                "ann_vol": metrics.ann_vol,
                "n_rebalances": metrics.n_rebalances,
                "time_in_market": metrics.time_in_market,
                "buyhold_cagr_equal_weight": metrics.buyhold_cagr,
                "excess_cagr_vs_buyhold": metrics.excess_cagr,
            }
        ]
    )
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    # Simple per-asset average rank report.
    rank_df = score.rank(axis=1, ascending=False, method="average")
    avg_rank = rank_df.replace([np.inf, -np.inf], np.nan).mean(axis=0).sort_values()
    rank_report = pd.DataFrame({"asset": avg_rank.index, "avg_rank": avg_rank.values})
    rank_report_path = out_dir / "ranked_rotation_avg_rank.csv"
    rank_report.to_csv(rank_report_path, index=False)

    print("===================================================")
    print("CROSS-ASSET RANKED ROTATION (2019-2024)")
    print("Daily points ranking | top-k rotation | 10 bps turnover cost")
    print("===================================================")
    print(f"Assets: {', '.join(score.columns.tolist())}")
    print(f"Window: {score.index.min().date()} -> {score.index.max().date()}")
    print(
        f"Top-k: {args.top_k} | Min score: {args.min_score} | Max gross: {args.max_gross:.2f} | "
        f"Min hold: {args.min_hold_days}d | Switch gap: {args.switch_gap:.2f}"
    )
    print("---------------------------------------------------")
    print(f"CAGR:          {metrics.cagr*100:.2f}%")
    print(f"Sharpe:        {metrics.sharpe:.3f}")
    print(f"MaxDD:         {metrics.max_dd*100:.2f}%")
    print(f"Ann Vol:       {metrics.ann_vol*100:.2f}%")
    print(f"Rebalances:    {metrics.n_rebalances}")
    print(f"Time-in-market:{metrics.time_in_market*100:.2f}%")
    print(f"Buy&Hold CAGR: {metrics.buyhold_cagr*100:.2f}%")
    print(f"Excess CAGR:   {metrics.excess_cagr*100:+.2f}%")
    print("---------------------------------------------------")
    print("Average rank (lower is better):")
    for row in rank_report.itertuples(index=False):
        print(f"  {row.asset:<8} {row.avg_rank:>5.2f}")
    print("===================================================")
    print(f"Saved daily:    {daily_path}")
    print(f"Saved scores:   {score_path}")
    print(f"Saved summary:  {out_summary}")
    print(f"Saved ranks:    {rank_report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
