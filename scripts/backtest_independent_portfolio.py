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
    {"name": "ETH", "ticker": "ETH-USD", "alloc": 0.25},
    {"name": "BTC", "ticker": "BTC-USD", "alloc": 0.25},
    {"name": "Oil/WTI", "ticker": "CL=F", "alloc": 0.25},
    {"name": "Nat Gas", "ticker": "NG=F", "alloc": 0.25},
]

ASSET_PROFILES: dict[str, list[dict[str, Any]]] = {
    "futures": ASSETS,
    "eth_only": [
        {"name": "ETH", "ticker": "ETH-USD", "alloc": 1.0},
    ],
    "etf_proxy": [
        {"name": "ETH", "ticker": "ETH-USD", "alloc": 0.25},
        {"name": "BTC", "ticker": "BTC-USD", "alloc": 0.25},
        {"name": "Oil/USO", "ticker": "USO", "alloc": 0.25},
        {"name": "NatGas/UNG", "ticker": "UNG", "alloc": 0.25},
    ],
}


@dataclass
class SleeveResult:
    name: str
    alloc: float
    cagr: float
    sharpe: float
    max_dd: float
    ann_vol: float
    n_trades: int
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
    if "Open" in d.columns:
        d["open"] = pd.to_numeric(d["Open"], errors="coerce")
    else:
        d["open"] = pd.to_numeric(d["price"], errors="coerce")
    return d[["Date", "price", "open"]].copy()


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
    d["open"] = pd.to_numeric(d["open"], errors="coerce").ffill(limit=3)
    still_nan = d["price"].isna()
    long_gap_days = _max_consecutive_true(still_nan)
    if long_gap_days > 0:
        print(
            f"Warning: Data quality issue for {name}: gaps >3 days detected (max unresolved gap {long_gap_days}d). Dropping unresolved rows."
        )
    d = d.dropna(subset=["price", "open"]).copy()
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


def _perf_stats(strategy_r: pd.Series, buyhold_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(strategy_r, errors="coerce").fillna(0.0)
    y = pd.to_numeric(buyhold_r, errors="coerce").fillna(0.0)
    years = len(x) / 252.0 if len(x) else np.nan

    eq = (1.0 + x).cumprod()
    total = float(eq.iloc[-1]) if len(eq) else np.nan
    cagr = (total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else np.nan

    excess = x - (0.05 / 252.0)
    sd = float(excess.std(ddof=0))
    sharpe = float(np.mean(excess) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))

    eq_bh = (1.0 + y).cumprod()
    total_bh = float(eq_bh.iloc[-1]) if len(eq_bh) else np.nan
    buyhold_cagr = (total_bh ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    excess_cagr = cagr - buyhold_cagr if np.isfinite(cagr) and np.isfinite(buyhold_cagr) else np.nan

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "ann_vol": float(ann_vol),
        "buyhold_cagr": float(buyhold_cagr),
        "excess_cagr": float(excess_cagr),
    }


def _run_sleeve(
    name: str,
    alloc: float,
    raw: pd.DataFrame,
    out_dir: Path,
    cost_bps: float,
    exec_lag_days: int,
) -> tuple[SleeveResult | None, pd.DataFrame | None]:
    d = _prep_prices(raw, name=name)
    if len(d) < 200:
        print(f"Warning: Insufficient data for {name}")
        return None, None

    df = d.copy().reset_index(drop=True)
    df["daily_return"] = pd.to_numeric(df["price"], errors="coerce").pct_change().fillna(0.0)
    # Realistic trade return for close-signal systems: enter at next open, hold open-to-open.
    df["exec_return"] = pd.to_numeric(df["open"], errors="coerce").shift(-1) / pd.to_numeric(df["open"], errors="coerce") - 1.0
    df["ema21"] = df["price"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["price"].ewm(span=55, adjust=False).mean()
    df["ema144"] = df["price"].ewm(span=144, adjust=False).mean()

    df["stack_aligned"] = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema144"])
    grp = (~df["stack_aligned"]).cumsum()
    days_aligned = df["stack_aligned"].groupby(grp).cumcount() + 1
    df["days_aligned"] = days_aligned.where(df["stack_aligned"], 0).astype(int)

    entry_confirm = (df["stack_aligned"].rolling(3, min_periods=3).min() == 1).fillna(False)
    exit_confirm = ((~df["stack_aligned"]).rolling(3, min_periods=3).min() == 1).fillna(False)

    df["regime"] = _compute_regime(df["price"])
    off_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    rv = df["daily_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["vol_scalar"] = vol_scalar.clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(df), dtype=int)
    active = 0
    for i in range(len(df)):
        if active == 0 and bool(entry_confirm.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_confirm.iloc[i]):
            active = 0
        pos[i] = active
    df["position_active"] = pos

    df["raw_signal"] = np.where(df["position_active"] == 1, df["vol_scalar"], 0.0)
    df["off_weight"] = df["regime"].map(off_map).fillna(0.0)
    # Signal-time desired weight (cannot be traded on same close).
    df["scaled_weight_signal"] = df["raw_signal"] * df["off_weight"]
    # Executed weight: signal from day N applied from day N+lag.
    df["scaled_weight_exec"] = df["scaled_weight_signal"].shift(int(exec_lag_days)).fillna(0.0)

    prev_w = df["scaled_weight_exec"].shift(1).fillna(0.0)
    delta_w = (df["scaled_weight_exec"] - prev_w).abs()
    cost_ret = (float(cost_bps) / 10000.0) * delta_w
    df["strategy_return"] = df["scaled_weight_exec"] * df["exec_return"].fillna(0.0) - cost_ret

    # Sleeve contribution on total portfolio capital.
    df["portfolio_contrib_return"] = float(alloc) * df["strategy_return"]
    df["portfolio_gross_exposure"] = float(alloc) * df["scaled_weight_exec"]

    # Warmup cutoff: EMA144 + lookbacks.
    df = df.iloc[144:].copy()
    df = df.dropna(subset=["exec_return"]).copy()
    if len(df) < 200:
        print(f"Warning: Insufficient post-warmup data for {name}")
        return None, None

    df["cumulative_strategy"] = (1.0 + df["strategy_return"]).cumprod()
    df["cumulative_spot"] = (1.0 + df["exec_return"]).cumprod()

    stats = _perf_stats(df["strategy_return"], df["exec_return"])
    prev_pos = pd.Series((df["scaled_weight_exec"] > 0).astype(int)).shift(1).fillna(0).astype(int)
    curr_pos = pd.Series((df["scaled_weight_exec"] > 0).astype(int)
                        ).astype(int)
    n_trades = int((prev_pos != curr_pos).sum())
    time_in_market = float((df["scaled_weight_exec"] > 0).mean())

    out = df[
        [
            "Date",
            "price",
            "ema21",
            "ema55",
            "ema144",
            "stack_aligned",
            "days_aligned",
            "regime",
            "vol_scalar",
            "scaled_weight_signal",
            "scaled_weight_exec",
            "daily_return",
            "exec_return",
            "strategy_return",
            "portfolio_contrib_return",
            "portfolio_gross_exposure",
            "cumulative_strategy",
            "cumulative_spot",
        ]
    ].copy()
    out = out.rename(columns={"Date": "date"})
    out_path = out_dir / f"{_sanitize_name(name)}_sleeve.csv"
    out.to_csv(out_path, index=False)

    res = SleeveResult(
        name=name,
        alloc=float(alloc),
        cagr=stats["cagr"],
        sharpe=stats["sharpe"],
        max_dd=stats["max_dd"],
        ann_vol=stats["ann_vol"],
        n_trades=n_trades,
        time_in_market=time_in_market,
        buyhold_cagr=stats["buyhold_cagr"],
        excess_cagr=stats["excess_cagr"],
    )
    return res, out


def _print_summary(port_summary: pd.Series, sleeves: pd.DataFrame) -> None:
    print("===================================================")
    print("INDEPENDENT MULTI-ASSET PORTFOLIO BACKTEST (2019-2024)")
    print("4 sleeves in parallel: ETH, BTC, Oil/WTI, Nat Gas (25% each)")
    print("===================================================")
    print(f"Combined CAGR:            {port_summary['cagr']*100:.2f}%")
    print(f"Combined Sharpe:          {port_summary['sharpe']:.3f}")
    print(f"Combined MaxDD:           {port_summary['max_dd']*100:.2f}%")
    print(f"Combined Ann Vol:         {port_summary['ann_vol']*100:.2f}%")
    print(f"Combined Time-in-market:  {port_summary['time_in_market']*100:.2f}%")
    print(f"Combined Buy&Hold CAGR:   {port_summary['buyhold_cagr']*100:.2f}%")
    print(f"Combined Excess CAGR:     {port_summary['excess_cagr']*100:+.2f}%")
    print("---------------------------------------------------")
    print("Sleeve Metrics")
    print("Asset      Alloc  CAGR    Sharpe  MaxDD   Trades  TiM")
    for r in sleeves.itertuples(index=False):
        print(
            f"{r.asset:<10} {r.alloc*100:>5.0f}%  {r.cagr*100:>6.1f}%  {r.sharpe:>6.3f}  {r.max_dd*100:>6.1f}%  {int(r.n_trades):>6}  {r.time_in_market*100:>5.1f}%"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest independent multi-sleeve portfolio")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out-dir", default="artifacts/backtest/independent_portfolio")
    p.add_argument("--summary-csv", default="artifacts/backtest/independent_portfolio_summary.csv")
    p.add_argument("--correlation-csv", default="artifacts/backtest/independent_portfolio_corr.csv")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--exec-lag-days", type=int, default=1)
    p.add_argument("--asset-profile", choices=["futures", "eth_only", "etf_proxy"], default="futures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = out_dir / ".yf_tz_cache"
    active_assets = ASSET_PROFILES[str(args.asset_profile)]

    sleeve_rows: list[dict[str, Any]] = []
    sleeve_daily: dict[str, pd.DataFrame] = {}

    for a in active_assets:
        name = a["name"]
        ticker = a["ticker"]
        alloc = float(a["alloc"])
        try:
            raw = _download_asset(ticker=ticker, start=args.start, end=args.end, yf_cache_dir=yf_cache_dir)
        except Exception as exc:
            print(f"Warning: Could not download {name}: {exc}")
            continue
        if len(raw) == 0:
            print(f"Warning: Could not download {name}: empty dataset")
            continue
        res, daily = _run_sleeve(
            name=name,
            alloc=alloc,
            raw=raw,
            out_dir=out_dir,
            cost_bps=float(args.cost_bps),
            exec_lag_days=int(args.exec_lag_days),
        )
        if res is None or daily is None:
            continue
        sleeve_rows.append(
            {
                "asset": res.name,
                "alloc": res.alloc,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "max_dd": res.max_dd,
                "ann_vol": res.ann_vol,
                "n_trades": res.n_trades,
                "time_in_market": res.time_in_market,
                "buyhold_cagr": res.buyhold_cagr,
                "excess_cagr": res.excess_cagr,
            }
        )
        sleeve_daily[name] = daily

    if len(sleeve_rows) < 1:
        print("Not enough sleeves produced valid results.")
        return 0

    sleeves = pd.DataFrame(sleeve_rows)
    sleeves = sleeves.sort_values("sharpe", ascending=False).reset_index(drop=True)

    merged: pd.DataFrame | None = None
    for name, d in sleeve_daily.items():
        x = d[["date", "portfolio_contrib_return", "portfolio_gross_exposure", "strategy_return", "exec_return"]].copy()
        x = x.rename(
            columns={
                "portfolio_contrib_return": f"{_sanitize_name(name)}_contrib",
                "portfolio_gross_exposure": f"{_sanitize_name(name)}_gross",
                "strategy_return": f"{_sanitize_name(name)}_strat_r",
                "exec_return": f"{_sanitize_name(name)}_spot_r",
            }
        )
        x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
        merged = x if merged is None else merged.merge(x, on="date", how="outer")

    assert merged is not None
    merged = merged.sort_values("date")
    for c in merged.columns:
        if c != "date":
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    contrib_cols = [c for c in merged.columns if c.endswith("_contrib")]
    gross_cols = [c for c in merged.columns if c.endswith("_gross")]
    strat_cols = [c for c in merged.columns if c.endswith("_strat_r")]
    spot_cols = [c for c in merged.columns if c.endswith("_spot_r")]

    merged["portfolio_return"] = merged[contrib_cols].sum(axis=1)
    merged["portfolio_gross_exposure"] = merged[gross_cols].sum(axis=1)
    merged["portfolio_eq"] = (1.0 + merged["portfolio_return"]).cumprod()

    # Comparator: equal-weight spot basket of same 4 assets (25% each).
    merged["equal_weight_spot_return"] = merged[spot_cols].mean(axis=1)
    merged["equal_weight_spot_eq"] = (1.0 + merged["equal_weight_spot_return"]).cumprod()

    # Portfolio-level stats.
    port_stats = _perf_stats(merged["portfolio_return"], merged["equal_weight_spot_return"])
    port_summary = pd.Series(
        {
            "start": str(pd.to_datetime(merged["date"].min()).date()),
            "end": str(pd.to_datetime(merged["date"].max()).date()),
            "assets": ",".join(sleeves["asset"].tolist()),
            "asset_profile": str(args.asset_profile),
            "alloc_rule": "equal_25_each",
            "cost_bps": float(args.cost_bps),
            "exec_lag_days": int(args.exec_lag_days),
            "cagr": port_stats["cagr"],
            "sharpe": port_stats["sharpe"],
            "max_dd": port_stats["max_dd"],
            "ann_vol": port_stats["ann_vol"],
            "time_in_market": float((merged["portfolio_gross_exposure"] > 0).mean()),
            "buyhold_cagr": port_stats["buyhold_cagr"],
            "excess_cagr": port_stats["excess_cagr"],
        }
    )

    # Contribution summary.
    total_contrib = merged[contrib_cols].sum()
    total_contrib_pct = total_contrib / (total_contrib.sum() if abs(total_contrib.sum()) > 1e-12 else 1.0)
    contrib_summary = pd.DataFrame(
        {
            "asset": [c.replace("_contrib", "") for c in contrib_cols],
            "total_contrib_return_points": total_contrib.values,
            "contrib_share_of_total": total_contrib_pct.values,
        }
    )

    # Correlation of sleeve strategy daily returns.
    corr = merged[strat_cols].corr()
    corr.index = [c.replace("_strat_r", "") for c in corr.index]
    corr.columns = [c.replace("_strat_r", "") for c in corr.columns]
    corr.to_csv(args.correlation_csv, index=True)

    merged_path = out_dir / "independent_portfolio_daily.csv"
    merged.to_csv(merged_path, index=False)
    sleeves_path = out_dir / "independent_sleeves_summary.csv"
    sleeves.to_csv(sleeves_path, index=False)
    contrib_path = out_dir / "independent_portfolio_contribution.csv"
    contrib_summary.to_csv(contrib_path, index=False)

    summary_out = pd.DataFrame([port_summary.to_dict()])
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(args.summary_csv, index=False)

    _print_summary(port_summary, sleeves)
    print("---------------------------------------------------")
    print("Contribution (share of total portfolio return points)")
    for r in contrib_summary.itertuples(index=False):
        print(
            f"{r.asset:<10} {r.total_contrib_return_points:>8.4f}   {r.contrib_share_of_total*100:>6.2f}%"
        )
    print("---------------------------------------------------")
    print(f"Saved summary:      {args.summary_csv}")
    print(f"Saved sleeve table: {sleeves_path}")
    print(f"Saved daily:        {merged_path}")
    print(f"Saved contrib:      {contrib_path}")
    print(f"Saved correlation:  {args.correlation_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
