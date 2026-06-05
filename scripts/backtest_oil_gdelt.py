from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


KNOWN_EVENTS: list[tuple[str, str]] = [
    ("2020-03-09", "Saudi-Russia oil price war"),
    ("2022-02-24", "Ukraine invasion"),
    ("2022-10-05", "OPEC+ production cuts"),
    ("2023-10-07", "Hamas attack"),
    ("2024-04-15", "Red Sea shipping attacks"),
]


def _perf(simple_returns: pd.Series) -> tuple[float, float, float, float, pd.Series]:
    r = pd.to_numeric(simple_returns, errors="coerce").fillna(0.0)
    if len(r) == 0:
        return np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float)
    years = len(r) / 252.0
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1])
    cagr = (total ** (1.0 / years) - 1.0) if years > 0 else np.nan
    excess = r - (0.05 / 252.0)
    sd = float(excess.std(ddof=0))
    sharpe = float(excess.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    peak = equity.cummax()
    maxdd = float(((equity - peak) / peak).min())
    ann_vol = float(r.std(ddof=0) * np.sqrt(252.0))
    return cagr, sharpe, maxdd, ann_vol, equity


def _read_gdelt_events(files: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in files:
        try:
            d = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"warning: could not read {path}: {e}")
            continue
        if d.empty:
            continue

        lower = {c.lower(): c for c in d.columns}
        date_col = next((lower[k] for k in ["sqldate", "date"] if k in lower), None)
        g_col = next((lower[k] for k in ["goldsteinscale", "goldstein"] if k in lower), None)
        t_col = next((lower[k] for k in ["avgtone", "avg_tone"] if k in lower), None)
        m_col = next((lower[k] for k in ["nummentions", "num_mentions"] if k in lower), None)
        a_col = next((lower[k] for k in ["numarticles", "num_articles"] if k in lower), None)
        if not (date_col and g_col and t_col and m_col and a_col):
            print(f"warning: skipping {path} missing required columns")
            continue

        x = d[[date_col, g_col, t_col, m_col, a_col]].copy()
        x.columns = ["date_raw", "goldstein", "avg_tone", "num_mentions", "num_articles"]

        # Support both YYYYMMDD and ISO-like date strings.
        dr = x["date_raw"].astype(str).str.strip()
        is_yyyymmdd = dr.str.fullmatch(r"\d{8}", na=False)
        dt1 = pd.to_datetime(dr.where(is_yyyymmdd), format="%Y%m%d", errors="coerce")
        dt2 = pd.to_datetime(dr.where(~is_yyyymmdd), errors="coerce", utc=True)
        x["date"] = dt1.fillna(dt2.dt.tz_localize(None))

        x["goldstein"] = pd.to_numeric(x["goldstein"], errors="coerce")
        x["avg_tone"] = pd.to_numeric(x["avg_tone"], errors="coerce")
        x["num_mentions"] = pd.to_numeric(x["num_mentions"], errors="coerce")
        x["num_articles"] = pd.to_numeric(x["num_articles"], errors="coerce")
        x = x.dropna(subset=["date", "goldstein", "avg_tone", "num_mentions", "num_articles"])
        if x.empty:
            continue
        rows.append(x[["date", "goldstein", "avg_tone", "num_mentions", "num_articles"]])

    if not rows:
        return pd.DataFrame(columns=["date", "goldstein", "avg_tone", "num_mentions", "num_articles"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out = out.dropna(subset=["date"])
    return out


def _build_daily_score(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    days = pd.date_range(start=start, end=end, freq="D")
    if events.empty:
        daily = pd.DataFrame({"date": days, "raw_score": 0.0})
    else:
        neg_tone = np.maximum(0.0, -events["avg_tone"].to_numpy())
        # Weighted by negative tone as requested.
        tone_weight = 1.0 + neg_tone
        component = np.abs(events["goldstein"].to_numpy()) * events["num_mentions"].to_numpy() * tone_weight
        events = events.copy()
        events["component"] = component
        g = events.groupby("date", as_index=False)["component"].sum()
        g.columns = ["date", "raw_score"]
        daily = pd.DataFrame({"date": days}).merge(g, on="date", how="left")
        daily["raw_score"] = pd.to_numeric(daily["raw_score"], errors="coerce").fillna(0.0)

    ma30 = daily["raw_score"].rolling(30, min_periods=10).mean()
    sd30 = daily["raw_score"].rolling(30, min_periods=10).std(ddof=0)
    z = (daily["raw_score"] - ma30) / sd30.replace(0.0, np.nan)
    daily["oil_news_zscore"] = z.fillna(0.0)

    def _label(v: float) -> str:
        if v > 2.0:
            return "HIGH_BULLISH"
        if v > 1.0:
            return "MEDIUM_BULLISH"
        if v < -1.0:
            return "MILD_BEARISH"
        return "NEUTRAL"

    daily["oil_news_signal"] = daily["oil_news_zscore"].map(_label)
    return daily


def _load_oil_prices(start: str, end: str) -> tuple[pd.DataFrame, str]:
    for ticker in ("USO", "XLE"):
        x = yf.download(ticker, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d", auto_adjust=True, progress=False)
        if x is None or x.empty:
            continue
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x = x.reset_index()
        date_col = "Date" if "Date" in x.columns else "index"
        x["date"] = pd.to_datetime(x[date_col], errors="coerce").dt.floor("D")
        x["open"] = pd.to_numeric(x["Open"], errors="coerce")
        x["close"] = pd.to_numeric(x["Close"], errors="coerce")
        x = x.dropna(subset=["date", "open", "close"]).sort_values("date")
        return x[["date", "open", "close"]].copy(), ticker
    raise RuntimeError("USO data unavailable and XLE fallback also failed.")


def _backtest(d: pd.DataFrame, strategy: str, cost_bps: float) -> dict[str, float]:
    x = d.copy()
    x["ret_close"] = x["close"].pct_change().fillna(0.0)
    x["ret_exec"] = x["open"].pct_change().fillna(0.0)
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema55"] = x["close"].ewm(span=55, adjust=False).mean()
    x["ema144"] = x["close"].ewm(span=144, adjust=False).mean()
    x["ema_aligned"] = (x["ema21"] > x["ema55"]) & (x["ema55"] > x["ema144"])
    x["ema_entry3"] = (x["ema_aligned"].rolling(3, min_periods=3).min() == 1).fillna(False)
    x["ema_break3"] = ((~x["ema_aligned"]).rolling(3, min_periods=3).min() == 1).fillna(False)
    rv = x["ret_close"].rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    x["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(x), dtype=int)
    active = 0
    hold_days = 0
    for i in range(len(x)):
        z = float(x["oil_news_zscore"].iloc[i])
        entry_ema = bool(x["ema_entry3"].iloc[i])
        exit_ema = bool(x["ema_break3"].iloc[i])

        if strategy == "A_EMA_ONLY":
            enter = entry_ema
            exit_now = exit_ema
        elif strategy == "B_NEWS_ONLY":
            enter = z > 2.0
            exit_now = (z < 0.5) or (hold_days >= 20)
        elif strategy == "C_NEWS_ENTRY_EMA_EXIT":
            enter = z > 2.0
            exit_now = exit_ema or (z < 0.5)
        elif strategy == "D_EMA_ENTRY_NEWS_FILTER":
            enter = entry_ema and (z > 0.5)
            exit_now = exit_ema
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        if active == 0:
            if enter:
                active = 1
                hold_days = 0
        else:
            hold_days += 1
            if exit_now:
                active = 0
                hold_days = 0
        pos[i] = active

    x["position"] = pos.astype(float)
    x["weight_target"] = x["position"] * x["vol_scalar"]
    x["weight_exec"] = x["weight_target"].shift(1).fillna(0.0)
    x["turnover"] = (x["weight_exec"] - x["weight_exec"].shift(1).fillna(0.0)).abs()
    x["cost"] = x["turnover"] * (cost_bps / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["ret_exec"] - x["cost"]

    cagr, sharpe, maxdd, ann_vol, _eq = _perf(x["strategy_return"])
    trades = int(((x["weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum() // 2)
    tim = float((x["weight_exec"] > 0).mean())
    return {
        "strategy": strategy,
        "cagr": cagr,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "trades": trades,
        "time_in_market": tim,
        "ann_vol": ann_vol,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest Oil strategy variants using GDELT daily news score.")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--gdelt-dir", default="data")
    p.add_argument("--score-out", default="data/oil_gdelt_daily_score.csv")
    p.add_argument("--summary-out", default="artifacts/backtest/oil_gdelt_summary.csv")
    p.add_argument("--events-out", default="artifacts/backtest/oil_gdelt_events.csv")
    args = p.parse_args()

    gdelt_dir = Path(args.gdelt_dir)
    files = sorted(gdelt_dir.glob("gdelt_oil_events_*.csv"))
    if not files:
        mono = gdelt_dir / "gdelt_oil_events.csv"
        if mono.exists():
            files = [mono]
    if not files:
        print(f"GDELT data not found at {gdelt_dir}")
        print("Expected files: data/gdelt_oil_events_*.csv")
        return 2

    events = _read_gdelt_events(files)
    if len(events) < 100:
        print("warning: Sparse GDELT data - results may be unreliable")

    daily_score = _build_daily_score(events, args.start, args.end)
    Path(args.score_out).parent.mkdir(parents=True, exist_ok=True)
    daily_score[["date", "raw_score", "oil_news_zscore", "oil_news_signal"]].to_csv(args.score_out, index=False)

    oil_px, proxy = _load_oil_prices(args.start, args.end)
    merged = oil_px.merge(daily_score[["date", "oil_news_zscore", "oil_news_signal"]], on="date", how="left")
    merged["oil_news_zscore"] = pd.to_numeric(merged["oil_news_zscore"], errors="coerce").fillna(0.0)
    merged["oil_news_signal"] = merged["oil_news_signal"].fillna("NEUTRAL")

    results = []
    for s in ["A_EMA_ONLY", "B_NEWS_ONLY", "C_NEWS_ENTRY_EMA_EXIT", "D_EMA_ENTRY_NEWS_FILTER"]:
        results.append(_backtest(merged, strategy=s, cost_bps=float(args.cost_bps)))
    res_df = pd.DataFrame(results)

    # Buy/hold comparator.
    bh_ret = merged["open"].pct_change().fillna(0.0)
    bh_cagr, bh_sharpe, bh_maxdd, _bh_vol, _ = _perf(bh_ret)

    # Event table.
    merged = merged.copy()
    merged["next_week_return"] = merged["close"].shift(-5) / merged["close"] - 1.0
    ev_rows = []
    for d_s, label in KNOWN_EVENTS:
        d0 = pd.Timestamp(d_s)
        # Use nearest trading day on/after event date.
        chunk = merged.loc[merged["date"] >= d0].head(1)
        if len(chunk) == 0:
            continue
        r = chunk.iloc[0]
        z = float(r["oil_news_zscore"])
        sig = str(r["oil_news_signal"])
        ev_rows.append(
            {
                "date": d_s,
                "zscore": z,
                "signal": sig,
                "event_detected": bool(z > 1.0),
                "oil_next_week_return": float(r["next_week_return"]) if np.isfinite(r["next_week_return"]) else np.nan,
                "event_name": label,
            }
        )
    ev_df = pd.DataFrame(ev_rows)
    Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(args.events_out, index=False)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    res_df[["strategy", "cagr", "sharpe", "maxdd", "trades", "time_in_market"]].to_csv(args.summary_out, index=False)

    # Verdict.
    best = res_df.sort_values("sharpe", ascending=False).iloc[0]
    best_name = str(best["strategy"])
    best_sh = float(best["sharpe"])
    best_dd = float(best["maxdd"])
    best_trades = int(best["trades"])
    if best_sh >= 0.7 and best_dd > -0.40 and best_trades >= 10:
        grade = "PASS"
    elif best_sh >= 0.5 and best_dd > -0.40 and best_trades >= 10:
        grade = "MARGINAL"
    else:
        grade = "FAIL"

    ema_sh = float(res_df.loc[res_df["strategy"] == "A_EMA_ONLY", "sharpe"].iloc[0])
    delta = best_sh - ema_sh
    news_adds = "YES" if delta > 0 else "NO"
    oil_viable = "YES" if grade == "PASS" else "NO"

    merged["next_month_return"] = merged["close"].shift(-21) / merged["close"] - 1.0
    corr_week = float(merged[["oil_news_zscore", "next_week_return"]].corr().iloc[0, 1])
    corr_month = float(merged[["oil_news_zscore", "next_month_return"]].corr().iloc[0, 1])

    print("===============================================")
    print("OIL GDELT NEWS BACKTEST (2019-2024)")
    print(f"Lag-1 realistic execution, 10bps cost | proxy={proxy}")
    print("===============================================")
    print("Strategy                  CAGR    Sharpe   MaxDD    Trades")
    for _, r in res_df.iterrows():
        print(
            f"{str(r['strategy']):24s} "
            f"{float(r['cagr'])*100:6.2f}%  "
            f"{float(r['sharpe']):6.3f}  "
            f"{float(r['maxdd'])*100:7.2f}%  "
            f"{int(r['trades']):6d}"
        )
    print("-----------------------------------------------")
    print(f"BUY_HOLD_{proxy:>8s}         {bh_cagr*100:6.2f}%  {bh_sharpe:6.3f}  {bh_maxdd*100:7.2f}%      --")
    print("===============================================")
    print(f"Best strategy: {best_name}")
    print(f"Sharpe improvement vs EMA alone: {delta:+.3f}")
    print(f"GDELT news adds value: {news_adds}")
    print(f"Oil viable with news signal: {oil_viable}")
    print(f"Recommended approach: {best_name}")
    print(f"Grade: {grade}")
    print("-----------------------------------------------")
    print("Known events")
    print("Date       Event                          Zscore   Signal          Oil move")
    for _, r in ev_df.iterrows():
        mv = r["oil_next_week_return"]
        mv_s = "n/a" if not np.isfinite(mv) else f"{float(mv)*100:6.2f}%"
        print(f"{r['date']} {str(r['event_name'])[:30]:30s} {float(r['zscore']):6.2f}  {str(r['signal'])[:14]:14s}  {mv_s}")
    print("-----------------------------------------------")
    print(f"Correlation (zscore vs next-week):  {corr_week:.3f}")
    print(f"Correlation (zscore vs next-month): {corr_month:.3f}")
    print(f"Saved: {args.score_out}")
    print(f"Saved: {args.summary_out}")
    print(f"Saved: {args.events_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
