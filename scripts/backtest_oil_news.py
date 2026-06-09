from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


GDELT_EVENTS_BASE = "http://data.gdeltproject.org/events"

# GDELT event file column indices (0-based).
COL_SQLDATE = 1
COL_ACTOR1_COUNTRY = 7
COL_ACTOR2_COUNTRY = 17
COL_EVENT_CODE = 26
COL_GOLDSTEIN = 30
COL_NUM_MENTIONS = 31
COL_NUM_ARTICLES = 33
COL_AVG_TONE = 34
COL_ACTION_COUNTRY = 49
COL_SOURCEURL = 57


@dataclass
class Perf:
    cagr: float
    sharpe: float
    max_dd: float
    trades: int


def _perf(simple_r: pd.Series) -> tuple[Perf, pd.Series]:
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
    return Perf(cagr=cagr, sharpe=sharpe, max_dd=max_dd, trades=0), eq


def _download_gdelt_zip(day: pd.Timestamp, cache_dir: Path, timeout_sec: float) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{day.strftime('%Y%m%d')}.export.CSV.zip"
    local_path = cache_dir / fn
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    url = f"{GDELT_EVENTS_BASE}/{fn}"
    req = Request(url, headers={"User-Agent": "LPBot-OilNewsBacktest/1.0"})
    try:
        with urlopen(req, timeout=float(timeout_sec)) as resp:
            blob = resp.read()
    except HTTPError as e:
        if int(getattr(e, "code", 0)) == 404:
            return None
        return None
    except URLError:
        return None
    if not blob:
        return None
    local_path.write_bytes(blob)
    return local_path


def _read_filtered_events_from_zip(path: Path, oil_cc: set[str], oil_kw_regex: str) -> pd.DataFrame:
    with zipfile.ZipFile(path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            return pd.DataFrame()
        with zf.open(members[0], "r") as f:
            d = pd.read_csv(
                f,
                sep="\t",
                header=None,
                usecols=[
                    COL_SQLDATE,
                    COL_ACTOR1_COUNTRY,
                    COL_ACTOR2_COUNTRY,
                    COL_EVENT_CODE,
                    COL_GOLDSTEIN,
                    COL_NUM_MENTIONS,
                    COL_NUM_ARTICLES,
                    COL_AVG_TONE,
                    COL_ACTION_COUNTRY,
                    COL_SOURCEURL,
                ],
                dtype="object",
                on_bad_lines="skip",
                low_memory=False,
            )
    if d.empty:
        return d
    d.columns = [
        "sqldate",
        "actor1_country",
        "actor2_country",
        "event_code",
        "goldstein",
        "num_mentions",
        "num_articles",
        "avg_tone",
        "action_country",
        "sourceurl",
    ]
    d["sqldate"] = pd.to_datetime(d["sqldate"], format="%Y%m%d", errors="coerce")
    d["event_code"] = d["event_code"].astype(str).str.strip()
    for c in ["actor1_country", "actor2_country", "action_country"]:
        d[c] = d[c].astype(str).str.strip().str.upper()
    d["sourceurl"] = d["sourceurl"].astype(str)
    d["goldstein"] = pd.to_numeric(d["goldstein"], errors="coerce")
    d["num_mentions"] = pd.to_numeric(d["num_mentions"], errors="coerce")
    d["num_articles"] = pd.to_numeric(d["num_articles"], errors="coerce")
    d["avg_tone"] = pd.to_numeric(d["avg_tone"], errors="coerce")
    d = d.dropna(subset=["sqldate", "event_code"]).copy()

    # Strict oil relevance filter to avoid exploding unrelated rows.
    is_event17 = d["event_code"].str.startswith("17", na=False)
    is_oil_cc = d["actor1_country"].isin(oil_cc) | d["actor2_country"].isin(oil_cc) | d["action_country"].isin(oil_cc)
    has_oil_kw = d["sourceurl"].str.contains(oil_kw_regex, case=False, regex=True, na=False)
    keep = (is_event17 & is_oil_cc) | (is_oil_cc & has_oil_kw)
    d = d[keep].copy()
    if d.empty:
        return d

    neg_tone = np.maximum(0.0, -pd.to_numeric(d["avg_tone"], errors="coerce").fillna(0.0))
    d["num_mentions"] = pd.to_numeric(d["num_mentions"], errors="coerce").fillna(1.0).clip(lower=1.0)
    d["goldstein"] = pd.to_numeric(d["goldstein"], errors="coerce").fillna(0.0)
    d["score_component"] = d["num_mentions"] * np.abs(d["goldstein"]) * (1.0 + neg_tone)
    return d


def _build_positions(
    x: pd.DataFrame,
    *,
    mode: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, Perf]:
    d = x.copy()
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema55"] = d["close"].ewm(span=55, adjust=False).mean()
    d["ema144"] = d["close"].ewm(span=144, adjust=False).mean()
    d["ema_aligned"] = (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])
    d["ema_break3"] = ((~d["ema_aligned"]).rolling(3, min_periods=3).min() == 1).fillna(False)
    d["ema_entry3"] = (d["ema_aligned"].rolling(3, min_periods=3).min() == 1).fillna(False)

    rv = d["ret"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    d["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(d), dtype=int)
    entry_px = np.nan
    active = 0
    for i in range(len(d)):
        z = float(d["oil_news_zscore"].iloc[i]) if np.isfinite(d["oil_news_zscore"].iloc[i]) else np.nan
        ema21_gt_55 = bool(d["ema21"].iloc[i] > d["ema55"].iloc[i]) if np.isfinite(d["ema21"].iloc[i]) and np.isfinite(d["ema55"].iloc[i]) else False
        entry_news = np.isfinite(z) and z > 2.0 and ema21_gt_55
        exit_news = np.isfinite(z) and z < 0.5
        exit_ema = bool(d["ema_break3"].iloc[i])
        px = float(d["close"].iloc[i]) if np.isfinite(d["close"].iloc[i]) else np.nan

        if mode == "ema_only":
            enter = bool(d["ema_entry3"].iloc[i])
            exit_now = exit_ema
        elif mode == "news_only":
            enter = np.isfinite(z) and z > 2.0
            exit_now = exit_news
        else:  # news_ema
            enter = entry_news
            exit_now = exit_news or exit_ema

        if active == 0:
            if enter:
                active = 1
                entry_px = px
        else:
            dd_from_entry = (px / entry_px - 1.0) if np.isfinite(px) and np.isfinite(entry_px) and entry_px > 0 else 0.0
            if exit_now or (dd_from_entry <= -0.08):
                active = 0
                entry_px = np.nan
        pos[i] = active

    d["position"] = pos
    d["weight_target"] = d["position"].astype(float) * d["vol_scalar"]
    d["weight_exec"] = d["weight_target"].shift(1).fillna(0.0)
    d["exec_return"] = d["open"].shift(-1) / d["open"] - 1.0
    d["turnover"] = (d["weight_exec"] - d["weight_exec"].shift(1).fillna(0.0)).abs()
    d["cost"] = d["turnover"] * (float(cost_bps) / 10000.0)
    d["strategy_return"] = d["weight_exec"] * d["exec_return"].fillna(0.0) - d["cost"]
    d = d.dropna(subset=["exec_return"]).copy()
    p, eq = _perf(d["strategy_return"])
    trades = int(((d["weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())
    p = Perf(cagr=p.cagr, sharpe=p.sharpe, max_dd=p.max_dd, trades=trades)
    d["eq"] = eq
    return d, p


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest oil news-driven strategy using GDELT events.")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--ticker", default="USO")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--cache-dir", default="data/gdelt/events_cache")
    p.add_argument("--events-out", default="data/gdelt_oil_events.csv")
    p.add_argument("--summary-out", default="artifacts/backtest/oil_news_summary.csv")
    p.add_argument("--daily-out", default="artifacts/backtest/oil_news_daily.csv")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--checkpoint-every", type=int, default=100)
    args = p.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    days = pd.date_range(start=start, end=end, freq="D")
    if int(args.max_days) > 0:
        days = days[: int(args.max_days)]

    oil_cc = {"SAU", "IRN", "IRQ", "ARE", "KWT", "RUS", "QAT", "OMN", "BHR"}
    oil_kw_regex = r"(?:oil|crude|opec|hormuz|tanker|pipeline|sanction|refiner|shipping|energy)"

    events_out = Path(args.events_out)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    if events_out.exists():
        events_out.unlink()
    wrote_header = False

    score_by_day: defaultdict[pd.Timestamp, float] = defaultdict(float)
    rows_processed = 0
    relevant_events = 0
    done = 0
    downloaded = 0
    missing = 0
    errors = 0
    total = len(days)

    def _process_day(day_ts: pd.Timestamp) -> tuple[pd.Timestamp, str, pd.DataFrame]:
        try:
            zpath = _download_gdelt_zip(day_ts, cache_dir=Path(args.cache_dir), timeout_sec=20.0)
        except Exception:
            return day_ts, "error", pd.DataFrame()
        if zpath is None:
            return day_ts, "missing", pd.DataFrame()
        try:
            fd = _read_filtered_events_from_zip(zpath, oil_cc=oil_cc, oil_kw_regex=oil_kw_regex)
        except Exception:
            return day_ts, "error", pd.DataFrame()
        return day_ts, "ok", fd

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_process_day, d) for d in days]
        for fut in as_completed(futs):
            done += 1
            _, status, fd = fut.result()
            if status == "ok":
                downloaded += 1
                if len(fd):
                    rows_processed += int(len(fd))
                    relevant_events += int(len(fd))
                    # Persist filtered events incrementally.
                    keep_cols = [
                        "sqldate",
                        "event_code",
                        "actor1_country",
                        "actor2_country",
                        "action_country",
                        "goldstein",
                        "num_mentions",
                        "num_articles",
                        "avg_tone",
                        "score_component",
                    ]
                    fd[keep_cols].to_csv(events_out, mode="a", index=False, header=(not wrote_header))
                    wrote_header = True
                    # Aggregate score by day incrementally.
                    g = fd.groupby(fd["sqldate"].dt.floor("D"))["score_component"].sum()
                    for k, v in g.items():
                        score_by_day[pd.Timestamp(k)] += float(v)
            elif status == "missing":
                missing += 1
            else:
                errors += 1

            if done % int(args.checkpoint_every) == 0 or done in {1, total}:
                print(
                    f"[GDELT] processed {done}/{total} "
                    f"(downloaded={downloaded}, missing={missing}, errors={errors}, rows={rows_processed})"
                )
                # Checkpoint summary while running.
                chk = pd.DataFrame(
                    [
                        {
                            "status": "running",
                            "processed_days": done,
                            "total_days": total,
                            "downloaded_days": downloaded,
                            "missing_days": missing,
                            "error_days": errors,
                            "oil_relevant_events": rows_processed,
                        }
                    ]
                )
                Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
                chk.to_csv(args.summary_out, index=False)

    if relevant_events == 0 or len(score_by_day) == 0:
        raise RuntimeError("No oil-relevant GDELT events extracted for selected range.")

    daily_score = (
        pd.DataFrame({"date": list(score_by_day.keys()), "oil_news_score": list(score_by_day.values())})
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_score["oil_news_ma30"] = daily_score["oil_news_score"].rolling(30, min_periods=10).mean()
    daily_score["oil_news_sd30"] = daily_score["oil_news_score"].rolling(30, min_periods=10).std(ddof=0)
    daily_score["oil_news_zscore"] = (
        (daily_score["oil_news_score"] - daily_score["oil_news_ma30"]) / daily_score["oil_news_sd30"].replace(0.0, np.nan)
    )

    uso = yf.download(
        args.ticker,
        start=args.start,
        end=(pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if uso is None or uso.empty:
        raise RuntimeError(f"{args.ticker} download failed.")
    if isinstance(uso.columns, pd.MultiIndex):
        uso.columns = uso.columns.get_level_values(0)
    uso = uso.reset_index()
    date_col = "Date" if "Date" in uso.columns else "index"
    uso["date"] = pd.to_datetime(uso[date_col], errors="coerce")
    uso["open"] = pd.to_numeric(uso["Open"], errors="coerce")
    uso["close"] = pd.to_numeric(uso["Close"], errors="coerce")
    uso = uso.dropna(subset=["date", "open", "close"]).sort_values("date")
    uso = uso[(uso["date"] >= start) & (uso["date"] <= end)].copy()

    merged = pd.merge_asof(
        uso.assign(date=pd.to_datetime(uso["date"]).values.astype("datetime64[ns]")).sort_values("date"),
        daily_score.assign(date=pd.to_datetime(daily_score["date"]).values.astype("datetime64[ns]")).sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["oil_news_score"] = merged["oil_news_score"].fillna(0.0)
    merged["oil_news_zscore"] = merged["oil_news_zscore"].fillna(0.0)

    merged["next_week_return"] = merged["close"].shift(-5) / merged["close"] - 1.0
    merged["next_month_return"] = merged["close"].shift(-21) / merged["close"] - 1.0
    corr_week = float(merged[["oil_news_zscore", "next_week_return"]].corr().iloc[0, 1])
    corr_month = float(merged[["oil_news_zscore", "next_month_return"]].corr().iloc[0, 1])

    ema_only_df, ema_only = _build_positions(merged, mode="ema_only", cost_bps=float(args.cost_bps))
    news_only_df, news_only = _build_positions(merged, mode="news_only", cost_bps=float(args.cost_bps))
    news_ema_df, news_ema = _build_positions(merged, mode="news_ema", cost_bps=float(args.cost_bps))

    known_events = [
        ("2020-03-09", "Saudi-Russia oil price war"),
        ("2022-02-24", "Ukraine invasion"),
        ("2022-10-05", "OPEC+ production cuts"),
        ("2023-10-07", "Hamas attack"),
        ("2024-01-15", "Red Sea shipping attacks period"),
        ("2026-04-08", "Iran blockade/oil shock"),
    ]
    signal_days = set(pd.to_datetime(news_ema_df.loc[news_ema_df["position"] == 1, "date"]).dt.floor("D"))
    event_rows = []
    for ds, label in known_events:
        dt = pd.Timestamp(ds)
        lo = dt - pd.Timedelta(days=3)
        hi = dt + pd.Timedelta(days=3)
        fired = any((d >= lo) and (d <= hi) for d in signal_days)
        oil_move_5d = np.nan
        row0 = merged[merged["date"] >= dt].head(1)
        if len(row0):
            idx = row0.index[0]
            if idx + 5 < len(merged):
                p0 = float(merged.loc[idx, "close"])
                p1 = float(merged.loc[idx + 5, "close"])
                if p0 > 0:
                    oil_move_5d = p1 / p0 - 1.0
        event_rows.append({"date": ds, "event": label, "signal_fired": bool(fired), "oil_move_5d": oil_move_5d})
    event_log = pd.DataFrame(event_rows)
    known_caught = int(event_log["signal_fired"].sum())

    summary = pd.DataFrame(
        [
            {
                "start": str(start.date()),
                "end": str(end.date()),
                "gdelt_event_rows_processed": int(rows_processed),
                "oil_relevant_events": int(relevant_events),
                "signal_fires_news_ema": int((news_ema_df["position"].diff().fillna(news_ema_df["position"]).abs() == 1).sum() // 2),
                "corr_news_z_next_week": corr_week,
                "corr_news_z_next_month": corr_month,
                "ema_only_cagr": ema_only.cagr,
                "ema_only_sharpe": ema_only.sharpe,
                "ema_only_max_dd": ema_only.max_dd,
                "ema_only_trades": int(ema_only.trades),
                "news_only_cagr": news_only.cagr,
                "news_only_sharpe": news_only.sharpe,
                "news_only_max_dd": news_only.max_dd,
                "news_only_trades": int(news_only.trades),
                "news_ema_cagr": news_ema.cagr,
                "news_ema_sharpe": news_ema.sharpe,
                "news_ema_max_dd": news_ema.max_dd,
                "news_ema_trades": int(news_ema.trades),
                "known_events_caught": int(known_caught),
                "known_events_total": int(len(event_log)),
                "verdict_viable": bool(np.isfinite(news_ema.sharpe) and news_ema.sharpe >= 0.7),
            }
        ]
    )
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    daily_out = Path(args.daily_out)
    daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily_export = merged[["date", "open", "close", "oil_news_score", "oil_news_zscore"]].copy()
    daily_export = daily_export.merge(
        news_ema_df[["date", "position", "weight_exec", "strategy_return"]].rename(
            columns={"position": "news_ema_position", "weight_exec": "news_ema_weight", "strategy_return": "news_ema_return"}
        ),
        on="date",
        how="left",
    )
    daily_export.to_csv(daily_out, index=False)
    event_log.to_csv(summary_out.with_name("oil_news_event_detection_log.csv"), index=False)

    print("===============================================")
    print("OIL NEWS-DRIVEN BACKTEST")
    print("===============================================")
    print(f"GDELT events processed: {rows_processed:,}")
    print(f"Oil-relevant events:    {relevant_events:,}")
    print(f"Signal fires (News+EMA): {int(summary.loc[0, 'signal_fires_news_ema'])}")
    print("-----------------------------------------------")
    print("Strategy comparison")
    print("Metric      EMA only   News only  News+EMA")
    print(
        f"CAGR        {ema_only.cagr*100:>7.2f}%   {news_only.cagr*100:>7.2f}%   {news_ema.cagr*100:>7.2f}%\n"
        f"Sharpe      {ema_only.sharpe:>7.3f}   {news_only.sharpe:>7.3f}   {news_ema.sharpe:>7.3f}\n"
        f"MaxDD       {ema_only.max_dd*100:>7.2f}%   {news_only.max_dd*100:>7.2f}%   {news_ema.max_dd*100:>7.2f}%\n"
        f"Trades      {ema_only.trades:>7d}   {news_only.trades:>7d}   {news_ema.trades:>7d}"
    )
    print("-----------------------------------------------")
    print(f"Correlation news_z vs next-week return:  {corr_week:.3f}")
    print(f"Correlation news_z vs next-month return: {corr_month:.3f}")
    print("-----------------------------------------------")
    print("Known-event detection (±3d):")
    for _, r in event_log.iterrows():
        mv = r["oil_move_5d"]
        mv_s = "n/a" if not np.isfinite(mv) else f"{mv*100:.2f}%"
        print(f"{r['date']} | {'FIRED' if r['signal_fired'] else 'MISS '} | {r['event']} | oil_5d={mv_s}")
    print("-----------------------------------------------")
    print(f"Verdict viable: {'YES' if bool(summary.loc[0, 'verdict_viable']) else 'NO'}")
    print("===============================================")
    print(f"Saved: {events_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {daily_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
