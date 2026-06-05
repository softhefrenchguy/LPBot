from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and (k not in os.environ):
            os.environ[k] = v


def _http_get_json(url: str, timeout_sec: float = 20.0) -> dict:
    req = Request(
        url,
        headers={
            "User-Agent": "LPBot-OilEIAResearch/1.0",
            "Accept": "application/json,*/*",
        },
    )
    with urlopen(req, timeout=float(timeout_sec)) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get_text(url: str, timeout_sec: float = 20.0) -> str:
    req = Request(url, headers={"User-Agent": "LPBot-OilEIAResearch/1.0", "Accept": "text/csv,*/*"})
    with urlopen(req, timeout=float(timeout_sec)) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_eia_weekly_inventory(api_key: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    errs: list[str] = []

    # Preferred EIA v2 path.
    # Weekly U.S. crude oil ending stocks (thousand barrels) series id used in EIA endpoints: WCESTUS1.
    v2_params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WCESTUS1",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    v2_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?{urlencode(v2_params)}"
    try:
        j = _http_get_json(v2_url)
        data = (((j or {}).get("response") or {}).get("data")) or []
        rows = []
        for r in data:
            dt = pd.to_datetime(r.get("period"), errors="coerce")
            val = pd.to_numeric(r.get("value"), errors="coerce")
            if pd.notna(dt) and np.isfinite(val):
                rows.append({"date": dt, "inventory_mbbl": float(val) / 1000.0})
        d = pd.DataFrame(rows).sort_values("date")
        if len(d):
            return d, "eia_v2"
    except Exception as e:  # noqa: BLE001
        errs.append(f"v2:{type(e).__name__}")

    # Legacy series path.
    legacy_url = (
        "https://api.eia.gov/series/?"
        + urlencode(
            {
                "api_key": api_key,
                "series_id": "PET.WCRSTUS1.W",
            }
        )
    )
    try:
        j = _http_get_json(legacy_url)
        series = ((j or {}).get("series") or [])
        points = series[0].get("data", []) if len(series) else []
        rows = []
        for p in points:
            # Legacy format: ["YYYYMMDD", value]
            dt = pd.to_datetime(str(p[0]), format="%Y%m%d", errors="coerce")
            val = pd.to_numeric(p[1], errors="coerce")
            if pd.notna(dt) and np.isfinite(val):
                rows.append({"date": dt, "inventory_mbbl": float(val) / 1000.0})
        d = pd.DataFrame(rows).sort_values("date")
        if len(d):
            d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))].copy()
            return d, "eia_legacy"
    except Exception as e:  # noqa: BLE001
        errs.append(f"legacy:{type(e).__name__}")

    raise RuntimeError(f"EIA API failed ({', '.join(errs)})")


def fetch_fred_weekly_inventory(start: str, end: str) -> tuple[pd.DataFrame, str]:
    # Fallback mirror of EIA weekly crude stocks via FRED (WCESTUS1, thousand barrels).
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WCESTUS1"
    txt = _http_get_text(url)
    d = pd.read_csv(pd.io.common.StringIO(txt))
    if "DATE" not in d.columns or "WCESTUS1" not in d.columns:
        raise RuntimeError("FRED fallback parse failed")
    d["date"] = pd.to_datetime(d["DATE"], errors="coerce")
    d["inventory_kbbl"] = pd.to_numeric(d["WCESTUS1"], errors="coerce")
    d = d.dropna(subset=["date", "inventory_kbbl"]).sort_values("date")
    d["inventory_mbbl"] = d["inventory_kbbl"] / 1000.0
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))].copy()
    return d[["date", "inventory_mbbl"]], "fred_fallback"


def fetch_eia_public_html_weekly_inventory(start: str, end: str) -> tuple[pd.DataFrame, str]:
    url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCRSTUS1&f=W"
    html = _http_get_text(url, timeout_sec=40)
    # Parse rows from the monthly table.
    tr_blocks = re.findall(r"<tr>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    rows: list[dict[str, object]] = []
    for tr in tr_blocks:
        if "B6" not in tr:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.IGNORECASE | re.DOTALL)
        if len(tds) < 3:
            continue
        header = re.sub(r"<[^>]+>", "", tds[0]).replace("&nbsp;", "").strip()
        m_head = re.search(r"(\d{4})-([A-Za-z]{3})", header)
        if not m_head:
            continue
        year = int(m_head.group(1))
        for i in range(1, len(tds), 2):
            if i + 1 >= len(tds):
                break
            d_txt = re.sub(r"<[^>]+>", "", tds[i]).replace("&nbsp;", "").strip()
            v_txt = re.sub(r"<[^>]+>", "", tds[i + 1]).replace("&nbsp;", "").strip().replace(",", "")
            if not d_txt or d_txt == "":
                continue
            md = re.search(r"(\d{2})/(\d{2})", d_txt)
            if not md:
                continue
            month = int(md.group(1))
            day = int(md.group(2))
            val = pd.to_numeric(v_txt, errors="coerce")
            if not np.isfinite(val):
                continue
            try:
                dt = pd.Timestamp(year=year, month=month, day=day)
            except Exception:
                continue
            rows.append({"date": dt, "inventory_mbbl": float(val) / 1000.0})
    d = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))].copy()
    if len(d) == 0:
        raise RuntimeError("EIA public HTML parse produced no rows")
    return d, "eia_public_html"


def _perf(simple_r: pd.Series) -> dict[str, float]:
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
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}


def _run_strategy(
    d: pd.DataFrame,
    *,
    use_eia_filter: bool,
    cost_bps: float = 10.0,
    confirm_days: int = 3,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    x = d.copy()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema55"] = x["close"].ewm(span=55, adjust=False).mean()
    x["ema144"] = x["close"].ewm(span=144, adjust=False).mean()
    stack_aligned = (x["ema21"] > x["ema55"]) & (x["ema55"] > x["ema144"])
    entry_stack = (stack_aligned.rolling(confirm_days, min_periods=confirm_days).min() == 1).fillna(False)
    exit_stack = ((~stack_aligned).rolling(confirm_days, min_periods=confirm_days).min() == 1).fillna(False)

    r = x["close"].pct_change().fillna(0.0)
    rv = r.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    vol_scalar = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.25, 1.0).fillna(0.25)

    pos = np.zeros(len(x), dtype=int)
    active = 0
    for i in range(len(x)):
        z = float(x["inventory_zscore"].iloc[i]) if np.isfinite(x["inventory_zscore"].iloc[i]) else np.nan
        eia_entry = (not use_eia_filter) or (np.isfinite(z) and z < -0.5)
        eia_exit = use_eia_filter and np.isfinite(z) and z > 1.5

        if active == 0:
            if bool(entry_stack.iloc[i]) and eia_entry:
                active = 1
        else:
            if bool(exit_stack.iloc[i]) or bool(eia_exit):
                active = 0
        pos[i] = active

    x["position"] = pos
    x["weight_target"] = x["position"].astype(float) * vol_scalar
    x["weight_exec"] = x["weight_target"].shift(1).fillna(0.0)  # lag-1 execution
    x["exec_return"] = x["open"].shift(-1) / x["open"] - 1.0
    x["turnover"] = (x["weight_exec"] - x["weight_exec"].shift(1).fillna(0.0)).abs()
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["exec_return"].fillna(0.0) - x["cost"]
    x = x.dropna(subset=["exec_return"]).copy()
    x["eq"] = (1.0 + x["strategy_return"]).cumprod()
    trades = int(((x["weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())
    return x, _perf(x["strategy_return"]), trades


def main() -> int:
    p = argparse.ArgumentParser(description="Oil EIA inventory research with EMA+EIA filter.")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out-eia-csv", default="data/oil_eia_weekly.csv")
    p.add_argument("--out-summary-csv", default="artifacts/backtest/oil_eia_summary.csv")
    p.add_argument("--cost-bps", type=float, default=10.0)
    args = p.parse_args()

    _load_env_file(Path(".env"))
    eia_key = os.environ.get("EIA_API_KEY", "").strip()

    source = "none"
    if eia_key:
        try:
            eia_weekly, source = fetch_eia_weekly_inventory(eia_key, args.start, args.end)
        except Exception:
            try:
                eia_weekly, source = fetch_eia_public_html_weekly_inventory(args.start, args.end)
            except Exception:
                eia_weekly, source = fetch_fred_weekly_inventory(args.start, args.end)
    else:
        try:
            eia_weekly, source = fetch_eia_public_html_weekly_inventory(args.start, args.end)
        except Exception:
            eia_weekly, source = fetch_fred_weekly_inventory(args.start, args.end)

    eia_weekly = eia_weekly.sort_values("date").drop_duplicates("date", keep="last").copy()
    eia_weekly["rolling_avg"] = eia_weekly["inventory_mbbl"].rolling(260, min_periods=52).mean()
    eia_weekly["rolling_std"] = eia_weekly["inventory_mbbl"].rolling(260, min_periods=52).std(ddof=0)
    eia_weekly["inventory_zscore"] = (
        (eia_weekly["inventory_mbbl"] - eia_weekly["rolling_avg"]) / eia_weekly["rolling_std"].replace(0.0, np.nan)
    )

    out_eia = Path(args.out_eia_csv)
    out_eia.parent.mkdir(parents=True, exist_ok=True)
    eia_weekly[["date", "inventory_mbbl", "rolling_avg", "rolling_std", "inventory_zscore"]].to_csv(out_eia, index=False)

    uso = yf.download("USO", start=args.start, end=pd.Timestamp(args.end) + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
    if uso is None or uso.empty:
        raise RuntimeError("USO download failed.")
    if isinstance(uso.columns, pd.MultiIndex):
        uso.columns = uso.columns.get_level_values(0)
    uso = uso.reset_index()
    date_col = "Date" if "Date" in uso.columns else "index"
    uso["date"] = pd.to_datetime(uso[date_col], errors="coerce")
    uso["open"] = pd.to_numeric(uso["Open"], errors="coerce")
    uso["close"] = pd.to_numeric(uso["Close"], errors="coerce")
    uso = uso.dropna(subset=["date", "open", "close"]).sort_values("date")
    uso = uso[(uso["date"] >= pd.Timestamp(args.start)) & (uso["date"] <= pd.Timestamp(args.end))].copy()

    d = pd.merge_asof(
        uso.assign(date=pd.to_datetime(uso["date"]).values.astype("datetime64[ns]")).sort_values("date"),
        eia_weekly[["date", "inventory_zscore"]]
        .assign(date=pd.to_datetime(eia_weekly["date"]).values.astype("datetime64[ns]"))
        .sort_values("date"),
        on="date",
        direction="backward",
    )
    d["inventory_zscore"] = d["inventory_zscore"].ffill()
    d["next_week_return"] = d["close"].shift(-5) / d["close"] - 1.0
    d["next_month_return"] = d["close"].shift(-21) / d["close"] - 1.0
    corr_week = float(d[["inventory_zscore", "next_week_return"]].corr().iloc[0, 1])
    corr_month = float(d[["inventory_zscore", "next_month_return"]].corr().iloc[0, 1])

    ema_only_df, ema_only_m, ema_only_trades = _run_strategy(d, use_eia_filter=False, cost_bps=float(args.cost_bps), confirm_days=3)
    ema_eia_df, ema_eia_m, ema_eia_trades = _run_strategy(d, use_eia_filter=True, cost_bps=float(args.cost_bps), confirm_days=3)

    verdict_adds = bool(
        np.isfinite(ema_eia_m["sharpe"])
        and np.isfinite(ema_only_m["sharpe"])
        and (ema_eia_m["sharpe"] > ema_only_m["sharpe"])
    )
    verdict_viable = bool(np.isfinite(ema_eia_m["sharpe"]) and ema_eia_m["sharpe"] >= 0.7 and ema_eia_m["max_dd"] > -0.35)

    summary = pd.DataFrame(
        [
            {
                "source": source,
                "weekly_obs": int(len(eia_weekly)),
                "data_start": str(eia_weekly["date"].min().date()) if len(eia_weekly) else "",
                "data_end": str(eia_weekly["date"].max().date()) if len(eia_weekly) else "",
                "corr_zscore_next_week": corr_week,
                "corr_zscore_next_month": corr_month,
                "ema_only_cagr": ema_only_m["cagr"],
                "ema_only_sharpe": ema_only_m["sharpe"],
                "ema_only_max_dd": ema_only_m["max_dd"],
                "ema_only_trades": int(ema_only_trades),
                "ema_eia_cagr": ema_eia_m["cagr"],
                "ema_eia_sharpe": ema_eia_m["sharpe"],
                "ema_eia_max_dd": ema_eia_m["max_dd"],
                "ema_eia_trades": int(ema_eia_trades),
                "eia_adds_value": bool(verdict_adds),
                "oil_full_stack_viable": bool(verdict_viable),
            }
        ]
    )
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print("===============================================")
    print("OIL EIA RESEARCH RESULTS")
    print("===============================================")
    print(f"Data source: {source}")
    print(
        f"Data range: {summary.loc[0, 'data_start']} to {summary.loc[0, 'data_end']} | "
        f"Observations: {int(summary.loc[0, 'weekly_obs'])}"
    )
    print(f"inventory_zscore corr vs next-week return:  {corr_week:.3f}")
    print(f"inventory_zscore corr vs next-month return: {corr_month:.3f}")
    print("-----------------------------------------------")
    print("Strategy comparison")
    print("Metric        EMA only   EMA+EIA")
    print(
        f"CAGR          {ema_only_m['cagr']*100:>6.2f}%   {ema_eia_m['cagr']*100:>6.2f}%\n"
        f"Sharpe        {ema_only_m['sharpe']:>6.3f}   {ema_eia_m['sharpe']:>6.3f}\n"
        f"MaxDD         {ema_only_m['max_dd']*100:>6.2f}%  {ema_eia_m['max_dd']*100:>6.2f}%\n"
        f"Trades        {ema_only_trades:>6d}   {ema_eia_trades:>6d}"
    )
    print("-----------------------------------------------")
    print(f"Verdict: EIA adds value: {'YES' if verdict_adds else 'NO'}")
    print(f"Oil full stack viable: {'YES' if verdict_viable else 'NO'}")
    print("===============================================")
    print(f"Saved: {out_eia}")
    print(f"Saved: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
