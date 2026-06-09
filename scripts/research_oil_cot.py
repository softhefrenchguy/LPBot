from __future__ import annotations

import csv
from io import StringIO
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yfinance as yf


CFTC_URLS = [
    "https://www.cftc.gov/dea/newcot/c_disagg.txt",
    "https://www.cftc.gov/dea/newcot/FinFutWk.txt",
]


def _download_text(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/plain,text/csv,*/*",
        },
    )
    with urlopen(req, timeout=40) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _download_bytes(url: str) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "*/*",
        },
    )
    with urlopen(req, timeout=40) as resp:
        return resp.read()


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def _perf(simple_r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_r, errors="coerce").fillna(0.0)
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(np.mean(ex) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    eq = (1.0 + x).cumprod()
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else np.nan
    years = len(x) / 252.0 if len(x) else np.nan
    total = float(eq.iloc[-1]) if len(eq) else np.nan
    cagr = (total ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    return {"sharpe": sharpe, "max_dd": max_dd, "cagr": cagr}


def load_cot_crude() -> pd.DataFrame:
    # Preferred path: historical disaggregated futures-only yearly ZIP files.
    current_year = pd.Timestamp.now("UTC").year
    start_year = max(2009, current_year - 7)
    yearly_frames: list[pd.DataFrame] = []
    for y in range(start_year, current_year + 1):
        zip_url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip"
        try:
            blob = _download_bytes(zip_url)
            with ZipFile(BytesIO(blob)) as zf:
                txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                if not txt_members:
                    continue
                content = zf.read(txt_members[0]).decode("utf-8", errors="replace")
            frame = pd.read_csv(StringIO(content), low_memory=False)
            yearly_frames.append(frame)
        except Exception:
            continue

    if yearly_frames:
        raw = pd.concat(yearly_frames, ignore_index=True)
        raw.columns = [str(c).strip() for c in raw.columns]
        cols = list(raw.columns)
        market_col = _find_col(cols, ["Market_and_Exchange_Names"])
        date_col = _find_col(cols, ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD"])
        noncom_long_col = _find_col(cols, ["M_Money_Positions_Long_All"])
        noncom_short_col = _find_col(cols, ["M_Money_Positions_Short_All"])
        comm_long_col = _find_col(cols, ["Prod_Merc_Positions_Long_All"])
        comm_short_col = _find_col(cols, ["Prod_Merc_Positions_Short_All"])
        oi_col = _find_col(cols, ["Open_Interest_All"])

        required = [market_col, date_col, noncom_long_col, noncom_short_col, comm_long_col, comm_short_col, oi_col]
        if all(c is not None for c in required):
            d = raw.copy()
            d = d[d[market_col].astype(str).str.upper().str.contains("CRUDE OIL|WTI|LIGHT SWEET", regex=True, na=False)].copy()
            d["report_date"] = pd.to_datetime(d[date_col], errors="coerce")
            for c in [noncom_long_col, noncom_short_col, comm_long_col, comm_short_col, oi_col]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=["report_date"]).sort_values("report_date")
            weekly = (
                d.groupby("report_date", as_index=False)
                .agg(
                    non_commercial_longs=(noncom_long_col, "sum"),
                    non_commercial_shorts=(noncom_short_col, "sum"),
                    commercial_longs=(comm_long_col, "sum"),
                    commercial_shorts=(comm_short_col, "sum"),
                    open_interest_total=(oi_col, "sum"),
                )
                .sort_values("report_date")
            )
            weekly["speculative_net"] = weekly["non_commercial_longs"] - weekly["non_commercial_shorts"]
            rmin = weekly["speculative_net"].rolling(52, min_periods=26).min()
            rmax = weekly["speculative_net"].rolling(52, min_periods=26).max()
            denom = (rmax - rmin).replace(0.0, np.nan)
            weekly["cot_score"] = ((weekly["speculative_net"] - rmin) / denom).clip(0.0, 1.0)
            return weekly

    # Fallback path: current weekly c_disagg snapshot.
    last_err: Exception | None = None
    txt = ""
    used_url = None
    for url in CFTC_URLS:
        try:
            txt = _download_text(url)
            used_url = url
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if not txt:
        raise RuntimeError(f"CFTC download failed from all sources: {last_err}")

    # c_disagg.txt is a headerless CSV with fixed column ordering.
    # Relevant 1-based fields:
    # 1 market, 3 report_date, 8 OI, 9/10 prod_merc long/short, 14/15 managed_money long/short.
    rows = []
    reader = csv.reader(StringIO(txt))
    for rec in reader:
        if len(rec) < 15:
            continue
        market = str(rec[0]).strip()
        if "CRUDE OIL" not in market.upper() and "WTI" not in market.upper() and "LIGHT SWEET" not in market.upper():
            continue
        rows.append(
            {
                "market": market,
                "report_date": rec[2].strip(),
                "open_interest_total": rec[7].strip(),
                "commercial_longs": rec[8].strip(),
                "commercial_shorts": rec[9].strip(),
                "non_commercial_longs": rec[13].strip(),
                "non_commercial_shorts": rec[14].strip(),
            }
        )
    if not rows:
        raise RuntimeError(f"No crude oil rows found in COT file from {used_url}")

    d = pd.DataFrame(rows)
    d["report_date"] = pd.to_datetime(d["report_date"], errors="coerce")
    for c in [
        "open_interest_total",
        "commercial_longs",
        "commercial_shorts",
        "non_commercial_longs",
        "non_commercial_shorts",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["report_date"]).sort_values("report_date")

    weekly = (
        d.groupby("report_date", as_index=False)
        .agg(
            non_commercial_longs=("non_commercial_longs", "sum"),
            non_commercial_shorts=("non_commercial_shorts", "sum"),
            commercial_longs=("commercial_longs", "sum"),
            commercial_shorts=("commercial_shorts", "sum"),
            open_interest_total=("open_interest_total", "sum"),
        )
        .sort_values("report_date")
    )
    weekly["speculative_net"] = weekly["non_commercial_longs"] - weekly["non_commercial_shorts"]
    rmin = weekly["speculative_net"].rolling(52, min_periods=26).min()
    rmax = weekly["speculative_net"].rolling(52, min_periods=26).max()
    denom = (rmax - rmin).replace(0.0, np.nan)
    weekly["cot_score"] = ((weekly["speculative_net"] - rmin) / denom).clip(0.0, 1.0)
    return weekly


def load_uso_daily(start_date: str, end_date: str) -> pd.DataFrame:
    px = yf.download("USO", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if px is None or px.empty:
        raise RuntimeError("USO download returned empty data.")
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px = px.reset_index()
    date_col = "Date" if "Date" in px.columns else "index"
    px["date"] = pd.to_datetime(px[date_col], errors="coerce")
    close_col = "Close" if "Close" in px.columns else None
    if close_col is None:
        raise RuntimeError("USO data missing Close column.")
    px["close"] = pd.to_numeric(px[close_col], errors="coerce")
    px = px.dropna(subset=["date", "close"]).sort_values("date")
    return px[["date", "close"]].copy()


def main() -> int:
    try:
        cot = load_cot_crude()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] COT parse failed: {exc}")
        return 1

    out_weekly = Path("data/oil_cot_weekly.csv")
    out_weekly.parent.mkdir(parents=True, exist_ok=True)
    cot.to_csv(out_weekly, index=False)

    start_date = str(cot["report_date"].min().date())
    end_date = str(pd.Timestamp.now("UTC").date())
    uso = load_uso_daily(start_date=start_date, end_date=end_date)

    cot_daily = cot[["report_date", "cot_score"]].rename(columns={"report_date": "date"}).sort_values("date")
    uso["date"] = pd.to_datetime(uso["date"]).values.astype("datetime64[ns]")
    cot_daily["date"] = pd.to_datetime(cot_daily["date"]).values.astype("datetime64[ns]")
    merged = pd.merge_asof(uso.sort_values("date"), cot_daily.sort_values("date"), on="date", direction="backward")
    merged["cot_score"] = merged["cot_score"].ffill()

    merged["next_week_return"] = merged["close"].shift(-5) / merged["close"] - 1.0
    merged["next_month_return"] = merged["close"].shift(-21) / merged["close"] - 1.0
    corr_week = float(merged[["cot_score", "next_week_return"]].corr().iloc[0, 1])
    corr_month = float(merged[["cot_score", "next_month_return"]].corr().iloc[0, 1])

    sig = np.where(merged["cot_score"] < 0.3, 1.0, np.where(merged["cot_score"] > 0.7, 0.0, np.nan))
    merged["signal_raw"] = sig
    merged["position"] = pd.Series(merged["signal_raw"]).ffill().fillna(0.0)
    merged["position_exec"] = merged["position"].shift(1).fillna(0.0)
    merged["daily_return"] = merged["close"].pct_change().fillna(0.0)
    merged["strategy_return"] = merged["position_exec"] * merged["daily_return"]

    m_sig = _perf(merged["strategy_return"])
    m_bh = _perf(merged["daily_return"])

    if np.isfinite(m_sig["sharpe"]) and np.isfinite(m_bh["sharpe"]) and m_sig["sharpe"] > m_bh["sharpe"]:
        conclusion = "COT data is viable as a defensive/positioning signal preview."
    else:
        conclusion = "COT data is not yet viable as a standalone defensive signal."

    print("===============================================")
    print("OIL COT DATA RESEARCH")
    print("===============================================")
    print(f"Data available: {cot['report_date'].min().date()} to {cot['report_date'].max().date()}")
    print(f"Weekly observations: {len(cot)}")
    print("")
    print(f"Correlation (COT vs next-week return): {corr_week:.3f}")
    print(f"Correlation (COT vs next-month return): {corr_month:.3f}")
    print("")
    print(f"Simple COT signal Sharpe: {m_sig['sharpe']:.3f}")
    print(f"vs Oil buy-and-hold Sharpe: {m_bh['sharpe']:.3f}")
    print("")
    print("Conclusion:")
    print(conclusion)
    print("===============================================")
    print(f"Saved: {out_weekly}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
