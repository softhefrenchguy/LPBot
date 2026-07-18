from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    x = _safe_num(value)
    if not np.isfinite(x):
        return "n/a"
    return f"{x * 100.0:.{decimals}f}%"


def _fmt_num(value: Any, decimals: int = 3) -> str:
    x = _safe_num(value)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{decimals}f}"


def _norm_pctile(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    finite = out[np.isfinite(out)]
    if not finite.empty and finite.max() > 2.0:
        out = out / 100.0
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise SystemExit(f"Empty input file: {path}")
    if "date" not in df.columns:
        raise SystemExit(f"Input has no date column: {path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _regime_from_pctile(pct: pd.Series) -> pd.Series:
    out = pd.Series("NORMAL", index=pct.index, dtype=object)
    out[pct >= 0.75] = "HIGH"
    out[pct <= 0.25] = "LOW"
    out[pct.isna()] = "NA"
    return out


def _summarize_group(df: pd.DataFrame, mask: pd.Series, ret_col: str) -> dict[str, float]:
    vals = pd.to_numeric(df.loc[mask, ret_col], errors="coerce").dropna()
    if vals.empty:
        return {"days": 0, "avg": np.nan, "positive": np.nan}
    return {
        "days": float(len(vals)),
        "avg": float(vals.mean()),
        "positive": float((vals > 0).mean()),
    }


def _corr(df: pd.DataFrame, a: str, b: str) -> float:
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def _filter_correct(regime: pd.Series, next_ret: pd.Series) -> pd.Series:
    r = regime.astype(str).str.upper()
    ret = pd.to_numeric(next_ret, errors="coerce")
    out = pd.Series(np.nan, index=regime.index, dtype=float)
    out[(r == "HIGH") & ret.notna()] = (ret[(r == "HIGH") & ret.notna()] < 0).astype(float)
    out[(r == "LOW") & ret.notna()] = (ret[(r == "LOW") & ret.notna()] > 0).astype(float)
    return out


def _load_merged(dvol_path: Path, daily_path: Path) -> pd.DataFrame:
    dvol = _read_csv(dvol_path)
    daily = _read_csv(daily_path)

    daily_cols = ["date"]
    for col in [
        "eth_price",
        "vol_percentile",
        "vol_regime",
        "portfolio_strategy_ret",
        "portfolio_excess_vs_basket",
    ]:
        if col in daily.columns:
            daily_cols.append(col)
    daily = daily[daily_cols].drop_duplicates(subset=["date"], keep="last")

    # Prefer the daily checklist ETH price if present because it is the live paper source.
    merged = dvol.merge(daily, on="date", how="left", suffixes=("_dvol", "_daily"))
    if "eth_price_daily" in merged.columns:
        merged["eth_price"] = pd.to_numeric(merged["eth_price_daily"], errors="coerce")
        if "eth_price_dvol" in merged.columns:
            merged["eth_price"] = merged["eth_price"].fillna(pd.to_numeric(merged["eth_price_dvol"], errors="coerce"))
    else:
        merged["eth_price"] = pd.to_numeric(merged.get("eth_price", np.nan), errors="coerce")

    merged["atm_iv_30d"] = pd.to_numeric(merged.get("atm_iv_30d", np.nan), errors="coerce")
    merged["iv_percentile"] = _norm_pctile(merged.get("iv_percentile", pd.Series(np.nan, index=merged.index)))
    merged["term_slope"] = pd.to_numeric(merged.get("term_slope", np.nan), errors="coerce")
    merged["rv_percentile"] = _norm_pctile(merged.get("vol_percentile", pd.Series(np.nan, index=merged.index)))

    if "options_vol_regime" not in merged.columns:
        merged["options_vol_regime"] = _regime_from_pctile(merged["iv_percentile"])
    else:
        merged["options_vol_regime"] = merged["options_vol_regime"].astype(str).str.upper()

    rv_col = _pick_col(merged, ["rv_vol_regime", "vol_regime"])
    if rv_col:
        merged["rv_vol_regime_effective"] = merged[rv_col].astype(str).str.upper()
    else:
        merged["rv_vol_regime_effective"] = _regime_from_pctile(merged["rv_percentile"])

    if "agreement" not in merged.columns:
        merged["agreement"] = np.where(
            merged["options_vol_regime"] == merged["rv_vol_regime_effective"],
            "AGREE",
            "DISAGREE",
        )
    else:
        merged["agreement"] = merged["agreement"].astype(str).str.upper()

    for horizon in [3, 5, 7]:
        merged[f"next_{horizon}d_return"] = merged["eth_price"].shift(-horizon) / merged["eth_price"] - 1.0

    return merged.sort_values("date").reset_index(drop=True)


def _build_report(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    valid_5d = df.dropna(subset=["next_5d_return"]).copy()
    start = df["date"].min().date().isoformat()
    end = df["date"].max().date().isoformat()

    corr_rows = [
        ("atm_iv_30d_vs_next_3d", _corr(df, "atm_iv_30d", "next_3d_return")),
        ("atm_iv_30d_vs_next_5d", _corr(df, "atm_iv_30d", "next_5d_return")),
        ("iv_percentile_vs_next_3d", _corr(df, "iv_percentile", "next_3d_return")),
        ("term_slope_vs_next_3d", _corr(df, "term_slope", "next_3d_return")),
        ("term_slope_vs_next_5d", _corr(df, "term_slope", "next_5d_return")),
        ("term_slope_vs_next_7d", _corr(df, "term_slope", "next_7d_return")),
    ]
    strongest = max(corr_rows, key=lambda x: abs(x[1]) if np.isfinite(x[1]) else -1)

    regime_stats = {
        regime: _summarize_group(df, df["options_vol_regime"].eq(regime), "next_5d_return")
        for regime in ["HIGH", "LOW", "NORMAL"]
    }

    agree_stats = _summarize_group(df, df["agreement"].eq("AGREE"), "next_5d_return")
    disagree_stats = _summarize_group(df, df["agreement"].eq("DISAGREE"), "next_5d_return")
    high_rv_normal = _summarize_group(
        df,
        df["options_vol_regime"].eq("HIGH") & df["rv_vol_regime_effective"].eq("NORMAL"),
        "next_5d_return",
    )
    low_rv_normal = _summarize_group(
        df,
        df["options_vol_regime"].eq("LOW") & df["rv_vol_regime_effective"].eq("NORMAL"),
        "next_5d_return",
    )

    inverted = _summarize_group(df, df["term_slope"] < 0, "next_5d_return")
    normal = _summarize_group(df, df["term_slope"] > 0, "next_5d_return")
    steep = _summarize_group(df, df["term_slope"] > 5, "next_5d_return")

    opt_correct = _filter_correct(df["options_vol_regime"], df["next_5d_return"])
    rv_correct = _filter_correct(df["rv_vol_regime_effective"], df["next_5d_return"])
    both = pd.concat([opt_correct.rename("options"), rv_correct.rename("realised")], axis=1)
    scored = both.dropna(how="all")
    options_better = int(((scored["options"] == 1.0) & (scored["realised"] != 1.0)).sum())
    realised_better = int(((scored["realised"] == 1.0) & (scored["options"] != 1.0)).sum())
    filter_agreed = int((df["options_vol_regime"] == df["rv_vol_regime_effective"]).sum())

    opt_acc = float(opt_correct.dropna().mean()) if not opt_correct.dropna().empty else np.nan
    rv_acc = float(rv_correct.dropna().mean()) if not rv_correct.dropna().empty else np.nan
    if np.isfinite(opt_acc) and np.isfinite(rv_acc):
        if opt_acc > rv_acc + 0.05:
            recommendation = "SWITCH to options IV"
        elif opt_acc + 0.05 < rv_acc:
            recommendation = "KEEP realised vol filter"
        else:
            recommendation = "USE BOTH (combined signal)"
    else:
        recommendation = "KEEP realised vol filter"

    summary_rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any) -> None:
        summary_rows.append({"section": section, "metric": metric, "value": value})

    add("meta", "rows", len(df))
    add("meta", "valid_next_5d_rows", len(valid_5d))
    add("meta", "start", start)
    add("meta", "end", end)
    for name, value in corr_rows:
        add("correlation", name, value)
    add("correlation", "strongest_abs_metric", strongest[0])
    add("correlation", "strongest_abs_corr", strongest[1])
    for regime, stats in regime_stats.items():
        for metric, value in stats.items():
            add(f"regime_{regime.lower()}", metric, value)
    for name, stats in [
        ("agree", agree_stats),
        ("disagree", disagree_stats),
        ("options_high_rv_normal", high_rv_normal),
        ("options_low_rv_normal", low_rv_normal),
        ("slope_inverted", inverted),
        ("slope_normal", normal),
        ("slope_steep", steep),
    ]:
        for metric, value in stats.items():
            add(name, metric, value)
    add("filter", "options_better_days", options_better)
    add("filter", "realised_better_days", realised_better)
    add("filter", "regime_agreement_days", filter_agreed)
    add("filter", "options_signal_accuracy", opt_acc)
    add("filter", "realised_signal_accuracy", rv_acc)
    add("filter", "recommendation", recommendation)

    report = f"""
========================================
OPTIONS VOL SURFACE ANALYSIS
{len(df)} days | {start} - {end}
========================================

QUESTION 1: IV AS PRICE PREDICTOR
ATM IV vs next 3d return corr: {_fmt_num(dict(corr_rows)['atm_iv_30d_vs_next_3d'])}
ATM IV vs next 5d return corr: {_fmt_num(dict(corr_rows)['atm_iv_30d_vs_next_5d'])}
IV pct vs next 3d return corr: {_fmt_num(dict(corr_rows)['iv_percentile_vs_next_3d'])}
Term slope vs next 3d corr:    {_fmt_num(dict(corr_rows)['term_slope_vs_next_3d'])}
Strongest absolute metric: {strongest[0]} ({_fmt_num(strongest[1])})

HIGH IV regime:
  Days: {int(regime_stats['HIGH']['days'])} | Avg next 5d: {_fmt_pct(regime_stats['HIGH']['avg'])} | Positive: {_fmt_pct(regime_stats['HIGH']['positive'], 1)}

LOW IV regime:
  Days: {int(regime_stats['LOW']['days'])} | Avg next 5d: {_fmt_pct(regime_stats['LOW']['avg'])} | Positive: {_fmt_pct(regime_stats['LOW']['positive'], 1)}

NORMAL IV regime:
  Days: {int(regime_stats['NORMAL']['days'])} | Avg next 5d: {_fmt_pct(regime_stats['NORMAL']['avg'])} | Positive: {_fmt_pct(regime_stats['NORMAL']['positive'], 1)}

----------------------------------------

QUESTION 2: AGREE/DISAGREE
AGREE days: {int(agree_stats['days'])} | Avg next 5d: {_fmt_pct(agree_stats['avg'])}
DISAGREE days: {int(disagree_stats['days'])} | Avg next 5d: {_fmt_pct(disagree_stats['avg'])}

Options HIGH / RV NORMAL: {int(high_rv_normal['days'])} days, avg next 5d: {_fmt_pct(high_rv_normal['avg'])}
Options LOW / RV NORMAL: {int(low_rv_normal['days'])} days, avg next 5d: {_fmt_pct(low_rv_normal['avg'])}

----------------------------------------

QUESTION 3: TERM SLOPE
Corr with next 5d return: {_fmt_num(dict(corr_rows)['term_slope_vs_next_5d'])}
Corr with next 7d return: {_fmt_num(dict(corr_rows)['term_slope_vs_next_7d'])}
Inverted (<0): {int(inverted['days'])} days, avg: {_fmt_pct(inverted['avg'])}
Normal (>0): {int(normal['days'])} days, avg: {_fmt_pct(normal['avg'])}
Steep (>5): {int(steep['days'])} days, avg: {_fmt_pct(steep['avg'])}

----------------------------------------

QUESTION 4: FILTER COMPARISON
Days options IV was better signal: {options_better}
Days realised vol was better signal: {realised_better}
Days they agreed: {filter_agreed}
Options signal accuracy: {_fmt_pct(opt_acc, 1)}
Realised signal accuracy: {_fmt_pct(rv_acc, 1)}

Recommendation:
  {recommendation}

----------------------------------------

CAVEAT:
  Sample size: {len(df)} days ({len(valid_5d)} with full next-5d forward returns)
  Minimum needed for significance: 60+ days
  Current findings: PRELIMINARY
  Revisit after 90 days of history
========================================
""".strip()

    return report, pd.DataFrame(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse ETH options vol history against future ETH returns.")
    parser.add_argument("--dvol-log", default="artifacts/options/dvol_log.csv")
    parser.add_argument("--daily-log", default="artifacts/paper_trade/daily_checks_log.csv")
    parser.add_argument("--summary-out", default="artifacts/options/vol_surface_analysis.csv")
    parser.add_argument("--report-out", default="artifacts/options/vol_surface_report.txt")
    args = parser.parse_args()

    df = _load_merged(Path(args.dvol_log), Path(args.daily_log))
    report, summary = _build_report(df)

    summary_path = Path(args.summary_out)
    report_path = Path(args.report_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
