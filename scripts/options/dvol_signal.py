from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from fetch_deribit import build_surface
except Exception:  # pragma: no cover - import failure is reported in main path
    build_surface = None


def _safe_float(value: object) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _load_surface(path: Path, refresh: bool, currency: str) -> pd.DataFrame:
    if refresh or not path.exists():
        if build_surface is None:
            raise SystemExit("Could not import build_surface from fetch_deribit.py")
        df = build_surface(currency)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        dated = path.parent / f"vol_surface_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
        df.to_csv(dated, index=False)
        return df
    df = pd.read_csv(path)
    for col in ["mark_iv", "underlying_price", "moneyness", "days_to_expiry", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], utc=True, errors="coerce")
    return df


def _atm_iv_by_expiry(surface: pd.DataFrame, atm_band: tuple[float, float] = (0.95, 1.05)) -> pd.DataFrame:
    d = surface.copy()
    if "option_type" in d.columns:
        d = d[d["option_type"].astype(str).str.lower().eq("call")]
    d = d[
        d["moneyness"].between(atm_band[0], atm_band[1], inclusive="both")
        & (d["mark_iv"] > 0)
        & (d["days_to_expiry"] > 0)
    ].copy()
    if d.empty:
        # Fallback: nearest option to ATM per expiry if the strict band is empty.
        d = surface.copy()
        if "option_type" in d.columns:
            d = d[d["option_type"].astype(str).str.lower().eq("call")]
        d = d[(d["mark_iv"] > 0) & (d["days_to_expiry"] > 0)].copy()
    if d.empty:
        return pd.DataFrame(columns=["days_to_expiry", "atm_iv", "eth_price"])

    d["atm_dist"] = (d["moneyness"] - 1.0).abs()
    if "open_interest" not in d.columns:
        d["open_interest"] = 0.0
    rows = []
    for days, g in d.groupby("days_to_expiry"):
        g = g.sort_values(["atm_dist", "open_interest"], ascending=[True, False])
        # Use up to the three nearest ATM contracts for stability.
        near = g.head(3)
        weights = pd.to_numeric(near.get("open_interest", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(weights.sum()) > 0:
            iv = float(np.average(near["mark_iv"], weights=weights))
        else:
            iv = float(near["mark_iv"].mean())
        rows.append(
            {
                "days_to_expiry": float(days),
                "atm_iv": iv,
                "eth_price": float(pd.to_numeric(near["underlying_price"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values("days_to_expiry").reset_index(drop=True)


def _interp_target_iv(atm: pd.DataFrame, target_days: float = 30.0) -> tuple[float, str]:
    if atm.empty:
        return float("nan"), "no_atm_data"
    x = atm.dropna(subset=["days_to_expiry", "atm_iv"]).sort_values("days_to_expiry")
    below = x[x["days_to_expiry"] <= target_days]
    above = x[x["days_to_expiry"] >= target_days]
    if below.empty or above.empty:
        nearest = x.iloc[(x["days_to_expiry"] - target_days).abs().argsort()].iloc[0]
        return float(nearest["atm_iv"]), f"nearest_{float(nearest['days_to_expiry']):.0f}d"
    b = below.iloc[-1]
    a = above.iloc[0]
    db = float(b["days_to_expiry"])
    da = float(a["days_to_expiry"])
    if abs(da - db) < 1e-9:
        return float(a["atm_iv"]), f"exact_{da:.0f}d"
    w = (target_days - db) / (da - db)
    iv = float(float(b["atm_iv"]) * (1.0 - w) + float(a["atm_iv"]) * w)
    return iv, f"interp_{db:.0f}d_{da:.0f}d"


def _target_or_nearest(atm: pd.DataFrame, target_days: float) -> float:
    iv, _ = _interp_target_iv(atm, target_days)
    return iv


def _iv_percentile(log_path: Path, today: str, atm_iv: float, window: int = 30) -> tuple[float, bool, int]:
    if not log_path.exists() or not np.isfinite(atm_iv):
        return float("nan"), True, 0
    try:
        log = pd.read_csv(log_path)
    except Exception:
        return float("nan"), True, 0
    if log.empty or "atm_iv_30d" not in log.columns:
        return float("nan"), True, 0
    if "date" in log.columns:
        log = log[log["date"].astype(str) != str(today)]
    vals = pd.to_numeric(log["atm_iv_30d"], errors="coerce").dropna().tail(window)
    n = int(len(vals))
    insufficient = n < window
    if n < 1:
        return float("nan"), True, n
    percentile = float(((vals <= atm_iv).sum() + 0.5) / (n + 1.0))
    return max(0.0, min(1.0, percentile)), insufficient, n


def _options_regime(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "LOG_ONLY"
    if percentile > 0.75:
        return "HIGH"
    if percentile < 0.25:
        return "LOW"
    return "NORMAL"


def _load_realised_vol(log_path: Path) -> tuple[str, float]:
    if not log_path.exists():
        return "NA", float("nan")
    try:
        d = pd.read_csv(log_path)
    except Exception:
        return "NA", float("nan")
    if d.empty:
        return "NA", float("nan")
    row = d.tail(1).iloc[0]
    regime = str(row.get("vol_regime", "NA")).upper()
    realised = _safe_float(row.get("rolling_vol_20d", np.nan))
    return regime, realised


def _append_log(log_path: Path, row: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if log_path.exists():
        old = pd.read_csv(log_path)
        if "date" in old.columns:
            old = old[old["date"].astype(str) != str(row["date"])]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out = out.sort_values("date") if "date" in out.columns else out
    out.to_csv(log_path, index=False)


def _fmt_pct(x: float, digits: int = 1) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.{digits}f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute log-only ETH Deribit options volatility signal.")
    ap.add_argument("--surface-csv", default="artifacts/options/vol_surface_latest.csv")
    ap.add_argument("--log-csv", default="artifacts/options/dvol_log.csv")
    ap.add_argument("--daily-checks-log", default="artifacts/paper_trade/daily_checks_log.csv")
    ap.add_argument("--currency", default="ETH")
    ap.add_argument("--refresh", action="store_true", help="Fetch a fresh Deribit surface before computing the signal.")
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    today = now.strftime("%Y-%m-%d")
    surface = _load_surface(Path(args.surface_csv), bool(args.refresh), str(args.currency).upper())
    atm = _atm_iv_by_expiry(surface)
    atm_iv_30d, iv_method = _interp_target_iv(atm, 30.0)
    iv_7d = _target_or_nearest(atm, 7.0)
    iv_90d = _target_or_nearest(atm, 90.0)
    # Negative means front-end IV is above 90d IV: inverted/fear.
    term_slope = iv_90d - iv_7d if np.isfinite(iv_90d) and np.isfinite(iv_7d) else float("nan")
    eth_price = float(pd.to_numeric(surface.get("underlying_price", pd.Series(dtype=float)), errors="coerce").dropna().median())

    iv_pct, insufficient, hist_n = _iv_percentile(Path(args.log_csv), today, atm_iv_30d, int(args.window))
    opt_regime = _options_regime(iv_pct)
    rv_regime, realised_vol = _load_realised_vol(Path(args.daily_checks_log))
    agreement = "LOG_ONLY" if opt_regime == "LOG_ONLY" or rv_regime in {"", "NA", "NAN"} else ("AGREE" if opt_regime == rv_regime else "DISAGREE")

    row = {
        "date": today,
        "timestamp_utc": now.isoformat(),
        "eth_price": eth_price,
        "atm_iv_30d": atm_iv_30d,
        "iv_percentile": iv_pct,
        "options_vol_regime": opt_regime,
        "rv_vol_regime": rv_regime,
        "realised_vol_20d": realised_vol,
        "agreement": agreement,
        "term_slope": term_slope,
        "iv_7d": iv_7d,
        "iv_90d": iv_90d,
        "iv_method": iv_method,
        "history_days": hist_n,
        "insufficient_history": bool(insufficient),
    }
    _append_log(Path(args.log_csv), row)

    pct_txt = "N/A" if not np.isfinite(iv_pct) else f"{iv_pct * 100:.0f}%"
    print("=" * 48)
    print("ETH OPTIONS VOL SIGNAL")
    print(today)
    print("=" * 48)
    print(f"ETH price:       ${eth_price:,.0f}" if np.isfinite(eth_price) else "ETH price:       N/A")
    print(f"ATM IV (30d):    {_fmt_pct(atm_iv_30d, 1)}")
    print(f"IV percentile:   {pct_txt}")
    print(f"Options regime:  {opt_regime}")
    print("")
    print(f"Realised vol:    {_fmt_pct(realised_vol * 100.0 if np.isfinite(realised_vol) and realised_vol < 5 else realised_vol, 1)}")
    print(f"RV regime:       {rv_regime}")
    print("")
    print(f"Agreement:       {agreement}")
    print(f"Term slope:      {term_slope:.1f} vol pts" if np.isfinite(term_slope) else "Term slope:      N/A")
    print("                 (negative = inverted = fear)")
    print("")
    note = "LOG ONLY"
    if insufficient:
        note += f" ({hist_n}/{int(args.window)} days history)"
    print(f"Signal:          {note}")
    print(f"Method:          {iv_method}")
    print("=" * 48)
    print(f"Saved: {args.log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
