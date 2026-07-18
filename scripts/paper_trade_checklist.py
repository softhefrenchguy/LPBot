from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error, request

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    ok: bool
    value: Optional[float | str] = None
    note: str = ""


def _read_csv_ts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, low_memory=False)
    if d.empty:
        return d
    for c in ["timestamp", "ts", "day", "date"]:
        if c in d.columns:
            d["timestamp"] = pd.to_datetime(d[c], utc=True, errors="coerce")
            break
    if "timestamp" not in d.columns:
        return pd.DataFrame()
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    return d


def _pick_col(d: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in d.columns:
            return c
    return None


def _weight_col(d: pd.DataFrame) -> Optional[str]:
    return _pick_col(d, ["weight_target", "weight_exec", "weight_daily", "weight"])


def _ret_col(d: pd.DataFrame) -> Optional[str]:
    return _pick_col(d, ["strat_r", "combined_r", "core_r", "r", "ret", "return"])


def _close_col(d: pd.DataFrame) -> Optional[str]:
    return _pick_col(d, ["eth_close", "close", "spot_close"])


def _safe_num(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _compute_returns_if_missing(d: pd.DataFrame) -> pd.Series:
    rc = _ret_col(d)
    if rc is not None:
        return pd.to_numeric(d[rc], errors="coerce").fillna(0.0)
    cc = _close_col(d)
    if cc is None:
        return pd.Series(np.zeros(len(d)), index=d.index, dtype=float)
    c = pd.to_numeric(d[cc], errors="coerce")
    return np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_spot_returns_if_missing(d: pd.DataFrame) -> pd.Series:
    if "spot_r" in d.columns:
        return pd.to_numeric(d["spot_r"], errors="coerce").fillna(0.0)
    cc = _close_col(d)
    if cc is None:
        return pd.Series(np.zeros(len(d)), index=d.index, dtype=float)
    c = pd.to_numeric(d[cc], errors="coerce")
    return np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bars_per_year_from_ts(ts: pd.Series) -> float:
    if len(ts) < 3:
        return 365.0
    dt_min = ts.diff().dt.total_seconds().dropna().median() / 60.0
    if not np.isfinite(dt_min) or dt_min <= 0:
        return 365.0
    return 365.0 * 24.0 * 60.0 / dt_min


def _rolling_sharpe_last_30d(d: pd.DataFrame) -> float:
    if d.empty:
        return float("nan")
    end_ts = d["timestamp"].iloc[-1]
    start_ts = end_ts - pd.Timedelta(days=30)
    w = d[d["timestamp"] >= start_ts].copy()
    if len(w) < 5:
        return float("nan")
    r = _compute_returns_if_missing(w)
    s = float(r.std(ddof=0))
    if s <= 0:
        return float("nan")
    bpy = _bars_per_year_from_ts(w["timestamp"])
    return float((r.mean() * bpy) / (s * np.sqrt(bpy)))


def _drawdown_from_peak(d: pd.DataFrame) -> float:
    if d.empty:
        return float("nan")
    r = _compute_returns_if_missing(d)
    eq = np.exp(np.cumsum(r.to_numpy(dtype=float)))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd[-1])


def _fmt_pct(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _read_daily_close_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "close"])
    d = pd.read_csv(path, low_memory=False)
    if d.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    ts_col = _pick_col(d, ["timestamp", "ts", "day", "date", "Date"])
    close_col = _pick_col(d, ["close", "Close", "adj_close", "Adj Close"])
    if ts_col is None or close_col is None:
        return pd.DataFrame(columns=["timestamp", "close"])
    out = d[[ts_col, close_col]].copy()
    out["timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    out = out[["timestamp", "close"]].drop_duplicates("timestamp", keep="last")
    return out


def _fetch_binance_daily_close(symbol: str, limit: int = 800, timeout_sec: float = 10.0) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={int(limit)}"
    req = request.Request(url, headers={"User-Agent": "LPBot-PaperCheck/1.0"})
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame(columns=["timestamp", "close"])
    if not isinstance(payload, list) or len(payload) == 0:
        return pd.DataFrame(columns=["timestamp", "close"])
    rows: list[dict[str, object]] = []
    for x in payload:
        try:
            ts = pd.to_datetime(int(x[0]), unit="ms", utc=True)
            close = float(x[4])
        except Exception:
            continue
        rows.append({"timestamp": ts, "close": close})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "close"])
    out = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out


def _load_json_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_portfolio_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cfg_bool(cfg: dict[str, object], key: str, default: bool) -> bool:
    if key not in cfg:
        return bool(default)
    v = cfg.get(key)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_float(cfg: dict[str, object], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def _cfg_int(cfg: dict[str, object], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _cfg_int_list(cfg: dict[str, object], key: str, default: list[int]) -> list[int]:
    v = cfg.get(key, default)
    if isinstance(v, list):
        vals = v
    elif isinstance(v, str):
        vals = [x.strip() for x in v.replace("/", ",").split(",") if x.strip()]
    else:
        return list(default)
    try:
        out = [int(x) for x in vals]
    except Exception:
        return list(default)
    return out if len(out) == len(default) else list(default)


def _vol_filter_state(daily_close: pd.Series, cfg: dict[str, object]) -> dict[str, object]:
    out = {
        "vol_regime": "NA",
        "vol_percentile": np.nan,
        "vol_multiplier": 1.0,
        "rolling_vol_20d": np.nan,
    }
    if len(daily_close) < 30:
        return out
    lookback = max(2, _cfg_int(cfg, "vol_lookback", 20))
    rank_window = max(lookback + 5, _cfg_int(cfg, "vol_rank_window", 252))
    r = daily_close.astype(float).pct_change()
    vol = r.rolling(lookback, min_periods=max(5, lookback // 2)).std(ddof=0) * np.sqrt(365.0)
    if len(vol.dropna()) == 0:
        return out
    latest_vol = float(vol.iloc[-1])
    hist = vol.dropna().iloc[-rank_window:]
    if len(hist) < 20 or not np.isfinite(latest_vol):
        return out
    pct = float((hist <= latest_vol).mean())
    high_th = _cfg_float(cfg, "vol_high_percentile", 0.75)
    low_th = _cfg_float(cfg, "vol_low_percentile", 0.25)
    if pct > high_th:
        regime = "HIGH"
        mult = _cfg_float(cfg, "vol_high_multiplier", 0.5)
    elif pct < low_th:
        regime = "LOW"
        mult = _cfg_float(cfg, "vol_low_multiplier", 1.2)
    else:
        regime = "NORMAL"
        mult = 1.0
    out.update(
        {
            "vol_regime": regime,
            "vol_percentile": pct,
            "vol_multiplier": float(mult),
            "rolling_vol_20d": latest_vol,
        }
    )
    return out


def _conviction_state(
    price: float,
    ema_mid: float,
    ema_slow: float,
    regime: str,
    vol_regime: str,
    cfg: dict[str, object],
    enabled: bool,
) -> dict[str, object]:
    out = {
        "conviction_score": np.nan,
        "conviction_bucket": "OFF",
        "conviction_multiplier": 1.0,
        "conviction_gap_score": np.nan,
        "conviction_regime_score": np.nan,
        "conviction_vol_score": np.nan,
    }
    if not enabled:
        return out
    px = _safe_num(price)
    mid = _safe_num(ema_mid)
    slow = _safe_num(ema_slow)
    if not (np.isfinite(px) and px > 0 and np.isfinite(mid) and np.isfinite(slow)):
        return out
    gap_pct = float((mid - slow) / px)
    if gap_pct > 0.02:
        f1 = 0.33
    elif gap_pct > 0.01:
        f1 = 0.20
    else:
        f1 = 0.10
    reg = str(regime).upper()
    f2 = 0.33 if reg == "BULL" else 0.17 if reg == "CHOP" else 0.0
    vol = str(vol_regime).upper()
    f3 = 0.33 if vol == "LOW" else 0.10 if vol == "HIGH" else 0.20
    score = float(f1 + f2 + f3)
    high_th = _cfg_float(cfg, "conviction_high_threshold", 0.80)
    mid_th = _cfg_float(cfg, "conviction_mid_threshold", 0.60)
    low_th = _cfg_float(cfg, "conviction_low_threshold", 0.40)
    if score > high_th:
        bucket = "HIGH"
        mult = _cfg_float(cfg, "conviction_high_multiplier", 1.2)
    elif score >= mid_th:
        bucket = "MID"
        mult = _cfg_float(cfg, "conviction_mid_multiplier", 1.0)
    elif score >= low_th:
        bucket = "LOW"
        mult = _cfg_float(cfg, "conviction_low_multiplier", 0.8)
    else:
        bucket = "FLOOR"
        mult = _cfg_float(cfg, "conviction_floor_multiplier", 0.6)
    out.update(
        {
            "conviction_score": score,
            "conviction_bucket": bucket,
            "conviction_multiplier": float(mult),
            "conviction_gap_score": f1,
            "conviction_regime_score": f2,
            "conviction_vol_score": f3,
        }
    )
    return out


def _transition_state(
    daily_close: pd.Series,
    regime: pd.DataFrame,
    regime_col: str,
    paper_start_ts: pd.Timestamp,
    entry_ts: pd.Timestamp,
    hold_return: float,
    cfg: dict[str, object],
) -> dict[str, object]:
    out = {
        "transition_strength": "NA",
        "transition_multiplier": 1.0,
        "transition_return": np.nan,
        "transition_flip_date": "",
    }
    if pd.isna(entry_ts) or daily_close.empty or regime.empty or regime_col not in regime.columns:
        return out
    rg = regime.copy()
    rg["day"] = rg["timestamp"].dt.floor("D")
    rg = rg[rg["day"] >= paper_start_ts].sort_values("day").drop_duplicates("day", keep="last")
    if rg.empty:
        return out
    entry_day = entry_ts.floor("D")
    rg = rg[rg["day"] <= entry_day]
    if rg.empty:
        return out
    prev = rg[regime_col].astype(str).shift(1)
    flips = rg[(rg[regime_col].astype(str) != prev) & rg[regime_col].astype(str).isin(["BULL", "CHOP"])]
    if flips.empty:
        return out
    flip_day = pd.Timestamp(flips["day"].iloc[-1])
    c = daily_close.sort_index()
    if flip_day not in c.index:
        prior_idx = c.index[c.index <= flip_day]
        if len(prior_idx) == 0:
            return out
        flip_day = pd.Timestamp(prior_idx[-1])
    loc = c.index.get_loc(flip_day)
    if isinstance(loc, slice) or int(loc) <= 0:
        return out
    prev_px = float(c.iloc[int(loc) - 1])
    flip_px = float(c.iloc[int(loc)])
    if prev_px <= 0:
        return out
    tr = float(flip_px / prev_px - 1.0)
    strong_th = _cfg_float(cfg, "transition_strong_threshold", 0.03)
    weak_th = _cfg_float(cfg, "transition_weak_threshold", 0.01)
    scale_up_days = _cfg_int(cfg, "transition_scale_up_days", 5)
    if tr > strong_th:
        strength = "STRONG"
        mult = _cfg_float(cfg, "transition_strong_multiplier", 1.0)
    elif tr < weak_th:
        strength = "WEAK"
        days_held = max(0, int((pd.Timestamp.now("UTC").floor("D") - entry_day).days))
        positive = np.isfinite(hold_return) and hold_return > 0
        mult = 1.0 if days_held >= scale_up_days and positive else _cfg_float(cfg, "transition_weak_multiplier", 0.6)
    else:
        strength = "NORMAL"
        mult = 1.0
    out.update(
        {
            "transition_strength": strength,
            "transition_multiplier": float(mult),
            "transition_return": tr,
            "transition_flip_date": str(flip_day.date()),
        }
    )
    return out


def _mean_reversion_overlay_state(
    daily_close: pd.Series,
    regime: str,
    base_weight: float,
    gross_cap: float,
    cfg: dict[str, object],
    now: pd.Timestamp,
) -> dict[str, object]:
    enabled = _cfg_bool(cfg, "mean_reversion_overlay_enabled", False)
    out = {
        "enabled": bool(enabled),
        "active": False,
        "weight": 0.0,
        "remaining_cap": max(0.0, float(gross_cap) - max(0.0, float(base_weight) if np.isfinite(base_weight) else 0.0)),
        "z_score": np.nan,
        "daily_ret": np.nan,
        "entry_signal": False,
        "exit_signal": False,
        "days_held": 0,
        "entry_date": "",
        "exit_reason": "",
    }
    if not enabled:
        return out

    lookback = max(5, _cfg_int(cfg, "mean_reversion_lookback", 20))
    z_entry = _cfg_float(cfg, "mean_reversion_z_entry", -1.5)
    z_exit = _cfg_float(cfg, "mean_reversion_z_exit", -0.5)
    ret_entry = _cfg_float(cfg, "mean_reversion_daily_ret_entry", -0.03)
    max_hold = max(1, _cfg_int(cfg, "mean_reversion_max_hold_days", 10))
    state_path = Path(str(cfg.get("mean_reversion_state_json", "artifacts/paper_trade/mean_reversion_overlay_state.json")))

    state = _load_json_state(state_path)
    active = bool(state.get("active", False))
    entry_ts = pd.to_datetime(state.get("entry_ts", ""), utc=True, errors="coerce")
    days_held = int(_safe_num(state.get("days_held", 0))) if np.isfinite(_safe_num(state.get("days_held", 0))) else 0

    c = daily_close.dropna().astype(float).sort_index()
    if len(c) >= max(lookback, 2):
        latest = float(c.iloc[-1])
        mu = float(c.rolling(lookback, min_periods=lookback).mean().iloc[-1])
        sd = float(c.rolling(lookback, min_periods=lookback).std(ddof=0).iloc[-1])
        if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
            out["z_score"] = float((latest - mu) / sd)
        if len(c) >= 2 and float(c.iloc[-2]) > 0:
            out["daily_ret"] = float(c.iloc[-1] / c.iloc[-2] - 1.0)

    if active:
        days_held += 1

    is_chop = str(regime).upper() == "CHOP"
    z = _safe_num(out["z_score"])
    r = _safe_num(out["daily_ret"])
    entry_signal = bool((not active) and is_chop and np.isfinite(z) and np.isfinite(r) and z < z_entry and r < ret_entry)
    exit_reason = ""
    exit_signal = False
    if active:
        if not is_chop:
            exit_signal = True
            exit_reason = "REGIME_NOT_CHOP"
        elif np.isfinite(z) and z > z_exit:
            exit_signal = True
            exit_reason = "Z_RECOVERY"
        elif days_held >= max_hold:
            exit_signal = True
            exit_reason = "MAX_HOLD"

    if entry_signal:
        active = True
        days_held = 1
        entry_ts = now.floor("D")
        exit_reason = ""
    elif exit_signal:
        active = False
        days_held = 0
        entry_ts = pd.NaT

    remaining = max(0.0, float(gross_cap) - max(0.0, float(base_weight) if np.isfinite(base_weight) else 0.0))
    weight = remaining if active and is_chop else 0.0
    _save_json_state(
        state_path,
        {
            "active": bool(active),
            "entry_ts": str(entry_ts) if pd.notna(entry_ts) else "",
            "days_held": int(days_held),
            "last_z_score": float(z) if np.isfinite(z) else None,
            "last_daily_ret": float(r) if np.isfinite(r) else None,
            "last_weight": float(weight),
            "last_exit_reason": exit_reason,
            "updated_utc": str(now),
        },
    )
    out.update(
        {
            "active": bool(active),
            "weight": float(weight),
            "remaining_cap": float(remaining),
            "entry_signal": bool(entry_signal),
            "exit_signal": bool(exit_signal),
            "days_held": int(days_held),
            "entry_date": str(entry_ts.date()) if pd.notna(entry_ts) else "",
            "exit_reason": exit_reason,
        }
    )
    return out


def _latest_news_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    if d.empty:
        return {}
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        d = d.sort_values("date")
    row = d.iloc[-1].to_dict()
    return row if isinstance(row, dict) else {}


def _latest_dvol_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    if d.empty:
        return {}
    if "date" in d.columns:
        d = d.sort_values("date")
    row = d.iloc[-1].to_dict()
    return row if isinstance(row, dict) else {}


def _latest_market_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    if d.empty or "date" not in d.columns or "asset" not in d.columns:
        return []
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return []
    dmax = d["date"].max()
    w = d[d["date"] == dmax].copy()
    for c in ["price", "pct_change", "trend", "stack_aligned"]:
        if c not in w.columns:
            w[c] = np.nan
    order = ["BTC", "Oil/WTI", "Nat Gas", "Gold", "Silver", "S&P 500", "FTSE 100", "TLT", "DXY", "Wheat"]
    lines: list[str] = []
    for asset in order:
        sub = w[w["asset"].astype(str) == asset]
        if sub.empty:
            continue
        r = sub.iloc[-1]
        px = _safe_num(r.get("price", np.nan))
        pct = _safe_num(r.get("pct_change", np.nan))
        trend = str(r.get("trend", "NA"))
        aligned = str(r.get("stack_aligned", "")).strip().lower() in {"true", "1", "yes"}
        if asset in {"S&P 500", "FTSE 100", "DXY", "Wheat"}:
            px_s = _fmt_num(px, 1)
        elif asset == "BTC":
            px_s = _fmt_num(px, 0)
        else:
            px_s = _fmt_num(px, 2)
        mk = " <- stack aligned" if aligned else ""
        lines.append(f"{asset:<9} {px_s:>9}  {_fmt_pct(pct, 1):>7}  {trend}{mk}")
    return lines


def _trailing_false_days_from_log(log_path: Path, paper_start_ts: pd.Timestamp, today: str, current_value: bool) -> float:
    rows: list[dict[str, object]] = [{"date": today, "stack_aligned": bool(current_value)}]
    if log_path.exists():
        try:
            lg = pd.read_csv(log_path, usecols=["date", "stack_aligned"], low_memory=False)
            lg["date_ts"] = pd.to_datetime(lg["date"], utc=True, errors="coerce").dt.floor("D")
            lg = lg.dropna(subset=["date_ts"]).copy()
            lg = lg[lg["date_ts"] >= paper_start_ts.floor("D")]
            lg = lg[lg["date"].astype(str) != str(today)]
            rows.extend(lg[["date", "stack_aligned"]].to_dict("records"))
        except Exception:
            pass
    d = pd.DataFrame(rows)
    if d.empty:
        return np.nan
    d["date_ts"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["date_ts"]).sort_values("date_ts").drop_duplicates("date_ts", keep="last")
    vals = d["stack_aligned"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if vals.iloc[-1]:
        return 0.0
    return float((~vals.iloc[::-1]).cumprod().sum())


def _send_discord_summary(webhook_url: str, summary: dict[str, object], timeout_sec: float = 10.0) -> tuple[bool, str]:
    status = str(summary.get("status", "REVIEW"))
    color_map = {"PASS": 0x00FF00, "REVIEW": 0xFFA500, "STOP": 0xFF0000}
    color = int(color_map.get(status, 0x808080))
    regime = str(summary.get("regime", "NA"))
    regime_map = {"BULL": "BULL", "CHOP": "CHOP", "BEAR": "BEAR"}
    regime_label = regime_map.get(regime, "NA")
    paper_start = str(summary.get("paper_start_date", "NA"))
    days_live = summary.get("days_live", "NA")
    off_w_raw = _safe_num(summary.get("off_weight_raw", summary.get("off_weight", np.nan)))
    off_w_scaled = _safe_num(summary.get("off_weight_scaled", np.nan))
    def_w_raw = _safe_num(summary.get("def_weight_raw", summary.get("def_weight", np.nan)))
    def_w_scaled = _safe_num(summary.get("def_weight_scaled", np.nan))

    market_lines = [
        f"ETH: ${_fmt_num(_safe_num(summary.get('eth_price', np.nan)), 0)} ({_fmt_pct(_safe_num(summary.get('eth_24h_pct', np.nan)), 1)})",
        f"Regime: {regime_label} (streak {summary.get('regime_days', 'NA')}d)",
        f"DD/20d: {_fmt_pct(_safe_num(summary.get('dd_20d', np.nan)), 1)}",
    ]
    markets_lines = summary.get("markets_lines", [])
    if not isinstance(markets_lines, list):
        markets_lines = []
    if not markets_lines:
        markets_lines = ["No market tracker data"]
    state_lines = [
        f"Offensive: {'LONG' if bool(summary.get('off_position', False)) else 'FLAT'}",
        f"Defensive: {'ACTIVE' if bool(summary.get('def_position', False)) else 'FLAT'}",
        f"Base ETH weight: {_fmt_pct(_safe_num(summary.get('combined_weight', np.nan)), 0)}",
        f"ETH execution weight: {_fmt_pct(_safe_num(summary.get('eth_execution_weight', summary.get('combined_weight', np.nan))), 0)}",
        f"Off contribution (scaled): {_fmt_num(off_w_scaled, 4)}",
        f"Def contribution (scaled): {_fmt_num(def_w_scaled, 4)}",
        f"Vol regime: {summary.get('vol_regime', 'NA')} ({_fmt_pct(_safe_num(summary.get('vol_percentile', np.nan)), 0)} pctile)",
        f"Transition: {summary.get('transition_strength', 'NA')} x{_fmt_num(_safe_num(summary.get('transition_multiplier', np.nan)), 2)}",
        f"ETH conviction: {summary.get('conviction_bucket', 'NA')} ({_fmt_num(_safe_num(summary.get('conviction_score', np.nan)), 2)} score)",
    ]
    eth_ema_fast = summary.get("eth_ema_fast", 21)
    eth_ema_mid = summary.get("eth_ema_mid", 55)
    eth_ema_slow = summary.get("eth_ema_slow", 144)
    off_gate_lines = [
        f"EMA{eth_ema_fast}/{eth_ema_mid}/{eth_ema_slow}: {_fmt_num(_safe_num(summary.get('ema21', np.nan)), 2)} / {_fmt_num(_safe_num(summary.get('ema55', np.nan)), 2)} / {_fmt_num(_safe_num(summary.get('ema144', np.nan)), 2)}",
        f"Stack aligned ({eth_ema_fast}>{eth_ema_mid}>{eth_ema_slow}): {'YES' if bool(summary.get('stack_aligned', False)) else 'NO'}",
        f"Days aligned: {summary.get('stack_aligned_days', 'NA')}",
        f"Days since break: {summary.get('off_days_since_break', 'NA')}",
        f"Entry threshold met: {'YES' if bool(summary.get('entry_threshold_met', False)) else 'NO'}",
        f"Raw signal: {_fmt_num(off_w_raw, 4)}",
        f"Scaled weight: {_fmt_num(off_w_scaled, 4)}",
    ]
    def_gate_lines = [
        f"Mode: {summary.get('defense_live_mode', 'NA')}",
        f"Signal ts: {summary.get('def_signal_ts', 'NA')}",
        f"P(up): {_fmt_num(_safe_num(summary.get('def_p_up', np.nan)), 4)}",
        f"Signal active: {'YES' if bool(summary.get('def_position', False)) else 'NO'}",
        f"Raw signal: {_fmt_num(def_w_raw, 4)}",
        f"Scaled weight: {_fmt_num(def_w_scaled, 4)}",
    ]
    gold_ema_default = _cfg_int_list(
        _load_portfolio_config(Path("config/portfolio_config.json")),
        "gold_ema",
        [21, 55, 144],
    )
    gold_ema_fast = int(summary.get("gold_ema_fast", gold_ema_default[0]))
    gold_ema_mid = int(summary.get("gold_ema_mid", gold_ema_default[1]))
    gold_ema_slow = int(summary.get("gold_ema_slow", gold_ema_default[2]))
    gold_lines = [
        f"PAXG: ${_fmt_num(_safe_num(summary.get('gold_price', np.nan)), 2)} ({_fmt_pct(_safe_num(summary.get('gold_24h_pct', np.nan)), 2)})",
        f"EMA{gold_ema_fast}/{gold_ema_mid}/{gold_ema_slow}: {_fmt_num(_safe_num(summary.get('gold_ema21', np.nan)), 2)} / {_fmt_num(_safe_num(summary.get('gold_ema55', np.nan)), 2)} / {_fmt_num(_safe_num(summary.get('gold_ema144', np.nan)), 2)}",
        f"EMA gate: {'YES' if bool(summary.get('gold_entry_threshold_met', False)) else 'NO'}",
        f"Flat+BEAR gate: {'YES' if bool(summary.get('gold_flat_bear_gate', False)) else 'NO'}",
        f"Condition met: {'YES' if bool(summary.get('gold_condition_met', False)) else 'NO'}",
        f"Paper sleeve: {'ACTIVE' if bool(summary.get('gold_position_active', False)) else 'FLAT'}",
        f"Paper return: {_fmt_pct(_safe_num(summary.get('gold_paper_return', np.nan)), 2)} | P&L: GBP {_fmt_num(_safe_num(summary.get('gold_paper_pnl_gbp', np.nan)), 2)}",
    ]
    mean_rev_lines = [
        f"Enabled: {'YES' if bool(summary.get('mean_reversion_enabled', False)) else 'NO'}",
        f"Status: {'ACTIVE' if bool(summary.get('mean_reversion_active', False)) else 'FLAT'}",
        f"Weight: {_fmt_pct(_safe_num(summary.get('mean_reversion_weight', np.nan)), 0)}",
        f"Remaining cap: {_fmt_pct(_safe_num(summary.get('mean_reversion_remaining_cap', np.nan)), 0)}",
        f"Z-score: {_fmt_num(_safe_num(summary.get('mean_reversion_z_score', np.nan)), 2)}",
        f"Daily ret: {_fmt_pct(_safe_num(summary.get('mean_reversion_daily_ret', np.nan)), 2)}",
        f"Entry signal: {'YES' if bool(summary.get('mean_reversion_entry_signal', False)) else 'NO'}",
        f"Exit signal: {'YES' if bool(summary.get('mean_reversion_exit_signal', False)) else 'NO'}",
        f"Days held: {summary.get('mean_reversion_days_held', 0)}",
    ]
    dvol_iv = _safe_num(summary.get("dvol_atm_iv_30d", np.nan))
    dvol_pct = _safe_num(summary.get("dvol_iv_percentile", np.nan))
    dvol_slope = _safe_num(summary.get("dvol_term_slope", np.nan))
    dvol_slope_label = "inverted" if np.isfinite(dvol_slope) and dvol_slope < 0 else ("normal" if np.isfinite(dvol_slope) else "NA")
    dvol_history_days = int(_safe_num(summary.get("dvol_history_days", 0), 0))
    dvol_insufficient = bool(summary.get("dvol_insufficient_history", True))
    dvol_mode = "LOG ONLY" if dvol_insufficient or dvol_history_days < 30 else "READY"
    options_vol_lines = [
        f"ATM IV (30d): {_fmt_pct(dvol_iv / 100.0 if dvol_iv > 5 else dvol_iv, 1)}",
        f"IV pct: {_fmt_pct(dvol_pct, 0)} | Regime: {summary.get('dvol_options_vol_regime', 'NA')}",
        f"Term slope: {_fmt_num(dvol_slope, 1)} ({dvol_slope_label})",
        f"vs Realised: {summary.get('dvol_agreement', 'NA')}",
        f"Note: {dvol_mode} ({dvol_history_days}/30 days history)",
    ]
    btc_signal_lines = [
        f"BTC: ${_fmt_num(_safe_num(summary.get('btc_price', np.nan)), 0)} ({_fmt_pct(_safe_num(summary.get('btc_24h_pct', np.nan)), 1)})",
        f"EMA15/40/120: {_fmt_num(_safe_num(summary.get('btc_ema15', summary.get('btc_ema21', np.nan))), 2)} / {_fmt_num(_safe_num(summary.get('btc_ema40', summary.get('btc_ema55', np.nan))), 2)} / {_fmt_num(_safe_num(summary.get('btc_ema120', summary.get('btc_ema144', np.nan))), 2)}",
        f"Stack aligned: {'YES' if bool(summary.get('btc_stack_aligned', False)) else 'NO'}",
        f"Days aligned: {summary.get('btc_stack_aligned_days', 'NA')}",
        f"Entry threshold met: {'YES' if bool(summary.get('btc_entry_threshold_met', False)) else 'NO'}",
        f"Regime: {summary.get('btc_regime', regime_label)}",
        f"Def signal (funding z): {_fmt_num(_safe_num(summary.get('btc_funding_z', np.nan)), 3)}",
        f"Raw signal: {_fmt_num(_safe_num(summary.get('btc_off_weight_raw', np.nan)), 4)}",
        f"Scaled weight: {_fmt_num(_safe_num(summary.get('btc_combined_weight', np.nan)), 4)}",
        f"Transition: {summary.get('btc_transition_strength', 'NA')} x{_fmt_num(_safe_num(summary.get('btc_transition_multiplier', np.nan)), 2)}",
        f"Conviction: {summary.get('btc_conviction_bucket', 'NA')} ({_fmt_num(_safe_num(summary.get('btc_conviction_score', np.nan)), 2)} score)",
    ]
    perf_lines = [
        f"Start: {paper_start} | days: {days_live}",
        f"ETH sleeve strategy: {_fmt_pct(_safe_num(summary.get('eth_sleeve_strategy_ret', summary.get('cum_strategy_ret', np.nan))), 2)}",
        f"ETH spot: {_fmt_pct(_safe_num(summary.get('eth_sleeve_spot_ret', summary.get('cum_spot_ret', np.nan))), 2)}",
        f"BTC sleeve strategy: {_fmt_pct(_safe_num(summary.get('btc_sleeve_strategy_ret', np.nan)), 2)}",
        f"BTC spot: {_fmt_pct(_safe_num(summary.get('btc_sleeve_spot_ret', np.nan)), 2)}",
        f"Combined strategy: {_fmt_pct(_safe_num(summary.get('portfolio_strategy_ret', summary.get('cum_strategy_ret', np.nan))), 2)}",
        f"Excess vs 50/50 basket: {_fmt_pct(_safe_num(summary.get('portfolio_excess_vs_basket', summary.get('excess', np.nan))), 2)}",
        f"Peak DD: {_fmt_pct(_safe_num(summary.get('portfolio_peak_dd', summary.get('peak_dd', np.nan))), 2)}",
    ]
    news_lines = [str(summary.get("news_alert_line", "News: No major macro events"))]
    health_lines = [
        f"Status: **{status}**",
        str(summary.get("flags_text", "No flags")),
    ]
    live_lines: list[str] = []
    if bool(summary.get("live_mode_requested", False)):
        live_lines = [
            f"Mode: {'DRY RUN' if bool(summary.get('live_execution_dry_run', True)) else 'REAL'}",
            f"ETH: {summary.get('live_eth_status', 'NA')}",
            f"BTC: {summary.get('live_btc_status', 'NA')}",
            f"Fees: ${_fmt_num(_safe_num(summary.get('live_total_fees_usd', np.nan)), 2)}",
        ]
        if str(summary.get("live_execution_error", "")).strip():
            live_lines.append(f"Error: {summary.get('live_execution_error')}")

    fields = [
        {"name": "Market", "value": "\n".join(market_lines), "inline": False},
        {"name": "Markets", "value": "\n".join(markets_lines), "inline": False},
        {"name": "Strategy State", "value": "\n".join(state_lines), "inline": False},
        {"name": "Offensive Signal", "value": "\n".join(off_gate_lines), "inline": False},
        {"name": "Defensive Signal", "value": "\n".join(def_gate_lines), "inline": False},
        {"name": "BTC Signal", "value": "\n".join(btc_signal_lines), "inline": False},
        {"name": "Gold Sleeve (Paper)", "value": "\n".join(gold_lines), "inline": False},
        {"name": "Mean-Reversion Overlay", "value": "\n".join(mean_rev_lines), "inline": False},
        {"name": f"Options Vol ({dvol_mode})", "value": "\n".join(options_vol_lines), "inline": False},
        {"name": "Performance (paper)", "value": "\n".join(perf_lines), "inline": False},
        {"name": "News", "value": "\n".join(news_lines), "inline": False},
        {"name": "Health", "value": "\n".join(health_lines), "inline": False},
    ]
    if live_lines:
        fields.insert(-2, {"name": "Live Execution", "value": "\n".join(live_lines), "inline": False})

    if bool(summary.get("off_position", False)):
        fields.insert(
            2,
            {
                "name": "Active Offensive Trade",
                "value": "\n".join(
                    [
                        f"Entry: {summary.get('off_entry_date', 'NA')}",
                        f"Days held: {summary.get('off_days_held', 'NA')}",
                        f"Return: {_fmt_pct(_safe_num(summary.get('off_trade_ret', np.nan)), 2)}",
                    ]
                ),
                "inline": False,
            },
        )
    elif bool(summary.get("off_last_trade_closed", False)):
        fields.insert(
            2,
            {
                "name": "Last Offensive Trade (CLOSED)",
                "value": "\n".join(
                    [
                        f"Entry: {summary.get('off_last_entry_date', 'NA')} at ${_fmt_num(_safe_num(summary.get('off_last_entry_price', np.nan)), 0)}",
                        f"Exit: {summary.get('off_last_exit_date', 'NA')} at ${_fmt_num(_safe_num(summary.get('off_last_exit_price', np.nan)), 0)}",
                        f"Reason: {summary.get('off_last_exit_reason', 'NA')}",
                        f"Return: {_fmt_pct(_safe_num(summary.get('off_last_return_pct', np.nan)), 2)}",
                        f"Days held: {summary.get('off_last_days_held', 'NA')}",
                        "Next Entry: waiting for signal",
                    ]
                ),
                "inline": False,
            },
        )

    if bool(summary.get("gold_position_active", False)):
        fields.insert(
            6,
            {
                "name": "Active Gold Trade (Paper)",
                "value": "\n".join(
                    [
                        f"Entry: {summary.get('gold_entry_date', 'NA')}",
                        f"Days held: {summary.get('gold_days_held', 'NA')}",
                        f"Return: {_fmt_pct(_safe_num(summary.get('gold_return_since_entry', np.nan)), 2)}",
                    ]
                ),
                "inline": False,
            },
        )

    if bool(summary.get("btc_position", False)):
        fields.insert(
            3,
            {
                "name": "Active BTC Trade",
                "value": "\n".join(
                    [
                        f"Entry: {summary.get('btc_entry_date', 'NA')}",
                        f"Days held: {summary.get('btc_days_held', 'NA')}",
                        f"Return: {_fmt_pct(_safe_num(summary.get('btc_trade_ret', np.nan)), 2)}",
                    ]
                ),
                "inline": False,
            },
        )

    embed = {
        "title": f"LPBot Daily Summary - {pd.Timestamp.now('UTC').strftime('%Y-%m-%d')}",
        "color": color,
        "fields": fields,
        "footer": {"text": "LPBot Paper Trading | Hetzner"},
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }
    payload = {"embeds": [embed]}

    req = request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # Discord webhooks can be blocked by Cloudflare without a UA header.
            "User-Agent": "LPBot-PaperCheck/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            code = int(getattr(resp, "status", 0))
            if code in (200, 204):
                return True, "discord_sent"
            return False, f"discord_http_{code}"
    except error.HTTPError as e:
        return False, f"discord_http_{int(e.code)}"
    except Exception as e:
        return False, f"discord_error_{type(e).__name__}"


def _run() -> int:
    p = argparse.ArgumentParser(description="Daily paper-trade checklist for promoted RouterA strategy.")
    p.add_argument("--portfolio-config", default="config/portfolio_config.json")
    p.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    p.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_6y.csv")
    p.add_argument("--regime-csv", default="artifacts/paper_trade/regime_snapshot_immutable.csv")
    p.add_argument("--regime-col", default="regime_v2")
    p.add_argument("--offense-log-csv", default="artifacts/backtest/offense_trend_follow_v1a_6y_ema21_55_144.csv")
    p.add_argument("--defense-log-csv", default="artifacts/paper_trade/defensive_live.csv")
    p.add_argument("--combined-log-csv", default="artifacts/backtest/combined_offtf_ema21_55_144_defv1_routerA_6y.csv")
    p.add_argument("--defense-live-mode", choices=["model", "artifact"], default="artifact")
    p.add_argument("--defense-script", default="scripts/direction_event_model_v1.py")
    p.add_argument("--defense-live-out-csv", default="artifacts/paper_trade/defensive_live.csv")
    p.add_argument("--defense-live-preds-csv", default="artifacts/paper_trade/defensive_live_preds.csv")
    p.add_argument("--defense-live-sweep-csv", default="artifacts/paper_trade/defensive_live_sweep.csv")
    p.add_argument("--def-perp-csv", default="data/backtest/ETH_perp_features_5m_6y.csv")
    p.add_argument("--def-window-days", type=int, default=1825)
    p.add_argument("--def-horizon-bars", type=int, default=288)
    p.add_argument("--def-breakout-lookback", type=int, default=96)
    p.add_argument("--def-vol-event-k", type=float, default=1.5)
    p.add_argument("--def-basis-z-th", type=float, default=1.5)
    p.add_argument("--def-funding-z-th", type=float, default=1.5)
    p.add_argument("--def-oi-z-th", type=float, default=1.5)
    p.add_argument("--def-wf-train-days", type=int, default=240)
    p.add_argument("--def-wf-test-days", type=int, default=30)
    p.add_argument("--def-wf-step-days", type=int, default=30)
    p.add_argument("--def-long-thresholds", default="0.80")
    p.add_argument("--def-short-thresholds", default="0.20")
    p.add_argument("--def-hold-bars-list", default="24")
    p.add_argument("--def-max-weight", type=float, default=1.0)
    p.add_argument("--def-trade-cost-bps", type=float, default=5.0)
    p.add_argument("--def-min-time-in-market-pct", type=float, default=0.0)
    p.add_argument("--def-timeout-sec", type=int, default=900)
    p.add_argument("--out-dir", default="artifacts/paper_trade")
    p.add_argument("--data-fresh-hours", type=float, default=2.0)
    p.add_argument("--funding-fresh-hours", type=float, default=9.0)
    p.add_argument("--max-gap-minutes", type=float, default=15.0)
    p.add_argument("--gap-lookback-hours", type=float, default=48.0)
    p.add_argument("--regime-stuck-days", type=int, default=60)
    p.add_argument("--stop-rolling-sharpe", type=float, default=-0.5)
    p.add_argument("--stop-drawdown", type=float, default=-0.25)
    p.add_argument("--review-chop-bps", type=float, default=-0.06)
    p.add_argument("--review-dd20", type=float, default=-0.15)
    p.add_argument("--discord-webhook", default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    p.add_argument("--discord-timeout-sec", type=float, default=10.0)
    p.add_argument("--live", action="store_true", help="Call Kraken execution engine. Still dry-run unless LIVE_TRADING_ENABLED=true.")
    p.add_argument("--off-ema-fast", type=int, default=21)
    p.add_argument("--off-ema-mid", type=int, default=55)
    p.add_argument("--off-ema-slow", type=int, default=144)
    p.add_argument("--off-confirm-days", type=int, default=3)
    p.add_argument("--off-exit-confirm-days", type=int, default=3)
    p.add_argument("--off-min-hold-days", type=int, default=0)
    p.add_argument("--off-target-vol", type=float, default=0.50)
    p.add_argument("--off-vol-window", type=int, default=20)
    p.add_argument("--off-vol-floor", type=float, default=0.25)
    p.add_argument("--off-vol-cap", type=float, default=1.00)
    p.add_argument("--btc-daily-csv", default="data/btc_daily.csv")
    p.add_argument("--btc-perp-csv", default="data/btc_perp_features.csv")
    p.add_argument("--btc-ema-fast", type=int, default=21)
    p.add_argument("--btc-ema-mid", type=int, default=55)
    p.add_argument("--btc-ema-slow", type=int, default=144)
    p.add_argument("--btc-confirm-days", type=int, default=5)
    p.add_argument("--btc-exit-confirm-days", type=int, default=3)
    p.add_argument("--btc-min-hold-days", type=int, default=0)
    p.add_argument("--btc-target-vol", type=float, default=0.50)
    p.add_argument("--btc-vol-window", type=int, default=20)
    p.add_argument("--btc-vol-floor", type=float, default=0.25)
    p.add_argument("--btc-vol-cap", type=float, default=1.00)
    p.add_argument("--eth-sleeve-capital", type=float, default=0.5)
    p.add_argument("--btc-sleeve-capital", type=float, default=0.5)
    p.add_argument("--router-bull-off", type=float, default=0.8)
    p.add_argument("--router-chop-off", type=float, default=0.4)
    p.add_argument("--router-bear-off", type=float, default=0.0)
    p.add_argument("--router-bull-def", type=float, default=0.0)
    p.add_argument("--router-chop-def", type=float, default=0.3)
    p.add_argument("--router-bear-def", type=float, default=0.8)
    p.add_argument("--gold-price-csv", "--paxg-csv", dest="gold_price_csv", default="data/etf/PAXGUSDT_binance_daily.csv")
    p.add_argument("--gold-symbol", default="PAXGUSDT")
    p.add_argument("--gold-fetch-limit", type=int, default=800)
    p.add_argument("--gold-fetch-timeout-sec", type=float, default=10.0)
    p.add_argument("--gold-ema-fast", type=int, default=21)
    p.add_argument("--gold-ema-mid", type=int, default=55)
    p.add_argument("--gold-ema-slow", type=int, default=144)
    p.add_argument("--gold-confirm-days", type=int, default=3)
    p.add_argument("--gold-notional-gbp", type=float, default=10000.0)
    p.add_argument("--gold-state-json", default="artifacts/paper_trade/gold_sleeve_state.json")
    p.add_argument("--news-log-csv", default="artifacts/news/news_log.csv")
    p.add_argument("--dvol-log-csv", default="artifacts/options/dvol_log.csv")
    p.add_argument("--market-log-csv", default="artifacts/markets/market_tracker.csv")
    p.add_argument(
        "--paper-start-date",
        default="",
        help="Paper inception date (YYYY-MM-DD or timestamp). If omitted, uses first date in daily_checks_log.csv, else today.",
    )
    p.add_argument("--completed-trades-csv", default="")
    args = p.parse_args()

    portfolio_cfg_path = Path(args.portfolio_config)
    portfolio_cfg = _load_portfolio_config(portfolio_cfg_path)
    if not portfolio_cfg and str(portfolio_cfg_path).replace("\\", "/") != "artifacts/backtest/portfolio_config.json":
        portfolio_cfg_path = Path("artifacts/backtest/portfolio_config.json")
        portfolio_cfg = _load_portfolio_config(portfolio_cfg_path)
    if portfolio_cfg:
        args.off_confirm_days = _cfg_int(portfolio_cfg, "eth_confirm_days", int(args.off_confirm_days))
        args.btc_confirm_days = _cfg_int(portfolio_cfg, "btc_confirm_days", int(args.btc_confirm_days))
        eth_ema = _cfg_int_list(
            portfolio_cfg,
            "eth_ema",
            [int(args.off_ema_fast), int(args.off_ema_mid), int(args.off_ema_slow)],
        )
        btc_ema = _cfg_int_list(
            portfolio_cfg,
            "btc_ema",
            [int(args.btc_ema_fast), int(args.btc_ema_mid), int(args.btc_ema_slow)],
        )
        args.off_ema_fast, args.off_ema_mid, args.off_ema_slow = eth_ema
        args.btc_ema_fast, args.btc_ema_mid, args.btc_ema_slow = btc_ema
        gold_ema = _cfg_int_list(
            portfolio_cfg,
            "gold_ema",
            [int(args.gold_ema_fast), int(args.gold_ema_mid), int(args.gold_ema_slow)],
        )
        args.gold_ema_fast, args.gold_ema_mid, args.gold_ema_slow = gold_ema

    now = pd.Timestamp.now("UTC")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paper inception date
    paper_start_ts = pd.to_datetime(args.paper_start_date, utc=True, errors="coerce") if args.paper_start_date else pd.NaT
    if pd.notna(paper_start_ts):
        paper_start_ts = paper_start_ts.floor("D")
    if pd.isna(paper_start_ts):
        prior_log = out_dir / "daily_checks_log.csv"
        if prior_log.exists():
            try:
                lg0 = pd.read_csv(prior_log, usecols=["date"])
                d0 = pd.to_datetime(lg0["date"], utc=True, errors="coerce").dropna()
                if len(d0):
                    paper_start_ts = d0.min().floor("D")
            except Exception:
                pass
    if pd.isna(paper_start_ts):
        paper_start_ts = now.floor("D")

    vol_filter_enabled = _cfg_bool(portfolio_cfg, "vol_filter", False)
    transition_momentum_enabled = _cfg_bool(portfolio_cfg, "transition_momentum", False)
    asymmetric_sizing_enabled = _cfg_bool(portfolio_cfg, "asymmetric_sizing", False)
    gross_cap = _cfg_float(portfolio_cfg, "gross_cap", 1.0)

    # Load inputs
    price = _read_csv_ts(Path(args.price_csv))
    funding = _read_csv_ts(Path(args.funding_csv))
    regime = _read_csv_ts(Path(args.regime_csv))
    off = _read_csv_ts(Path(args.offense_log_csv))
    deff = _read_csv_ts(Path(args.defense_log_csv))
    comb = _read_csv_ts(Path(args.combined_log_csv))
    comb_perf = comb[comb["timestamp"] >= paper_start_ts].copy() if len(comb) else comb.copy()

    flags: list[str] = []
    reviews: list[str] = []
    stops: list[str] = []

    # 1) Data freshness + gap checks
    price_last = price["timestamp"].iloc[-1] if len(price) else pd.NaT
    funding_last = funding["timestamp"].iloc[-1] if len(funding) else pd.NaT
    price_age_h = float((now - price_last).total_seconds() / 3600.0) if pd.notna(price_last) else float("nan")
    funding_age_h = float((now - funding_last).total_seconds() / 3600.0) if pd.notna(funding_last) else float("nan")

    price_fresh = bool(np.isfinite(price_age_h) and price_age_h <= float(args.data_fresh_hours))
    funding_fresh = bool(np.isfinite(funding_age_h) and funding_age_h <= float(args.funding_fresh_hours))
    if not price_fresh:
        flags.append("price_not_fresh")
    if not funding_fresh:
        flags.append("funding_not_fresh")

    gap_count = np.nan
    max_gap_min = np.nan
    if len(price):
        start_gap = now - pd.Timedelta(hours=float(args.gap_lookback_hours))
        pw = price[price["timestamp"] >= start_gap]
        if len(pw) >= 2:
            gaps = pw["timestamp"].diff().dt.total_seconds().dropna() / 60.0
            gap_count = int((gaps > float(args.max_gap_minutes)).sum())
            max_gap_min = float(gaps.max()) if len(gaps) else float("nan")
            if gap_count > 0:
                flags.append("ohlcv_gap_detected")

    # 2) Regime + EMA + DD(20d)
    current_regime = "NA"
    regime_days = np.nan
    if len(regime) and args.regime_col in regime.columns:
        regime["day"] = regime["timestamp"].dt.floor("D")
        rg = regime.sort_values("day").drop_duplicates("day", keep="last").copy()
        current_regime = str(rg[args.regime_col].iloc[-1])
        # Regime day counter is scoped to paper period to avoid counting historical backtest runs.
        rg_live = rg[rg["day"] >= paper_start_ts].copy()
        if len(rg_live) == 0:
            rg_live = rg
        rev = rg_live[args.regime_col].astype(str).iloc[::-1]
        regime_days = int((rev == current_regime).cumprod().sum())
        if int(regime_days) > int(args.regime_stuck_days):
            reviews.append("regime_stuck_gt_threshold")

    ema21 = ema55 = ema144 = np.nan
    ema_state = "NA"
    stack_aligned = False
    stack_aligned_days = np.nan
    entry_threshold_met = False
    dd_20d = np.nan
    eth_price = np.nan
    eth_24h_pct = np.nan
    daily_close = pd.Series(dtype=float)
    vol_state = {"vol_regime": "NA", "vol_percentile": np.nan, "vol_multiplier": 1.0, "rolling_vol_20d": np.nan}
    if len(price):
        cc = _close_col(price)
        if cc is not None:
            pdaily = (
                price.assign(day=price["timestamp"].dt.floor("D"))
                .groupby("day", as_index=False)[cc]
                .last()
                .rename(columns={cc: "close"})
            )
            c = pd.to_numeric(pdaily["close"], errors="coerce")
            daily_close = pd.Series(c.to_numpy(dtype=float), index=pdaily["day"])
            ema21_s = c.ewm(span=int(args.off_ema_fast), adjust=False).mean()
            ema55_s = c.ewm(span=int(args.off_ema_mid), adjust=False).mean()
            ema144_s = c.ewm(span=int(args.off_ema_slow), adjust=False).mean()
            stack = (ema21_s > ema55_s) & (ema55_s > ema144_s)
            stack_confirm = stack.rolling(int(args.off_confirm_days), min_periods=int(args.off_confirm_days)).min() == 1
            ema21 = float(ema21_s.iloc[-1])
            ema55 = float(ema55_s.iloc[-1])
            ema144 = float(ema144_s.iloc[-1])
            stack_aligned = bool(stack.iloc[-1])
            rev_stack = stack.astype(int).iloc[::-1]
            stack_aligned_days = int(rev_stack.cumprod().sum()) if len(rev_stack) else np.nan
            entry_threshold_met = bool(stack_confirm.iloc[-1]) if len(stack_confirm) else False
            if ema21 > ema55 > ema144:
                ema_state = f"bullish_{int(args.off_ema_fast)}>{int(args.off_ema_mid)}>{int(args.off_ema_slow)}"
            elif ema21 < ema55 < ema144:
                ema_state = f"bearish_{int(args.off_ema_fast)}<{int(args.off_ema_mid)}<{int(args.off_ema_slow)}"
            else:
                ema_state = "mixed"

            dd_20d = float(c.iloc[-1] / c.rolling(20).max().iloc[-1] - 1.0) if len(c) >= 20 else np.nan
            if np.isfinite(dd_20d) and dd_20d < float(args.review_dd20):
                reviews.append("dd20_below_review_threshold")

            p2 = price[["timestamp", cc]].copy()
            p2[cc] = pd.to_numeric(p2[cc], errors="coerce")
            p2 = p2.dropna(subset=[cc])
            if len(p2):
                eth_price = float(p2[cc].iloc[-1])
                ts_24h = p2["timestamp"].iloc[-1] - pd.Timedelta(hours=24)
                prev = p2[p2["timestamp"] <= ts_24h]
                if len(prev):
                    c0 = float(prev[cc].iloc[-1])
                    if c0 > 0:
                        eth_24h_pct = float(eth_price / c0 - 1.0)
            min_vol_days = max(60, _cfg_int(portfolio_cfg, "vol_rank_window", 252))
            if len(daily_close.dropna()) < min_vol_days:
                fetched_daily = _fetch_binance_daily_close("ETHUSDC", limit=max(800, min_vol_days + 50))
                if len(fetched_daily):
                    fetched_daily["day"] = fetched_daily["timestamp"].dt.floor("D")
                    daily_close = pd.Series(
                        pd.to_numeric(fetched_daily["close"], errors="coerce").to_numpy(dtype=float),
                        index=fetched_daily["day"],
                    ).dropna()
            if vol_filter_enabled:
                vol_state = _vol_filter_state(daily_close, portfolio_cfg)

    # 3) Signal states (offensive = computed live from price)
    off_w = 0.0
    off_pos = False
    off_entry_ts = pd.NaT
    off_hold_ret = np.nan
    off_days_held = np.nan
    off_stack_broken = False
    off_days_since_break = np.nan
    off_exit_confirmed = False
    off_closed_today = False
    off_close_reason = ""
    off_close_entry_ts = pd.NaT
    off_close_entry_price = np.nan
    off_close_exit_ts = pd.NaT
    off_close_exit_price = np.nan
    off_close_days_held = np.nan
    off_close_return = np.nan
    if len(daily_close):
        dc = daily_close.dropna().sort_index()
        if len(dc) >= 5:
            e_fast = dc.ewm(span=int(args.off_ema_fast), adjust=False).mean()
            e_mid = dc.ewm(span=int(args.off_ema_mid), adjust=False).mean()
            e_slow = dc.ewm(span=int(args.off_ema_slow), adjust=False).mean()
            trend_aligned_s = (e_fast > e_mid) & (e_mid > e_slow)
            trend_confirmed_s = trend_aligned_s.rolling(int(args.off_confirm_days), min_periods=int(args.off_confirm_days)).min() == 1
            trend_broken_s = ~trend_aligned_s
            exit_confirmed_s = trend_broken_s.rolling(
                int(args.off_exit_confirm_days), min_periods=int(args.off_exit_confirm_days)
            ).min() == 1
            off_stack_broken = bool(trend_broken_s.iloc[-1]) if len(trend_broken_s) else False
            if off_stack_broken:
                rev_break = trend_broken_s.astype(int).iloc[::-1]
                off_days_since_break = int(rev_break.cumprod().sum()) if len(rev_break) else 0
            else:
                off_days_since_break = 0
            off_exit_confirmed = bool(exit_confirmed_s.iloc[-1]) if len(exit_confirmed_s) else False

            daily_r = np.log(dc / dc.shift(1)).fillna(0.0)
            rv = daily_r.rolling(int(args.off_vol_window), min_periods=max(5, int(args.off_vol_window) // 2)).std() * np.sqrt(365.0)
            vol_scalar_s = (float(args.off_target_vol) / (rv.replace(0.0, np.nan))).clip(
                lower=float(args.off_vol_floor), upper=float(args.off_vol_cap)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            pos_state = 0
            days_in_pos = 0
            pos_state_vals: list[int] = []
            for day in dc.index:
                if pos_state == 1:
                    days_in_pos += 1
                else:
                    days_in_pos = 0

                if pos_state == 0:
                    if bool(trend_confirmed_s.loc[day]):
                        # Entry is a close-of-day decision once confirmation is met.
                        pos_state = 1
                        days_in_pos = 1
                else:
                    can_exit = days_in_pos >= int(args.off_min_hold_days)
                    if bool(exit_confirmed_s.loc[day]) and can_exit:
                        # Exit is also applied on the current close-of-day state.
                        pos_state = 0
                        days_in_pos = 0

                pos_state_vals.append(int(pos_state))

            pos_state_s = pd.Series(pos_state_vals, index=dc.index).astype(int)
            starts = pos_state_s[(pos_state_s == 1) & (pos_state_s.shift(1, fill_value=0) == 0)]
            exits = pos_state_s[(pos_state_s == 0) & (pos_state_s.shift(1, fill_value=0) == 1)]

            off_weight_daily = pos_state_s.astype(float) * vol_scalar_s
            off_w = float(off_weight_daily.iloc[-1]) if len(off_weight_daily) else 0.0
            off_pos = bool(pos_state_s.iloc[-1] == 1) if len(pos_state_s) else False
            if off_pos and len(starts):
                off_entry_ts = pd.Timestamp(starts.index[-1])
                if off_entry_ts.tzinfo is None:
                    off_entry_ts = off_entry_ts.tz_localize("UTC")
                else:
                    off_entry_ts = off_entry_ts.tz_convert("UTC")
                c0 = _safe_num(dc.loc[starts.index[-1]])
                c1 = _safe_num(dc.iloc[-1])
                if np.isfinite(c0) and c0 > 0 and np.isfinite(c1):
                    off_hold_ret = float(c1 / c0 - 1.0)
                off_days_held = float((now - off_entry_ts).total_seconds() / 86400.0)

            # EMA-break close today (close-of-day) if signal engine transitioned 1 -> 0 on latest bar.
            if len(exits):
                last_exit_day = pd.Timestamp(exits.index[-1])
                if last_exit_day.floor("D") == pd.Timestamp(dc.index[-1]).floor("D"):
                    ent_candidates = starts[starts.index <= last_exit_day]
                    if len(ent_candidates):
                        ent_day = pd.Timestamp(ent_candidates.index[-1])
                        c0 = _safe_num(dc.loc[ent_day])
                        c1 = _safe_num(dc.loc[last_exit_day])
                        if np.isfinite(c0) and c0 > 0 and np.isfinite(c1):
                            off_close_return = float(c1 / c0 - 1.0)
                        off_close_entry_ts = ent_day.tz_localize("UTC") if ent_day.tzinfo is None else ent_day.tz_convert("UTC")
                        off_close_exit_ts = last_exit_day.tz_localize("UTC") if last_exit_day.tzinfo is None else last_exit_day.tz_convert("UTC")
                        off_close_entry_price = c0
                        off_close_exit_price = c1
                        off_close_days_held = float((off_close_exit_ts - off_close_entry_ts).total_seconds() / 86400.0)
                        off_close_reason = "EMA_BREAK_3D"
                        off_closed_today = True

            # Immediate execution-state close overrides: BEAR regime or DD override.
            forced_reason = ""
            if off_pos and str(current_regime) == "BEAR":
                forced_reason = "BEAR_REGIME"
            elif off_pos and np.isfinite(dd_20d) and dd_20d <= float(args.review_dd20):
                forced_reason = "DD_OVERRIDE"
            if forced_reason:
                if pd.notna(off_entry_ts):
                    c0 = np.nan
                    try:
                        c0 = _safe_num(dc.loc[pd.Timestamp(off_entry_ts)])
                    except Exception:
                        try:
                            alt = pd.Timestamp(off_entry_ts).tz_convert(None) if pd.Timestamp(off_entry_ts).tzinfo is not None else pd.Timestamp(off_entry_ts)
                            c0 = _safe_num(dc.loc[alt])
                        except Exception:
                            c0 = np.nan
                    c1 = _safe_num(dc.iloc[-1])
                    if np.isfinite(c0) and c0 > 0 and np.isfinite(c1):
                        off_close_return = float(c1 / c0 - 1.0)
                    off_close_entry_ts = off_entry_ts
                    off_close_exit_ts = pd.Timestamp(dc.index[-1])
                    off_close_exit_ts = off_close_exit_ts.tz_localize("UTC") if off_close_exit_ts.tzinfo is None else off_close_exit_ts.tz_convert("UTC")
                    off_close_entry_price = c0
                    off_close_exit_price = c1
                    off_close_days_held = float((off_close_exit_ts - off_close_entry_ts).total_seconds() / 86400.0)
                    off_close_reason = forced_reason
                    off_closed_today = True
                # Execution state is flat immediately.
                off_pos = False
                off_w = 0.0
                off_entry_ts = pd.NaT
                off_hold_ret = 0.0
                off_days_held = np.nan

    def_w = np.nan
    def_pos = False
    def_signal_ts = pd.NaT
    def_p_up = np.nan
    if args.defense_live_mode == "model":
        try:
            out_live = Path(args.defense_live_out_csv)
            out_live.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(args.defense_script)),
                "--price-csv",
                str(args.price_csv),
                "--btc-csv",
                "data/BTCUSDC_5m.csv",
                "--perp-csv",
                str(args.def_perp_csv),
                "--window-days",
                str(int(args.def_window_days)),
                "--horizon-bars",
                str(int(args.def_horizon_bars)),
                "--breakout-lookback",
                str(int(args.def_breakout_lookback)),
                "--vol-event-k",
                str(float(args.def_vol_event_k)),
                "--basis-z-th",
                str(float(args.def_basis_z_th)),
                "--funding-z-th",
                str(float(args.def_funding_z_th)),
                "--oi-z-th",
                str(float(args.def_oi_z_th)),
                "--wf-train-days",
                str(int(args.def_wf_train_days)),
                "--wf-test-days",
                str(int(args.def_wf_test_days)),
                "--wf-step-days",
                str(int(args.def_wf_step_days)),
                "--long-thresholds",
                str(args.def_long_thresholds),
                "--short-thresholds",
                str(args.def_short_thresholds),
                "--hold-bars-list",
                str(args.def_hold_bars_list),
                "--max-weight",
                str(float(args.def_max_weight)),
                "--trade-cost-bps",
                str(float(args.def_trade_cost_bps)),
                "--min-time-in-market-pct",
                str(float(args.def_min_time_in_market_pct)),
                "--out-preds-csv",
                str(args.defense_live_preds_csv),
                "--out-sweep-csv",
                str(args.defense_live_sweep_csv),
                "--out-best-csv",
                str(args.defense_live_out_csv),
                "--out-html",
                str(out_dir / "defensive_live.html"),
            ]
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(args.def_timeout_sec),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                err_msg = err[-1] if err else "unknown_error"
                reviews.append(f"defense_live_compute_failed:{err_msg[:140]}")
                raise RuntimeError("defensive live compute failed")
            d_live = _read_csv_ts(out_live)
            if len(d_live):
                dwc = _weight_col(d_live)
                if dwc is not None:
                    def_w = float(pd.to_numeric(d_live[dwc], errors="coerce").fillna(0.0).iloc[-1])
                    def_pos = bool(abs(def_w) > 1e-9)
                def_signal_ts = d_live["timestamp"].iloc[-1]
                if "p_up" in d_live.columns:
                    def_p_up = _safe_num(d_live["p_up"].iloc[-1])
        except Exception:
            if not any(str(x).startswith("defense_live_compute_failed") for x in reviews):
                reviews.append("defense_live_compute_failed")

    if (not np.isfinite(def_w)) and len(deff):
        dwc = _weight_col(deff)
        if dwc is not None:
            def_w = float(pd.to_numeric(deff[dwc], errors="coerce").fillna(0.0).iloc[-1])
            def_pos = bool(abs(def_w) > 1e-9)
        if len(deff):
            def_signal_ts = deff["timestamp"].iloc[-1]
        if "p_up" in deff.columns:
            def_p_up = _safe_num(deff["p_up"].iloc[-1])

    # Combined weight (live router calc from live offense + current defensive state)
    comb_w = np.nan
    cap_bind_today = False
    off_scale_map = {"BULL": float(args.router_bull_off), "CHOP": float(args.router_chop_off), "BEAR": float(args.router_bear_off)}
    def_scale_map = {"BULL": float(args.router_bull_def), "CHOP": float(args.router_chop_def), "BEAR": float(args.router_bear_def)}
    eth_transition = _transition_state(
        daily_close=daily_close,
        regime=regime,
        regime_col=args.regime_col,
        paper_start_ts=paper_start_ts,
        entry_ts=off_entry_ts,
        hold_return=off_hold_ret,
        cfg=portfolio_cfg,
    )
    if not transition_momentum_enabled:
        eth_transition["transition_multiplier"] = 1.0
        eth_transition["transition_strength"] = "OFF"
    vol_multiplier = float(vol_state.get("vol_multiplier", 1.0)) if vol_filter_enabled else 1.0
    eth_transition_multiplier = float(eth_transition.get("transition_multiplier", 1.0))
    off_scaled = float(off_w) * float(off_scale_map.get(current_regime, 0.0))
    off_scaled = off_scaled * vol_multiplier * eth_transition_multiplier
    eth_conviction = _conviction_state(
        price=eth_price,
        ema_mid=ema55,
        ema_slow=ema144,
        regime=current_regime,
        vol_regime=str(vol_state.get("vol_regime", "NA")),
        cfg=portfolio_cfg,
        enabled=asymmetric_sizing_enabled,
    )
    off_scaled = off_scaled * float(eth_conviction.get("conviction_multiplier", 1.0))
    def_scaled = (float(def_w) if np.isfinite(def_w) else 0.0) * float(def_scale_map.get(current_regime, 0.0))
    w_raw = off_scaled + def_scaled
    comb_w = float(min(gross_cap, max(0.0, w_raw)))
    cap_bind_today = bool(w_raw > gross_cap + 1e-9)
    if cap_bind_today:
        flags.append("cap_bind_today")

    # Router sanity (on scaled sleeve weights, not raw sleeve signals)
    if current_regime == "BEAR" and abs(off_scaled) > 1e-9:
        flags.append("router_violation_offense_in_bear")
    if current_regime == "BULL" and abs(def_scaled) > 1e-9:
        flags.append("router_violation_defense_in_bull")

    # 3b) Gold sleeve (paper only): track strict conditional PAXG rotation.
    gold_price = np.nan
    gold_24h_pct = np.nan
    gold_signal_ts = pd.NaT
    gold_ema21 = np.nan
    gold_ema55 = np.nan
    gold_ema144 = np.nan
    gold_stack_aligned = False
    gold_entry_threshold_met = False
    # Gold sleeve should key off ETH offensive exposure being flat, not total combined weight.
    # Defensive sleeve can be active without implying ETH spot is long.
    gold_off_flat_gate = bool((not off_pos) or (np.isfinite(off_w) and abs(off_w) <= 1e-9))
    gold_flat_bear_gate = bool(current_regime == "BEAR" and gold_off_flat_gate)
    gold_condition_met = False
    gold_notional_gbp = float(max(args.gold_notional_gbp, 1.0))
    gold_position_active = False
    gold_entry_ts = pd.NaT
    gold_entry_price = np.nan
    gold_days_held = np.nan
    gold_return_since_entry = np.nan
    gold_trade_count = 0
    gold_paper_equity_gbp = gold_notional_gbp
    gold_paper_return = 0.0
    gold_paper_pnl_gbp = 0.0

    gold_daily = _read_daily_close_csv(Path(args.gold_price_csv))
    if len(gold_daily) == 0:
        gold_daily = _fetch_binance_daily_close(
            symbol=str(args.gold_symbol),
            limit=int(args.gold_fetch_limit),
            timeout_sec=float(args.gold_fetch_timeout_sec),
        )
    if len(gold_daily):
        g = gold_daily.sort_values("timestamp").dropna(subset=["timestamp", "close"]).copy()
        gc = pd.to_numeric(g["close"], errors="coerce").dropna()
        if len(gc):
            gold_signal_ts = g["timestamp"].iloc[-1]
            gold_price = float(gc.iloc[-1])
            if len(gc) >= 2:
                gold_24h_pct = float(gc.iloc[-1] / gc.iloc[-2] - 1.0)
            ge_fast = gc.ewm(span=int(args.gold_ema_fast), adjust=False).mean()
            ge_mid = gc.ewm(span=int(args.gold_ema_mid), adjust=False).mean()
            ge_slow = gc.ewm(span=int(args.gold_ema_slow), adjust=False).mean()
            gold_ema21 = float(ge_fast.iloc[-1])
            gold_ema55 = float(ge_mid.iloc[-1])
            gold_ema144 = float(ge_slow.iloc[-1])
            gold_stack = (ge_fast > ge_mid) & (ge_mid > ge_slow)
            gold_stack_aligned = bool(gold_stack.iloc[-1])
            gold_entry_threshold_met = bool(
                (gold_stack.rolling(int(args.gold_confirm_days), min_periods=int(args.gold_confirm_days)).min() == 1).iloc[-1]
            )
    else:
        reviews.append("gold_price_unavailable")

    gold_condition_met = bool(gold_flat_bear_gate and gold_entry_threshold_met and np.isfinite(gold_price))
    gold_state_path = Path(args.gold_state_json)
    gold_state = _load_json_state(gold_state_path)
    gold_position_active = bool(gold_state.get("active", False))
    gold_entry_ts = pd.to_datetime(gold_state.get("entry_ts", ""), utc=True, errors="coerce")
    gold_entry_price = _safe_num(gold_state.get("entry_price", np.nan))
    gold_capital_pre_entry = _safe_num(gold_state.get("capital_pre_entry", gold_notional_gbp))
    gold_paper_equity_gbp = _safe_num(gold_state.get("paper_equity", gold_notional_gbp))
    trade_count_raw = _safe_num(gold_state.get("trade_count", 0))
    gold_trade_count = int(trade_count_raw) if np.isfinite(trade_count_raw) and trade_count_raw >= 0 else 0
    if (not np.isfinite(gold_paper_equity_gbp)) or gold_paper_equity_gbp <= 0:
        gold_paper_equity_gbp = gold_notional_gbp
    if (not np.isfinite(gold_capital_pre_entry)) or gold_capital_pre_entry <= 0:
        gold_capital_pre_entry = gold_paper_equity_gbp

    if np.isfinite(gold_price):
        if gold_condition_met:
            if not gold_position_active:
                gold_position_active = True
                gold_entry_ts = gold_signal_ts if pd.notna(gold_signal_ts) else now
                gold_entry_price = float(gold_price)
                gold_capital_pre_entry = float(gold_paper_equity_gbp)
                gold_trade_count += 1
        else:
            if gold_position_active:
                if np.isfinite(gold_entry_price) and gold_entry_price > 0:
                    gold_paper_equity_gbp = float(gold_capital_pre_entry * (gold_price / gold_entry_price))
                gold_position_active = False
                gold_entry_ts = pd.NaT
                gold_entry_price = np.nan
                gold_capital_pre_entry = float(gold_paper_equity_gbp)

    if gold_position_active and np.isfinite(gold_price) and np.isfinite(gold_entry_price) and gold_entry_price > 0:
        gold_return_since_entry = float(gold_price / gold_entry_price - 1.0)
        gold_paper_equity_gbp = float(gold_capital_pre_entry * (gold_price / gold_entry_price))
        if pd.notna(gold_entry_ts):
            gold_days_held = float((now - gold_entry_ts).total_seconds() / 86400.0)
    gold_paper_return = float(gold_paper_equity_gbp / gold_notional_gbp - 1.0)
    gold_paper_pnl_gbp = float(gold_paper_equity_gbp - gold_notional_gbp)

    _save_json_state(
        gold_state_path,
        {
            "active": bool(gold_position_active),
            "entry_ts": str(gold_entry_ts) if pd.notna(gold_entry_ts) else "",
            "entry_price": float(gold_entry_price) if np.isfinite(gold_entry_price) else None,
            "capital_pre_entry": float(gold_capital_pre_entry),
            "paper_equity": float(gold_paper_equity_gbp),
            "paper_notional": float(gold_notional_gbp),
            "paper_return": float(gold_paper_return),
            "paper_pnl_gbp": float(gold_paper_pnl_gbp),
            "trade_count": int(gold_trade_count),
            "last_price_ts": str(gold_signal_ts) if pd.notna(gold_signal_ts) else "",
            "last_price": float(gold_price) if np.isfinite(gold_price) else None,
            "updated_utc": str(now),
        },
    )

    # 3c) BTC sleeve (paper): offensive EMA sleeve.
    # Funding is reported as a diagnostic only; it must not create BTC exposure by itself.
    btc_price = np.nan
    btc_24h_pct = np.nan
    btc_ema21 = np.nan
    btc_ema55 = np.nan
    btc_ema144 = np.nan
    btc_stack_aligned = False
    btc_stack_aligned_days = np.nan
    btc_entry_threshold_met = False
    btc_off_w = 0.0
    btc_off_pos = False
    btc_off_entry_ts = pd.NaT
    btc_off_hold_ret = np.nan
    btc_off_days_held = np.nan
    btc_funding_z = np.nan
    btc_def_w = 0.0
    btc_def_pos = False
    btc_def_signal_ts = pd.NaT
    btc_off_scaled = 0.0
    btc_def_scaled = 0.0
    btc_comb_w = 0.0
    btc_daily = _read_daily_close_csv(Path(args.btc_daily_csv))
    btc_close_s = pd.Series(dtype=float)
    btc_vol_scalar_last = 0.0
    if len(btc_daily):
        bd = btc_daily.sort_values("timestamp").copy()
        bc = pd.to_numeric(bd["close"], errors="coerce").dropna()
        if len(bc):
            btc_price = float(bc.iloc[-1])
            if len(bc) >= 2:
                btc_24h_pct = float(bc.iloc[-1] / bc.iloc[-2] - 1.0)
        bday = bd["timestamp"].dt.floor("D")
        btc_close_s = pd.Series(pd.to_numeric(bd["close"], errors="coerce").to_numpy(dtype=float), index=bday)
        btc_close_s = btc_close_s[~btc_close_s.index.duplicated(keep="last")].dropna().sort_index()
        if len(btc_close_s) >= 5:
            b_fast = btc_close_s.ewm(span=int(args.btc_ema_fast), adjust=False).mean()
            b_mid = btc_close_s.ewm(span=int(args.btc_ema_mid), adjust=False).mean()
            b_slow = btc_close_s.ewm(span=int(args.btc_ema_slow), adjust=False).mean()
            btc_ema21 = float(b_fast.iloc[-1])
            btc_ema55 = float(b_mid.iloc[-1])
            btc_ema144 = float(b_slow.iloc[-1])
            b_stack = (b_fast > b_mid) & (b_mid > b_slow)
            btc_stack_aligned = bool(b_stack.iloc[-1])
            rev_bstack = b_stack.astype(int).iloc[::-1]
            btc_stack_aligned_days = int(rev_bstack.cumprod().sum()) if len(rev_bstack) else np.nan
            b_entry = (b_stack.rolling(int(args.btc_confirm_days), min_periods=int(args.btc_confirm_days)).min() == 1).fillna(False)
            b_exit = ((~b_stack).rolling(int(args.btc_exit_confirm_days), min_periods=int(args.btc_exit_confirm_days)).min() == 1).fillna(False)
            btc_entry_threshold_met = bool(b_entry.iloc[-1]) if len(b_entry) else False

            b_r = np.log(btc_close_s / btc_close_s.shift(1)).fillna(0.0)
            b_rv = b_r.rolling(int(args.btc_vol_window), min_periods=max(5, int(args.btc_vol_window) // 2)).std() * np.sqrt(365.0)
            b_vs = (float(args.btc_target_vol) / (b_rv.replace(0.0, np.nan))).clip(
                lower=float(args.btc_vol_floor), upper=float(args.btc_vol_cap)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            btc_vol_scalar_last = float(b_vs.iloc[-1]) if len(b_vs) else 0.0

            b_pos = 0
            b_days = 0
            b_states: list[int] = []
            b_regime_by_day = pd.Series(dtype=object)
            if len(regime) and args.regime_col in regime.columns:
                brg = regime.copy()
                brg["day"] = brg["timestamp"].dt.floor("D")
                b_regime_by_day = brg.sort_values("day").drop_duplicates("day", keep="last").set_index("day")[args.regime_col].astype(str)
            for day in btc_close_s.index:
                day_regime = str(b_regime_by_day.get(day, current_regime)) if len(b_regime_by_day) else str(current_regime)
                if day_regime == "BEAR":
                    b_pos = 0
                    b_days = 0
                    b_states.append(int(b_pos))
                    continue
                if b_pos == 1:
                    b_days += 1
                else:
                    b_days = 0
                if b_pos == 0:
                    if bool(b_entry.loc[day]):
                        b_pos = 1
                        b_days = 1
                else:
                    can_exit = b_days >= int(args.btc_min_hold_days)
                    if bool(b_exit.loc[day]) and can_exit:
                        b_pos = 0
                        b_days = 0
                b_states.append(int(b_pos))
            b_state_s = pd.Series(b_states, index=btc_close_s.index).astype(int)
            b_w = b_state_s.astype(float) * b_vs
            btc_off_w = float(b_w.iloc[-1]) if len(b_w) else 0.0
            btc_off_pos = bool(b_state_s.iloc[-1] == 1) if len(b_state_s) else False
            if btc_off_pos:
                b_starts = b_state_s[(b_state_s == 1) & (b_state_s.shift(1, fill_value=0) == 0)]
                if len(b_starts):
                    btc_off_entry_ts = pd.Timestamp(b_starts.index[-1])
                    if btc_off_entry_ts.tzinfo is None:
                        btc_off_entry_ts = btc_off_entry_ts.tz_localize("UTC")
                    else:
                        btc_off_entry_ts = btc_off_entry_ts.tz_convert("UTC")
                    b0 = _safe_num(btc_close_s.loc[b_starts.index[-1]])
                    b1 = _safe_num(btc_close_s.iloc[-1])
                    if np.isfinite(b0) and b0 > 0 and np.isfinite(b1):
                        btc_off_hold_ret = float(b1 / b0 - 1.0)
                    btc_off_days_held = float((now - btc_off_entry_ts).total_seconds() / 86400.0)

    btc_perp = _read_csv_ts(Path(args.btc_perp_csv))
    if len(btc_perp):
        fcol = _pick_col(btc_perp, ["funding_rate", "fundingRate", "funding"])
        if fcol is not None:
            bpf = btc_perp[["timestamp", fcol]].copy()
            bpf["day"] = bpf["timestamp"].dt.floor("D")
            bpf[fcol] = pd.to_numeric(bpf[fcol], errors="coerce")
            bpf = bpf.dropna(subset=[fcol]).sort_values("day")
            bpf_day = bpf.groupby("day", as_index=False)[fcol].mean()
            bpf_day["roll_mean"] = bpf_day[fcol].rolling(30, min_periods=10).mean()
            bpf_day["roll_std"] = bpf_day[fcol].rolling(30, min_periods=10).std(ddof=0)
            bpf_day["funding_z"] = (bpf_day[fcol] - bpf_day["roll_mean"]) / bpf_day["roll_std"].replace(0.0, np.nan)
            if len(bpf_day):
                btc_funding_z = float(pd.to_numeric(bpf_day["funding_z"], errors="coerce").iloc[-1])
                btc_def_signal_ts = pd.Timestamp(bpf_day["day"].iloc[-1])
                if btc_def_signal_ts.tzinfo is None:
                    btc_def_signal_ts = btc_def_signal_ts.tz_localize("UTC")
                else:
                    btc_def_signal_ts = btc_def_signal_ts.tz_convert("UTC")
                # Contrarian funding defensive trigger: deeply negative funding => potential rebound sleeve.
                if np.isfinite(btc_funding_z) and btc_funding_z < -1.0:
                    btc_def_pos = True
                    btc_def_w = float(btc_vol_scalar_last)

    btc_off_scaled = float(btc_off_w) * float(off_scale_map.get(current_regime, 0.0))
    btc_transition = _transition_state(
        daily_close=btc_close_s,
        regime=regime,
        regime_col=args.regime_col,
        paper_start_ts=paper_start_ts,
        entry_ts=btc_off_entry_ts,
        hold_return=btc_off_hold_ret,
        cfg=portfolio_cfg,
    )
    if not transition_momentum_enabled:
        btc_transition["transition_multiplier"] = 1.0
        btc_transition["transition_strength"] = "OFF"
    btc_transition_multiplier = float(btc_transition.get("transition_multiplier", 1.0))
    btc_off_scaled = btc_off_scaled * vol_multiplier * btc_transition_multiplier
    btc_conviction = _conviction_state(
        price=btc_price,
        ema_mid=btc_ema55,
        ema_slow=btc_ema144,
        regime=current_regime,
        vol_regime=str(vol_state.get("vol_regime", "NA")),
        cfg=portfolio_cfg,
        enabled=asymmetric_sizing_enabled,
    )
    btc_off_scaled = btc_off_scaled * float(btc_conviction.get("conviction_multiplier", 1.0))
    btc_def_scaled = float(btc_def_w) * float(def_scale_map.get(current_regime, 0.0))
    # Deployed BTC weight is the final offensive signal only. In particular, when
    # btc_off_w/raw signal is zero, displayed and executable BTC weight must be zero.
    btc_comb_w = float(min(gross_cap, max(0.0, btc_off_scaled)))

    mean_reversion_overlay = _mean_reversion_overlay_state(
        daily_close=daily_close,
        regime=current_regime,
        base_weight=comb_w,
        gross_cap=gross_cap,
        cfg=portfolio_cfg,
        now=now,
    )
    mean_reversion_weight = _safe_num(mean_reversion_overlay.get("weight", 0.0))
    if not np.isfinite(mean_reversion_weight):
        mean_reversion_weight = 0.0
    eth_execution_weight = float(min(gross_cap, max(0.0, (comb_w if np.isfinite(comb_w) else 0.0) + mean_reversion_weight)))

    # 4) Live monitoring metrics
    rolling_30d_sharpe = np.nan
    peak_dd = 0.0
    chop_bps_60d = np.nan
    if len(comb_perf):
        rolling_30d_sharpe = _rolling_sharpe_last_30d(comb_perf)
        peak_dd = _drawdown_from_peak(comb_perf)
        if len(regime) and args.regime_col in regime.columns:
            cd = comb_perf.copy()
            cd["day"] = cd["timestamp"].dt.floor("D")
            if args.regime_col not in cd.columns:
                rg_day = regime.copy()
                rg_day["day"] = rg_day["timestamp"].dt.floor("D")
                rg_day = rg_day.sort_values("day").drop_duplicates("day", keep="last")[["day", args.regime_col]]
                cd = cd.merge(rg_day, on="day", how="left")
            cd[args.regime_col] = cd[args.regime_col].fillna("CHOP")
            end_ts = cd["timestamp"].iloc[-1]
            start_ts = end_ts - pd.Timedelta(days=60)
            w = cd[(cd["timestamp"] >= start_ts) & (cd[args.regime_col] == "CHOP")].copy()
            if len(w):
                sr = _compute_returns_if_missing(w)
                br = _compute_spot_returns_if_missing(w)
                chop_bps_60d = float((sr - br).mean() * 1e4)

    # Live mark-to-market performance since paper inception.
    # ETH sleeve mirrors existing method; BTC sleeve uses daily BTC close and BTC sleeve weight.
    # Combined portfolio is equal-weight across sleeves.
    cum_strategy_ret = 0.0
    cum_spot_ret = np.nan
    excess = np.nan
    eth_sleeve_strategy_ret = np.nan
    eth_sleeve_spot_ret = np.nan
    btc_sleeve_strategy_ret = np.nan
    btc_sleeve_spot_ret = np.nan
    portfolio_strategy_ret = np.nan
    portfolio_spot_ret = np.nan
    portfolio_excess_vs_basket = np.nan
    portfolio_peak_dd = np.nan
    live_log_path = out_dir / "daily_checks_log.csv"
    if len(price):
        cc = _close_col(price)
        if cc is not None:
            p3 = price[["timestamp", cc]].copy().dropna()
            p3[cc] = pd.to_numeric(p3[cc], errors="coerce")
            p3 = p3.dropna(subset=[cc]).sort_values("timestamp")
            pday = (
                p3.assign(day=p3["timestamp"].dt.floor("D"))
                .groupby("day", as_index=False)[cc]
                .last()
                .rename(columns={cc: "close"})
            )
            pday = pday[pday["day"] >= paper_start_ts].copy()
            if len(pday):
                c0 = _safe_num(pday["close"].iloc[0])
                c1 = _safe_num(pday["close"].iloc[-1])
                if np.isfinite(c0) and c0 > 0 and np.isfinite(c1):
                    cum_spot_ret = float(c1 / c0 - 1.0)
                    eth_sleeve_spot_ret = cum_spot_ret

                # Exposure by day: prior logged combined weight + today's combined weight.
                ex_rows = []
                if live_log_path.exists():
                    try:
                        lg_prev = pd.read_csv(live_log_path, usecols=["date", "combined_weight"])
                        lg_prev["day"] = pd.to_datetime(lg_prev["date"], utc=True, errors="coerce").dt.floor("D")
                        lg_prev["combined_weight"] = pd.to_numeric(lg_prev["combined_weight"], errors="coerce")
                        lg_prev = lg_prev.dropna(subset=["day"]).sort_values("day")
                        ex_rows.append(lg_prev[["day", "combined_weight"]].copy())
                    except Exception:
                        pass
                ex_today = pd.DataFrame([{"day": now.floor("D"), "combined_weight": float(comb_w) if np.isfinite(comb_w) else 0.0}])
                ex_rows.append(ex_today)
                ex = pd.concat(ex_rows, ignore_index=True) if ex_rows else ex_today
                ex = ex.sort_values("day").drop_duplicates(subset=["day"], keep="last")
                ex = ex[ex["day"] >= paper_start_ts].copy()

                close_s = pd.Series(pd.to_numeric(pday["close"], errors="coerce").to_numpy(dtype=float), index=pday["day"])
                spot_daily_ret = close_s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                w_s = (
                    ex.set_index("day")["combined_weight"]
                    .reindex(close_s.index)
                    .astype(float)
                    .ffill()
                    .fillna(0.0)
                    .clip(lower=0.0, upper=1.0)
                )
                strat_daily_ret = w_s.shift(1).fillna(0.0) * spot_daily_ret
                cum_strategy_ret = float((1.0 + strat_daily_ret).prod() - 1.0) if len(strat_daily_ret) else 0.0
                eth_sleeve_strategy_ret = cum_strategy_ret

    # BTC sleeve performance since paper inception.
    btc_log_path = out_dir / "btc_daily_checks_log.csv"
    if len(btc_close_s):
        bcs = btc_close_s[btc_close_s.index >= paper_start_ts].copy()
        if len(bcs):
            b0 = _safe_num(bcs.iloc[0])
            b1 = _safe_num(bcs.iloc[-1])
            if np.isfinite(b0) and b0 > 0 and np.isfinite(b1):
                btc_sleeve_spot_ret = float(b1 / b0 - 1.0)

            b_ex_rows = []
            if btc_log_path.exists():
                try:
                    bl_prev = pd.read_csv(btc_log_path, usecols=["date", "btc_combined_weight"])
                    bl_prev["day"] = pd.to_datetime(bl_prev["date"], utc=True, errors="coerce").dt.floor("D")
                    bl_prev["btc_combined_weight"] = pd.to_numeric(bl_prev["btc_combined_weight"], errors="coerce")
                    bl_prev = bl_prev.dropna(subset=["day"]).sort_values("day")
                    b_ex_rows.append(bl_prev[["day", "btc_combined_weight"]].copy())
                except Exception:
                    pass
            b_ex_rows.append(pd.DataFrame([{"day": now.floor("D"), "btc_combined_weight": float(btc_comb_w)}]))
            bex = pd.concat(b_ex_rows, ignore_index=True)
            bex = bex.sort_values("day").drop_duplicates(subset=["day"], keep="last")
            bex = bex[bex["day"] >= paper_start_ts].copy()
            b_spot_ret = bcs.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            b_w = (
                bex.set_index("day")["btc_combined_weight"]
                .reindex(bcs.index)
                .astype(float)
                .ffill()
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
            b_strat_ret = b_w.shift(1).fillna(0.0) * b_spot_ret
            btc_sleeve_strategy_ret = float((1.0 + b_strat_ret).prod() - 1.0) if len(b_strat_ret) else np.nan

            # Combined portfolio daily return + drawdown
            idx = bcs.index
            if np.isfinite(eth_sleeve_spot_ret):
                # Rebuild ETH spot/strategy daily series on same daily index.
                eth_pday = (
                    price[[ "timestamp", _close_col(price)]]
                    .dropna()
                    .assign(day=lambda z: z["timestamp"].dt.floor("D"))
                    .groupby("day", as_index=False)[_close_col(price)]
                    .last()
                    .rename(columns={_close_col(price): "close"})
                )
                eth_pday = eth_pday[eth_pday["day"] >= paper_start_ts].copy()
                eth_close_s = pd.Series(pd.to_numeric(eth_pday["close"], errors="coerce").to_numpy(dtype=float), index=eth_pday["day"])
                eth_close_s = eth_close_s.reindex(idx).ffill()
                eth_spot_daily = eth_close_s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

                e_ex_rows = []
                if live_log_path.exists():
                    try:
                        lg_prev = pd.read_csv(live_log_path, usecols=["date", "combined_weight"])
                        lg_prev["day"] = pd.to_datetime(lg_prev["date"], utc=True, errors="coerce").dt.floor("D")
                        lg_prev["combined_weight"] = pd.to_numeric(lg_prev["combined_weight"], errors="coerce")
                        lg_prev = lg_prev.dropna(subset=["day"]).sort_values("day")
                        e_ex_rows.append(lg_prev[["day", "combined_weight"]].copy())
                    except Exception:
                        pass
                e_ex_rows.append(pd.DataFrame([{"day": now.floor("D"), "combined_weight": float(comb_w) if np.isfinite(comb_w) else 0.0}]))
                eex = pd.concat(e_ex_rows, ignore_index=True)
                eex = eex.sort_values("day").drop_duplicates(subset=["day"], keep="last")
                e_w = (
                    eex.set_index("day")["combined_weight"]
                    .reindex(idx)
                    .astype(float)
                    .ffill()
                    .fillna(0.0)
                    .clip(lower=0.0, upper=1.0)
                )
                mr_w = pd.Series(0.0, index=e_w.index)
                if _cfg_bool(portfolio_cfg, "mean_reversion_overlay_enabled", False):
                    try:
                        lg_mr = pd.read_csv(live_log_path, usecols=["date", "mean_reversion_weight"])
                        lg_mr["day"] = pd.to_datetime(lg_mr["date"], utc=True, errors="coerce").dt.floor("D")
                        lg_mr["mean_reversion_weight"] = pd.to_numeric(lg_mr["mean_reversion_weight"], errors="coerce")
                        lg_mr = lg_mr.dropna(subset=["day"]).sort_values("day")
                    except Exception:
                        lg_mr = pd.DataFrame(columns=["day", "mean_reversion_weight"])
                    lg_mr = pd.concat(
                        [
                            lg_mr[["day", "mean_reversion_weight"]],
                            pd.DataFrame([{"day": now.floor("D"), "mean_reversion_weight": float(mean_reversion_weight)}]),
                        ],
                        ignore_index=True,
                    )
                    lg_mr = lg_mr.sort_values("day").drop_duplicates(subset=["day"], keep="last")
                    mr_w = (
                        lg_mr.set_index("day")["mean_reversion_weight"]
                        .reindex(idx)
                        .astype(float)
                        .ffill()
                        .fillna(0.0)
                        .clip(lower=0.0, upper=1.0)
                    )
                eth_strat_daily = e_w.shift(1).fillna(0.0) * eth_spot_daily
                eth_strat_daily = eth_strat_daily + mr_w.shift(1).fillna(0.0) * eth_spot_daily

                sleeve_eth_cap = float(max(0.0, args.eth_sleeve_capital))
                sleeve_btc_cap = float(max(0.0, args.btc_sleeve_capital))
                if sleeve_eth_cap + sleeve_btc_cap <= 0:
                    sleeve_eth_cap, sleeve_btc_cap = 0.5, 0.5
                cap_sum = sleeve_eth_cap + sleeve_btc_cap
                sleeve_eth_cap /= cap_sum
                sleeve_btc_cap /= cap_sum

                combined_daily = sleeve_eth_cap * eth_strat_daily + sleeve_btc_cap * b_strat_ret.reindex(idx).fillna(0.0)
                basket_daily = sleeve_eth_cap * eth_spot_daily + sleeve_btc_cap * b_spot_ret.reindex(idx).fillna(0.0)

                portfolio_strategy_ret = float((1.0 + combined_daily).prod() - 1.0)
                portfolio_spot_ret = float((1.0 + basket_daily).prod() - 1.0)
                portfolio_excess_vs_basket = float(portfolio_strategy_ret - portfolio_spot_ret)
                eqp = (1.0 + combined_daily).cumprod()
                ddp = eqp / eqp.cummax() - 1.0
                portfolio_peak_dd = float(ddp.min()) if len(ddp) else np.nan

    if np.isfinite(cum_strategy_ret) and np.isfinite(cum_spot_ret):
        excess = float(cum_strategy_ret - cum_spot_ret)
    days_live = int((now.floor("D") - paper_start_ts.floor("D")).days + 1)
    if days_live < 1:
        days_live = 1

    live_execution_report: dict[str, object] = {}
    live_execution_error = ""
    live_execution_dry_run = True
    if bool(args.live):
        try:
            from execution_kraken import execute_strategy_signal

            live_execution_dry_run = os.getenv("LIVE_TRADING_ENABLED", "").strip().lower() != "true"
            live_execution_report = execute_strategy_signal(
                eth_target_weight=float(eth_execution_weight) if np.isfinite(eth_execution_weight) else 0.0,
                btc_target_weight=float(btc_comb_w) if np.isfinite(btc_comb_w) else 0.0,
                total_capital_usd=None,
                dry_run=live_execution_dry_run,
            )
        except Exception as exc:
            live_execution_error = f"{type(exc).__name__}: {exc}"
            reviews.append(f"live_execution_failed:{str(exc)[:120]}")

    news_row = _latest_news_row(Path(args.news_log_csv))
    markets_lines = _latest_market_lines(Path(args.market_log_csv))
    news_major_event = False
    news_severity = "low"
    news_affected = ""
    news_direction = "mixed"
    news_summary = "No major macro events"
    if news_row:
        mv = str(news_row.get("major_event", "")).strip().lower()
        news_major_event = mv in {"true", "1", "yes", "y"}
        news_severity = str(news_row.get("severity", "low") or "low")
        news_affected = str(news_row.get("affected", "") or "")
        news_direction = str(news_row.get("direction", "mixed") or "mixed")
        news_summary = str(news_row.get("summary", "No major macro events") or "No major macro events")
    if news_major_event:
        affected_txt = news_affected.replace("|", ", ") if news_affected else "broad markets"
        news_alert_line = f"NEWS ALERT: {news_summary}\nAffected: {affected_txt} | {news_direction}"
    else:
        news_alert_line = "News: No major macro events"
    dvol_row = _latest_dvol_row(Path(args.dvol_log_csv))

    if np.isfinite(rolling_30d_sharpe) and rolling_30d_sharpe < float(args.stop_rolling_sharpe):
        stops.append("rolling_30d_sharpe_below_stop")
    if np.isfinite(peak_dd) and peak_dd < float(args.stop_drawdown):
        stops.append("drawdown_below_stop")
    if np.isfinite(chop_bps_60d) and chop_bps_60d < float(args.review_chop_bps):
        reviews.append("chop_bps_60d_below_review")

    all_reasons = stops + flags + reviews
    any_flag = bool(len(all_reasons) > 0)
    status = "STOP" if len(stops) > 0 else ("REVIEW" if len(flags + reviews) > 0 else "PASS")

    # Completed offensive trades log (execution-state closes).
    today = now.strftime("%Y-%m-%d")
    log_counter = _trailing_false_days_from_log(out_dir / "daily_checks_log.csv", paper_start_ts, today, bool(stack_aligned))
    if np.isfinite(log_counter):
        off_days_since_break = log_counter
    completed_trades_path = Path(args.completed_trades_csv) if str(args.completed_trades_csv).strip() else (out_dir / "completed_trades.csv")
    completed_cols = [
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "exit_reason",
        "days_held",
        "return_pct",
        "regime_at_exit",
    ]
    if completed_trades_path.exists():
        try:
            completed_df = pd.read_csv(completed_trades_path, low_memory=False)
        except Exception:
            completed_df = pd.DataFrame(columns=completed_cols)
    else:
        completed_df = pd.DataFrame(columns=completed_cols)
    for c in completed_cols:
        if c not in completed_df.columns:
            completed_df[c] = np.nan if c not in {"entry_date", "exit_date", "exit_reason", "regime_at_exit"} else ""
    completed_df = completed_df[completed_cols]
    # Normalize key columns for stable dedupe/locking logic.
    completed_df["entry_date"] = pd.to_datetime(completed_df["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    completed_df["exit_date"] = pd.to_datetime(completed_df["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Repair historical duplicates: one closed trade per entry_date.
    # Keep the earliest exit (first close event wins and locks the trade).
    if len(completed_df):
        ctmp = completed_df.copy()
        ctmp["_exit_dt"] = pd.to_datetime(ctmp["exit_date"], errors="coerce")
        ctmp = ctmp.sort_values(["entry_date", "_exit_dt"], ascending=[True, True], na_position="last")
        completed_df = ctmp.drop_duplicates(subset=["entry_date"], keep="first").drop(columns=["_exit_dt"])

    if off_closed_today and pd.notna(off_close_entry_ts) and pd.notna(off_close_exit_ts):
        lock_entry_date = str(pd.Timestamp(off_close_entry_ts).date())
        # Trade already closed previously: lock it, skip any further exit logic/update.
        already_closed = bool((completed_df["entry_date"].astype(str) == lock_entry_date).any())
        if already_closed:
            off_closed_today = False
            off_close_reason = ""
            off_close_entry_ts = pd.NaT
            off_close_exit_ts = pd.NaT
            off_close_entry_price = np.nan
            off_close_exit_price = np.nan
            off_close_days_held = np.nan
            off_close_return = np.nan
            off_pos = False
            off_w = 0.0
            off_hold_ret = 0.0

    if off_closed_today and pd.notna(off_close_entry_ts) and pd.notna(off_close_exit_ts):
        tr = {
            "entry_date": str(pd.Timestamp(off_close_entry_ts).date()),
            "entry_price": float(off_close_entry_price) if np.isfinite(off_close_entry_price) else np.nan,
            "exit_date": str(pd.Timestamp(off_close_exit_ts).date()),
            "exit_price": float(off_close_exit_price) if np.isfinite(off_close_exit_price) else np.nan,
            "exit_reason": str(off_close_reason),
            "days_held": float(off_close_days_held) if np.isfinite(off_close_days_held) else np.nan,
            "return_pct": float(off_close_return) if np.isfinite(off_close_return) else np.nan,
            "regime_at_exit": str(current_regime),
        }
        # Lock by entry_date: once closed, never append another exit for same entry.
        same_entry = (completed_df["entry_date"].astype(str) == tr["entry_date"])
        if not bool(same_entry.any()):
            completed_df = pd.concat([completed_df, pd.DataFrame([tr], columns=completed_cols)], ignore_index=True)

    if len(completed_df):
        completed_df["_exit_dt"] = pd.to_datetime(completed_df["exit_date"], errors="coerce")
        completed_df = completed_df.sort_values("_exit_dt")
        completed_df["exit_date"] = completed_df["_exit_dt"].dt.strftime("%Y-%m-%d")
        completed_df = completed_df.drop(columns=["_exit_dt"])
    completed_trades_path.parent.mkdir(parents=True, exist_ok=True)
    completed_df.to_csv(completed_trades_path, index=False)

    last_closed = completed_df.iloc[-1].to_dict() if len(completed_df) else {}
    off_last_trade_closed = bool(len(last_closed) > 0)

    # Emit console summary
    print(f"=== PAPER TRADE CHECK {today} ===")
    print(f"Data:       {'PASS' if (price_fresh and funding_fresh and (not np.isfinite(gap_count) or gap_count == 0)) else 'WARN'} | "
          f"price_age_h={price_age_h:.2f} funding_age_h={funding_age_h:.2f} gap_count_48h={gap_count if np.isfinite(gap_count) else 'NA'}")
    print(f"Regime:     {current_regime} (streak {int(regime_days) if np.isfinite(regime_days) else 'NA'}d)")
    print(
        f"EMA state:  {ema_state} | "
        f"ema{int(args.off_ema_fast)}={ema21:.2f} "
        f"ema{int(args.off_ema_mid)}={ema55:.2f} "
        f"ema{int(args.off_ema_slow)}={ema144:.2f}"
    )
    print(
        f"Off gate:   stack_aligned={stack_aligned} aligned_days={int(stack_aligned_days) if np.isfinite(stack_aligned_days) else 'NA'} "
        f"entry_threshold_met={entry_threshold_met} days_since_break={int(off_days_since_break) if np.isfinite(off_days_since_break) else 'NA'}"
    )
    print(f"DD/20d hi:  {dd_20d:.4f}")
    print(
        f"Offensive:  {'IN' if off_pos else 'FLAT'} | w_raw={off_w:.4f} | w_scaled={off_scaled:.4f} "
        f"| entry={off_entry_ts if pd.notna(off_entry_ts) else 'NA'} | hold_ret={off_hold_ret:.4f}"
    )
    if off_closed_today:
        print(
            f"Off close:  CLOSED | reason={off_close_reason} | entry={off_close_entry_ts if pd.notna(off_close_entry_ts) else 'NA'} "
            f"| exit={off_close_exit_ts if pd.notna(off_close_exit_ts) else 'NA'} | ret={_fmt_pct(off_close_return, 2)}"
        )
    print(
        f"Defensive:  {'IN' if def_pos else 'FLAT'} | w_raw={def_w:.4f} | w_scaled={def_scaled:.4f} | mode={args.defense_live_mode} "
        f"| signal_ts={def_signal_ts if pd.notna(def_signal_ts) else 'NA'} | p_up={def_p_up:.4f}"
    )
    print(f"Combined:   w={comb_w:.4f} | cap_bind_today={cap_bind_today}")
    print(
        f"Vol filter: regime={vol_state.get('vol_regime', 'NA')} "
        f"| pct={_fmt_pct(_safe_num(vol_state.get('vol_percentile', np.nan)), 0)} "
        f"| mult={_fmt_num(_safe_num(vol_state.get('vol_multiplier', np.nan)), 2)}"
    )
    print(
        f"Transition: ETH={eth_transition.get('transition_strength', 'NA')} "
        f"x{_fmt_num(_safe_num(eth_transition.get('transition_multiplier', np.nan)), 2)} "
        f"| BTC={btc_transition.get('transition_strength', 'NA')} "
        f"x{_fmt_num(_safe_num(btc_transition.get('transition_multiplier', np.nan)), 2)}"
    )
    print(
        f"Conviction: ETH={eth_conviction.get('conviction_bucket', 'NA')} "
        f"({_fmt_num(_safe_num(eth_conviction.get('conviction_score', np.nan)), 2)}) "
        f"x{_fmt_num(_safe_num(eth_conviction.get('conviction_multiplier', np.nan)), 2)} "
        f"| BTC={btc_conviction.get('conviction_bucket', 'NA')} "
        f"({_fmt_num(_safe_num(btc_conviction.get('conviction_score', np.nan)), 2)}) "
        f"x{_fmt_num(_safe_num(btc_conviction.get('conviction_multiplier', np.nan)), 2)}"
    )
    print(
        f"Gold paper: {'IN' if gold_position_active else 'FLAT'} | gate_flat_bear={gold_flat_bear_gate} "
        f"| ema_gate={gold_entry_threshold_met} | cond={gold_condition_met} | px={_fmt_num(gold_price, 2)} "
        f"| entry={gold_entry_ts if pd.notna(gold_entry_ts) else 'NA'} | ret={_fmt_pct(gold_return_since_entry, 2)} "
        f"| pnl_gbp={_fmt_num(gold_paper_pnl_gbp, 2)}"
    )
    print(
        f"BTC sleeve: {'IN' if btc_off_pos else 'FLAT'} | w_raw={btc_off_w:.4f} | w_scaled={btc_comb_w:.4f} "
        f"| funding_z={_fmt_num(btc_funding_z, 3)} | entry={btc_off_entry_ts if pd.notna(btc_off_entry_ts) else 'NA'} "
        f"| hold_ret={_fmt_pct(btc_off_hold_ret, 2)}"
    )
    print(f"30d Sharpe: {rolling_30d_sharpe:.4f}")
    print(f"Peak DD:    {peak_dd:.4f}")
    print(f"CHOP bps 60d: {chop_bps_60d:.4f}")
    if markets_lines:
        print("Markets:")
        for ln in markets_lines:
            print(f"  {ln}")
    news_console_line = news_alert_line
    if os.name == "nt":
        news_console_line = news_console_line.encode("cp1252", errors="ignore").decode("cp1252").strip()
    print(f"News:       {news_console_line}")
    print(f"Paper start: {paper_start_ts.date()} | days_live={days_live}")
    print(
        f"Perf sleeves: ETH strat={_fmt_pct(eth_sleeve_strategy_ret, 2)} spot={_fmt_pct(eth_sleeve_spot_ret, 2)} | "
        f"BTC strat={_fmt_pct(btc_sleeve_strategy_ret, 2)} spot={_fmt_pct(btc_sleeve_spot_ret, 2)}"
    )
    print(
        f"Perf combined: strat={_fmt_pct(portfolio_strategy_ret, 2)} basket={_fmt_pct(portfolio_spot_ret, 2)} "
        f"excess={_fmt_pct(portfolio_excess_vs_basket, 2)} peak_dd={_fmt_pct(portfolio_peak_dd, 2)}"
    )
    if bool(args.live):
        eth_live_status = str((live_execution_report.get("eth_trade") or {}).get("status", "NA")) if live_execution_report else "NA"
        btc_live_status = str((live_execution_report.get("btc_trade") or {}).get("status", "NA")) if live_execution_report else "NA"
        print(
            f"Live execution: {'DRY RUN' if live_execution_dry_run else 'REAL'} | "
            f"ETH={eth_live_status} "
            f"BTC={btc_live_status} "
            f"error={live_execution_error or 'none'}"
        )
    print("---")
    print(f"STATUS: {status}")
    if all_reasons:
        print("Flags:", "; ".join(all_reasons))

    # Prepare row
    out_row = {
        "date": today,
        "timestamp_utc": str(now),
        "status": status,
        "any_flag": any_flag,
        "stop_count": len(stops),
        "warn_count": len(flags + reviews),
        "flag_reasons": ";".join(all_reasons),
        "price_csv": args.price_csv,
        "funding_csv": args.funding_csv,
        "regime_csv": args.regime_csv,
        "offense_log_csv": args.offense_log_csv,
        "defense_log_csv": args.defense_log_csv,
        "combined_log_csv": args.combined_log_csv,
        "paper_start_date": str(paper_start_ts.date()),
        "days_live": days_live,
        "price_last_ts": str(price_last) if pd.notna(price_last) else "",
        "funding_last_ts": str(funding_last) if pd.notna(funding_last) else "",
        "price_age_hours": price_age_h,
        "funding_age_hours": funding_age_h,
        "data_fresh": bool(price_fresh and funding_fresh),
        "ohlcv_gap_count_lookback": gap_count,
        "ohlcv_max_gap_minutes_lookback": max_gap_min,
        "regime": current_regime,
        "regime_days": regime_days,
        "ema_state": ema_state,
        "eth_ema_fast": int(args.off_ema_fast),
        "eth_ema_mid": int(args.off_ema_mid),
        "eth_ema_slow": int(args.off_ema_slow),
        "ema21": ema21,
        "ema55": ema55,
        "ema144": ema144,
        "stack_aligned": bool(stack_aligned),
        "stack_aligned_days": stack_aligned_days,
        "off_days_since_break": off_days_since_break,
        "off_exit_confirmed": bool(off_exit_confirmed),
        "entry_threshold_met": bool(entry_threshold_met),
        "dd_20d": dd_20d,
        "eth_price": eth_price,
        "eth_24h_pct": eth_24h_pct,
        "btc_price": btc_price,
        "btc_24h_pct": btc_24h_pct,
        "btc_ema15": btc_ema21,
        "btc_ema40": btc_ema55,
        "btc_ema120": btc_ema144,
        "btc_ema21": btc_ema21,
        "btc_ema55": btc_ema55,
        "btc_ema144": btc_ema144,
        "btc_stack_aligned": bool(btc_stack_aligned),
        "btc_stack_aligned_days": btc_stack_aligned_days,
        "btc_entry_threshold_met": bool(btc_entry_threshold_met),
        "btc_regime": current_regime,
        "btc_funding_z": btc_funding_z,
        "btc_def_signal_ts": str(btc_def_signal_ts) if pd.notna(btc_def_signal_ts) else "",
        "btc_position": bool(btc_off_pos),
        "btc_off_weight_raw": btc_off_w,
        "btc_off_weight_scaled": btc_off_scaled,
        "btc_def_weight_raw": btc_def_w,
        "btc_def_weight_scaled": btc_def_scaled,
        "btc_combined_weight": btc_comb_w,
        "btc_off_entry_ts": str(btc_off_entry_ts) if pd.notna(btc_off_entry_ts) else "",
        "btc_off_hold_return": btc_off_hold_ret,
        "btc_off_days_held": btc_off_days_held,
        "gold_symbol": args.gold_symbol,
        "gold_price_csv": args.gold_price_csv,
        "gold_state_json": args.gold_state_json,
        "gold_ema_fast": int(args.gold_ema_fast),
        "gold_ema_mid": int(args.gold_ema_mid),
        "gold_ema_slow": int(args.gold_ema_slow),
        "gold_price": gold_price,
        "gold_24h_pct": gold_24h_pct,
        "gold_price_ts": str(gold_signal_ts) if pd.notna(gold_signal_ts) else "",
        "gold_ema21": gold_ema21,
        "gold_ema55": gold_ema55,
        "gold_ema144": gold_ema144,
        "gold_stack_aligned": bool(gold_stack_aligned),
        "gold_entry_threshold_met": bool(gold_entry_threshold_met),
        "gold_flat_bear_gate": bool(gold_flat_bear_gate),
        "gold_condition_met": bool(gold_condition_met),
        "gold_position_active": bool(gold_position_active),
        "gold_entry_ts": str(gold_entry_ts) if pd.notna(gold_entry_ts) else "",
        "gold_entry_price": gold_entry_price,
        "gold_days_held": gold_days_held,
        "gold_return_since_entry": gold_return_since_entry,
        "gold_trade_count": int(gold_trade_count),
        "gold_paper_notional_gbp": gold_notional_gbp,
        "gold_paper_equity_gbp": gold_paper_equity_gbp,
        "gold_paper_return": gold_paper_return,
        "gold_paper_pnl_gbp": gold_paper_pnl_gbp,
        "news_major_event": bool(news_major_event),
        "news_severity": news_severity,
        "news_affected": news_affected,
        "news_direction": news_direction,
        "news_summary": news_summary,
        "news_alert_line": news_alert_line,
        "dvol_atm_iv_30d": _safe_num(dvol_row.get("atm_iv_30d", np.nan)),
        "dvol_iv_percentile": _safe_num(dvol_row.get("iv_percentile", np.nan)),
        "dvol_options_vol_regime": str(dvol_row.get("options_vol_regime", "NA")),
        "dvol_rv_vol_regime": str(dvol_row.get("rv_vol_regime", "NA")),
        "dvol_agreement": str(dvol_row.get("agreement", "NA")),
        "dvol_term_slope": _safe_num(dvol_row.get("term_slope", np.nan)),
        "dvol_history_days": int(_safe_num(dvol_row.get("history_days", 0))) if np.isfinite(_safe_num(dvol_row.get("history_days", 0))) else 0,
        "dvol_insufficient_history": bool(dvol_row.get("insufficient_history", True)) if dvol_row else True,
        "markets_count": int(len(markets_lines)),
        "off_position": off_pos,
        "off_weight": off_w,
        "off_weight_scaled": off_scaled,
        "off_entry_ts": str(off_entry_ts) if pd.notna(off_entry_ts) else "",
        "off_hold_return": off_hold_ret,
        "off_days_held": off_days_held,
        "off_closed_today": bool(off_closed_today),
        "off_close_reason": off_close_reason,
        "off_close_entry_ts": str(off_close_entry_ts) if pd.notna(off_close_entry_ts) else "",
        "off_close_exit_ts": str(off_close_exit_ts) if pd.notna(off_close_exit_ts) else "",
        "off_close_return": off_close_return,
        "off_close_days_held": off_close_days_held,
        "off_last_trade_closed": bool(off_last_trade_closed),
        "off_last_entry_date": str(last_closed.get("entry_date", "")) if off_last_trade_closed else "",
        "off_last_entry_price": _safe_num(last_closed.get("entry_price", np.nan)) if off_last_trade_closed else np.nan,
        "off_last_exit_date": str(last_closed.get("exit_date", "")) if off_last_trade_closed else "",
        "off_last_exit_price": _safe_num(last_closed.get("exit_price", np.nan)) if off_last_trade_closed else np.nan,
        "off_last_exit_reason": str(last_closed.get("exit_reason", "")) if off_last_trade_closed else "",
        "off_last_days_held": _safe_num(last_closed.get("days_held", np.nan)) if off_last_trade_closed else np.nan,
        "off_last_return_pct": _safe_num(last_closed.get("return_pct", np.nan)) if off_last_trade_closed else np.nan,
        "def_position": def_pos,
        "def_weight": def_w,
        "def_weight_scaled": def_scaled,
        "defense_live_mode": args.defense_live_mode,
        "def_signal_ts": str(def_signal_ts) if pd.notna(def_signal_ts) else "",
        "def_p_up": def_p_up,
        "combined_weight": comb_w,
        "eth_execution_weight": eth_execution_weight,
        "mean_reversion_enabled": bool(mean_reversion_overlay.get("enabled", False)),
        "mean_reversion_active": bool(mean_reversion_overlay.get("active", False)),
        "mean_reversion_weight": mean_reversion_weight,
        "mean_reversion_remaining_cap": _safe_num(mean_reversion_overlay.get("remaining_cap", np.nan)),
        "mean_reversion_z_score": _safe_num(mean_reversion_overlay.get("z_score", np.nan)),
        "mean_reversion_daily_ret": _safe_num(mean_reversion_overlay.get("daily_ret", np.nan)),
        "mean_reversion_entry_signal": bool(mean_reversion_overlay.get("entry_signal", False)),
        "mean_reversion_exit_signal": bool(mean_reversion_overlay.get("exit_signal", False)),
        "mean_reversion_days_held": int(_safe_num(mean_reversion_overlay.get("days_held", 0))) if np.isfinite(_safe_num(mean_reversion_overlay.get("days_held", 0))) else 0,
        "mean_reversion_entry_date": str(mean_reversion_overlay.get("entry_date", "")),
        "mean_reversion_exit_reason": str(mean_reversion_overlay.get("exit_reason", "")),
        "vol_regime": str(vol_state.get("vol_regime", "NA")),
        "vol_percentile": _safe_num(vol_state.get("vol_percentile", np.nan)),
        "vol_multiplier": _safe_num(vol_state.get("vol_multiplier", np.nan)),
        "rolling_vol_20d": _safe_num(vol_state.get("rolling_vol_20d", np.nan)),
        "transition_strength": str(eth_transition.get("transition_strength", "NA")),
        "transition_multiplier": _safe_num(eth_transition.get("transition_multiplier", np.nan)),
        "transition_return": _safe_num(eth_transition.get("transition_return", np.nan)),
        "transition_flip_date": str(eth_transition.get("transition_flip_date", "")),
        "btc_transition_strength": str(btc_transition.get("transition_strength", "NA")),
        "btc_transition_multiplier": _safe_num(btc_transition.get("transition_multiplier", np.nan)),
        "btc_transition_return": _safe_num(btc_transition.get("transition_return", np.nan)),
        "btc_transition_flip_date": str(btc_transition.get("transition_flip_date", "")),
        "asymmetric_sizing": bool(asymmetric_sizing_enabled),
        "conviction_score": _safe_num(eth_conviction.get("conviction_score", np.nan)),
        "conviction_bucket": str(eth_conviction.get("conviction_bucket", "NA")),
        "conviction_multiplier": _safe_num(eth_conviction.get("conviction_multiplier", np.nan)),
        "conviction_gap_score": _safe_num(eth_conviction.get("conviction_gap_score", np.nan)),
        "conviction_regime_score": _safe_num(eth_conviction.get("conviction_regime_score", np.nan)),
        "conviction_vol_score": _safe_num(eth_conviction.get("conviction_vol_score", np.nan)),
        "btc_conviction_score": _safe_num(btc_conviction.get("conviction_score", np.nan)),
        "btc_conviction_bucket": str(btc_conviction.get("conviction_bucket", "NA")),
        "btc_conviction_multiplier": _safe_num(btc_conviction.get("conviction_multiplier", np.nan)),
        "btc_conviction_gap_score": _safe_num(btc_conviction.get("conviction_gap_score", np.nan)),
        "btc_conviction_regime_score": _safe_num(btc_conviction.get("conviction_regime_score", np.nan)),
        "btc_conviction_vol_score": _safe_num(btc_conviction.get("conviction_vol_score", np.nan)),
        "portfolio_config": args.portfolio_config,
        "combined_weight_le_1": bool(np.isfinite(comb_w) and comb_w <= gross_cap + 1e-9),
        "cap_bind_today": cap_bind_today,
        "rolling_30d_sharpe": rolling_30d_sharpe,
        "peak_dd": peak_dd,
        "cum_strategy_ret": cum_strategy_ret,
        "cum_spot_ret": cum_spot_ret,
        "excess": excess,
        "eth_sleeve_strategy_ret": eth_sleeve_strategy_ret,
        "eth_sleeve_spot_ret": eth_sleeve_spot_ret,
        "btc_sleeve_strategy_ret": btc_sleeve_strategy_ret,
        "btc_sleeve_spot_ret": btc_sleeve_spot_ret,
        "portfolio_strategy_ret": portfolio_strategy_ret,
        "portfolio_spot_ret": portfolio_spot_ret,
        "portfolio_excess_vs_basket": portfolio_excess_vs_basket,
        "portfolio_peak_dd": portfolio_peak_dd,
        "live_mode_requested": bool(args.live),
        "live_execution_dry_run": bool(live_execution_dry_run),
        "live_execution_error": live_execution_error,
        "live_execution_time_ms": _safe_num(live_execution_report.get("execution_time_ms", np.nan)) if live_execution_report else np.nan,
        "live_total_fees_usd": _safe_num(live_execution_report.get("total_fees_usd", np.nan)) if live_execution_report else np.nan,
        "live_eth_status": str((live_execution_report.get("eth_trade") or {}).get("status", "")) if live_execution_report else "",
        "live_btc_status": str((live_execution_report.get("btc_trade") or {}).get("status", "")) if live_execution_report else "",
        "chop_bps_bar_60d": chop_bps_60d,
        "stop_rolling_30d_sharpe": bool(np.isfinite(rolling_30d_sharpe) and rolling_30d_sharpe < float(args.stop_rolling_sharpe)),
        "stop_drawdown": bool(np.isfinite(peak_dd) and peak_dd < float(args.stop_drawdown)),
        "review_chop_bps": bool(np.isfinite(chop_bps_60d) and chop_bps_60d < float(args.review_chop_bps)),
        "review_regime_stuck": bool(np.isfinite(regime_days) and regime_days > int(args.regime_stuck_days)),
        "router_ok_bear_offense_zero": bool(not (current_regime == "BEAR" and abs(off_scaled) > 1e-9)),
        "router_ok_bull_defense_zero": bool(not (current_regime == "BULL" and abs(def_scaled) > 1e-9)),
    }
    out_row["flags_text"] = "No flags" if not all_reasons else "; ".join(all_reasons)

    # Output files
    daily_path = out_dir / f"daily_check_{now.strftime('%Y%m%d')}.csv"
    log_path = out_dir / "daily_checks_log.csv"
    transitions_path = out_dir / "regime_transitions.csv"

    pd.DataFrame([out_row]).to_csv(daily_path, index=False)

    if log_path.exists():
        lg = pd.read_csv(log_path)
        lg = lg[lg["date"].astype(str) != today]
        lg = pd.concat([lg, pd.DataFrame([out_row])], ignore_index=True)
    else:
        lg = pd.DataFrame([out_row])
    lg = lg.sort_values("date")
    lg.to_csv(log_path, index=False)

    # BTC sleeve log (mirrors ETH daily log style with BTC-specific fields).
    btc_log_path = out_dir / "btc_daily_checks_log.csv"
    btc_row = {
        "date": today,
        "timestamp_utc": str(now),
        "status": status,
        "paper_start_date": str(paper_start_ts.date()),
        "days_live": days_live,
        "regime": current_regime,
        "btc_price": btc_price,
        "btc_24h_pct": btc_24h_pct,
        "btc_ema15": btc_ema21,
        "btc_ema40": btc_ema55,
        "btc_ema120": btc_ema144,
        "btc_ema21": btc_ema21,
        "btc_ema55": btc_ema55,
        "btc_ema144": btc_ema144,
        "btc_stack_aligned": bool(btc_stack_aligned),
        "btc_stack_aligned_days": btc_stack_aligned_days,
        "btc_entry_threshold_met": bool(btc_entry_threshold_met),
        "btc_funding_z": btc_funding_z,
        "btc_position": bool(btc_off_pos),
        "btc_combined_weight": btc_comb_w,
        "vol_regime": str(vol_state.get("vol_regime", "NA")),
        "vol_percentile": _safe_num(vol_state.get("vol_percentile", np.nan)),
        "transition_strength": str(btc_transition.get("transition_strength", "NA")),
        "transition_multiplier": _safe_num(btc_transition.get("transition_multiplier", np.nan)),
        "btc_conviction_score": _safe_num(btc_conviction.get("conviction_score", np.nan)),
        "btc_conviction_bucket": str(btc_conviction.get("conviction_bucket", "NA")),
        "btc_conviction_multiplier": _safe_num(btc_conviction.get("conviction_multiplier", np.nan)),
        "btc_off_entry_ts": str(btc_off_entry_ts) if pd.notna(btc_off_entry_ts) else "",
        "btc_off_days_held": btc_off_days_held,
        "btc_off_hold_return": btc_off_hold_ret,
        "btc_sleeve_strategy_ret": btc_sleeve_strategy_ret,
        "btc_sleeve_spot_ret": btc_sleeve_spot_ret,
    }
    if btc_log_path.exists():
        blg = pd.read_csv(btc_log_path)
        blg = blg[blg["date"].astype(str) != today]
        blg = pd.concat([blg, pd.DataFrame([btc_row])], ignore_index=True)
    else:
        blg = pd.DataFrame([btc_row])
    blg = blg.sort_values("date")
    blg.to_csv(btc_log_path, index=False)

    # Regime transition log (daily level)
    lg2 = lg.copy()
    if len(lg2):
        lg2["date"] = pd.to_datetime(lg2["date"], errors="coerce")
        lg2 = lg2.sort_values("date")
        last_idx = lg2["date"].idxmax()
        prev = lg2[lg2["date"] < lg2.loc[last_idx, "date"]]
        if len(prev):
            prev_row = prev.iloc[-1]
            curr_row = lg2.loc[last_idx]
            prev_reg = str(prev_row.get("regime", "NA"))
            curr_reg = str(curr_row.get("regime", "NA"))
            if prev_reg != curr_reg and curr_reg != "NA" and prev_reg != "NA":
                tr = {
                    "date": str(curr_row["date"].date()),
                    "from_regime": prev_reg,
                    "to_regime": curr_reg,
                    "days_in_prior": _safe_num(prev_row.get("regime_days", np.nan)),
                    "price_at_transition": _safe_num(price[_close_col(price)].iloc[-1]) if len(price) and _close_col(price) else np.nan,
                }
                if transitions_path.exists():
                    t = pd.read_csv(transitions_path)
                    t = pd.concat([t, pd.DataFrame([tr])], ignore_index=True)
                    t = t.drop_duplicates(subset=["date", "from_regime", "to_regime"], keep="last")
                else:
                    t = pd.DataFrame([tr])
                t = t.sort_values("date")
                t.to_csv(transitions_path, index=False)
        elif not transitions_path.exists():
            pd.DataFrame(columns=["date", "from_regime", "to_regime", "days_in_prior", "price_at_transition"]).to_csv(
                transitions_path, index=False
            )

    print(f"wrote {daily_path}")
    print(f"wrote {log_path}")
    print(f"wrote {btc_log_path}")
    print(f"wrote {completed_trades_path}")
    if transitions_path.exists():
        print(f"wrote {transitions_path}")

    # Optional Discord summary
    webhook = str(args.discord_webhook or "").strip()
    if webhook:
        summary_payload = {
            "status": status,
            "regime": current_regime,
            "regime_days": int(regime_days) if np.isfinite(regime_days) else "NA",
            "dd_20d": dd_20d,
            "eth_price": eth_price,
            "eth_24h_pct": eth_24h_pct,
            "btc_price": btc_price,
            "btc_24h_pct": btc_24h_pct,
            "off_position": off_pos,
            "off_weight_raw": off_w,
            "off_weight_scaled": off_scaled,
            "def_position": def_pos,
            "def_weight_raw": def_w,
            "def_weight_scaled": def_scaled,
            "combined_weight": comb_w,
            "eth_execution_weight": eth_execution_weight,
            "mean_reversion_enabled": out_row["mean_reversion_enabled"],
            "mean_reversion_active": out_row["mean_reversion_active"],
            "mean_reversion_weight": out_row["mean_reversion_weight"],
            "mean_reversion_remaining_cap": out_row["mean_reversion_remaining_cap"],
            "mean_reversion_z_score": out_row["mean_reversion_z_score"],
            "mean_reversion_daily_ret": out_row["mean_reversion_daily_ret"],
            "mean_reversion_entry_signal": out_row["mean_reversion_entry_signal"],
            "mean_reversion_exit_signal": out_row["mean_reversion_exit_signal"],
            "mean_reversion_days_held": out_row["mean_reversion_days_held"],
            "mean_reversion_exit_reason": out_row["mean_reversion_exit_reason"],
            "vol_regime": out_row["vol_regime"],
            "vol_percentile": out_row["vol_percentile"],
            "vol_multiplier": out_row["vol_multiplier"],
            "transition_strength": out_row["transition_strength"],
            "transition_multiplier": out_row["transition_multiplier"],
            "btc_transition_strength": out_row["btc_transition_strength"],
            "btc_transition_multiplier": out_row["btc_transition_multiplier"],
            "conviction_score": out_row["conviction_score"],
            "conviction_bucket": out_row["conviction_bucket"],
            "conviction_multiplier": out_row["conviction_multiplier"],
            "btc_conviction_score": out_row["btc_conviction_score"],
            "btc_conviction_bucket": out_row["btc_conviction_bucket"],
            "btc_conviction_multiplier": out_row["btc_conviction_multiplier"],
            "cum_strategy_ret": cum_strategy_ret,
            "cum_spot_ret": cum_spot_ret,
            "excess": excess,
            "peak_dd": peak_dd,
            "eth_sleeve_strategy_ret": eth_sleeve_strategy_ret,
            "eth_sleeve_spot_ret": eth_sleeve_spot_ret,
            "btc_sleeve_strategy_ret": btc_sleeve_strategy_ret,
            "btc_sleeve_spot_ret": btc_sleeve_spot_ret,
            "portfolio_strategy_ret": portfolio_strategy_ret,
            "portfolio_spot_ret": portfolio_spot_ret,
            "portfolio_excess_vs_basket": portfolio_excess_vs_basket,
            "portfolio_peak_dd": portfolio_peak_dd,
            "dvol_atm_iv_30d": out_row["dvol_atm_iv_30d"],
            "dvol_iv_percentile": out_row["dvol_iv_percentile"],
            "dvol_options_vol_regime": out_row["dvol_options_vol_regime"],
            "dvol_rv_vol_regime": out_row["dvol_rv_vol_regime"],
            "dvol_agreement": out_row["dvol_agreement"],
            "dvol_term_slope": out_row["dvol_term_slope"],
            "dvol_history_days": out_row["dvol_history_days"],
            "dvol_insufficient_history": out_row["dvol_insufficient_history"],
            "live_mode_requested": out_row["live_mode_requested"],
            "live_execution_dry_run": out_row["live_execution_dry_run"],
            "live_execution_error": out_row["live_execution_error"],
            "live_total_fees_usd": out_row["live_total_fees_usd"],
            "live_eth_status": out_row["live_eth_status"],
            "live_btc_status": out_row["live_btc_status"],
            "flags_text": out_row["flags_text"],
            "paper_start_date": out_row["paper_start_date"],
            "days_live": out_row["days_live"],
            "eth_ema_fast": out_row["eth_ema_fast"],
            "eth_ema_mid": out_row["eth_ema_mid"],
            "eth_ema_slow": out_row["eth_ema_slow"],
            "ema21": ema21,
            "ema55": ema55,
            "ema144": ema144,
            "stack_aligned": bool(stack_aligned),
            "stack_aligned_days": int(stack_aligned_days) if np.isfinite(stack_aligned_days) else "NA",
            "off_days_since_break": int(off_days_since_break) if np.isfinite(off_days_since_break) else "NA",
            "off_exit_confirmed": bool(off_exit_confirmed),
            "entry_threshold_met": bool(entry_threshold_met),
            "defense_live_mode": out_row["defense_live_mode"],
            "def_signal_ts": out_row["def_signal_ts"] or "NA",
            "def_p_up": out_row["def_p_up"],
            "gold_price": out_row["gold_price"],
            "gold_24h_pct": out_row["gold_24h_pct"],
            "gold_ema21": out_row["gold_ema21"],
            "gold_ema55": out_row["gold_ema55"],
            "gold_ema144": out_row["gold_ema144"],
            "gold_entry_threshold_met": out_row["gold_entry_threshold_met"],
            "gold_flat_bear_gate": out_row["gold_flat_bear_gate"],
            "gold_condition_met": out_row["gold_condition_met"],
            "gold_position_active": out_row["gold_position_active"],
            "gold_entry_date": str(gold_entry_ts.date()) if pd.notna(gold_entry_ts) else "NA",
            "gold_days_held": int(gold_days_held) if np.isfinite(gold_days_held) else "NA",
            "gold_return_since_entry": out_row["gold_return_since_entry"],
            "gold_paper_return": out_row["gold_paper_return"],
            "gold_paper_pnl_gbp": out_row["gold_paper_pnl_gbp"],
            "news_alert_line": out_row["news_alert_line"],
            "markets_lines": markets_lines,
            "off_entry_date": str(off_entry_ts.date()) if pd.notna(off_entry_ts) else "NA",
            "off_days_held": int(off_days_held) if np.isfinite(off_days_held) else "NA",
            "off_trade_ret": off_hold_ret,
            "off_closed_today": bool(off_closed_today),
            "off_close_reason": off_close_reason,
            "off_close_entry_date": str(off_close_entry_ts.date()) if pd.notna(off_close_entry_ts) else "NA",
            "off_close_entry_price": off_close_entry_price,
            "off_close_exit_date": str(off_close_exit_ts.date()) if pd.notna(off_close_exit_ts) else "NA",
            "off_close_exit_price": off_close_exit_price,
            "off_close_days_held": int(off_close_days_held) if np.isfinite(off_close_days_held) else "NA",
            "off_close_return": off_close_return,
            "off_last_trade_closed": bool(off_last_trade_closed),
            "off_last_entry_date": str(last_closed.get("entry_date", "NA")) if off_last_trade_closed else "NA",
            "off_last_entry_price": _safe_num(last_closed.get("entry_price", np.nan)) if off_last_trade_closed else np.nan,
            "off_last_exit_date": str(last_closed.get("exit_date", "NA")) if off_last_trade_closed else "NA",
            "off_last_exit_price": _safe_num(last_closed.get("exit_price", np.nan)) if off_last_trade_closed else np.nan,
            "off_last_exit_reason": str(last_closed.get("exit_reason", "NA")) if off_last_trade_closed else "NA",
            "off_last_days_held": int(_safe_num(last_closed.get("days_held", np.nan))) if off_last_trade_closed and np.isfinite(_safe_num(last_closed.get("days_held", np.nan))) else "NA",
            "off_last_return_pct": _safe_num(last_closed.get("return_pct", np.nan)) if off_last_trade_closed else np.nan,
            "btc_regime": current_regime,
            "btc_ema15": btc_ema21,
            "btc_ema40": btc_ema55,
            "btc_ema120": btc_ema144,
            "btc_ema21": btc_ema21,
            "btc_ema55": btc_ema55,
            "btc_ema144": btc_ema144,
            "btc_stack_aligned": bool(btc_stack_aligned),
            "btc_stack_aligned_days": int(btc_stack_aligned_days) if np.isfinite(btc_stack_aligned_days) else "NA",
            "btc_entry_threshold_met": bool(btc_entry_threshold_met),
            "btc_funding_z": btc_funding_z,
            "btc_position": bool(btc_off_pos),
            "btc_off_weight_raw": btc_off_w,
            "btc_combined_weight": btc_comb_w,
            "btc_entry_date": str(btc_off_entry_ts.date()) if pd.notna(btc_off_entry_ts) else "NA",
            "btc_days_held": int(btc_off_days_held) if np.isfinite(btc_off_days_held) else "NA",
            "btc_trade_ret": btc_off_hold_ret,
        }
        ok, msg = _send_discord_summary(webhook, summary_payload, timeout_sec=float(args.discord_timeout_sec))
        print(msg if ok else f"WARNING: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
