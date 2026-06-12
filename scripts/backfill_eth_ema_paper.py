from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cfg_list(cfg: dict[str, object], key: str, default: list[int]) -> list[int]:
    raw = cfg.get(key, default)
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.replace("/", ",").split(",") if x.strip()]
    elif isinstance(raw, list):
        parts = raw
    else:
        return default
    try:
        vals = [int(x) for x in parts]
    except Exception:
        return default
    return vals if len(vals) == 3 else default


def _cfg_float(cfg: dict[str, object], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def _cfg_int(cfg: dict[str, object], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return default


def _read_eth_daily(price_csv: Path) -> pd.Series:
    d = pd.read_csv(price_csv, low_memory=False)
    ts_col = next((c for c in ["timestamp", "ts", "date", "datetime", "Date"] if c in d.columns), None)
    close_col = next((c for c in ["close", "Close", "adj_close", "Adj Close"] if c in d.columns), None)
    if ts_col is None or close_col is None:
        raise SystemExit(f"{price_csv} needs timestamp/date and close columns")
    d["timestamp"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d[close_col], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    daily = d.assign(day=d["timestamp"].dt.floor("D")).groupby("day")["close"].last()
    return daily.dropna().sort_index()


def _consecutive_true(s: pd.Series) -> pd.Series:
    vals: list[int] = []
    n = 0
    for v in s.fillna(False).astype(bool):
        n = n + 1 if v else 0
        vals.append(n)
    return pd.Series(vals, index=s.index)


def _router_weight(regime: str) -> float:
    r = str(regime).upper()
    if r == "BULL":
        return 0.8
    if r == "CHOP":
        return 0.4
    return 0.0


def _conviction_multiplier(price: float, ema_mid: float, ema_slow: float, regime: str, vol_regime: str) -> tuple[float, str, float]:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(ema_mid) or not np.isfinite(ema_slow):
        return 1.0, "NA", np.nan
    gap_pct = (ema_mid - ema_slow) / price
    if gap_pct > 0.02:
        f1 = 0.33
    elif gap_pct > 0.01:
        f1 = 0.20
    else:
        f1 = 0.10
    r = str(regime).upper()
    f2 = 0.33 if r == "BULL" else 0.17 if r == "CHOP" else 0.0
    v = str(vol_regime).upper()
    f3 = 0.33 if v == "LOW" else 0.10 if v == "HIGH" else 0.20
    score = float(f1 + f2 + f3)
    if score > 0.80:
        return 1.2, "HIGH", score
    if score >= 0.60:
        return 1.0, "MID", score
    if score >= 0.40:
        return 0.8, "LOW", score
    return 0.6, "FLOOR", score


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill paper log ETH EMA/weight/performance fields from portfolio_config.json.")
    p.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    p.add_argument("--log-csv", default="artifacts/paper_trade/daily_checks_log.csv")
    p.add_argument("--config", default="config/portfolio_config.json")
    p.add_argument("--out-csv", default="artifacts/paper_trade/daily_checks_log.csv")
    p.add_argument("--backup-csv", default="")
    args = p.parse_args()

    cfg = _load_config(Path(args.config))
    eth_ema = _cfg_list(cfg, "eth_ema", [21, 55, 144])
    confirm_days = _cfg_int(cfg, "eth_confirm_days", 3)
    exit_confirm_days = 3
    target_vol = 0.50
    vol_floor = 0.25
    vol_cap = 1.00
    gross_cap = _cfg_float(cfg, "gross_cap", 0.8)

    log_path = Path(args.log_csv)
    log = pd.read_csv(log_path, low_memory=False)
    log["day"] = pd.to_datetime(log["date"], utc=True, errors="coerce").dt.floor("D")
    log = log.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    if log.empty:
        raise SystemExit("No daily log rows to backfill")

    close = _read_eth_daily(Path(args.price_csv))
    idx = pd.date_range(log["day"].min(), log["day"].max(), freq="D", tz="UTC")
    close = close.reindex(close.index.union(idx)).sort_index().ffill().reindex(idx)

    e_fast = close.ewm(span=eth_ema[0], adjust=False).mean()
    e_mid = close.ewm(span=eth_ema[1], adjust=False).mean()
    e_slow = close.ewm(span=eth_ema[2], adjust=False).mean()
    stack = (e_fast > e_mid) & (e_mid > e_slow)
    confirmed = (stack.rolling(confirm_days, min_periods=confirm_days).min() == 1).fillna(False)
    broken = ~stack
    exit_confirmed = (broken.rolling(exit_confirm_days, min_periods=exit_confirm_days).min() == 1).fillna(False)

    daily_r = np.log(close / close.shift(1)).fillna(0.0)
    rv = daily_r.rolling(20, min_periods=10).std() * np.sqrt(365.0)
    vol_scalar = (target_vol / rv.replace(0.0, np.nan)).clip(lower=vol_floor, upper=vol_cap).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos = 0
    pos_vals: list[int] = []
    for day in idx:
        if pos == 0 and bool(confirmed.loc[day]):
            pos = 1
        elif pos == 1 and bool(exit_confirmed.loc[day]):
            pos = 0
        pos_vals.append(pos)
    pos_s = pd.Series(pos_vals, index=idx)

    aligned_days = _consecutive_true(stack)
    broken_days = _consecutive_true(broken)
    eth_spot_ret = close.pct_change().fillna(0.0)

    keyed = pd.DataFrame(
        {
            "day": idx,
            "eth_price_new": close.values,
            "ema_fast_new": e_fast.values,
            "ema_mid_new": e_mid.values,
            "ema_slow_new": e_slow.values,
            "stack_aligned_new": stack.values,
            "stack_aligned_days_new": aligned_days.values,
            "off_days_since_break_new": broken_days.values,
            "entry_threshold_met_new": confirmed.values,
            "off_exit_confirmed_new": exit_confirmed.values,
            "off_position_new": pos_s.astype(bool).values,
            "off_weight_raw_new": (pos_s * vol_scalar).values,
            "eth_spot_return_new": eth_spot_ret.values,
        }
    )
    log = log.merge(keyed, on="day", how="left")

    regimes = log["regime"].astype(str).str.upper()
    vol_regimes = log.get("vol_regime", pd.Series(["NORMAL"] * len(log))).astype(str)
    router = regimes.map(_router_weight).fillna(0.0).astype(float)
    convictions = [
        _conviction_multiplier(
            float(row["eth_price_new"]),
            float(row["ema_mid_new"]),
            float(row["ema_slow_new"]),
            str(row["regime"]),
            str(row.get("vol_regime", "NORMAL")),
        )
        for _, row in log.iterrows()
    ]
    conv_mult = pd.Series([x[0] for x in convictions], index=log.index)
    conv_bucket = pd.Series([x[1] for x in convictions], index=log.index)
    conv_score = pd.Series([x[2] for x in convictions], index=log.index)

    off_weight_raw = pd.to_numeric(log["off_weight_raw_new"], errors="coerce").fillna(0.0)
    off_scaled = (off_weight_raw * router * conv_mult).clip(lower=0.0, upper=gross_cap)
    eth_strategy_day = off_scaled.shift(1).fillna(0.0) * pd.to_numeric(log["eth_spot_return_new"], errors="coerce").fillna(0.0)
    eth_sleeve_strategy_ret = (1.0 + eth_strategy_day).cumprod() - 1.0
    eth_sleeve_spot_ret = (1.0 + pd.to_numeric(log["eth_spot_return_new"], errors="coerce").fillna(0.0)).cumprod() - 1.0

    btc_sleeve = pd.to_numeric(log.get("btc_sleeve_strategy_ret", 0.0), errors="coerce").fillna(0.0)
    btc_day = (1.0 + btc_sleeve).div((1.0 + btc_sleeve).shift(1).fillna(1.0)) - 1.0
    port_day = 0.5 * eth_strategy_day + 0.5 * btc_day
    port_ret = (1.0 + port_day).cumprod() - 1.0
    eth_spot = eth_sleeve_spot_ret
    btc_spot = pd.to_numeric(log.get("btc_sleeve_spot_ret", 0.0), errors="coerce").fillna(0.0)
    basket = 0.5 * eth_spot + 0.5 * btc_spot
    port_eq = 1.0 + port_ret
    port_peak_dd = port_eq / port_eq.cummax() - 1.0

    log["eth_ema_fast"] = eth_ema[0]
    log["eth_ema_mid"] = eth_ema[1]
    log["eth_ema_slow"] = eth_ema[2]
    log["ema_state"] = np.where(
        log["ema_fast_new"] > log["ema_mid_new"],
        np.where(log["ema_mid_new"] > log["ema_slow_new"], f"bullish_{eth_ema[0]}>{eth_ema[1]}>{eth_ema[2]}", "mixed"),
        np.where(log["ema_mid_new"] < log["ema_slow_new"], f"bearish_{eth_ema[0]}<{eth_ema[1]}<{eth_ema[2]}", "mixed"),
    )
    log["ema21"] = log["ema_fast_new"]
    log["ema55"] = log["ema_mid_new"]
    log["ema144"] = log["ema_slow_new"]
    log["eth_price"] = log["eth_price_new"].combine_first(pd.to_numeric(log.get("eth_price"), errors="coerce"))
    log["stack_aligned"] = log["stack_aligned_new"].astype(bool)
    log["stack_aligned_days"] = log["stack_aligned_days_new"]
    log["off_days_since_break"] = log["off_days_since_break_new"]
    log["entry_threshold_met"] = log["entry_threshold_met_new"].astype(bool)
    log["off_exit_confirmed"] = log["off_exit_confirmed_new"].astype(bool)
    log["off_position"] = log["off_position_new"].astype(bool)
    log["off_weight"] = off_weight_raw
    log["off_weight_scaled"] = off_scaled
    log["combined_weight"] = off_scaled
    log["eth_execution_weight"] = off_scaled
    log["conviction_bucket"] = conv_bucket
    log["conviction_score"] = conv_score
    log["conviction_multiplier"] = conv_mult
    log["eth_sleeve_strategy_ret"] = eth_sleeve_strategy_ret
    log["eth_sleeve_spot_ret"] = eth_sleeve_spot_ret
    log["portfolio_strategy_ret"] = port_ret
    log["portfolio_excess_vs_basket"] = port_ret - basket
    log["portfolio_peak_dd"] = port_peak_dd.cummin()
    log["cum_strategy_ret"] = eth_sleeve_strategy_ret
    log["cum_spot_ret"] = eth_sleeve_spot_ret
    log["excess"] = eth_sleeve_strategy_ret - eth_sleeve_spot_ret

    drop_cols = [c for c in log.columns if c.endswith("_new")] + ["day"]
    log = log.drop(columns=drop_cols, errors="ignore")

    backup = Path(args.backup_csv) if args.backup_csv else log_path.with_suffix(".pre_eth_ema_backfill.csv")
    if Path(args.out_csv).resolve() == log_path.resolve():
        log_path.replace(backup)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    print(f"ETH EMA backfill complete: {eth_ema[0]}/{eth_ema[1]}/{eth_ema[2]}")
    print(f"Rows: {len(log)}")
    print(f"Backup: {backup}")
    print(f"Wrote: {out}")
    print(f"Latest portfolio strategy ret: {float(log['portfolio_strategy_ret'].iloc[-1]) * 100:.2f}%")
    print(f"Latest excess vs basket: {float(log['portfolio_excess_vs_basket'].iloc[-1]) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
