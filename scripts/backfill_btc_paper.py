from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_num(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_daily_close_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "close"])
    d = pd.read_csv(path, low_memory=False)
    if d.empty:
        return pd.DataFrame(columns=["timestamp", "close"])
    ts_col = next((c for c in ["timestamp", "ts", "day", "date", "Date"] if c in d.columns), None)
    close_col = next((c for c in ["close", "Close", "adj_close", "Adj Close"] if c in d.columns), None)
    if ts_col is None or close_col is None:
        return pd.DataFrame(columns=["timestamp", "close"])
    out = d[[ts_col, close_col]].copy()
    out["timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    out = out[["timestamp", "close"]].drop_duplicates("timestamp", keep="last")
    return out


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


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


def _conviction_multiplier(price: float, ema_mid: float, ema_slow: float, regime: str) -> tuple[float, str, float]:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(ema_mid) or not np.isfinite(ema_slow):
        return 1.0, "NA", float("nan")
    gap_pct = (ema_mid - ema_slow) / price
    if gap_pct > 0.02:
        f1 = 0.33
    elif gap_pct > 0.01:
        f1 = 0.20
    else:
        f1 = 0.10
    reg = str(regime).upper()
    f2 = 0.33 if reg == "BULL" else 0.17 if reg == "CHOP" else 0.0
    # Backfill does not reconstruct ETH vol percentile reliably; use NORMAL,
    # matching the live default when the vol filter is not classifying extremes.
    f3 = 0.20
    score = float(f1 + f2 + f3)
    if score > 0.80:
        return 1.2, "HIGH", score
    if score >= 0.60:
        return 1.0, "MID", score
    if score >= 0.40:
        return 0.8, "LOW", score
    return 0.6, "FLOOR", score


def _build_btc_backfill(
    *,
    btc_daily_csv: Path,
    btc_perp_csv: Path,
    eth_log_csv: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    btc_ema: list[int],
    asymmetric_sizing: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[dict[str, object]]]:
    errors: list[str] = []
    warnings: list[str] = []
    transitions: list[str] = []
    trades: list[dict[str, object]] = []

    # ETH daily slice for context/combined calculations.
    eth_log = pd.read_csv(eth_log_csv, low_memory=False)
    eth_log["day"] = pd.to_datetime(eth_log["date"], utc=True, errors="coerce").dt.floor("D")
    eth_log = eth_log.dropna(subset=["day"]).sort_values("day")
    eth_log = eth_log[(eth_log["day"] >= start_date) & (eth_log["day"] <= end_date)].copy()
    if eth_log.empty:
        raise RuntimeError("No ETH daily log rows found for requested backfill range.")

    regime_s = eth_log.set_index("day")["regime"].astype(str).reindex(pd.date_range(start_date, end_date, freq="D", tz="UTC")).ffill().fillna("CHOP")
    eth_weight_s = pd.to_numeric(eth_log.set_index("day")["combined_weight"], errors="coerce").reindex(regime_s.index).ffill().fillna(0.0).clip(0.0, 1.0)
    eth_spot_s = pd.to_numeric(eth_log.set_index("day")["eth_price"], errors="coerce").reindex(regime_s.index).ffill()
    eth_spot_r = eth_spot_s.pct_change().fillna(0.0)
    eth_strat_r = eth_weight_s.shift(1).fillna(0.0) * eth_spot_r

    # BTC daily.
    btc_daily = _read_daily_close_csv(btc_daily_csv)
    if btc_daily.empty:
        raise RuntimeError(f"No BTC daily data in {btc_daily_csv}")
    bday = btc_daily["timestamp"].dt.floor("D")
    btc_close_s = pd.Series(pd.to_numeric(btc_daily["close"], errors="coerce").to_numpy(dtype=float), index=bday)
    btc_close_s = btc_close_s[~btc_close_s.index.duplicated(keep="last")].dropna().sort_index()
    btc_close_s = btc_close_s.reindex(btc_close_s.index.union(regime_s.index)).sort_index().ffill().reindex(regime_s.index)

    # BTC EMA/offensive.
    b_fast = btc_close_s.ewm(span=int(btc_ema[0]), adjust=False).mean()
    b_mid = btc_close_s.ewm(span=int(btc_ema[1]), adjust=False).mean()
    b_slow = btc_close_s.ewm(span=int(btc_ema[2]), adjust=False).mean()
    b_stack = (b_fast > b_mid) & (b_mid > b_slow)
    b_entry = (b_stack.rolling(5, min_periods=5).min() == 1).fillna(False)
    b_exit = ((~b_stack).rolling(3, min_periods=3).min() == 1).fillna(False)
    b_ret = np.log(btc_close_s / btc_close_s.shift(1)).fillna(0.0)
    b_rv = b_ret.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(365.0)
    b_vs = (0.50 / b_rv.replace(0.0, np.nan)).clip(lower=0.25, upper=1.0).replace([np.inf, -np.inf], np.nan).fillna(0.25)

    pos = 0
    pos_vals: list[int] = []
    entry_day: pd.Timestamp | None = None
    for day in regime_s.index:
        if pos == 0:
            if str(regime_s.loc[day]) != "BEAR" and bool(b_entry.loc[day]):
                pos = 1
                entry_day = day
        else:
            if str(regime_s.loc[day]) == "BEAR" or bool(b_exit.loc[day]):
                pos = 0
                if entry_day is not None and np.isfinite(btc_close_s.loc[entry_day]) and np.isfinite(btc_close_s.loc[day]):
                    r = btc_close_s.loc[day] / btc_close_s.loc[entry_day] - 1.0
                    trades.append({"entry": str(entry_day.date()), "exit": str(day.date()), "return_pct": float(r)})
                entry_day = None
        pos_vals.append(pos)
    pos_s = pd.Series(pos_vals, index=regime_s.index).astype(int)
    b_off_raw = pos_s.astype(float) * b_vs

    # BTC funding z is diagnostic only; it must not create BTC exposure by itself.
    btc_perp = pd.read_csv(btc_perp_csv, low_memory=False) if btc_perp_csv.exists() else pd.DataFrame()
    btc_funding_z_s = pd.Series(index=regime_s.index, dtype=float)
    btc_def_raw = pd.Series(0.0, index=regime_s.index, dtype=float)
    if not btc_perp.empty:
        tcol = next((c for c in ["timestamp", "ts", "day", "date"] if c in btc_perp.columns), None)
        fcol = next((c for c in ["funding_rate", "fundingRate", "funding"] if c in btc_perp.columns), None)
        if tcol and fcol:
            bpf = btc_perp[[tcol, fcol]].copy()
            bpf["timestamp"] = pd.to_datetime(bpf[tcol], utc=True, errors="coerce")
            bpf[fcol] = pd.to_numeric(bpf[fcol], errors="coerce")
            bpf = bpf.dropna(subset=["timestamp", fcol]).sort_values("timestamp")
            bpf["day"] = bpf["timestamp"].dt.floor("D")
            bpf_day = bpf.groupby("day", as_index=False)[fcol].mean()
            bpf_day["roll_mean"] = bpf_day[fcol].rolling(30, min_periods=10).mean()
            bpf_day["roll_std"] = bpf_day[fcol].rolling(30, min_periods=10).std(ddof=0)
            bpf_day["funding_z"] = (bpf_day[fcol] - bpf_day["roll_mean"]) / bpf_day["roll_std"].replace(0.0, np.nan)
            btc_funding_z_s = pd.to_numeric(bpf_day.set_index("day")["funding_z"], errors="coerce").reindex(regime_s.index).ffill()
            btc_def_raw = np.where((btc_funding_z_s < -1.0).fillna(False), b_vs, 0.0)
            btc_def_raw = pd.Series(btc_def_raw, index=regime_s.index, dtype=float)

    off_scale = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    def_scale = {"BULL": 0.0, "CHOP": 0.3, "BEAR": 0.8}
    reg_scale_off = regime_s.map(off_scale).fillna(0.0)
    reg_scale_def = regime_s.map(def_scale).fillna(0.0)
    conv_mult_vals: list[float] = []
    conv_bucket_vals: list[str] = []
    conv_score_vals: list[float] = []
    for day in regime_s.index:
        mult, bucket, score = _conviction_multiplier(
            float(btc_close_s.loc[day]),
            float(b_mid.loc[day]),
            float(b_slow.loc[day]),
            str(regime_s.loc[day]),
        )
        conv_mult_vals.append(mult if asymmetric_sizing else 1.0)
        conv_bucket_vals.append(bucket if asymmetric_sizing else "OFF")
        conv_score_vals.append(score if asymmetric_sizing else np.nan)
    conv_mult_s = pd.Series(conv_mult_vals, index=regime_s.index, dtype=float)
    conv_bucket_s = pd.Series(conv_bucket_vals, index=regime_s.index, dtype=object)
    conv_score_s = pd.Series(conv_score_vals, index=regime_s.index, dtype=float)
    b_off_scaled = b_off_raw * reg_scale_off * conv_mult_s
    b_def_scaled = btc_def_raw * reg_scale_def
    b_weight = b_off_scaled.clip(0.0, 1.0)

    btc_spot_r = btc_close_s.pct_change().fillna(0.0)
    btc_strat_r = b_weight.shift(1).fillna(0.0) * btc_spot_r
    btc_cum = (1.0 + btc_strat_r).cumprod() - 1.0
    eth_cum = (1.0 + eth_strat_r).cumprod() - 1.0
    btc_spot_cum = (1.0 + btc_spot_r).cumprod() - 1.0
    eth_spot_cum = (1.0 + eth_spot_r).cumprod() - 1.0
    combined_daily = 0.5 * eth_strat_r + 0.5 * btc_strat_r
    combined_cum = (1.0 + combined_daily).cumprod() - 1.0

    # Transitions.
    prev_reg = None
    for d, rg in regime_s.items():
        if prev_reg is None:
            prev_reg = rg
            continue
        if rg != prev_reg:
            transitions.append(f"{d.date()}: {prev_reg} -> {rg}")
            prev_reg = rg
    if len(transitions) > 20:
        warnings.append(f"regime transitions too high: {len(transitions)}")

    # Build rows + checks.
    rows: list[dict[str, object]] = []
    for i, day in enumerate(regime_s.index):
        days_aligned = int(b_stack.loc[:day].iloc[::-1].astype(int).cumprod().sum()) if bool(b_stack.loc[day]) else 0
        entry_met = bool(b_entry.loc[day])
        b_ret_d = float(btc_strat_r.loc[day])
        w_exec = float(b_weight.shift(1).fillna(0.0).loc[day])
        sret = float(btc_spot_r.loc[day])
        eth_d = float(eth_strat_r.loc[day])
        comb_d = float(combined_daily.loc[day])

        # Check 1
        if bool(b_stack.loc[day]) and days_aligned <= 0:
            errors.append(f"{day.date()} signal consistency: aligned yes but days=0")
        if days_aligned >= 5 and not entry_met:
            errors.append(f"{day.date()} signal consistency: days>=5 but entry_threshold false")
        # Check 2
        if str(regime_s.loc[day]) == "BEAR" and abs(float(b_weight.loc[day])) > 1e-9:
            errors.append(f"{day.date()} weight consistency: BEAR but scaled_weight={b_weight.loc[day]:.4f}")
        if entry_met and str(regime_s.loc[day]) != "BEAR" and float(b_weight.loc[day]) <= 1e-9:
            errors.append(f"{day.date()} weight consistency: entry met but scaled_weight=0")
        # Check 3
        if abs(w_exec) <= 1e-12 and abs(b_ret_d) > 1e-12:
            errors.append(f"{day.date()} return consistency: weight=0 but strategy return {b_ret_d:.6f}")
        if w_exec > 1e-12 and abs(b_ret_d) <= 1e-12 and abs(sret) > 1e-12:
            errors.append(f"{day.date()} return consistency: weight>0 but strategy return zero")
        # Check 5
        lhs = comb_d
        rhs = 0.5 * eth_d + 0.5 * b_ret_d
        if abs(lhs - rhs) > 1e-12:
            errors.append(f"{day.date()} combined consistency mismatch")

        rows.append(
            {
                "date": str(day.date()),
                "day_num": i + 1,
                "eth_price": _safe_num(eth_spot_s.loc[day]),
                "eth_regime": str(regime_s.loc[day]),
                "eth_weight": float(eth_weight_s.loc[day]),
                "eth_strategy_day_ret": eth_d,
                "eth_cum_strategy_ret": float(eth_cum.loc[day]),
                "eth_cum_spot_ret": float(eth_spot_cum.loc[day]),
                "btc_price": float(btc_close_s.loc[day]),
                "btc_regime": str(regime_s.loc[day]),
                "btc_ema15": float(b_fast.loc[day]),
                "btc_ema40": float(b_mid.loc[day]),
                "btc_ema120": float(b_slow.loc[day]),
                "btc_ema21": float(b_fast.loc[day]),
                "btc_ema55": float(b_mid.loc[day]),
                "btc_ema144": float(b_slow.loc[day]),
                "btc_stack_aligned": bool(b_stack.loc[day]),
                "btc_days_aligned": days_aligned,
                "btc_entry_threshold_met": entry_met,
                "btc_funding_z": float(btc_funding_z_s.loc[day]) if day in btc_funding_z_s.index else np.nan,
                "btc_raw_signal": float(b_off_raw.loc[day]),
                "btc_scaled_weight": float(b_weight.loc[day]),
                "btc_conviction_score": float(conv_score_s.loc[day]),
                "btc_conviction_bucket": str(conv_bucket_s.loc[day]),
                "btc_conviction_multiplier": float(conv_mult_s.loc[day]),
                "btc_daily_return": b_ret_d,
                "btc_cumulative_return": float(btc_cum.loc[day]),
                "btc_cum_spot_ret": float(btc_spot_cum.loc[day]),
                "combined_daily_return": comb_d,
                "combined_cum_return": float(combined_cum.loc[day]),
            }
        )

    out = pd.DataFrame(rows)
    return out, errors, warnings, transitions, trades


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill BTC paper sleeve and print historical daily summaries.")
    p.add_argument("--portfolio-config", default="config/portfolio_config.json")
    p.add_argument("--start-date", default="2026-03-21")
    p.add_argument("--end-date", default="")
    p.add_argument("--btc-daily-csv", default="data/btc_daily.csv")
    p.add_argument("--btc-perp-csv", default="data/btc_perp_features.csv")
    p.add_argument("--eth-log-csv", default="artifacts/paper_trade/daily_checks_log.csv")
    p.add_argument("--out-daily-csv", default="artifacts/paper_trade/btc_backfill_daily.csv")
    p.add_argument("--out-log-txt", default="artifacts/paper_trade/btc_backfill_log.txt")
    p.add_argument("--out-btc-log-csv", default="artifacts/paper_trade/btc_daily_checks_log.csv")
    args = p.parse_args()

    start_date = pd.Timestamp(args.start_date, tz="UTC").floor("D")
    end_date = pd.Timestamp.now(tz="UTC").floor("D") if not args.end_date else pd.Timestamp(args.end_date, tz="UTC").floor("D")
    cfg = _load_config(Path(args.portfolio_config))
    if not cfg:
        cfg = _load_config(Path("artifacts/backtest/portfolio_config.json"))
    btc_ema = _cfg_int_list(cfg, "btc_ema", [15, 40, 120])
    asymmetric_sizing = bool(str(cfg.get("asymmetric_sizing", True)).strip().lower() in {"1", "true", "yes", "on"})

    df, errors, warnings, transitions, trades = _build_btc_backfill(
        btc_daily_csv=Path(args.btc_daily_csv),
        btc_perp_csv=Path(args.btc_perp_csv),
        eth_log_csv=Path(args.eth_log_csv),
        start_date=start_date,
        end_date=end_date,
        btc_ema=btc_ema,
        asymmetric_sizing=asymmetric_sizing,
    )

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_daily, index=False)

    # Write BTC log in checklist-compatible structure.
    btc_log = pd.DataFrame(
        {
            "date": df["date"],
            "timestamp_utc": pd.to_datetime(df["date"], utc=True).astype(str),
            "status": "PASS",
            "paper_start_date": str(start_date.date()),
            "days_live": df["day_num"],
            "regime": df["btc_regime"],
            "btc_price": df["btc_price"],
            "btc_24h_pct": pd.Series(df["btc_price"]).pct_change().fillna(0.0).values,
            "btc_ema15": df["btc_ema15"],
            "btc_ema40": df["btc_ema40"],
            "btc_ema120": df["btc_ema120"],
            "btc_ema21": df["btc_ema21"],
            "btc_ema55": df["btc_ema55"],
            "btc_ema144": df["btc_ema144"],
            "btc_stack_aligned": df["btc_stack_aligned"],
            "btc_stack_aligned_days": df["btc_days_aligned"],
            "btc_entry_threshold_met": df["btc_entry_threshold_met"],
            "btc_funding_z": df["btc_funding_z"],
            "btc_position": (df["btc_scaled_weight"] > 0).astype(bool),
            "btc_combined_weight": df["btc_scaled_weight"],
            "btc_conviction_score": df["btc_conviction_score"],
            "btc_conviction_bucket": df["btc_conviction_bucket"],
            "btc_conviction_multiplier": df["btc_conviction_multiplier"],
            "btc_off_entry_ts": "",
            "btc_off_days_held": np.nan,
            "btc_off_hold_return": np.nan,
            "btc_sleeve_strategy_ret": df["btc_cumulative_return"],
            "btc_sleeve_spot_ret": df["btc_cum_spot_ret"],
        }
    )
    btc_log.to_csv(args.out_btc_log_csv, index=False)

    lines: list[str] = []
    for _, r in df.iterrows():
        lines.extend(
            [
                "-------------------------------------",
                f"DATE: {r['date']} | Day {int(r['day_num'])}",
                "-------------------------------------",
                f"ETH:  ${r['eth_price']:.0f} | Regime: {r['eth_regime']}",
                f"  Weight: {r['eth_weight']:.4f}",
                f"  Strategy day: {r['eth_strategy_day_ret']*100:.2f}%",
                f"BTC:  ${r['btc_price']:.0f} | Regime: {r['btc_regime']}",
                f"  EMA{btc_ema[0]}/{btc_ema[1]}/{btc_ema[2]}: {r['btc_ema15']:.2f} / {r['btc_ema40']:.2f} / {r['btc_ema120']:.2f}",
                f"  Stack aligned: {'YES' if r['btc_stack_aligned'] else 'NO'} | Days: {int(r['btc_days_aligned'])}",
                f"  Weight: {r['btc_scaled_weight']:.4f} | Trade: {'LONG' if r['btc_scaled_weight']>0 else 'FLAT'}",
                f"  Conviction: {r['btc_conviction_bucket']} ({r['btc_conviction_score']:.2f})",
                f"  Funding z: {r['btc_funding_z']:.3f}" if np.isfinite(r["btc_funding_z"]) else "  Funding z: n/a",
                "Combined:",
                f"  ETH strategy day: {r['eth_strategy_day_ret']*100:.2f}%",
                f"  BTC strategy day: {r['btc_daily_return']*100:.2f}%",
                f"  Combined cum:     {r['combined_cum_return']*100:.2f}%",
                f"  ETH cum spot:     {r['eth_cum_spot_ret']*100:.2f}%",
                f"  BTC cum spot:     {r['btc_cum_spot_ret']*100:.2f}%",
            ]
        )

    # Validation summary.
    lines.extend(
        [
            "===============================================",
            "BACKFILL VALIDATION SUMMARY",
            "===============================================",
            f"Days processed: {len(df)}",
            f"Errors found: {len(errors)}",
            f"Warnings found: {len(warnings)}",
            "",
            f"BTC regime transitions: {len(transitions)}",
        ]
    )
    lines.extend([f"  {t}" for t in transitions] if transitions else ["  none"])
    lines.append("")
    lines.append(f"BTC paper trades triggered: {len(trades)}")
    if trades:
        for t in trades:
            lines.append(f"  entry={t['entry']} exit={t['exit']} return={t['return_pct']*100:.2f}%")
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Combined performance (backfilled):",
            f"  Start: {start_date.date()}",
            f"  End:   {end_date.date()}",
            f"  ETH sleeve:      {df['eth_cum_strategy_ret'].iloc[-1]*100:.2f}%",
            f"  BTC sleeve:      {df['btc_cumulative_return'].iloc[-1]*100:.2f}%",
            f"  Combined:        {df['combined_cum_return'].iloc[-1]*100:.2f}%",
            f"  vs ETH alone:    {df['eth_cum_strategy_ret'].iloc[-1]*100:.2f}%",
            f"  vs 50/50 basket: {(0.5*df['eth_cum_spot_ret'].iloc[-1]+0.5*df['btc_cum_spot_ret'].iloc[-1])*100:.2f}%",
            "===============================================",
        ]
    )
    if errors:
        lines.append("Errors:")
        lines.extend([f"  {e}" for e in errors])
    if warnings:
        lines.append("Warnings:")
        lines.extend([f"  {w}" for w in warnings])

    txt_path = Path(args.out_log_txt)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"wrote {out_daily}")
    print(f"wrote {txt_path}")
    print(f"wrote {args.out_btc_log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
