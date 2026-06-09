from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_btc_full_stack import classify_regime_v2_on_btc


def _safe_float(x: object) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d["timestamp"] = pd.to_datetime(d.get("timestamp", d["day"]), utc=True, errors="coerce")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    return d


def _load_raw(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    ts_col = next((c for c in ["timestamp", "date", "day", "datetime", "Date"] if c in d.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp/date column")
    d["timestamp"] = pd.to_datetime(d[ts_col], utc=True, errors="coerce").dt.floor("D")
    rename = {c: c.lower() for c in d.columns if str(c).lower() in {"open", "high", "low", "close", "volume"}}
    d = d.rename(columns=rename)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return d.reset_index(drop=True)


def _with_manual_signals(raw: pd.DataFrame, ema: tuple[int, int, int]) -> pd.DataFrame:
    d = raw.copy()
    d = classify_regime_v2_on_btc(d)
    e1, e2, e3 = ema
    d["ema_fast"] = d["close"].ewm(span=e1, adjust=False).mean()
    d["ema_mid"] = d["close"].ewm(span=e2, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=e3, adjust=False).mean()
    d["stack_aligned_manual"] = (d["ema_fast"] > d["ema_mid"]) & (d["ema_mid"] > d["ema_slow"])
    d["exec_return_manual"] = d["open"].shift(-1) / d["open"] - 1.0
    d = d.rename(columns={"timestamp": "day", "regime_v2": "regime_manual"})
    return d[["day", "open", "close", "ema_fast", "ema_mid", "ema_slow", "stack_aligned_manual", "regime_manual", "exec_return_manual"]]


def _asset_frame(daily: pd.DataFrame, raw: pd.DataFrame, asset: str, ema: tuple[int, int, int], cost_bps: float) -> pd.DataFrame:
    manual = _with_manual_signals(raw, ema)
    a = daily.merge(manual, on="day", how="left", suffixes=("", "_raw"))
    pref = f"{asset}_"
    a["asset"] = asset.upper()
    a["bt_open"] = pd.to_numeric(a[f"{pref}open"], errors="coerce")
    a["bt_close"] = pd.to_numeric(a[f"{pref}close"], errors="coerce")
    a["bt_regime"] = a[f"{pref}regime"].astype(str)
    a["bt_weight"] = pd.to_numeric(a[f"{pref}weight_exec"], errors="coerce").fillna(0.0)
    a["bt_turnover"] = pd.to_numeric(a[f"{pref}turnover"], errors="coerce").fillna(0.0)
    a["bt_strategy_return"] = pd.to_numeric(a[f"{pref}strategy_return"], errors="coerce").fillna(0.0)
    a["bt_spot_return"] = pd.to_numeric(a[f"{pref}spot_return"], errors="coerce").fillna(0.0)
    a["bt_off_active"] = pd.to_numeric(a[f"{pref}off_active"], errors="coerce").fillna(0).astype(int)
    a["bt_cost_manual"] = a["bt_turnover"] * (float(cost_bps) / 10000.0)
    a["strategy_return_manual"] = a["bt_weight"] * pd.to_numeric(a["exec_return_manual"], errors="coerce").fillna(0.0) - a["bt_cost_manual"]
    a["return_error"] = a["bt_strategy_return"] - a["strategy_return_manual"]
    a["open_error_pct"] = (a["bt_open"] / pd.to_numeric(a["open"], errors="coerce") - 1.0).replace([np.inf, -np.inf], np.nan)
    a["close_error_pct"] = (a["bt_close"] / pd.to_numeric(a["close"], errors="coerce") - 1.0).replace([np.inf, -np.inf], np.nan)
    a["regime_match"] = a["bt_regime"].astype(str).str.upper().eq(a["regime_manual"].astype(str).str.upper())
    return a


def _trade_checks(a: pd.DataFrame, sample_limit: int = 12) -> pd.DataFrame:
    w = pd.to_numeric(a["bt_weight"], errors="coerce").fillna(0.0)
    active = w > 1e-12
    starts = list(a.index[active & ~active.shift(1, fill_value=False)])
    exits = list(a.index[~active & active.shift(1, fill_value=False)])
    if exits and starts and exits[0] < starts[0]:
        exits = exits[1:]
    if len(exits) < len(starts):
        exits.append(len(a) - 1)
    rows = []
    chosen = starts[:3] + starts[max(0, len(starts)//2 - 2):len(starts)//2 + 2] + starts[-3:]
    chosen = [] if not starts else sorted(set(chosen))[:sample_limit]
    for s in chosen:
        e_candidates = [x for x in exits if x > s]
        if not e_candidates:
            continue
        e = int(e_candidates[0])
        g = a.loc[s:e-1].copy() if e > s else a.loc[s:s].copy()
        entry_open = _safe_float(a.loc[s, "bt_open"])
        exit_open = _safe_float(a.loc[e, "bt_open"]) if e < len(a) else _safe_float(a.loc[e, "bt_close"])
        gross_price_return = exit_open / entry_open - 1.0 if entry_open > 0 and exit_open > 0 else np.nan
        compounded_strategy = float((1.0 + g["bt_strategy_return"]).prod() - 1.0) if not g.empty else 0.0
        manual_compounded = float((1.0 + g["strategy_return_manual"]).prod() - 1.0) if not g.empty else 0.0
        rows.append({
            "asset": str(a.loc[s, "asset"]),
            "entry_date": str(a.loc[s, "day"].date()),
            "entry_signal_date": str((a.loc[s, "day"] - pd.Timedelta(days=1)).date()),
            "entry_execution_price": entry_open,
            "entry_signal_close": _safe_float(a.loc[max(0, s-1), "bt_close"]),
            "entry_weight": _safe_float(a.loc[s, "bt_weight"]),
            "entry_stack_aligned_manual": bool(a.loc[s, "stack_aligned_manual"]),
            "entry_regime_backtest": str(a.loc[s, "bt_regime"]),
            "entry_regime_manual": str(a.loc[s, "regime_manual"]),
            "exit_date": str(a.loc[e, "day"].date()) if e < len(a) else str(a.loc[len(a)-1, "day"].date()),
            "exit_execution_price": exit_open,
            "exit_regime_backtest": str(a.loc[e, "bt_regime"]) if e < len(a) else "NA",
            "exit_regime_manual": str(a.loc[e, "regime_manual"]) if e < len(a) else "NA",
            "gross_price_return": gross_price_return,
            "backtest_compounded_return": compounded_strategy,
            "manual_compounded_return": manual_compounded,
            "return_error": compounded_strategy - manual_compounded,
            "days_held": int(max(0, e - s)),
        })
    return pd.DataFrame(rows)


def _event_checks(daily: pd.DataFrame, eth_a: pd.DataFrame) -> pd.DataFrame:
    events = [
        ("2018-01-13", "ETH 2018 peak"),
        ("2018-01-17", "2018 BEAR fired"),
        ("2022-11-08", "FTX collapse"),
        ("2020-03-12", "COVID crash"),
    ]
    rows = []
    for date, label in events:
        day = pd.Timestamp(date, tz="UTC")
        win = eth_a[(eth_a["day"] >= day - pd.Timedelta(days=5)) & (eth_a["day"] <= day + pd.Timedelta(days=5))].copy()
        exact = eth_a[eth_a["day"] == day]
        used_nearest = False
        if exact.empty:
            near = eth_a[(eth_a["day"] >= day - pd.Timedelta(days=3)) & (eth_a["day"] <= day + pd.Timedelta(days=3))].copy()
            if near.empty:
                rows.append({"date": date, "event": label, "found": False, "used_nearest": False})
                continue
            near["dist"] = (near["day"] - day).abs()
            exact = near.sort_values("dist").head(1)
            used_nearest = True
        r = exact.iloc[0]
        rows.append({
            "date": date,
            "event": label,
            "found": True,
            "used_nearest": bool(used_nearest),
            "checked_date": str(r["day"].date()),
            "eth_close": _safe_float(r["bt_close"]),
            "ema_fast": _safe_float(r["ema_fast"]),
            "ema_mid": _safe_float(r["ema_mid"]),
            "ema_slow": _safe_float(r["ema_slow"]),
            "stack_aligned_manual": bool(r["stack_aligned_manual"]),
            "regime_backtest": str(r["bt_regime"]),
            "regime_manual": str(r["regime_manual"]),
            "eth_weight": _safe_float(r["bt_weight"]),
            "eth_strategy_return_day": _safe_float(r["bt_strategy_return"]),
            "eth_spot_return_day": _safe_float(r["bt_spot_return"]),
            "window_strategy_return": float((1.0 + win["bt_strategy_return"]).prod() - 1.0) if not win.empty else np.nan,
            "window_min_strategy_day": float(win["bt_strategy_return"].min()) if not win.empty else np.nan,
            "position": "LONG" if _safe_float(r["bt_weight"]) > 1e-12 else "FLAT",
        })
    return pd.DataFrame(rows)


def _random_day_checks(a: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pos = a[a["bt_weight"] > 1e-12].copy()
    if pos.empty:
        return pd.DataFrame()
    sample = pos.sample(n=min(n, len(pos)), random_state=42).sort_values("day")
    return sample[["asset", "day", "bt_weight", "exec_return_manual", "bt_turnover", "bt_cost_manual", "bt_strategy_return", "strategy_return_manual", "return_error"]].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Manual audit of ETH/BTC extended backtest calculations.")
    ap.add_argument("--daily", default="artifacts/backtest/extended_base_daily.csv")
    ap.add_argument("--eth-raw", default="data/eth_daily_extended.csv")
    ap.add_argument("--btc-raw", default="data/btc_daily_extended.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out-report", default="artifacts/backtest/audit_report.csv")
    ap.add_argument("--out-trades", default="artifacts/backtest/audit_trade_checks.csv")
    args = ap.parse_args()

    daily = _load_daily(Path(args.daily))
    eth = _asset_frame(daily, _load_raw(Path(args.eth_raw)), "eth", (21, 55, 144), args.cost_bps)
    btc = _asset_frame(daily, _load_raw(Path(args.btc_raw)), "btc", (15, 40, 120), args.cost_bps)
    all_a = pd.concat([eth, btc], ignore_index=True)

    trades = pd.concat([_trade_checks(eth), _trade_checks(btc)], ignore_index=True)
    random_checks = pd.concat([_random_day_checks(eth), _random_day_checks(btc)], ignore_index=True)
    events = _event_checks(daily, eth)

    price_match = (all_a["open_error_pct"].abs().fillna(0) < 1e-10) & (all_a["close_error_pct"].abs().fillna(0) < 1e-10)
    ema_match = all_a["regime_match"].fillna(False)
    return_abs = all_a["return_error"].abs().fillna(0.0)
    cost_error = (all_a["bt_cost_manual"] - all_a["bt_turnover"] * (args.cost_bps / 10000.0)).abs().fillna(0.0)
    systematic_bias = bool(abs(float(all_a["return_error"].mean())) > 0.0001)

    event_2018 = events[events["date"] == "2018-01-17"]
    bear_2018 = bool((not event_2018.empty) and str(event_2018.iloc[0].get("regime_backtest", "")).upper() == "BEAR")
    ftx = events[events["date"] == "2022-11-08"]
    ftx_position = str(ftx.iloc[0].get("position", "NA")) if not ftx.empty else "NA"
    covid = events[events["date"] == "2020-03-12"]
    covid_loss = _safe_float(covid.iloc[0].get("window_min_strategy_day", np.nan)) if not covid.empty else np.nan

    issues = []
    if int(price_match.sum()) != len(all_a):
        issues.append("price_mismatch")
    if int(ema_match.sum()) != len(all_a):
        issues.append("regime_mismatch")
    if float(return_abs.max()) > 0.0001:
        issues.append("return_error_gt_1bp")
    if float(cost_error.max()) > 1e-12:
        issues.append("cost_error")
    if not bear_2018:
        issues.append("2018_bear_not_confirmed")
    verdict = "BACKTEST RELIABLE" if not issues else "ISSUES FOUND"

    report = pd.DataFrame([
        {"metric": "rows_audited", "value": int(len(all_a))},
        {"metric": "trades_audited", "value": int(len(trades))},
        {"metric": "price_matches", "value": f"{int(price_match.sum())}/{len(all_a)}"},
        {"metric": "regime_matches", "value": f"{int(ema_match.sum())}/{len(all_a)}"},
        {"metric": "return_max_error_pct", "value": float(return_abs.max() * 100.0)},
        {"metric": "return_mean_error_pct", "value": float(all_a["return_error"].mean() * 100.0)},
        {"metric": "systematic_bias", "value": systematic_bias},
        {"metric": "cost_max_error", "value": float(cost_error.max())},
        {"metric": "2018_01_17_bear_fired", "value": bear_2018},
        {"metric": "2022_11_08_position", "value": ftx_position},
        {"metric": "2020_03_12_window_max_loss_pct", "value": float(covid_loss * 100.0) if np.isfinite(covid_loss) else np.nan},
        {"metric": "issues", "value": ";".join(issues)},
        {"metric": "verdict", "value": verdict},
    ])

    out_report = Path(args.out_report)
    out_trades = Path(args.out_trades)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_report, index=False)
    trades.to_csv(out_trades, index=False)
    events.to_csv(out_report.with_name("audit_event_checks.csv"), index=False)
    random_checks.to_csv(out_report.with_name("audit_return_day_checks.csv"), index=False)

    print("=" * 48)
    print("BACKTEST AUDIT REPORT")
    print("=" * 48)
    print(f"Rows audited: {len(all_a)}")
    print(f"Trades audited: {len(trades)}")
    print("")
    print("Price verification:")
    print(f"  Matches: {int(price_match.sum())}/{len(all_a)}")
    print(f"  Mismatches: {len(all_a) - int(price_match.sum())}")
    print("")
    print("Regime verification:")
    print(f"  Matches: {int(ema_match.sum())}/{len(all_a)}")
    print(f"  Mismatches: {len(all_a) - int(ema_match.sum())}")
    print("")
    print("Return calculation:")
    print(f"  Max error: {float(return_abs.max() * 100.0):.6f}%")
    print(f"  Mean error: {float(all_a['return_error'].mean() * 100.0):.6f}%")
    print(f"  Systematic bias: {'YES' if systematic_bias else 'NO'}")
    print("")
    print("Cost calculation:")
    print(f"  Max error: {float(cost_error.max()):.12f}")
    print("")
    print("Known event checks:")
    print(f"  2018-01-17 BEAR fired: {'YES' if bear_2018 else 'NO'}")
    print(f"  2022-11-08 position: {ftx_position}")
    print(f"  2020-03-12 max daily strategy loss in +/-5d window: {float(covid_loss * 100.0):.2f}%" if np.isfinite(covid_loss) else "  2020-03-12 max loss: NA")
    print("")
    print(f"Overall verdict: {verdict}")
    if issues:
        print(f"Issues: {'; '.join(issues)}")
    print("=" * 48)
    print(f"Saved: {out_report}")
    print(f"Saved: {out_trades}")
    print(f"Saved: {out_report.with_name('audit_event_checks.csv')}")
    print(f"Saved: {out_report.with_name('audit_return_day_checks.csv')}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
