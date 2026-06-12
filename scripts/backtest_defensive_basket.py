from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_overlay_strategies import _mean_reversion_overlay, _stats


ASSETS = {
    "PAXG": "PAXG-USD",
    "TLT": "TLT",
    "GLD": "GLD",
    "UUP": "UUP",
}


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _download_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d = yf.download(symbol, start=start, end=end_plus, interval="1d", auto_adjust=True, progress=False)
    if d is None or d.empty:
        raise RuntimeError(f"No yfinance data for {symbol}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()
    date_col = "Date" if "Date" in d.columns else "index"
    out = pd.DataFrame(
        {
            "day": pd.to_datetime(d[date_col], errors="coerce").dt.floor("D"),
            "open": pd.to_numeric(d["Open"], errors="coerce"),
            "close": pd.to_numeric(d["Close"], errors="coerce"),
        }
    )
    return out.dropna(subset=["day", "open", "close"]).sort_values("day").reset_index(drop=True)


def _consecutive_true(s: pd.Series) -> pd.Series:
    vals: list[int] = []
    n = 0
    for v in s.fillna(False).astype(bool):
        n = n + 1 if v else 0
        vals.append(n)
    return pd.Series(vals, index=s.index)


def _prepare_asset_frame(days: pd.Series, prices: pd.DataFrame, prefix: str) -> pd.DataFrame:
    base = pd.DataFrame({"day": pd.to_datetime(days, errors="coerce")}).drop_duplicates().sort_values("day")
    base["day"] = base["day"].dt.tz_localize(None)
    px = prices.copy()
    px["day"] = pd.to_datetime(px["day"], errors="coerce")
    if getattr(px["day"].dt, "tz", None) is not None:
        px["day"] = px["day"].dt.tz_localize(None)
    x = base.merge(px, on="day", how="left").sort_values("day")
    x[["open", "close"]] = x[["open", "close"]].ffill()
    x[f"{prefix}_ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x[f"{prefix}_ema55"] = x["close"].ewm(span=55, adjust=False).mean()
    x[f"{prefix}_ema144"] = x["close"].ewm(span=144, adjust=False).mean()
    x[f"{prefix}_aligned"] = (x[f"{prefix}_ema21"] > x[f"{prefix}_ema55"]) & (
        x[f"{prefix}_ema55"] > x[f"{prefix}_ema144"]
    )
    x[f"{prefix}_aligned_days"] = _consecutive_true(x[f"{prefix}_aligned"])
    x[f"{prefix}_gap"] = (x[f"{prefix}_ema55"] / x[f"{prefix}_ema144"] - 1.0).replace([np.inf, -np.inf], np.nan)
    x[f"{prefix}_ret_exec"] = x["open"].pct_change().fillna(0.0)
    return x[
        [
            "day",
            f"{prefix}_aligned",
            f"{prefix}_aligned_days",
            f"{prefix}_gap",
            f"{prefix}_ret_exec",
        ]
    ].copy()


def _bear_streak(regime: pd.Series) -> pd.Series:
    vals: list[int] = []
    n = 0
    for r in regime.astype(str).str.upper():
        n = n + 1 if r == "BEAR" else 0
        vals.append(n)
    return pd.Series(vals, index=regime.index)


def _add_paxg_only(base: pd.DataFrame, asset_frames: dict[str, pd.DataFrame], cap: float, cost_bps: float) -> pd.DataFrame:
    x = base.copy()
    x = x.merge(asset_frames["PAXG"], on="day", how="left")
    idle_cap = (1.0 - (x["alloc_eth"] + x["alloc_btc"])).clip(lower=0.0)
    both_flat = (pd.to_numeric(x["eth_off_active"], errors="coerce").fillna(0) <= 0) & (
        pd.to_numeric(x["btc_off_active"], errors="coerce").fillna(0) <= 0
    )
    bear_gate = x["eth_regime"].astype(str).str.upper().eq("BEAR")
    gate = both_flat & bear_gate & x["PAXG_aligned"].fillna(False)
    x["reserve_asset_target"] = np.where(gate, "PAXG", "")
    x["reserve_weight_target"] = np.where(gate, np.minimum(idle_cap, float(cap)), 0.0)
    x["reserve_weight_exec"] = pd.to_numeric(x["reserve_weight_target"], errors="coerce").fillna(0.0).shift(1).fillna(0.0)
    x["reserve_asset_exec"] = x["reserve_asset_target"].shift(1).fillna("")
    prev_w = x["reserve_weight_exec"].shift(1).fillna(0.0)
    x["reserve_turnover"] = (x["reserve_weight_exec"] - prev_w).abs()
    x["reserve_cost"] = x["reserve_turnover"] * (float(cost_bps) / 10000.0)
    x["reserve_return_gross"] = x["reserve_weight_exec"] * pd.to_numeric(x["PAXG_ret_exec"], errors="coerce").fillna(0.0)
    x["reserve_return"] = x["reserve_return_gross"] - x["reserve_cost"]
    return x


def _add_defensive_basket(
    base: pd.DataFrame,
    asset_frames: dict[str, pd.DataFrame],
    cap: float,
    cost_bps: float,
    transfer_drag: float,
    delay_days: int,
    min_hold_days: int,
) -> pd.DataFrame:
    x = base.copy()
    for name, frame in asset_frames.items():
        x = x.merge(frame, on="day", how="left")

    idle_cap = (1.0 - (x["alloc_eth"] + x["alloc_btc"])).clip(lower=0.0)
    both_flat = (pd.to_numeric(x["eth_off_active"], errors="coerce").fillna(0) <= 0) & (
        pd.to_numeric(x["btc_off_active"], errors="coerce").fillna(0) <= 0
    )
    regime = x["eth_regime"].astype(str).str.upper()
    bear_gate = regime.eq("BEAR")
    x["bear_streak"] = _bear_streak(regime)

    paxg_gate = both_flat & bear_gate & x["PAXG_aligned"].fillna(False)

    target_assets: list[str] = []
    target_weights: list[float] = []
    entry_events = np.zeros(len(x), dtype=bool)
    exit_events = np.zeros(len(x), dtype=bool)

    active_asset = ""
    active_weight = 0.0
    active_days = 0
    pending_entry: tuple[int, str, float] | None = None
    pending_exit_day: int | None = None

    tier2_assets = ["TLT", "GLD", "UUP"]
    for i, row in x.iterrows():
        if pending_exit_day is not None and i >= pending_exit_day:
            active_asset = ""
            active_weight = 0.0
            active_days = 0
            pending_exit_day = None
            exit_events[i] = True

        if pending_entry is not None and i >= pending_entry[0] and not active_asset:
            _, asset, weight = pending_entry
            active_asset = asset
            active_weight = weight
            active_days = 0
            pending_entry = None
            entry_events[i] = True

        # Tier 1 is instant PAXG-style reserve. It supersedes any pending Tier 2 entry.
        if bool(paxg_gate.iloc[i]):
            pending_entry = None
            active_asset = "PAXG"
            active_weight = float(min(float(idle_cap.iloc[i]), float(cap)))
            active_days = active_days + 1 if active_weight > 0 else 0
            target_assets.append(active_asset)
            target_weights.append(active_weight)
            continue

        # Existing Tier 1 exits immediately when its gate closes; Tier 2 respects delayed exits.
        if active_asset == "PAXG":
            active_asset = ""
            active_weight = 0.0
            active_days = 0

        if active_asset in tier2_assets:
            still_aligned = bool(row.get(f"{active_asset}_aligned", False))
            exit_signal = (not still_aligned) or (not bool(bear_gate.iloc[i])) or (not bool(both_flat.iloc[i]))
            if exit_signal and active_days >= int(min_hold_days) and pending_exit_day is None:
                pending_exit_day = min(i + int(delay_days), len(x) - 1)
            active_days += 1
        elif not active_asset and pending_entry is None:
            if bool(both_flat.iloc[i]) and bool(bear_gate.iloc[i]) and int(row["bear_streak"]) >= 10:
                choices: list[tuple[float, str]] = []
                for asset in tier2_assets:
                    aligned_days = row.get(f"{asset}_aligned_days", 0)
                    gap = row.get(f"{asset}_gap", np.nan)
                    if pd.notna(aligned_days) and float(aligned_days) >= 5 and pd.notna(gap):
                        choices.append((float(gap), asset))
                if choices:
                    choices.sort(reverse=True)
                    asset = choices[0][1]
                    weight = float(min(float(idle_cap.iloc[i]), float(cap)))
                    if weight > 0:
                        exec_day = min(i + int(delay_days), len(x) - 1)
                        pending_entry = (exec_day, asset, weight)

        target_assets.append(active_asset)
        target_weights.append(active_weight if active_asset else 0.0)

    x["reserve_asset_exec"] = target_assets
    x["reserve_weight_exec"] = target_weights
    prev_w = x["reserve_weight_exec"].shift(1).fillna(0.0)
    x["reserve_turnover"] = (x["reserve_weight_exec"] - prev_w).abs()
    x["tier2_trade_event"] = entry_events | exit_events
    x["reserve_transfer_cost"] = np.where(x["tier2_trade_event"], x["reserve_turnover"] * float(transfer_drag), 0.0)
    x["reserve_trading_cost"] = x["reserve_turnover"] * (float(cost_bps) / 10000.0)
    ret = np.zeros(len(x), dtype=float)
    for asset in ["PAXG", "TLT", "GLD", "UUP"]:
        mask = x["reserve_asset_exec"].astype(str).eq(asset)
        ret += np.where(mask, pd.to_numeric(x[f"{asset}_ret_exec"], errors="coerce").fillna(0.0), 0.0)
    x["reserve_return_gross"] = x["reserve_weight_exec"] * ret
    x["reserve_return"] = x["reserve_return_gross"] - x["reserve_trading_cost"] - x["reserve_transfer_cost"]
    return x


def _trade_diagnostics(d: pd.DataFrame) -> dict[str, object]:
    tier2_assets = {"TLT", "GLD", "UUP"}
    trades: list[dict[str, object]] = []
    active = False
    start_i = 0
    asset = ""
    eq = 1.0
    for i, row in d.iterrows():
        a = str(row.get("reserve_asset_exec", ""))
        w = float(row.get("reserve_weight_exec", 0.0) or 0.0)
        if (not active) and a in tier2_assets and w > 0:
            active = True
            start_i = i
            asset = a
            eq = 1.0
        if active:
            eq *= 1.0 + float(row.get("reserve_return", 0.0) or 0.0)
            next_a = str(d["reserve_asset_exec"].iloc[i + 1]) if i + 1 < len(d) else ""
            next_w = float(d["reserve_weight_exec"].iloc[i + 1]) if i + 1 < len(d) else 0.0
            if next_a != asset or next_w <= 0 or i + 1 == len(d):
                trades.append({"asset": asset, "days": i - start_i + 1, "return": eq - 1.0})
                active = False
                asset = ""
    counts = Counter(t["asset"] for t in trades)
    return {
        "tier2_rotations": len(trades),
        "tlt_rotations": counts.get("TLT", 0),
        "gld_rotations": counts.get("GLD", 0),
        "uup_rotations": counts.get("UUP", 0),
        "avg_hold_days": float(np.mean([t["days"] for t in trades])) if trades else 0.0,
        "avg_tier2_return": float(np.mean([t["return"] for t in trades])) if trades else 0.0,
    }


def _scenario_row(name: str, d: pd.DataFrame, ret_col: str, extra: dict[str, object]) -> dict[str, object]:
    st = _stats(pd.to_numeric(d[ret_col], errors="coerce").fillna(0.0))
    return {"config": name, "sharpe": st["sharpe"], "maxdd": st["maxdd"], "cagr": st["cagr"], **extra}


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest BEAR+FLAT defensive basket with cross-venue friction.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--work-dir", default="artifacts/backtest/defensive_basket")
    ap.add_argument("--out-summary", default="artifacts/backtest/defensive_basket_summary.csv")
    ap.add_argument("--gold-cap", type=float, default=0.30)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-hold-days", type=int, default=10)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base_daily = work / "base_no_reserve_daily.csv"
    base_summary = work / "base_no_reserve_summary.csv"

    _run(
        [
            sys.executable,
            "scripts/backtest_eth_btc_portfolio.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--vol-filter",
            "--transition-momentum",
            "--asymmetric-sizing",
            "--allocation-mode",
            "signal_weighted",
            "--cost-mode",
            "weight_change",
            "--cost-bps",
            str(float(args.cost_bps)),
            "--eth-confirm-days",
            "3",
            "--btc-confirm-days",
            "5",
            "--eth-ema",
            "21,55,144",
            "--btc-ema",
            "15,40,120",
            "--out-summary-csv",
            str(base_summary),
            "--out-daily-csv",
            str(base_daily),
        ]
    )

    base = pd.read_csv(base_daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    base["day"] = base["day"].dt.tz_localize(None)
    base = base.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    asset_frames: dict[str, pd.DataFrame] = {}
    for name, symbol in ASSETS.items():
        print(f"Downloading {name} ({symbol})...", flush=True)
        px = _download_daily(symbol, args.start, args.end)
        asset_frames[name] = _prepare_asset_frame(base["day"], px, name)

    scenarios = [
        ("A) PAXG only", "paxg", 0.0, 1),
        ("B) +Basket GBP1k", "basket_1k", 0.020, 3),
        ("C) +Basket GBP5k", "basket_5k", 0.004, 3),
        ("D) +Basket GBP20k", "basket_20k", 0.001, 3),
        ("E) +Basket zero friction", "basket_zero", 0.0, 0),
    ]

    rows: list[dict[str, object]] = []
    daily_outputs: dict[str, pd.DataFrame] = {}
    for label, slug, transfer_drag, delay_days in scenarios:
        if slug == "paxg":
            d = _add_paxg_only(base, asset_frames, cap=float(args.gold_cap), cost_bps=float(args.cost_bps))
        else:
            d = _add_defensive_basket(
                base,
                asset_frames,
                cap=float(args.gold_cap),
                cost_bps=float(args.cost_bps),
                transfer_drag=float(transfer_drag),
                delay_days=int(delay_days),
                min_hold_days=int(args.min_hold_days),
            )
        d["combined_return_with_reserve"] = pd.to_numeric(d["combined_return"], errors="coerce").fillna(0.0) + d[
            "reserve_return"
        ].fillna(0.0)
        d["gold_weight_exec"] = pd.to_numeric(d["reserve_weight_exec"], errors="coerce").fillna(0.0)
        d["combined_return"] = d["combined_return_with_reserve"]
        mr = _mean_reversion_overlay(
            d,
            gross_cap=0.8,
            cost_bps=float(args.cost_bps),
            z_entry=-1.5,
            z_exit=-0.5,
            ret_entry=-0.03,
            max_hold_days=10,
        )
        d["mean_reversion_overlay_return"] = mr["mr_return"]
        d["final_return"] = d["combined_return_with_reserve"] + d["mean_reversion_overlay_return"]
        diag = _trade_diagnostics(d)
        rows.append(_scenario_row(label, d, "final_return", diag))
        d.to_csv(work / f"{slug}_daily.csv", index=False)
        daily_outputs[slug] = d

    summary = pd.DataFrame(rows)
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print("")
    print("================================================")
    print("DEFENSIVE BASKET RESULTS")
    print("================================================")
    print("Config                        Sharpe  MaxDD    CAGR")
    print("------------------------------------------------")
    for _, r in summary.iterrows():
        print(
            f"{str(r['config'])[:28]:<28}"
            f"{float(r['sharpe']):>7.3f} "
            f"{float(r['maxdd']) * 100:>7.2f}% "
            f"{float(r['cagr']) * 100:>6.2f}%"
        )
    print("------------------------------------------------")
    basket = summary[summary["config"].astype(str).str.contains("Basket")]
    if len(basket):
        best = basket.sort_values(["sharpe", "maxdd"], ascending=[False, False]).iloc[0]
        print(f"Best basket config: {best['config']} (Sharpe {float(best['sharpe']):.3f})")
    diag_row = summary[summary["config"].astype(str).str.contains("zero friction")]
    if len(diag_row):
        r = diag_row.iloc[0]
        print(f"Tier 2 rotations: {int(r['tier2_rotations'])}")
        print(f"Winning asset breakdown: TLT={int(r['tlt_rotations'])} GLD={int(r['gld_rotations'])} UUP={int(r['uup_rotations'])}")
        print(f"Avg hold: {float(r['avg_hold_days']):.1f} days")
        print(f"Avg Tier 2 return: {float(r['avg_tier2_return']) * 100:.2f}%")
    print("================================================")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
