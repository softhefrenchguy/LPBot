from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_overlay_strategies import _mean_reversion_overlay


PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
}

EMA_GRID = {
    "10/25/75": (10, 25, 75),
    "15/35/100": (15, 35, 100),
    "20/50/200": (20, 50, 200),
    "25/60/150": (25, 60, 150),
    "30/75/200": (30, 75, 200),
    "50/120/300": (50, 120, 300),
}

# Approximate average annual carry for long base/short quote.
CARRY_ANN = {
    "EURUSD": -0.0125,
    "GBPUSD": 0.0000,
    "USDJPY": 0.0515,
    "AUDUSD": -0.0090,
    "NZDUSD": 0.0025,
    "USDCAD": 0.0050,
    "USDCHF": 0.0350,
}


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _stats(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return {"return": np.nan, "cagr": np.nan, "sharpe": np.nan, "raw_sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan}
    eq = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    sd = float(r.std(ddof=0))
    raw_sharpe = float(r.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else np.nan
    ex = r - (0.05 / 252.0)
    ex_sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / ex_sd * np.sqrt(252.0)) if ex_sd > 1e-12 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {
        "return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "raw_sharpe": raw_sharpe,
        "maxdd": maxdd,
        "ann_vol": float(sd * np.sqrt(252.0)),
    }


def _load_price(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns and "day" in d.columns:
        d["timestamp"] = d["day"]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.floor("D")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    d["open"] = pd.to_numeric(d.get("open", d["close"]), errors="coerce").fillna(d["close"])
    return d.reset_index(drop=True)


def _download_yfinance(symbol: str, start: str, end: str, out: Path, refresh: bool) -> pd.DataFrame:
    if out.exists() and not refresh:
        return _load_price(out)
    try:
        import yfinance as yf
    except Exception as exc:
        raise SystemExit("yfinance is required. Install with: pip install yfinance") from exc
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbol, start=start, end=end_plus, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit(f"No yfinance data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"date", "datetime", "index"}), None)
    if date_col is None:
        raise SystemExit(f"Could not identify date column for {symbol}. Columns: {list(raw.columns)}")
    raw = raw.rename(columns={date_col: "timestamp"})
    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raw[col] = raw.get("close", np.nan)
    if "volume" not in raw.columns:
        raw["volume"] = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out, index=False)
    return _load_price(out)


def _vol_multiplier(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    vol = close.pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    pct = vol.rolling(252, min_periods=60).rank(pct=True)
    mult = pd.Series(1.0, index=close.index)
    mult[pct > 0.75] = 0.5
    mult[pct < 0.25] = 1.2
    return mult.fillna(1.0), pct


def _trade_stats(d: pd.DataFrame) -> dict[str, float | int]:
    active = pd.to_numeric(d["weight_exec"], errors="coerce").fillna(0.0).abs() > 1e-12
    ret = pd.to_numeric(d["strategy_return"], errors="coerce").fillna(0.0)
    side = pd.to_numeric(d["signal_exec"], errors="coerce").fillna(0.0)
    rows: list[dict[str, float | int]] = []
    in_trade = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_trade:
            start = i
            in_trade = True
        if in_trade and ((not on) or i == len(active) - 1):
            end = i - 1 if not on else i
            tr = float((1.0 + ret.iloc[start : end + 1]).prod() - 1.0)
            nonzero = side.iloc[start : end + 1][side.iloc[start : end + 1].abs() > 0]
            rows.append({"return": tr, "side": int(np.sign(nonzero.iloc[0])) if len(nonzero) else 0})
            in_trade = False
    if not rows:
        return {"trades": 0, "win_rate": np.nan, "long_trades": 0, "short_trades": 0, "long_win_rate": np.nan, "short_win_rate": np.nan, "avg_trade_return": np.nan}
    t = pd.DataFrame(rows)
    longs = t[t["side"] > 0]
    shorts = t[t["side"] < 0]
    return {
        "trades": int(len(t)),
        "win_rate": float((t["return"] > 0).mean()),
        "long_trades": int(len(longs)),
        "short_trades": int(len(shorts)),
        "long_win_rate": float((longs["return"] > 0).mean()) if len(longs) else np.nan,
        "short_win_rate": float((shorts["return"] > 0).mean()) if len(shorts) else np.nan,
        "avg_trade_return": float(t["return"].mean()),
    }


def _backtest_pair(
    pair: str,
    px: pd.DataFrame,
    ema: tuple[int, int, int],
    confirm_days: int,
    long_short: bool,
    vol_filter: bool,
    asymmetric: bool,
    transition: bool,
    carry: bool,
    weight: float,
    cost_bps: float,
) -> pd.DataFrame:
    x = px.copy()
    f, m, s = ema
    x["ema_fast"] = x["close"].ewm(span=f, adjust=False).mean()
    x["ema_mid"] = x["close"].ewm(span=m, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=s, adjust=False).mean()
    bull = (x["ema_fast"] > x["ema_mid"]) & (x["ema_mid"] > x["ema_slow"])
    bear = (x["ema_fast"] < x["ema_mid"]) & (x["ema_mid"] < x["ema_slow"])
    raw = np.select([bull, bear & bool(long_short)], [1.0, -1.0], default=0.0)

    confirmed = np.zeros(len(x), dtype=float)
    streak_side = 0.0
    streak = 0
    entry_i: int | None = None
    trans_mult = np.ones(len(x), dtype=float)
    for i, side in enumerate(raw):
        if side != 0 and side == streak_side:
            streak += 1
        elif side != 0:
            streak_side = float(side)
            streak = 1
        else:
            streak_side = 0.0
            streak = 0
            entry_i = None
        sig = float(side) if side != 0 and streak >= int(confirm_days) else 0.0
        if sig != 0 and (i == 0 or confirmed[i - 1] == 0):
            entry_i = i
            if transition:
                mom = x["close"].pct_change(5).iloc[i]
                aligned_mom = np.isfinite(mom) and np.sign(mom) == np.sign(sig)
                trans_mult[i] = 1.0 if aligned_mom and abs(float(mom)) >= 0.01 else 0.6
        elif sig != 0 and transition and entry_i is not None:
            trans_mult[i] = trans_mult[entry_i]
        confirmed[i] = sig

    x["signal_target"] = confirmed
    x["signal_exec"] = x["signal_target"].shift(1).fillna(0.0)
    x["size_mult"] = 1.0
    if vol_filter:
        vm, vp = _vol_multiplier(x["close"])
        x["vol_multiplier"] = vm
        x["vol_percentile"] = vp
        x["size_mult"] *= x["vol_multiplier"]
    else:
        x["vol_multiplier"] = 1.0
        x["vol_percentile"] = np.nan
    if asymmetric:
        gap = ((x["ema_fast"] - x["ema_mid"]).abs() / x["close"]).clip(upper=0.08) / 0.08
        regime_score = pd.Series(np.where(x["signal_target"].abs() > 0, 0.30, 0.0), index=x.index)
        vol_score = np.select([x["vol_percentile"] < 0.25, x["vol_percentile"] > 0.75], [0.33, 0.10], default=0.20)
        conv = gap.fillna(0.0) + regime_score + pd.Series(vol_score, index=x.index)
        asym = pd.Series(np.select([conv > 0.80, conv >= 0.60, conv >= 0.40], [1.2, 1.0, 0.8], default=0.6), index=x.index)
        x["size_mult"] *= asym
        x["conviction"] = conv
    else:
        x["conviction"] = np.nan
    if transition:
        x["transition_multiplier"] = pd.Series(trans_mult, index=x.index).shift(1).fillna(1.0)
        x["size_mult"] *= x["transition_multiplier"]
    else:
        x["transition_multiplier"] = 1.0

    x["weight_exec"] = x["signal_exec"] * float(weight) * pd.to_numeric(x["size_mult"], errors="coerce").fillna(1.0)
    x["weight_exec"] = x["weight_exec"].clip(lower=-float(weight) * 1.2, upper=float(weight) * 1.2)
    x["pair_return"] = x["close"].pct_change().fillna(0.0)
    x["turnover"] = (x["weight_exec"] - x["weight_exec"].shift(1).fillna(0.0)).abs()
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    carry_r = x["weight_exec"] * (float(CARRY_ANN.get(pair, 0.0)) / 252.0) if carry else 0.0
    x["strategy_return"] = x["weight_exec"] * x["pair_return"] + carry_r - x["cost"]
    return x


def _crypto_main(start: str, end: str, work: Path, cost_bps: float, refresh: bool) -> pd.DataFrame:
    daily = work / "validated_crypto_daily.csv"
    summary = work / "validated_crypto_summary.csv"
    if refresh or not daily.exists():
        _run(
            [
                sys.executable,
                "scripts/backtest_eth_btc_portfolio.py",
                "--start",
                start,
                "--end",
                end,
                "--vol-filter",
                "--transition-momentum",
                "--asymmetric-sizing",
                "--allocation-mode",
                "signal_weighted",
                "--cost-mode",
                "weight_change",
                "--cost-bps",
                str(float(cost_bps)),
                "--gross-cap",
                "0.8",
                "--eth-confirm-days",
                "3",
                "--btc-confirm-days",
                "5",
                "--eth-ema",
                "50,120,300",
                "--btc-ema",
                "15,40,120",
                "--include-gold",
                "--gold-symbol",
                "PAXG-USD",
                "--gold-ema",
                "25,65,180",
                "--gold-cap",
                "0.3",
                "--gold-cost-bps",
                str(float(cost_bps)),
                "--out-summary-csv",
                str(summary),
                "--out-daily-csv",
                str(daily),
            ]
        )
    base = pd.read_csv(daily, low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    mr = _mean_reversion_overlay(base, gross_cap=0.8, cost_bps=float(cost_bps), z_entry=-1.5, z_exit=-0.5, ret_entry=-0.03, max_hold_days=10)
    mr["main_return"] = pd.to_numeric(mr["combined_return"], errors="coerce").fillna(0.0) + pd.to_numeric(mr["mr_return"], errors="coerce").fillna(0.0)
    return mr[["day", "main_return"]].dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimise forex trend following with LPBot-style improvements.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--data-dir", default="data/forex")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--out-summary", default="artifacts/backtest/forex_optimised_summary.csv")
    ap.add_argument("--out-combined", default="artifacts/backtest/forex_optimised_combined.csv")
    ap.add_argument("--position-size", type=float, default=0.15)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--crypto-cost-bps", type=float, default=20.0)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--refresh-crypto", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    main = _crypto_main(args.start, args.end, work, float(args.crypto_cost_bps), bool(args.refresh_crypto))
    main_stats = _stats(main["main_return"])

    rows: list[dict[str, object]] = []
    for pair, symbol in PAIRS.items():
        px = _download_yfinance(symbol, args.start, args.end, data_dir / f"{pair}_daily.csv", bool(args.refresh_data))
        for ema_label, ema in EMA_GRID.items():
            for confirm_days, long_short, vol_filter, asymmetric, transition, carry in itertools.product(
                [3, 5, 7, 10], [False, True], [False, True], [False, True], [False, True], [False, True]
            ):
                bt = _backtest_pair(
                    pair,
                    px,
                    ema,
                    int(confirm_days),
                    bool(long_short),
                    bool(vol_filter),
                    bool(asymmetric),
                    bool(transition),
                    bool(carry),
                    float(args.position_size),
                    float(args.cost_bps),
                )
                st = _stats(bt["strategy_return"])
                tm = _trade_stats(bt)
                aligned = main.merge(bt[["timestamp", "strategy_return"]].rename(columns={"timestamp": "day", "strategy_return": "fx_return"}), on="day", how="inner")
                corr = float(aligned["main_return"].corr(aligned["fx_return"])) if len(aligned) > 2 else np.nan
                daily_name = f"{pair}_{ema_label.replace('/', '_')}_c{confirm_days}_{'ls' if long_short else 'lo'}_{'vf' if vol_filter else 'novf'}_{'asym' if asymmetric else 'noasym'}_{'tm' if transition else 'notm'}_{'carry' if carry else 'nocarry'}.csv"
                row = {
                    "pair": pair,
                    "symbol": symbol,
                    "ema": ema_label,
                    "confirm_days": int(confirm_days),
                    "mode": "long_short" if long_short else "long_only",
                    "vol_filter": bool(vol_filter),
                    "asymmetric": bool(asymmetric),
                    "transition": bool(transition),
                    "carry": bool(carry),
                    **st,
                    **tm,
                    "corr_vs_main": corr,
                    "daily_csv": str(work / daily_name),
                }
                rows.append(row)
                if bool(st["raw_sharpe"] > 0.45):
                    bt.to_csv(row["daily_csv"], index=False)

    summary = pd.DataFrame(rows)
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    best = summary.sort_values(["pair", "raw_sharpe", "maxdd"], ascending=[True, False, False]).groupby("pair", as_index=False).head(1)
    # Ensure best daily files exist even if below the high-sharpe persistence threshold.
    for _, r in best.iterrows():
        p = Path(str(r["daily_csv"]))
        if not p.exists():
            px = _load_price(data_dir / f"{r['pair']}_daily.csv")
            bt = _backtest_pair(
                str(r["pair"]),
                px,
                EMA_GRID[str(r["ema"])],
                int(r["confirm_days"]),
                str(r["mode"]) == "long_short",
                bool(r["vol_filter"]),
                bool(r["asymmetric"]),
                bool(r["transition"]),
                bool(r["carry"]),
                float(args.position_size),
                float(args.cost_bps),
            )
            bt.to_csv(p, index=False)

    combined_rows: list[dict[str, object]] = [{"configuration": "main_only", "pairs": "", "fx_weight_each": 0.0, **main_stats}]
    ranked_pairs = list(best.sort_values("raw_sharpe", ascending=False)["pair"])
    best_by_pair = best.set_index("pair")
    for n in [1, 2, 3]:
        for combo in itertools.combinations(ranked_pairs, n):
            for fx_w in [0.05, 0.10, 0.15, 0.20]:
                main_w = 1.0 - fx_w * n
                if main_w <= 0:
                    continue
                portfolio = main.copy()
                r_total = main_w * pd.to_numeric(portfolio["main_return"], errors="coerce").fillna(0.0)
                corrs: list[float] = []
                for pair in combo:
                    row = best_by_pair.loc[pair]
                    bt = pd.read_csv(str(row["daily_csv"]))
                    bt["day"] = pd.to_datetime(bt["timestamp"], utc=True, errors="coerce").dt.floor("D")
                    bt["fx_unit_return"] = pd.to_numeric(bt["strategy_return"], errors="coerce").fillna(0.0) / float(args.position_size)
                    portfolio = portfolio.merge(bt[["day", "fx_unit_return"]].rename(columns={"fx_unit_return": f"{pair}_unit_return"}), on="day", how="left")
                    portfolio[f"{pair}_unit_return"] = pd.to_numeric(portfolio[f"{pair}_unit_return"], errors="coerce").fillna(0.0)
                    r_total += fx_w * portfolio[f"{pair}_unit_return"]
                    corrs.append(float(row["corr_vs_main"]))
                st = _stats(r_total)
                combined_rows.append(
                    {
                        "configuration": f"main_{int(main_w*100)}_" + "_".join(combo) + f"_{int(fx_w*100)}each",
                        "pairs": ",".join(combo),
                        "fx_weight_each": fx_w,
                        "main_weight": main_w,
                        **st,
                        "avg_corr_vs_main": float(np.nanmean(corrs)) if corrs else np.nan,
                    }
                )
    combined = pd.DataFrame(combined_rows)
    out_combined = Path(args.out_combined)
    combined.to_csv(out_combined, index=False)

    best_combo = combined.sort_values(["sharpe", "maxdd"], ascending=[False, False]).iloc[0]
    feasible = combined[
        (pd.to_numeric(combined["sharpe"], errors="coerce") > float(main_stats["sharpe"]))
        & (pd.to_numeric(combined["cagr"], errors="coerce") > 0.35)
        & (pd.to_numeric(combined["maxdd"], errors="coerce") >= float(main_stats["maxdd"]))
    ].copy()
    best_feasible = feasible.sort_values(["sharpe", "maxdd"], ascending=[False, False]).iloc[0] if not feasible.empty else None
    print("")
    print("=" * 88)
    print("OPTIMISED FOREX RESULTS")
    print("2019-2024 | Full LPBot-style stack | Long-only/Long-short | Carry tested")
    print("=" * 88)
    print("Best per pair:")
    for _, r in best.sort_values("raw_sharpe", ascending=False).iterrows():
        print(
            f"  {r['pair']:<6} EMA {r['ema']:<10} c{int(r['confirm_days']):<2} {r['mode']:<10} "
            f"vol={r['vol_filter']} asym={r['asymmetric']} trans={r['transition']} carry={r['carry']} "
            f"raw_sharpe={float(r['raw_sharpe']):.3f} excess_sharpe={float(r['sharpe']):.3f} "
            f"CAGR={float(r['cagr'])*100:.2f}% MaxDD={float(r['maxdd'])*100:.2f}% Corr={float(r['corr_vs_main']):.3f}"
        )
    print("-" * 88)
    print("Best combined rows:")
    show = combined.sort_values(["sharpe", "maxdd"], ascending=[False, False]).head(10)
    for _, r in show.iterrows():
        print(
            f"  {r['configuration']:<42} Sharpe {float(r['sharpe']):.3f} "
            f"CAGR {float(r['cagr'])*100:.2f}% MaxDD {float(r['maxdd'])*100:.2f}%"
        )
    print("-" * 88)
    print(f"Main only: Sharpe {main_stats['sharpe']:.3f} CAGR {main_stats['cagr']*100:.2f}% MaxDD {main_stats['maxdd']*100:.2f}%")
    print(f"Best combo: {best_combo['configuration']} Sharpe {float(best_combo['sharpe']):.3f}")
    if best_feasible is not None:
        print(
            f"Best feasible: {best_feasible['configuration']} Sharpe {float(best_feasible['sharpe']):.3f} "
            f"CAGR {float(best_feasible['cagr'])*100:.2f}% MaxDD {float(best_feasible['maxdd'])*100:.2f}%"
        )
    decision = "ADD" if best_feasible is not None else "SKIP"
    print(f"Decision: {decision}")
    print("=" * 88)
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
