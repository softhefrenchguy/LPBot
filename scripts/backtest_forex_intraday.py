from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from backtest_forex_optimised import _crypto_main, _stats


PAIRS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X"}
SESSIONS = {"london": (7, 12), "london_ny": (7, 17), "ny": (12, 17), "all": (0, 23)}
EMA_GRID = {"10/25": (10, 25), "20/50": (20, 50), "30/75": (30, 75)}


def _load_hourly(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _download_hourly(symbol: str, out: Path, refresh: bool) -> pd.DataFrame:
    if out.exists() and not refresh:
        return _load_hourly(out)
    try:
        import yfinance as yf
    except Exception as exc:
        raise SystemExit("yfinance is required. Install with: pip install yfinance") from exc
    raw = yf.download(symbol, period="729d", interval="1h", auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit(f"No hourly yfinance data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"datetime", "date", "index"}), None)
    if date_col is None:
        raise SystemExit(f"Could not identify datetime column for {symbol}. Columns: {list(raw.columns)}")
    raw = raw.rename(columns={date_col: "timestamp"})
    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raw[col] = raw.get("close", np.nan)
    if "volume" not in raw.columns:
        raw["volume"] = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out, index=False)
    return _load_hourly(out)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _daily_vol_ok(d: pd.DataFrame) -> np.ndarray:
    # resample("1D") inserts all-NaN placeholder rows for non-trading calendar days
    # (weekends). Left in place, a 14-row ATR window always spans a weekend gap, so it
    # never contains 14 valid trading days and ATR comes out NaN everywhere -> filter
    # blocks every trade. Drop the non-trading days first so the rolling windows are
    # over actual trading days.
    daily = d.set_index("timestamp").resample("1D").agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna(subset=["close"])
    daily["atr"] = _atr(daily["high"], daily["low"], daily["close"], 14)
    daily["atr20"] = daily["atr"].rolling(20, min_periods=20).mean()
    daily["ok"] = (daily["atr"] < daily["atr20"]).shift(1).fillna(False)
    x = pd.DataFrame({"timestamp": d["timestamp"], "day": d["timestamp"].dt.floor("D")})
    return x.merge(daily[["ok"]].reset_index().rename(columns={"timestamp": "day"}), on="day", how="left")["ok"].fillna(False).to_numpy(dtype=bool)


def _features(d: pd.DataFrame, ema_label: str, session: str, daily_filter: bool) -> dict[str, np.ndarray]:
    f, s = EMA_GRID[ema_label]
    close = d["close"].astype(float)
    ema_f = close.ewm(span=f, adjust=False).mean()
    ema_s = close.ewm(span=s, adjust=False).mean()
    atr = _atr(d["high"].astype(float), d["low"].astype(float), close, 14).replace(0.0, np.nan)
    dev = (((close - ((ema_f + ema_s) / 2.0)) / atr).replace([np.inf, -np.inf], np.nan)).to_numpy(float)
    start, end = SESSIONS[session]
    ok = d["timestamp"].dt.hour.between(start, end, inclusive="both").to_numpy(dtype=bool)
    if daily_filter:
        ok = ok & _daily_vol_ok(d)
    pair_ret = close.pct_change().fillna(0.0).to_numpy(float)
    return {"dev": dev, "close": close.to_numpy(float), "atr": atr.to_numpy(float), "ok": ok, "pair_ret": pair_ret}


@njit(cache=True, nogil=True)
def _simulate_core(close: np.ndarray, atr: np.ndarray, dev: np.ndarray, ok: np.ndarray, tp: float, sl: float, entry: float, max_hold: int) -> np.ndarray:
    # Entry price/ATR are pinned at trade-open and every later exit check (TP/SL) is
    # measured against that pinned value, so whether bar j is "in a trade" depends on
    # exactly which earlier bar opened it. That circular dependency (next entry time
    # depends on prior exit time, which depends on the prior entry price) can't be
    # expressed as independent boolean masks -- it's an inherently sequential scan.
    # Numba JIT-compiles this exact scan to machine code, removing the per-bar
    # Python/numpy-scalar overhead that made the pure-Python loop slow.
    n = close.shape[0]
    target = np.zeros(n, dtype=np.float64)
    active = 0.0
    held = 0
    entry_px = np.nan
    entry_atr = np.nan
    for i in range(n):
        z = dev[i]
        px = close[i]
        a = atr[i]
        if active == 0.0:
            if ok[i] and not np.isnan(z) and not np.isnan(a) and z < -entry:
                active = 1.0; held = 0; entry_px = px; entry_atr = a
            elif ok[i] and not np.isnan(z) and not np.isnan(a) and z > entry:
                active = -1.0; held = 0; entry_px = px; entry_atr = a
        else:
            if active > 0.0:
                hit_tp = px >= entry_px + tp * entry_atr
                hit_sl = px <= entry_px - sl * entry_atr
            else:
                hit_tp = px <= entry_px - tp * entry_atr
                hit_sl = px >= entry_px + sl * entry_atr
            recovered = (not np.isnan(z)) and abs(z) <= 0.5
            timeout = held >= max_hold
            if hit_tp or hit_sl or recovered or timeout:
                active = 0.0; held = 0; entry_px = np.nan; entry_atr = np.nan
        target[i] = active
        if active != 0.0:
            held += 1
    return target


def _simulate(feat: dict[str, np.ndarray], cost_bps: float, weight: float, tp: float, sl: float, entry: float, max_hold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = _simulate_core(feat["close"], feat["atr"], feat["dev"], feat["ok"], float(tp), float(sl), float(entry), int(max_hold))
    signal_exec = np.roll(target, 1); signal_exec[0] = 0.0
    weight_exec = signal_exec * weight
    prev_w = np.roll(weight_exec, 1); prev_w[0] = 0.0
    strat = weight_exec * feat["pair_ret"] - np.abs(weight_exec - prev_w) * (cost_bps / 10000.0)
    return strat, weight_exec, signal_exec


def _trade_metrics(ts: pd.Series, strat: np.ndarray, weight_exec: np.ndarray, signal_exec: np.ndarray) -> dict[str, float | int]:
    # Fully vectorized: unlike _simulate, trade boundaries here only depend on an
    # already-computed array (weight_exec), so segment extraction has no circular
    # dependency and can be done with pure numpy (no python-level bar loop).
    active = np.abs(weight_exec) > 1e-12
    if not active.any():
        return {"trades": 0, "win_rate": np.nan, "avg_hold_hours": np.nan, "trades_per_week": 0.0, "avg_trade_return": np.nan, "expectancy": np.nan, "profit_factor": np.nan, "avg_winner": np.nan, "avg_loser": np.nan}
    n = active.shape[0]
    diff = np.diff(active.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if active[-1] else end_candidates
    log_ret = np.log1p(strat)
    cum0 = np.concatenate(([0.0], np.cumsum(log_ret)))
    trade_returns = np.expm1(cum0[ends + 1] - cum0[starts])
    hours = ends - starts + 1
    n_trades = trade_returns.size
    wins_mask = trade_returns > 0
    losses_mask = ~wins_mask
    wr = wins_mask.sum() / n_trades
    avg_win = float(trade_returns[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss = float((-trade_returns[losses_mask]).mean()) if losses_mask.any() else 0.0
    gross_profit = float(trade_returns[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss = float((-trade_returns[losses_mask]).sum()) if losses_mask.any() else 0.0
    weeks = max((ts.max() - ts.min()).days / 7.0, 1e-9)
    return {"trades": int(n_trades), "win_rate": float(wr), "avg_hold_hours": float(hours.mean()), "trades_per_week": float(n_trades / weeks), "avg_trade_return": float(trade_returns.mean()), "expectancy": float(wr * avg_win - (1.0 - wr) * avg_loss), "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf, "avg_winner": avg_win, "avg_loser": avg_loss}


def _full_frame(d: pd.DataFrame, feat: dict[str, np.ndarray], strat: np.ndarray, weight_exec: np.ndarray, signal_exec: np.ndarray) -> pd.DataFrame:
    out = d[["timestamp", "open", "high", "low", "close"]].copy()
    out["dev"] = feat["dev"]
    out["signal_exec"] = signal_exec
    out["weight_exec"] = weight_exec
    out["pair_return"] = feat["pair_ret"]
    out["strategy_return"] = strat
    out["day"] = out["timestamp"].dt.floor("D")
    return out


def _daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("day", as_index=False)["strategy_return"].apply(lambda s: (1.0 + s).prod() - 1.0)


def _day_codes(ts: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(ts.dt.floor("D"), sort=True)
    return codes.astype(np.int64), len(uniques)


def _to_daily_stats(day_codes: np.ndarray, n_days: int, strat: np.ndarray) -> dict[str, float]:
    # _stats() assumes a DAILY return series (years=len(r)/252, sqrt(252) annualization,
    # a 0.05/252 daily hurdle). Feeding it hourly bars directly inflates "years" ~24x and,
    # because most hourly bars are exactly 0 (flat), the constant hurdle subtraction
    # collapses variance while keeping the mean negative -- producing nonsense Sharpe
    # values. Compound hourly returns into daily returns first so the annualization is valid.
    daily_log = np.bincount(day_codes, weights=np.log1p(strat), minlength=n_days)
    return _stats(pd.Series(np.expm1(daily_log)))


def _combined_with_main(gbp_frame: pd.DataFrame, weight: float, out_combined: Path) -> tuple[dict, dict, dict, float]:
    best_daily = _daily_returns(gbp_frame)
    start = str(best_daily["day"].min().date()); end = str(best_daily["day"].max().date())
    main = _crypto_main(start, end, Path("artifacts/backtest/forex_intraday_main"), 20.0, False)
    merged = main.merge(best_daily.rename(columns={"strategy_return": "fx_return_15"}), on="day", how="inner")
    fx_unit = merged["fx_return_15"] / weight
    main_stats = _stats(merged["main_return"]); fx_stats = _stats(merged["fx_return_15"]); comb_stats = _stats(0.90 * merged["main_return"] + 0.10 * fx_unit)
    corr = float(merged["main_return"].corr(merged["fx_return_15"])) if len(merged) > 2 else np.nan
    pd.DataFrame([{"configuration": "main_only", **main_stats, "corr_vs_main": 1.0}, {"configuration": "forex_mr_gbpusd_15pct_sleeve", **fx_stats, "corr_vs_main": corr}, {"configuration": "main_90_forex_10", **comb_stats, "corr_vs_main": np.nan}]).to_csv(out_combined, index=False)
    return main_stats, fx_stats, comb_stats, corr


FINE_TP = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
FINE_SL = [0.8, 1.0, 1.2, 1.5]
FINE_ENTRY = [2.0, 2.2, 2.5, 2.8, 3.0]
FINE_HOLD = [8, 12, 16, 24]
FINE_EMA = "20/50"
FINE_SESSION = "ny"


def _run_fine(args: argparse.Namespace, gbp: pd.DataFrame, gbp_codes: np.ndarray, gbp_ndays: int) -> int:
    feat_by_filter = {False: _features(gbp, FINE_EMA, FINE_SESSION, False), True: _features(gbp, FINE_EMA, FINE_SESSION, True)}
    n_ok = {k: int(v["ok"].sum()) for k, v in feat_by_filter.items()}
    print(f"Vol filter eligible hourly bars: without={n_ok[False]} with={n_ok[True]} ({n_ok[True]/n_ok[False]*100:.1f}% of unfiltered)")

    rows = []
    for tp, sl, entry, max_hold, vol_filter in itertools.product(FINE_TP, FINE_SL, FINE_ENTRY, FINE_HOLD, [False, True]):
        feat = feat_by_filter[vol_filter]
        strat, w, sig = _simulate(feat, args.cost_bps, args.weight, tp, sl, entry, max_hold)
        rows.append({"pair": "GBPUSD", "tp_atr": tp, "sl_atr": sl, "entry_atr": entry, "ema": FINE_EMA, "max_hold": max_hold, "session": FINE_SESSION, "daily_vol_filter": vol_filter, **_to_daily_stats(gbp_codes, gbp_ndays, strat), **_trade_metrics(gbp["timestamp"], strat, w, sig)})
    grid = pd.DataFrame(rows).sort_values(["raw_sharpe", "profit_factor", "expectancy"], ascending=[False, False, False])
    Path(args.out_fine_summary).parent.mkdir(parents=True, exist_ok=True); grid.to_csv(args.out_fine_summary, index=False)
    best = grid.iloc[0]

    print("=" * 90); print("GBPUSD FINE GRID SEARCH (NY session, EMA 20/50)"); print(f"{gbp['timestamp'].min().date()} to {gbp['timestamp'].max().date()} | 1bps costs"); print("=" * 90)
    print("TP   SL   Entry Hold Filter RawSharpe Sharpe  CAGR    WinRate Trades Expect  PF")
    print("-" * 90)
    for _, r in grid.head(20).iterrows():
        print(f"{r.tp_atr:<4.1f} {r.sl_atr:<4.1f} {r.entry_atr:<5.1f} {int(r.max_hold):<4d} {str(bool(r.daily_vol_filter)):<6} {r.raw_sharpe:>9.3f} {r.sharpe:>7.3f} {r.cagr*100:>6.2f}% {r.win_rate*100:>6.1f}% {int(r.trades):>6d} {r.expectancy*10000:>6.2f}bp {r.profit_factor:>4.2f}")
    print("=" * 90)

    by_filter = grid.groupby("daily_vol_filter").agg(mean_raw_sharpe=("raw_sharpe", "mean"), mean_expectancy_bps=("expectancy", lambda s: (s * 10000).mean()), mean_pf=("profit_factor", "mean"), n_combos_positive=("raw_sharpe", lambda s: int((s > 0).sum())))
    print("Vol filter effect (mean across all fine-grid combos):")
    print(by_filter.round(4).to_string())
    filter_helps = bool(by_filter.loc[True, "mean_raw_sharpe"] > by_filter.loc[False, "mean_raw_sharpe"]) if True in by_filter.index and False in by_filter.index else False
    print(f"Vol filter helps on average: {filter_helps}")
    print("=" * 90)

    n_clear_half = int((grid["raw_sharpe"] > 0.5).sum())
    print(f"Combos clearing raw_sharpe > 0.5: {n_clear_half} / {len(grid)}")
    print("Best fine-grid config:")
    print(f"  TP {best.tp_atr:.1f} ATR | SL {best.sl_atr:.1f} ATR | Entry {best.entry_atr:.1f} ATR | Hold {int(best.max_hold)}h | VolFilter {bool(best.daily_vol_filter)}")
    print(f"  RawSharpe {best.raw_sharpe:.3f} | Sharpe {best.sharpe:.3f} | CAGR {best.cagr*100:.2f}% | Win {best.win_rate*100:.1f}% | Trades {int(best.trades)} | Expectancy {best.expectancy*10000:.2f} bps/trade | PF {best.profit_factor:.2f}")

    best_feat = feat_by_filter[bool(best.daily_vol_filter)]
    best_strat, best_w, best_sig = _simulate(best_feat, args.cost_bps, args.weight, float(best.tp_atr), float(best.sl_atr), float(best.entry_atr), int(best.max_hold))
    best_frame = _full_frame(gbp, best_feat, best_strat, best_w, best_sig)
    main_stats, fx_stats, comb_stats, corr = _combined_with_main(best_frame, args.weight, Path(args.out_fine_combined))
    print("Best fine-grid combo combined with main strategy:")
    print(f"  Main only:       Sharpe {main_stats['sharpe']:.3f}")
    print(f"  Main + forex MR: Sharpe {comb_stats['sharpe']:.3f}  (corr_vs_main={corr:.4f})")
    decision = "IMPLEMENT" if best.raw_sharpe > 0.8 and corr < 0.3 and comb_stats["sharpe"] > main_stats["sharpe"] else "RESEARCH" if best.raw_sharpe > 0.5 else "SKIP"
    print(f"Decision: {decision}")
    print(f"Saved: {args.out_fine_summary}"); print(f"Saved: {args.out_fine_combined}")
    return 0


BEST_CONFIRM = {"tp": 2.0, "sl": 1.2, "entry": 2.5, "hold": 16, "ema": "20/50", "session": "ny"}
ALLOC_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]


def _pair_sleeve(px: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    feat = _features(px, BEST_CONFIRM["ema"], BEST_CONFIRM["session"], True)
    strat, w, sig = _simulate(feat, args.cost_bps, args.weight, BEST_CONFIRM["tp"], BEST_CONFIRM["sl"], BEST_CONFIRM["entry"], BEST_CONFIRM["hold"])
    codes, ndays = _day_codes(px["timestamp"])
    stats = {**_to_daily_stats(codes, ndays, strat), **_trade_metrics(px["timestamp"], strat, w, sig)}
    daily = _daily_returns(_full_frame(px, feat, strat, w, sig))
    return stats, daily


def _run_confirm(args: argparse.Namespace, gbp: pd.DataFrame, data_dir: Path) -> int:
    print("=" * 90)
    print(f"CONFIRM BEST CONFIG ACROSS PAIRS: TP {BEST_CONFIRM['tp']} / SL {BEST_CONFIRM['sl']} / Entry {BEST_CONFIRM['entry']} / Hold {BEST_CONFIRM['hold']}h / {BEST_CONFIRM['session']} session / EMA {BEST_CONFIRM['ema']} / vol filter ON")
    print("=" * 90)

    pair_stats: dict[str, dict] = {}
    pair_daily: dict[str, pd.DataFrame] = {}
    for pair, symbol in PAIRS.items():
        px = gbp if pair == "GBPUSD" else _download_hourly(symbol, data_dir / f"{pair}_1h.csv", bool(args.refresh_data))
        stats, daily = _pair_sleeve(px, args)
        pair_stats[pair] = stats
        pair_daily[pair] = daily.rename(columns={"strategy_return": f"fx_{pair}"})
        print(f"{pair}: raw_sharpe={stats['raw_sharpe']:.3f}  sharpe={stats['sharpe']:.3f}  win_rate={stats['win_rate']*100:.1f}%  trades={stats['trades']}  expectancy={stats['expectancy']*10000:.2f}bp  pf={stats['profit_factor']:.2f}")

    n_clear = sum(1 for p in PAIRS if pair_stats[p]["raw_sharpe"] is not None and not np.isnan(pair_stats[p]["raw_sharpe"]) and pair_stats[p]["raw_sharpe"] > 0.5)
    print()
    print(f"{'UNIVERSAL EDGE' if n_clear == len(PAIRS) else 'NOT UNIVERSAL'}: {n_clear}/{len(PAIRS)} pairs clear raw Sharpe > 0.5")

    start = str(pair_daily["GBPUSD"]["day"].min().date()); end = str(pair_daily["GBPUSD"]["day"].max().date())
    main = _crypto_main(start, end, Path("artifacts/backtest/forex_intraday_main"), 20.0, False)
    merged = main.copy()
    for pair in PAIRS:
        merged = merged.merge(pair_daily[pair], on="day", how="inner")
    print(f"(merged sample: {len(merged)} common trading days)")

    main_stats = _stats(merged["main_return"])
    print(f"\nMain only Sharpe: {main_stats['sharpe']:.3f}")

    alloc_rows = []
    for pair in PAIRS:
        fx_unit = merged[f"fx_{pair}"] / args.weight
        print(f"\n{pair} MR allocation sweep (single-sleeve blend with main):")
        for a in ALLOC_GRID:
            blend = (1.0 - a) * merged["main_return"] + a * fx_unit
            st = _stats(blend)
            alloc_rows.append({"pair": pair, "allocation": a, **st})
            flag = "  <-- beats main" if st["sharpe"] > main_stats["sharpe"] else ""
            print(f"  {a*100:>4.0f}%: Sharpe {st['sharpe']:.3f}{flag}")

    alloc_df = pd.DataFrame(alloc_rows)
    best_per_pair = alloc_df.loc[alloc_df.groupby("pair")["sharpe"].idxmax()]
    print("\nBest single-sleeve allocation per pair:")
    print(best_per_pair[["pair", "allocation", "sharpe"]].to_string(index=False))
    Path(args.out_confirm_alloc).parent.mkdir(parents=True, exist_ok=True)
    alloc_df.to_csv(args.out_confirm_alloc, index=False)

    print("\nAll three pairs combined simultaneously (X% each):")
    multi_rows = []
    for a in ALLOC_GRID:
        total_fx = sum(a * (merged[f"fx_{pair}"] / args.weight) for pair in PAIRS)
        blend = (1.0 - len(PAIRS) * a) * merged["main_return"] + total_fx
        st = _stats(blend)
        multi_rows.append({"allocation_each": a, "total_fx_allocation": len(PAIRS) * a, **st})
        flag = "  <-- beats main" if st["sharpe"] > main_stats["sharpe"] else ""
        print(f"  {a*100:>4.0f}% each ({len(PAIRS)*a*100:.0f}% total fx): Sharpe {st['sharpe']:.3f}{flag}")
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(args.out_confirm_multi, index=False)

    print(f"\nSaved: {args.out_confirm_alloc}")
    print(f"Saved: {args.out_confirm_multi}")
    return 0


def _run_full(args: argparse.Namespace, gbp: pd.DataFrame, gbp_codes: np.ndarray, gbp_ndays: int, data_dir: Path, work: Path) -> int:
    feature_cache = {(ema, session, False): _features(gbp, ema, session, False) for ema in EMA_GRID for session in SESSIONS}
    rows = []
    for tp, sl, entry, ema, max_hold, session in itertools.product([0.5,1.0,1.5,2.0,2.5,3.0], [0.5,1.0,1.5,2.0], [1.0,1.2,1.5,2.0,2.5], EMA_GRID, [12,24,48,72], SESSIONS):
        feat = feature_cache[(ema, session, False)]
        strat, w, sig = _simulate(feat, args.cost_bps, args.weight, tp, sl, entry, max_hold)
        rows.append({"pair":"GBPUSD", "tp_atr":tp, "sl_atr":sl, "entry_atr":entry, "ema":ema, "max_hold":max_hold, "session":session, "daily_vol_filter":False, **_to_daily_stats(gbp_codes, gbp_ndays, strat), **_trade_metrics(gbp["timestamp"], strat, w, sig)})
    grid = pd.DataFrame(rows).sort_values(["sharpe", "profit_factor", "expectancy"], ascending=[False, False, False])
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True); grid.to_csv(args.out_summary, index=False)
    best = grid.iloc[0]

    confirm_rows = []
    best_gbp_frame = None
    for pair, symbol in PAIRS.items():
        px = _download_hourly(symbol, data_dir / f"{pair}_1h.csv", bool(args.refresh_data))
        px_codes, px_ndays = _day_codes(px["timestamp"])
        feat = _features(px, str(best["ema"]), str(best["session"]), False)
        strat, w, sig = _simulate(feat, args.cost_bps, args.weight, float(best["tp_atr"]), float(best["sl_atr"]), float(best["entry_atr"]), int(best["max_hold"]))
        row = {"pair":pair, "tp_atr":float(best["tp_atr"]), "sl_atr":float(best["sl_atr"]), "entry_atr":float(best["entry_atr"]), "ema":str(best["ema"]), "max_hold":int(best["max_hold"]), "session":str(best["session"]), "daily_vol_filter":False, **_to_daily_stats(px_codes, px_ndays, strat), **_trade_metrics(px["timestamp"], strat, w, sig)}
        confirm_rows.append(row)
        frame = _full_frame(px, feat, strat, w, sig)
        frame.to_csv(work / f"{pair}_best_config_daily.csv", index=False)
        if pair == "GBPUSD": best_gbp_frame = frame

    feat_off = feature_cache[(str(best["ema"]), str(best["session"]), False)]
    off_strat, off_w, off_sig = _simulate(feat_off, args.cost_bps, args.weight, float(best["tp_atr"]), float(best["sl_atr"]), float(best["entry_atr"]), int(best["max_hold"]))
    feat_on = _features(gbp, str(best["ema"]), str(best["session"]), True)
    on_strat, on_w, on_sig = _simulate(feat_on, args.cost_bps, args.weight, float(best["tp_atr"]), float(best["sl_atr"]), float(best["entry_atr"]), int(best["max_hold"]))
    confirm_rows.append({"pair":"GBPUSD_VOL_FILTER_ON", "tp_atr":float(best["tp_atr"]), "sl_atr":float(best["sl_atr"]), "entry_atr":float(best["entry_atr"]), "ema":str(best["ema"]), "max_hold":int(best["max_hold"]), "session":str(best["session"]), "daily_vol_filter":True, **_to_daily_stats(gbp_codes, gbp_ndays, on_strat), **_trade_metrics(gbp["timestamp"], on_strat, on_w, on_sig)})
    pd.DataFrame(confirm_rows).to_csv(args.out_confirm, index=False)

    assert best_gbp_frame is not None
    main_stats, fx_stats, comb_stats, corr = _combined_with_main(best_gbp_frame, args.weight, Path(args.out_combined))

    print("="*80); print("GBPUSD INTRADAY GRID SEARCH"); print(f"{gbp['timestamp'].min().date()} to {gbp['timestamp'].max().date()} | 1bps costs"); print("="*80)
    print("TP   SL   Entry EMA    Session    Sharpe  CAGR  WinRate Expect PF")
    print("-"*80)
    for _, r in grid.head(20).iterrows():
        print(f"{r.tp_atr:<4.1f} {r.sl_atr:<4.1f} {r.entry_atr:<5.1f} {str(r.ema):<6} {str(r.session):<10} {r.sharpe:>6.3f} {r.cagr*100:>5.2f}% {r.win_rate*100:>6.1f}% {r.expectancy*10000:>6.2f} {r.profit_factor:>4.2f}")
    print("="*80)
    print("Best config found:")
    print(f"  TP {best.tp_atr:.1f} ATR | SL {best.sl_atr:.1f} ATR | Entry {best.entry_atr:.1f} ATR | EMA {best.ema} | Session {best.session} | Max hold {int(best.max_hold)}h")
    print(f"  Sharpe {best.sharpe:.3f} | Raw {best.raw_sharpe:.3f} | CAGR {best.cagr*100:.2f}% | Win {best.win_rate*100:.1f}% | Expectancy {best.expectancy*10000:.2f} bps/trade | PF {best.profit_factor:.2f} | Trades/week {best.trades_per_week:.2f}")
    print("Confirmed on other pairs:")
    for _, r in pd.DataFrame(confirm_rows)[lambda x: ~x['pair'].astype(str).str.contains('VOL_FILTER')].iterrows(): print(f"  {r.pair}: Sharpe {r.sharpe:.3f} Raw {r.raw_sharpe:.3f} CAGR {r.cagr*100:.2f}%")
    print("Daily vol filter impact:"); print(f"  Without filter: Sharpe {_to_daily_stats(gbp_codes, gbp_ndays, off_strat)['sharpe']:.3f}"); print(f"  With filter:    Sharpe {_to_daily_stats(gbp_codes, gbp_ndays, on_strat)['sharpe']:.3f}")
    print("Combined with main strategy:"); print(f"  Main only:       Sharpe {main_stats['sharpe']:.3f}"); print(f"  Main + forex MR: Sharpe {comb_stats['sharpe']:.3f}")
    decision = "IMPLEMENT" if best.raw_sharpe > 0.8 and corr < 0.3 and comb_stats["sharpe"] > main_stats["sharpe"] else "RESEARCH" if best.raw_sharpe > 0.5 else "SKIP"
    print(f"Decision: {decision}"); print(f"Saved: {args.out_summary}"); print(f"Saved: {args.out_confirm}"); print(f"Saved: {args.out_combined}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday forex mean-reversion grid search on 1h bars.")
    ap.add_argument("--mode", choices=["full", "fine", "confirm"], default="full", help="full = original 6-axis GBPUSD grid across all sessions/EMAs; fine = tightened grid around the confirmed NY/20-50 region, with/without the daily vol filter; confirm = test the best fine-grid config on all three pairs plus an allocation sweep against the main strategy")
    ap.add_argument("--data-dir", default="data/forex_intraday")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_intraday")
    ap.add_argument("--out-summary", default="artifacts/backtest/forex_intraday_grid_summary.csv")
    ap.add_argument("--out-confirm", default="artifacts/backtest/forex_intraday_confirm_summary.csv")
    ap.add_argument("--out-combined", default="artifacts/backtest/forex_intraday_combined_summary.csv")
    ap.add_argument("--out-fine-summary", default="artifacts/backtest/forex_intraday_fine_grid_summary.csv")
    ap.add_argument("--out-fine-combined", default="artifacts/backtest/forex_intraday_fine_combined_summary.csv")
    ap.add_argument("--out-confirm-alloc", default="artifacts/backtest/forex_intraday_confirm_alloc.csv")
    ap.add_argument("--out-confirm-multi", default="artifacts/backtest/forex_intraday_confirm_multi.csv")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--weight", type=float, default=0.15)
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir); work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    gbp = _download_hourly(PAIRS["GBPUSD"], data_dir / "GBPUSD_1h.csv", bool(args.refresh_data))
    gbp_codes, gbp_ndays = _day_codes(gbp["timestamp"])

    if args.mode == "fine":
        return _run_fine(args, gbp, gbp_codes, gbp_ndays)
    if args.mode == "confirm":
        return _run_confirm(args, gbp, data_dir)
    return _run_full(args, gbp, gbp_codes, gbp_ndays, data_dir, work)


if __name__ == "__main__":
    raise SystemExit(main())
