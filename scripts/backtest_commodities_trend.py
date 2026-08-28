from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_forex_optimised import _crypto_main, _stats, _vol_multiplier


COMMODITIES = {
    "GOLD": {"futures": "GC=F", "etf": "GLD"},
    "SILVER": {"futures": "SI=F", "etf": "SLV"},
    "OIL": {"futures": "CL=F", "etf": "USO"},
    "NATGAS": {"futures": "NG=F", "etf": "UNG"},
}
DATA_FILENAMES = {"GOLD": "GOLD_daily.csv", "SILVER": "SILVER_daily.csv", "OIL": "OIL_daily.csv", "NATGAS": "NATGAS_daily.csv"}

EMA_GRID = {
    "10/25/75": (10, 25, 75),
    "15/35/100": (15, 35, 100),
    "20/50/200": (20, 50, 200),
    "25/60/150": (25, 60, 150),
    "30/75/200": (30, 75, 200),
    "50/120/300": (50, 120, 300),
    "100/200/400": (100, 200, 400),
}
CONFIRM_GRID = [3, 5, 7, 10]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _download_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower().replace(" ", "_") for c in raw.columns]
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"date", "datetime", "index"}), None)
    if date_col is None:
        raise SystemExit(f"Could not identify date column for {ticker}. Columns: {list(raw.columns)}")
    raw = raw.rename(columns={date_col: "timestamp"})
    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raw[col] = raw.get("close", np.nan)
    if "volume" not in raw.columns:
        raw["volume"] = 0
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    return raw[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _load_or_download(ticker: str, start: str, end: str, cache_csv: Path, refresh: bool) -> pd.DataFrame:
    if cache_csv.exists() and not refresh:
        d = pd.read_csv(cache_csv)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        return d
    d = _download_daily(ticker, start, end)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(cache_csv, index=False)
    return d


def _load_commodity(name: str, data_dir: Path, start: str, end: str, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (primary_df_for_signal, etf_df_for_carry_proxy, source_used)."""
    spec = COMMODITIES[name]
    fut = _load_or_download(spec["futures"], start, end, data_dir / f"_raw_{name}_futures.csv", refresh)
    etf = _load_or_download(spec["etf"], start, end, data_dir / f"_raw_{name}_etf.csv", refresh)

    expected_days = len(pd.bdate_range(start, end))
    use_futures = len(fut) >= 0.7 * expected_days
    primary = fut if use_futures else etf
    source = "futures" if use_futures else "etf"

    out_csv = data_dir / DATA_FILENAMES[name]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(out_csv, index=False)
    return primary, etf, source


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _ema_stack(close: pd.Series, spans: tuple[int, int, int]) -> tuple[pd.Series, pd.Series, pd.Series]:
    f, m, s = spans
    return close.ewm(span=f, adjust=False).mean(), close.ewm(span=m, adjust=False).mean(), close.ewm(span=s, adjust=False).mean()


def _build_regime(d: pd.DataFrame, spans: tuple[int, int, int], confirm_days: int) -> pd.DataFrame:
    close = d["close"].astype(float)
    ema_f, ema_m, ema_s = _ema_stack(close, spans)
    bull_aligned = (ema_f > ema_m) & (ema_m > ema_s)
    bear_aligned = (ema_f < ema_m) & (ema_m < ema_s)
    regime = pd.Series(np.select([bull_aligned, bear_aligned], ["BULL", "BEAR"], default="CHOP"), index=d.index)
    conf = max(1, int(confirm_days))
    entry_long = (bull_aligned.rolling(conf, min_periods=conf).min() == 1).fillna(False)
    entry_short = (bear_aligned.rolling(conf, min_periods=conf).min() == 1).fillna(False)
    exit_long = ((~bull_aligned).rolling(conf, min_periods=conf).min() == 1).fillna(False)
    exit_short = ((~bear_aligned).rolling(conf, min_periods=conf).min() == 1).fillna(False)
    return pd.DataFrame({
        "ema_fast": ema_f, "ema_mid": ema_m, "ema_slow": ema_s,
        "bull_aligned": bull_aligned, "bear_aligned": bear_aligned, "regime": regime,
        "entry_long": entry_long, "entry_short": entry_short, "exit_long": exit_long, "exit_short": exit_short,
    }, index=d.index)


def _flip_classifications(close: pd.Series, regime: pd.Series) -> tuple[pd.Series, pd.Series]:
    """For each bar, the STRONG/MID/WEAK label frozen from the most recent flip into
    BULL (resp. BEAR), forward-filled. Mirrors the eth_btc transition-momentum scheme:
    classify the size of the move that triggered the regime flip, freeze it for the
    life of any trade opened afterward."""
    ret = close.pct_change()
    prev_regime = regime.shift(1)
    flip_into_bull = (regime == "BULL") & (prev_regime != "BULL")
    flip_into_bear = (regime == "BEAR") & (prev_regime != "BEAR")
    cls = np.select([ret.abs() > 0.03, ret.abs() < 0.01], ["STRONG", "WEAK"], default="MID")
    bull_cls = pd.Series(np.where(flip_into_bull.to_numpy(), cls, None), index=close.index).ffill().fillna("MID")
    bear_cls = pd.Series(np.where(flip_into_bear.to_numpy(), cls, None), index=close.index).ffill().fillna("MID")
    return bull_cls, bear_cls


def _position_loop(close: pd.Series, entry_long: pd.Series, entry_short: pd.Series, exit_long: pd.Series, exit_short: pd.Series,
                    bull_cls: pd.Series, bear_cls: pd.Series, long_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    el = entry_long.to_numpy()
    es = entry_short.to_numpy() if not long_only else np.zeros(n, dtype=bool)
    xl = exit_long.to_numpy()
    xs = exit_short.to_numpy()
    bcls = bull_cls.to_numpy()
    rcls = bear_cls.to_numpy()
    px = close.to_numpy()

    direction = np.zeros(n, dtype=np.int64)
    trans_mult = np.ones(n, dtype=np.float64)

    active = 0
    cur_entry_i = -1
    cur_class = ""
    entry_price = np.nan
    for i in range(n):
        if active == 0:
            if el[i]:
                active = 1; cur_entry_i = i; cur_class = bcls[i]; entry_price = px[i]
            elif es[i]:
                active = -1; cur_entry_i = i; cur_class = rcls[i]; entry_price = px[i]
        elif active == 1 and xl[i]:
            active = 0; cur_entry_i = -1
        elif active == -1 and xs[i]:
            active = 0; cur_entry_i = -1

        direction[i] = active
        if active != 0 and cur_class == "WEAK":
            held = i - cur_entry_i
            cur_ret = (px[i] / entry_price - 1.0) * active
            trans_mult[i] = 1.0 if (held >= 5 and cur_ret > 0.0) else 0.6
    return direction, trans_mult, (direction != 0).astype(np.int64)


def _conviction_multiplier(gap_pct: pd.Series, regime: pd.Series, vol_pct: pd.Series) -> pd.Series:
    gap_score = np.select([gap_pct.abs() > 0.02, gap_pct.abs() >= 0.01], [0.33, 0.20], default=0.10)
    regime_score = np.where(regime.isin(["BULL", "BEAR"]), 0.33, 0.17)
    vol_score = np.select([vol_pct < 0.25, vol_pct > 0.75], [0.33, 0.10], default=0.20)
    total = pd.Series(gap_score, index=regime.index) + pd.Series(regime_score, index=regime.index) + pd.Series(vol_score, index=regime.index).fillna(0.20)
    return pd.Series(np.select([total > 0.80, total >= 0.60, total >= 0.40], [1.2, 1.0, 0.8], default=0.6), index=regime.index)


def _carry_signal(fut: pd.DataFrame, etf: pd.DataFrame) -> pd.Series:
    """Approximate roll-yield/term-structure proxy. yfinance does not reliably expose
    individual next-contract-month futures across a multi-year rolling window, so a
    literal front-month-vs-next-month ratio is not obtainable here. Instead we use the
    realized spread between the continuous front-month future (which bears the full
    roll cost/gain) and its tracking ETF (whose NAV already absorbs the fund's realized
    roll cost) as an observable, data-available proxy for backwardation/contango:
    futures outperforming its ETF over a trailing window implies backwardation-like
    conditions (positive carry for longs); futures underperforming implies contango-like
    drag (negative carry for longs)."""
    f = fut[["timestamp", "close"]].rename(columns={"close": "fut_close"})
    e = etf[["timestamp", "close"]].rename(columns={"close": "etf_close"})
    m = f.merge(e, on="timestamp", how="left").sort_values("timestamp")
    m["etf_close"] = m["etf_close"].ffill()
    # Same clip as exec_return, and for the same reason (WTI's 2020-04-20 negative print).
    fut_ret = m["fut_close"].pct_change().clip(lower=-0.5, upper=0.5)
    etf_ret = m["etf_close"].pct_change().clip(lower=-0.5, upper=0.5)
    spread = (fut_ret - etf_ret).rolling(20, min_periods=20).sum()
    return pd.Series(spread.to_numpy(), index=fut.index)


def _gold_rate_ok(d: pd.DataFrame, data_dir: Path, start: str, end: str, refresh: bool) -> np.ndarray:
    tlt = _load_or_download("TLT", start, end, data_dir / "_raw_TLT.csv", refresh)
    dxy = _load_or_download("DX-Y.NYB", start, end, data_dir / "_raw_DXY.csv", refresh)
    tlt_bull = (tlt["close"].ewm(span=20, adjust=False).mean() > tlt["close"].ewm(span=50, adjust=False).mean())
    dxy_bear = (dxy["close"].ewm(span=20, adjust=False).mean() < dxy["close"].ewm(span=50, adjust=False).mean())
    tlt_frame = pd.DataFrame({"timestamp": tlt["timestamp"], "tlt_bull": tlt_bull})
    dxy_frame = pd.DataFrame({"timestamp": dxy["timestamp"], "dxy_bear": dxy_bear})
    merged = pd.DataFrame({"timestamp": d["timestamp"]}).merge(tlt_frame, on="timestamp", how="left").merge(dxy_frame, on="timestamp", how="left")
    merged[["tlt_bull", "dxy_bear"]] = merged[["tlt_bull", "dxy_bear"]].ffill().fillna(False)
    return (merged["tlt_bull"] | merged["dxy_bear"]).to_numpy(dtype=bool)


def _oil_seasonal_multiplier(timestamps: pd.Series) -> np.ndarray:
    month = timestamps.dt.month
    strong = month.isin([10, 11, 12, 1, 2, 3])
    return np.where(strong, 1.2, 0.8)


def _natgas_seasonal_direction(timestamps: pd.Series) -> np.ndarray:
    month = timestamps.dt.month
    bullish = month.isin([11, 12, 1, 2])
    bearish = month.isin([5, 6, 7, 8, 9])
    return np.select([bullish, bearish], [1, -1], default=0)


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(
    d: pd.DataFrame, etf: pd.DataFrame, name: str, spans: tuple[int, int, int], confirm_days: int, *,
    long_only: bool = False, vol_filter: bool = True, asym_sizing: bool = True, transition: bool = True,
    carry: bool = True, carry_scale: float = 1.0, gold_rate_filter: bool = False, oil_seasonal: bool = False,
    natgas_mode: str = "ema", cost_bps: float = 20.0, gross_cap: float = 1.0,
    data_dir: Path | None = None, start: str = "", end: str = "", refresh: bool = False,
) -> pd.DataFrame:
    close = d["close"].astype(float)
    reg = _build_regime(d, spans, confirm_days)

    entry_long, entry_short = reg["entry_long"].copy(), reg["entry_short"].copy()

    if name == "GOLD" and gold_rate_filter:
        assert data_dir is not None
        ok = _gold_rate_ok(d, data_dir, start, end, refresh)
        entry_long = entry_long & ok

    if name == "NATGAS" and natgas_mode != "ema":
        seas_dir = _natgas_seasonal_direction(d["timestamp"])
        if natgas_mode == "seasonal":
            entry_long = pd.Series(seas_dir == 1, index=d.index) & (~pd.Series(seas_dir == 1, index=d.index).shift(1).fillna(False))
            entry_short = pd.Series(seas_dir == -1, index=d.index) & (~pd.Series(seas_dir == -1, index=d.index).shift(1).fillna(False))
            reg = reg.copy()
            reg["exit_long"] = pd.Series(seas_dir != 1, index=d.index)
            reg["exit_short"] = pd.Series(seas_dir != -1, index=d.index)
        elif natgas_mode == "combined":
            seas = pd.Series(seas_dir, index=d.index)
            entry_long = entry_long & (seas != -1)
            entry_short = entry_short & (seas != 1)

    bull_cls, bear_cls = _flip_classifications(close, reg["regime"])
    direction, trans_mult, active_flag = _position_loop(close, entry_long, entry_short, reg["exit_long"], reg["exit_short"], bull_cls, bear_cls, long_only)

    vol_mult, vol_pct = _vol_multiplier(close)
    if not vol_filter:
        vol_mult = pd.Series(1.0, index=close.index)

    gap_pct = ((reg["ema_mid"] - reg["ema_slow"]) / close.replace(0.0, np.nan))
    asym_mult = _conviction_multiplier(gap_pct, reg["regime"], vol_pct) if asym_sizing else pd.Series(1.0, index=close.index)

    weight_target = pd.Series(direction, index=d.index).astype(float) * vol_mult.to_numpy() * asym_mult.to_numpy()
    if transition:
        weight_target = weight_target * trans_mult
    if name == "OIL" and oil_seasonal:
        weight_target = weight_target * _oil_seasonal_multiplier(d["timestamp"])
    weight_target = weight_target.clip(lower=-gross_cap, upper=gross_cap)

    weight_exec = weight_target.shift(1).fillna(0.0)
    # WTI (CL=F) traded briefly negative on 2020-04-20 (the widely documented one-off
    # contract-expiry event). open.shift(-1)/open - 1 is mathematically well-defined but
    # not a realizable trading outcome when the reference price crosses through zero --
    # it produced a fabricated +90%/+104% two-day return here. Clip to a bound wide
    # enough to keep every legitimate move in this dataset (max observed elsewhere is
    # ~32%) while killing that one artifact.
    exec_return = (d["open"].shift(-1) / d["open"] - 1.0).clip(lower=-0.5, upper=0.5)
    prev_w = weight_exec.shift(1).fillna(0.0)
    turnover = (weight_exec - prev_w).abs()
    cost = turnover * (float(cost_bps) / 10000.0)

    carry_contrib = pd.Series(0.0, index=d.index)
    if carry:
        carry_sig = _carry_signal(d, etf).fillna(0.0)
        # Proportional to position size (like the price-return P&L), not just its sign,
        # so a bigger position accrues proportionally more carry -- and so this formula
        # is directly reusable at any rescaled weight (e.g. the portfolio-priority sweep).
        carry_contrib = weight_exec * (carry_sig.shift(1).fillna(0.0) / 20.0) * float(carry_scale)

    strategy_return = weight_exec * exec_return.fillna(0.0) - cost + carry_contrib

    out = d[["timestamp", "open", "high", "low", "close"]].copy()
    out["day"] = out["timestamp"].dt.floor("D")
    out["regime"] = reg["regime"].to_numpy()
    out["direction"] = direction
    out["weight_target"] = weight_target.to_numpy()
    out["weight_exec"] = weight_exec.to_numpy()
    out["exec_return"] = exec_return.to_numpy()
    out["cost"] = cost.to_numpy()
    out["carry_contrib"] = carry_contrib.to_numpy()
    out["strategy_return"] = strategy_return.to_numpy()
    out = out.dropna(subset=["exec_return"]).reset_index(drop=True)
    return out


def _trade_stats(frame: pd.DataFrame) -> dict:
    w = frame["weight_exec"].to_numpy()
    ret = frame["strategy_return"].to_numpy()
    direction = frame["direction"].to_numpy()
    active = np.abs(w) > 1e-9
    if not active.any():
        return {"trades": 0, "trades_per_year": 0.0, "win_rate": np.nan, "long_win_rate": np.nan, "short_win_rate": np.nan}
    n = len(active)
    diff = np.diff(active.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if active[-1] else end_candidates
    log_ret = np.log1p(np.clip(ret, -0.999999, None))
    cum0 = np.concatenate(([0.0], np.cumsum(log_ret)))
    trade_returns = np.expm1(cum0[ends + 1] - cum0[starts])
    trade_side = direction[starts]
    years = max((frame["timestamp"].max() - frame["timestamp"].min()).days / 365.25, 1e-9)
    wins = trade_returns > 0
    long_mask = trade_side == 1
    short_mask = trade_side == -1
    return {
        "trades": int(len(trade_returns)),
        "trades_per_year": float(len(trade_returns) / years),
        "win_rate": float(wins.mean()) if len(wins) else np.nan,
        "long_win_rate": float(wins[long_mask].mean()) if long_mask.any() else np.nan,
        "short_win_rate": float(wins[short_mask].mean()) if short_mask.any() else np.nan,
    }


def _daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("day", as_index=False)["strategy_return"].apply(lambda s: (1.0 + s).prod() - 1.0)


# ---------------------------------------------------------------------------
# Grid search / ablations / combined portfolio
# ---------------------------------------------------------------------------

def _run_grid(name: str, d: pd.DataFrame, etf: pd.DataFrame, data_dir: Path, start: str, end: str, cost_bps: float, refresh: bool) -> pd.DataFrame:
    rows = []
    for label, spans in EMA_GRID.items():
        for conf in CONFIRM_GRID:
            frame = run_backtest(d, etf, name, spans, conf, cost_bps=cost_bps, data_dir=data_dir, start=start, end=end, refresh=refresh)
            daily = _daily_returns(frame)
            st = _stats(daily["strategy_return"])
            ts = _trade_stats(frame)
            rows.append({"commodity": name, "ema": label, "confirm_days": conf, **st, **ts})
    return pd.DataFrame(rows).sort_values("raw_sharpe", ascending=False).reset_index(drop=True)


def _ablation_variants(name: str) -> dict[str, dict]:
    variants: dict[str, dict] = {
        "full_stack": {},
        "long_only": {"long_only": True},
        "vol_filter_off": {"vol_filter": False},
        "carry_off": {"carry": False},
    }
    if name == "GOLD":
        variants["gold_rate_filter_on"] = {"gold_rate_filter": True}
    if name == "OIL":
        variants["oil_seasonal_on"] = {"oil_seasonal": True}
    if name == "NATGAS":
        variants["natgas_seasonal_only"] = {"natgas_mode": "seasonal"}
        variants["natgas_combined"] = {"natgas_mode": "combined"}
    return variants


def _run_ablations(name: str, d: pd.DataFrame, etf: pd.DataFrame, spans: tuple[int, int, int], confirm: int, data_dir: Path, start: str, end: str, cost_bps: float, refresh: bool) -> pd.DataFrame:
    rows = []
    for label, overrides in _ablation_variants(name).items():
        kwargs: dict = dict(cost_bps=cost_bps, data_dir=data_dir, start=start, end=end, refresh=refresh)
        kwargs.update(overrides)
        frame = run_backtest(d, etf, name, spans, confirm, **kwargs)
        daily = _daily_returns(frame)
        st = _stats(daily["strategy_return"])
        ts = _trade_stats(frame)
        rows.append({"commodity": name, "variant": label, **st, **ts})
    return pd.DataFrame(rows)


def _main_occupied_weight(work_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(work_dir / "validated_crypto_daily.csv", low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    occ = (
        pd.to_numeric(base.get("alloc_eth", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(base.get("alloc_btc", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(base.get("gold_weight_exec", 0.0), errors="coerce").fillna(0.0).abs()
    ).clip(lower=0.0, upper=1.0)
    return pd.DataFrame({"day": base["day"], "main_occupied_weight": occ})


def _priority_allocation_sweep(commodity_frame: pd.DataFrame, main_df: pd.DataFrame, work_dir: Path, allocations: list[float], cost_bps: float) -> pd.DataFrame:
    """Main gets its full return untouched every day ('first claim'). The commodity
    sleeve is capped, per day, at min(allocation ceiling, 1 - main's occupied weight)
    -- so it only uses capital main isn't already using that day."""
    occ = _main_occupied_weight(work_dir)
    cf = commodity_frame[["day", "weight_exec", "exec_return"]].rename(columns={"weight_exec": "commodity_weight_exec", "exec_return": "commodity_exec_return"})
    m = main_df.merge(occ, on="day", how="left").merge(cf, on="day", how="inner")
    m["main_occupied_weight"] = m["main_occupied_weight"].fillna(0.0)
    rows = []
    for a in allocations:
        idle_cap = (1.0 - m["main_occupied_weight"]).clip(lower=0.0)
        cap = np.minimum(a, idle_cap)
        final_w = m["commodity_weight_exec"].clip(lower=-cap, upper=cap)
        prev_w = final_w.shift(1).fillna(0.0)
        cost = (final_w - prev_w).abs() * (float(cost_bps) / 10000.0)
        commodity_contrib = final_w * m["commodity_exec_return"].fillna(0.0) - cost
        blend = m["main_return"] + commodity_contrib
        rows.append({"allocation": a, **_stats(blend)})
    return pd.DataFrame(rows)


GOLD_LONGONLY_EMA = {
    "10/25/75": (10, 25, 75), "15/35/100": (15, 35, 100), "20/50/200": (20, 50, 200), "25/60/150": (25, 60, 150),
    "30/75/200": (30, 75, 200), "50/120/300": (50, 120, 300), "75/150/400": (75, 150, 400), "100/200/500": (100, 200, 500),
}
GOLD_LONGONLY_CONFIRM = [3, 5, 7, 10, 14]


def _run_gold_longonly_grid(d: pd.DataFrame, etf: pd.DataFrame, data_dir: Path, start: str, end: str, cost_bps: float, refresh: bool) -> pd.DataFrame:
    rows = []
    for label, spans in GOLD_LONGONLY_EMA.items():
        for conf in GOLD_LONGONLY_CONFIRM:
            for vol_filter in (True, False):
                for rate_filter in (True, False):
                    frame = run_backtest(
                        d, etf, "GOLD", spans, conf, long_only=True, vol_filter=vol_filter, asym_sizing=True,
                        transition=True, carry=True, gold_rate_filter=rate_filter, cost_bps=cost_bps,
                        data_dir=data_dir, start=start, end=end, refresh=refresh,
                    )
                    daily = _daily_returns(frame)
                    st = _stats(daily["strategy_return"])
                    ts = _trade_stats(frame)
                    rows.append({"ema": label, "confirm_days": conf, "vol_filter": vol_filter, "rate_filter": rate_filter, **st, **ts})
    return pd.DataFrame(rows).sort_values("raw_sharpe", ascending=False).reset_index(drop=True)


def _fill_gap_analysis(commodity_frame: pd.DataFrame, main_df: pd.DataFrame, work_dir: Path) -> dict:
    """When main is flat (no occupied capital that day), does the commodity actually
    signal and fill the gap -- or does it mostly trade at the same time as main, which
    would be the worst case (no diversification benefit, just added correlated risk)?"""
    occ = _main_occupied_weight(work_dir)
    m = main_df.merge(occ, on="day", how="left").merge(
        commodity_frame[["day", "weight_exec", "exec_return"]].rename(columns={"weight_exec": "commodity_w", "exec_return": "commodity_ret"}),
        on="day", how="inner",
    )
    m["main_occupied_weight"] = m["main_occupied_weight"].fillna(0.0)
    main_flat = m["main_occupied_weight"] <= 1e-9
    commodity_active = m["commodity_w"].abs() > 1e-9
    n_main_flat = int(main_flat.sum())
    n_fill = int((main_flat & commodity_active).sum())
    n_overlap = int((~main_flat & commodity_active).sum())
    n_commodity_active = int(commodity_active.sum())
    return {
        "main_flat_days": n_main_flat,
        "total_days": int(len(m)),
        "commodity_active_on_main_flat_days": n_fill,
        "pct_of_main_flat_days_filled": float(n_fill / n_main_flat) if n_main_flat else np.nan,
        "avg_commodity_return_on_fill_days": float(m.loc[main_flat & commodity_active, "commodity_ret"].mean()) if n_fill else np.nan,
        "commodity_active_days_total": n_commodity_active,
        "commodity_active_days_overlapping_main": n_overlap,
        "pct_of_commodity_activity_overlapping_main": float(n_overlap / n_commodity_active) if n_commodity_active else np.nan,
    }


def _priority_allocation_multi(sleeves: dict[str, tuple[pd.DataFrame, float]], main_df: pd.DataFrame, work_dir: Path, cost_bps: float) -> dict:
    """Same 'main first claim, idle capital only' rule as _priority_allocation_sweep,
    generalized to multiple secondary sleeves that may compete for the same leftover
    capacity on the same day -- if their combined want exceeds what's idle, both are
    scaled down proportionally rather than one silently crowding out the other."""
    occ = _main_occupied_weight(work_dir)
    m = main_df.merge(occ, on="day", how="left")
    m["main_occupied_weight"] = m["main_occupied_weight"].fillna(0.0)
    for name, (frame, _alloc) in sleeves.items():
        cf = frame[["day", "weight_exec", "exec_return"]].rename(columns={"weight_exec": f"{name}_w", "exec_return": f"{name}_ret"})
        m = m.merge(cf, on="day", how="inner")

    idle_cap = (1.0 - m["main_occupied_weight"]).clip(lower=0.0)
    wants = {name: m[f"{name}_w"].clip(lower=-alloc, upper=alloc) for name, (_frame, alloc) in sleeves.items()}
    total_want = sum(w.abs() for w in wants.values())
    scale = np.where(total_want > 1e-12, np.minimum(1.0, idle_cap / total_want.replace(0.0, np.nan)), 1.0)
    scale = pd.Series(scale, index=m.index).fillna(1.0)

    contrib = pd.Series(0.0, index=m.index)
    for name in sleeves:
        final_w = wants[name] * scale
        prev_w = final_w.shift(1).fillna(0.0)
        cost = (final_w - prev_w).abs() * (float(cost_bps) / 10000.0)
        contrib = contrib + final_w * m[f"{name}_ret"].fillna(0.0) - cost

    blend = m["main_return"] + contrib
    return _stats(blend)


def run_longonly_tests(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    main_df = _crypto_main(args.start, args.end, Path(args.work_dir), float(args.crypto_cost_bps), bool(args.refresh_crypto))
    main_stats = _stats(main_df["main_return"])

    # ---- TEST 1: Gold long-only optimised grid ----
    gold_d, gold_etf, gold_source = _load_commodity("GOLD", data_dir, args.start, args.end, bool(args.refresh_data))
    gold_grid = _run_gold_longonly_grid(gold_d, gold_etf, data_dir, args.start, args.end, args.cost_bps, bool(args.refresh_data))
    gold_grid.to_csv(args.out_gold, index=False)
    gold_best = gold_grid.iloc[0]
    gold_spans = GOLD_LONGONLY_EMA[str(gold_best["ema"])]
    gold_frame = run_backtest(
        gold_d, gold_etf, "GOLD", gold_spans, int(gold_best["confirm_days"]), long_only=True,
        vol_filter=bool(gold_best["vol_filter"]), asym_sizing=True, transition=True, carry=True,
        gold_rate_filter=bool(gold_best["rate_filter"]), cost_bps=args.cost_bps, data_dir=data_dir,
        start=args.start, end=args.end, refresh=bool(args.refresh_data),
    )
    gold_daily = _daily_returns(gold_frame)
    gold_merged = main_df.merge(gold_daily, on="day", how="inner")
    gold_corr = float(gold_merged["main_return"].corr(gold_merged["strategy_return"])) if len(gold_merged) > 2 else np.nan

    print("=" * 90); print("GOLD LONG-ONLY OPTIMISED"); print("=" * 90)
    print(f"Best config: EMA {gold_best['ema']} | Confirm {int(gold_best['confirm_days'])}d | Vol filter {'ON' if gold_best['vol_filter'] else 'OFF'} | Rate filter {'ON' if gold_best['rate_filter'] else 'OFF'}")
    print(f"Raw Sharpe {gold_best['raw_sharpe']:.3f} | CAGR {gold_best['cagr']*100:.1f}% | MaxDD {gold_best['maxdd']*100:.1f}% | Trades/yr {gold_best['trades_per_year']:.1f} | Corr to main {gold_corr:.3f}")

    gold_qualifies = float(gold_best["raw_sharpe"]) > 0.8
    gold_sweep = pd.DataFrame()
    if gold_qualifies:
        gold_sweep = _priority_allocation_sweep(gold_frame, main_df, Path(args.work_dir), [0.05, 0.10, 0.15, 0.20], args.cost_bps)
        print("Combined (priority allocation):")
        for _, r in gold_sweep.iterrows():
            print(f"  Main + Gold {r['allocation']*100:.0f}%: Sharpe {r['sharpe']:.3f}")
        print(f"Beats main only ({main_stats['sharpe']:.3f}): {bool((gold_sweep['sharpe'] > main_stats['sharpe']).any())}")
    else:
        print(f"Raw Sharpe {gold_best['raw_sharpe']:.3f} does not clear 0.8 -- combined allocation test skipped.")
    print("=" * 90)

    # ---- TEST 2: Oil long-only combined (no grid, fixed config) ----
    oil_d, oil_etf, oil_source = _load_commodity("OIL", data_dir, args.start, args.end, bool(args.refresh_data))
    oil_frame = run_backtest(
        oil_d, oil_etf, "OIL", (10, 25, 75), 3, long_only=True, vol_filter=False, asym_sizing=True,
        transition=True, carry=True, oil_seasonal=False, cost_bps=args.cost_bps, data_dir=data_dir,
        start=args.start, end=args.end, refresh=bool(args.refresh_data),
    )
    oil_daily = _daily_returns(oil_frame)
    oil_merged = main_df.merge(oil_daily, on="day", how="inner")
    oil_corr = float(oil_merged["main_return"].corr(oil_merged["strategy_return"])) if len(oil_merged) > 2 else np.nan
    oil_ts = _trade_stats(oil_frame)
    oil_stats_standalone = _stats(oil_daily["strategy_return"])

    oil_allocations = [0.02, 0.05, 0.10, 0.15, 0.20]
    oil_sweep = _priority_allocation_sweep(oil_frame, main_df, Path(args.work_dir), oil_allocations, args.cost_bps)
    fill_gap = _fill_gap_analysis(oil_frame, main_df, Path(args.work_dir))

    print("OIL LONG-ONLY COMBINED"); print("=" * 90)
    print(f"Standalone raw Sharpe {oil_stats_standalone['raw_sharpe']:.3f} | Corr to main {oil_corr:.3f}")
    print("Combined (priority allocation):")
    for _, r in oil_sweep.iterrows():
        print(f"  Main + Oil {r['allocation']*100:.0f}%: Sharpe {r['sharpe']:.3f}  CAGR {r['cagr']*100:.1f}%  MaxDD {r['maxdd']*100:.1f}%")
    oil_beats_main = bool((oil_sweep["sharpe"] > main_stats["sharpe"]).any())
    print(f"Beats main only ({main_stats['sharpe']:.3f}): {oil_beats_main}")
    if oil_beats_main:
        best_alloc_row = oil_sweep.loc[oil_sweep["sharpe"].idxmax()]
        print(f"Best allocation: {best_alloc_row['allocation']*100:.0f}% (Sharpe {best_alloc_row['sharpe']:.3f})")
    print()
    print("Gap-fill analysis (main flat vs oil active):")
    print(f"  Main flat days: {fill_gap['main_flat_days']} / {fill_gap['total_days']}")
    print(f"  Oil active on main-flat days: {fill_gap['commodity_active_on_main_flat_days']} ({fill_gap['pct_of_main_flat_days_filled']*100:.1f}% of main-flat days)")
    print(f"  Avg oil return on those fill days: {fill_gap['avg_commodity_return_on_fill_days']*100:.3f}%")
    print(f"  Oil active days overlapping with main-active days: {fill_gap['commodity_active_days_overlapping_main']} / {fill_gap['commodity_active_days_total']} ({fill_gap['pct_of_commodity_activity_overlapping_main']*100:.1f}% of oil's activity)")
    print("=" * 90)

    oil_summary_df = pd.concat([
        pd.DataFrame([{"metric": "standalone", **oil_stats_standalone, **oil_ts, "corr_vs_main": oil_corr}]),
        oil_sweep.assign(configuration="priority_allocation"),
        pd.DataFrame([fill_gap]),
    ], ignore_index=True)
    oil_summary_df.to_csv(args.out_oil, index=False)

    # ---- Best combined: main + gold + oil simultaneously ----
    combined_rows = [{"configuration": "main_only", "allocation_gold": np.nan, "allocation_oil": np.nan, **main_stats}]
    if gold_qualifies:
        for _, r in gold_sweep.iterrows():
            combined_rows.append({"configuration": "main_plus_gold", "allocation_gold": r["allocation"], "allocation_oil": np.nan, **{k: r[k] for k in ["return", "cagr", "sharpe", "raw_sharpe", "maxdd", "ann_vol"]}})
    for _, r in oil_sweep.iterrows():
        combined_rows.append({"configuration": "main_plus_oil", "allocation_gold": np.nan, "allocation_oil": r["allocation"], **{k: r[k] for k in ["return", "cagr", "sharpe", "raw_sharpe", "maxdd", "ann_vol"]}})

    best_joint = None
    if gold_qualifies:
        gold_best_alloc = float(gold_sweep.loc[gold_sweep["sharpe"].idxmax(), "allocation"])
        oil_best_alloc = float(oil_sweep.loc[oil_sweep["sharpe"].idxmax(), "allocation"])
        joint_stats = _priority_allocation_multi({"gold": (gold_frame, gold_best_alloc), "oil": (oil_frame, oil_best_alloc)}, main_df, Path(args.work_dir), args.cost_bps)
        combined_rows.append({"configuration": "main_plus_gold_plus_oil", "allocation_gold": gold_best_alloc, "allocation_oil": oil_best_alloc, **joint_stats})
        best_joint = (gold_best_alloc, oil_best_alloc, joint_stats)

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(args.out_longonly_combined, index=False)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    if best_joint is not None:
        ga, oa, js = best_joint
        print(f"Best combined: Main + Gold ({ga*100:.0f}%) + Oil ({oa*100:.0f}%): Sharpe {js['sharpe']:.3f} | CAGR {js['cagr']*100:.1f}% | MaxDD {js['maxdd']*100:.1f}%")
    print()
    print("Decision:")
    gold_decision = "IMPLEMENT" if (gold_qualifies and gold_corr < 0.3 and not gold_sweep.empty and bool((gold_sweep["sharpe"] > main_stats["sharpe"]).any())) else "RESEARCH" if float(gold_best["raw_sharpe"]) > 0.5 else "SKIP"
    oil_decision = "IMPLEMENT" if (oil_corr < 0.3 and oil_beats_main) else "RESEARCH" if float(oil_stats_standalone["raw_sharpe"]) > 0.5 else "SKIP"
    print(f"  Gold: {gold_decision}  (raw Sharpe {gold_best['raw_sharpe']:.3f}, corr {gold_corr:.3f})")
    print(f"  Oil: {oil_decision}  (raw Sharpe {oil_stats_standalone['raw_sharpe']:.3f}, corr {oil_corr:.3f})")
    print(f"\nSaved: {args.out_gold}")
    print(f"Saved: {args.out_oil}")
    print(f"Saved: {args.out_longonly_combined}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Commodity EMA trend-following backtest with the full LPBot improvement stack.")
    ap.add_argument("--mode", choices=["full", "longonly"], default="full", help="full = original 4-commodity long+short grid; longonly = Gold long-only optimisation + Oil long-only combined-portfolio test")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--data-dir", default="data/commodities")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised", help="Cache dir for the crypto main strategy (reuses the existing 2019-2024 cache here if present).")
    ap.add_argument("--out-summary", default="artifacts/backtest/commodities_trend_summary.csv")
    ap.add_argument("--out-combined", default="artifacts/backtest/commodities_combined_summary.csv")
    ap.add_argument("--out-grid", default="artifacts/backtest/commodities_trend_grid.csv")
    ap.add_argument("--out-ablations", default="artifacts/backtest/commodities_trend_ablations.csv")
    ap.add_argument("--out-gold", default="artifacts/backtest/gold_optimised_summary.csv")
    ap.add_argument("--out-oil", default="artifacts/backtest/oil_combined_summary.csv")
    ap.add_argument("--out-longonly-combined", default="artifacts/backtest/commodities_longonly_combined.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--crypto-cost-bps", type=float, default=20.0)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--refresh-crypto", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "longonly":
        return run_longonly_tests(args)

    all_grid_rows = []
    ablation_rows = []
    summary_rows = []
    best_frames: dict[str, pd.DataFrame] = {}

    for name in COMMODITIES:
        d, etf, source = _load_commodity(name, data_dir, args.start, args.end, bool(args.refresh_data))
        grid = _run_grid(name, d, etf, data_dir, args.start, args.end, args.cost_bps, bool(args.refresh_data))
        grid["data_source"] = source
        all_grid_rows.append(grid)
        best = grid.iloc[0]
        spans = EMA_GRID[str(best["ema"])]
        confirm = int(best["confirm_days"])

        ablation_rows.append(_run_ablations(name, d, etf, spans, confirm, data_dir, args.start, args.end, args.cost_bps, bool(args.refresh_data)))

        best_frame = run_backtest(d, etf, name, spans, confirm, cost_bps=args.cost_bps, data_dir=data_dir, start=args.start, end=args.end, refresh=bool(args.refresh_data))
        best_frames[name] = best_frame

        summary_rows.append({
            "commodity": name, "best_ema": best["ema"], "best_confirm_days": confirm, "data_source": source,
            "sharpe": best["sharpe"], "raw_sharpe": best["raw_sharpe"], "cagr": best["cagr"], "maxdd": best["maxdd"],
            "ann_vol": best["ann_vol"], "trades": best["trades"], "trades_per_year": best["trades_per_year"],
            "win_rate": best["win_rate"], "long_win_rate": best["long_win_rate"], "short_win_rate": best["short_win_rate"],
        })

    grid_df = pd.concat(all_grid_rows, ignore_index=True)
    Path(args.out_grid).parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(args.out_grid, index=False)
    ablations_df = pd.concat(ablation_rows, ignore_index=True)
    ablations_df.to_csv(args.out_ablations, index=False)

    main_df = _crypto_main(args.start, args.end, Path(args.work_dir), float(args.crypto_cost_bps), bool(args.refresh_crypto))
    main_stats = _stats(main_df["main_return"])

    for row in summary_rows:
        daily = _daily_returns(best_frames[row["commodity"]])
        merged = main_df.merge(daily, on="day", how="inner")
        row["corr_vs_main"] = float(merged["main_return"].corr(merged["strategy_return"])) if len(merged) > 2 else np.nan

    summary_df = pd.DataFrame(summary_rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)

    allocations = [0.05, 0.10, 0.15, 0.20]
    combined_rows = [{"configuration": "main_only", "allocation": np.nan, **main_stats}]
    qualifying = [n for n in COMMODITIES if float(summary_df.loc[summary_df["commodity"] == n, "raw_sharpe"].iloc[0]) > 0.8]
    for name in qualifying:
        sweep = _priority_allocation_sweep(best_frames[name], main_df, Path(args.work_dir), allocations, args.cost_bps)
        for _, r in sweep.iterrows():
            combined_rows.append({"configuration": f"main_plus_{name.lower()}", "allocation": r["allocation"], **{k: r[k] for k in ["return", "cagr", "sharpe", "raw_sharpe", "maxdd", "ann_vol"]}})
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(args.out_combined, index=False)

    print("=" * 90); print("COMMODITY TREND FOLLOWING RESULTS"); print(f"{args.start} to {args.end} | Full improvement stack | {args.cost_bps:.0f}bps costs"); print("=" * 90)
    print(f"{'Commodity':10} {'Best EMA':14} {'Confirm':7} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Corr':>7}")
    print("-" * 90)
    for _, r in summary_df.iterrows():
        print(f"{r['commodity']:10} {str(r['best_ema']):14} {int(r['best_confirm_days']):>7} {r['raw_sharpe']:>7.3f} {r['cagr']*100:>6.1f}% {r['maxdd']*100:>6.1f}% {r['corr_vs_main']:>7.3f}")
    print("-" * 90)

    if qualifying:
        print("Best combined (priority allocation):")
        for name in qualifying:
            sub = combined_df[combined_df["configuration"] == f"main_plus_{name.lower()}"]
            best_row = sub.loc[sub["sharpe"].idxmax()]
            print(f"  Main + {name.title()} ({best_row['allocation']*100:.0f}%): Sharpe {best_row['sharpe']:.3f}")
        best_combo_row = combined_df[combined_df["configuration"] != "main_only"].loc[combined_df[combined_df["configuration"] != "main_only"]["sharpe"].idxmax()]
        print(f"  Best combined overall: {best_combo_row['configuration']} ({best_combo_row['allocation']*100:.0f}%): Sharpe {best_combo_row['sharpe']:.3f}")
    else:
        print("No commodity cleared raw Sharpe > 0.8 standalone -- combined-portfolio test skipped.")
    print("=" * 90)

    print("Decision per commodity:")
    for _, r in summary_df.iterrows():
        rs = float(r["raw_sharpe"]); corr = float(r["corr_vs_main"])
        beats_main = qualifying and r["commodity"] in qualifying and combined_df.loc[combined_df["configuration"] == f"main_plus_{r['commodity'].lower()}", "sharpe"].max() > main_stats["sharpe"]
        decision = "IMPLEMENT" if (rs > 0.8 and corr < 0.3 and beats_main) else "RESEARCH" if rs > 0.5 else "SKIP"
        print(f"  {r['commodity']}: {decision}  (raw Sharpe {rs:.3f}, corr {corr:.3f})")

    print()
    print(f"Does any commodity clear Sharpe > 1.5 standalone: {bool((summary_df['raw_sharpe'] > 1.5).any())}")
    if qualifying:
        any_beats_main = bool((combined_df[combined_df['configuration'] != 'main_only']['sharpe'] > main_stats['sharpe']).any())
        print(f"Does any combination beat main only ({main_stats['sharpe']:.3f}): {any_beats_main}")
    else:
        print(f"Does any combination beat main only ({main_stats['sharpe']:.3f}): N/A (nothing qualified for combined test)")

    print(f"\nSaved: {args.out_summary}")
    print(f"Saved: {args.out_combined}")
    print(f"Saved: {args.out_grid}")
    print(f"Saved: {args.out_ablations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
