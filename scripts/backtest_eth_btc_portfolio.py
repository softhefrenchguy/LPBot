from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_btc_full_stack import (
    _download_binance_daily,
    classify_regime_v2_on_btc,
    perf,
)


def _download_gold_daily(start: str, end: str, symbol: str) -> tuple[pd.DataFrame, str]:
    tickers = [symbol]
    if symbol.upper() != "GLD":
        tickers.append("GLD")
    for t in tickers:
        x = yf.download(
            t,
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if x is None or x.empty:
            continue
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x = x.reset_index()
        date_col = "Date" if "Date" in x.columns else "index"
        x["day"] = pd.to_datetime(x[date_col], errors="coerce").dt.floor("D")
        x["open"] = pd.to_numeric(x["Open"], errors="coerce")
        x["close"] = pd.to_numeric(x["Close"], errors="coerce")
        x = x.dropna(subset=["day", "open", "close"]).sort_values("day")
        return x[["day", "open", "close"]].copy(), t
    raise RuntimeError(f"Gold data unavailable for {symbol} and GLD fallback failed.")


def load_eth_defensive_proxy(path: Path, perp_path: Path | None = None, pup_fallback: bool = False) -> pd.DataFrame:
    d = pd.read_csv(path)
    if not {"timestamp", "weight"}.issubset(d.columns):
        raise ValueError(f"{path} must contain timestamp and weight")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    if "p_up" in d.columns:
        d["p_up"] = pd.to_numeric(d["p_up"], errors="coerce")
    else:
        d["p_up"] = np.nan
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

    out = d.set_index("timestamp")[["weight", "p_up"]].resample("1D").last().rename(columns={"weight": "def_proxy_signal"})
    out.index.name = "day"
    out = out.reset_index()
    out["def_proxy_signal"] = pd.to_numeric(out["def_proxy_signal"], errors="coerce").ffill().fillna(0.0)
    out["funding_z"] = np.nan

    if perp_path is not None and perp_path.exists():
        p = pd.read_csv(perp_path)
        if {"timestamp", "funding_rate"}.issubset(p.columns):
            p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
            p["funding_rate"] = pd.to_numeric(p["funding_rate"], errors="coerce")
            p = p.dropna(subset=["timestamp"]).sort_values("timestamp")
            f = p.set_index("timestamp")["funding_rate"].resample("1D").mean().to_frame("funding_rate")
            mu = f["funding_rate"].rolling(60, min_periods=20).mean()
            sd = f["funding_rate"].rolling(60, min_periods=20).std(ddof=0)
            f["funding_z"] = ((f["funding_rate"] - mu) / sd.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
            f.index.name = "day"
            out = out.drop(columns=["funding_z"]).merge(f[["funding_z"]].reset_index(), on="day", how="left")

    out["p_up_fallback"] = np.nan
    z = pd.to_numeric(out["funding_z"], errors="coerce")
    out.loc[z > 1.5, "p_up_fallback"] = 0.60
    out.loc[(z > 0.5) & (z <= 1.5), "p_up_fallback"] = 0.52
    out.loc[z < -1.5, "p_up_fallback"] = 0.40
    out.loc[(z < -0.5) & (z >= -1.5), "p_up_fallback"] = 0.48
    out.loc[z.between(-0.5, 0.5, inclusive="both"), "p_up_fallback"] = 0.50
    out["pup_fallback_used"] = bool(pup_fallback) & out["p_up"].isna() & out["p_up_fallback"].notna()
    out["p_up_eff"] = out["p_up"]
    if pup_fallback:
        out.loc[out["pup_fallback_used"], "p_up_eff"] = out.loc[out["pup_fallback_used"], "p_up_fallback"]

    # Conservative fallback: only synthesize a defensive long signal when the
    # fallback is at least directionally bullish. The fallback is intentionally
    # capped below the original model's strongest signals.
    out["def_proxy_signal_original"] = out["def_proxy_signal"]
    if pup_fallback:
        mask = out["pup_fallback_used"] & (out["def_proxy_signal"] <= 0.0) & (out["p_up_fallback"] > 0.50)
        out.loc[mask, "def_proxy_signal"] = out.loc[mask, "p_up_fallback"]
    return out


def _derisk_multiplier(p_up_eff: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lows: list[int] = []
    mults: list[float] = []
    counter = 0
    for v in pd.to_numeric(p_up_eff, errors="coerce"):
        if pd.isna(v):
            pass
        elif float(v) < 0.35:
            counter += 1
        else:
            counter = 0
        lows.append(counter)
        if counter >= 3:
            mults.append(0.50)
        elif counter == 2:
            mults.append(0.75)
        else:
            mults.append(1.0)
    return np.asarray(lows, dtype=int), np.asarray(mults, dtype=float)


def run_sleeve(
    symbol: str,
    confirm_days: int,
    def_proxy: pd.DataFrame,
    cost_bps: float,
    cost_mode: str,
    derisking: bool,
    start: str,
    end: str,
    out_price_csv: Path,
) -> pd.DataFrame:
    d = _download_binance_daily(symbol=symbol, start=start, end=end)
    if d.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")
    out_price_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_price_csv, index=False)

    d["day"] = d["timestamp"].dt.floor("D")
    d = classify_regime_v2_on_btc(d)

    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema55"] = d["close"].ewm(span=55, adjust=False).mean()
    d["ema144"] = d["close"].ewm(span=144, adjust=False).mean()
    d["stack_aligned"] = (d["ema21"] > d["ema55"]) & (d["ema55"] > d["ema144"])
    conf = max(1, int(confirm_days))
    entry = (d["stack_aligned"].rolling(conf, min_periods=conf).min() == 1).fillna(False)
    exit_ = ((~d["stack_aligned"]).rolling(conf, min_periods=conf).min() == 1).fillna(False)

    rv = d["close"].pct_change().rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    d["vol_scalar"] = (0.50 / rv.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.25, upper=1.0).fillna(0.25)

    pos = np.zeros(len(d), dtype=int)
    active = 0
    for i in range(len(d)):
        if active == 0 and bool(entry.iloc[i]):
            active = 1
        elif active == 1 and bool(exit_.iloc[i]):
            active = 0
        pos[i] = active
    d["off_active"] = pos
    d["off_raw_signal"] = np.where(d["off_active"] == 1, d["vol_scalar"], 0.0)

    off_scale_map = {"BULL": 0.8, "CHOP": 0.4, "BEAR": 0.0}
    def_scale_map = {"BULL": 0.0, "CHOP": 0.4, "BEAR": 1.0}
    d["off_scale"] = d["regime_v2"].map(off_scale_map).fillna(0.0)
    d["off_target"] = d["off_raw_signal"] * d["off_scale"]

    x = d.merge(def_proxy, on="day", how="left")
    x["def_proxy_signal"] = pd.to_numeric(x["def_proxy_signal"], errors="coerce").fillna(0.0)
    for c in ["p_up", "p_up_eff", "funding_z", "pup_fallback_used"]:
        if c not in x.columns:
            x[c] = np.nan if c != "pup_fallback_used" else False
    lows, mults = _derisk_multiplier(x["p_up_eff"])
    x["p_up_low_days"] = lows
    x["derisk_multiplier"] = mults
    if derisking:
        x["off_target_pre_derisk"] = x["off_target"]
        active_off = x["off_active"] == 1
        x.loc[active_off, "off_target"] = x.loc[active_off, "off_target"] * x.loc[active_off, "derisk_multiplier"]
    else:
        x["off_target_pre_derisk"] = x["off_target"]
        x["derisk_multiplier"] = 1.0
    x["def_scale"] = x["regime_v2"].map(def_scale_map).fillna(0.0)
    x["def_target"] = x["def_proxy_signal"] * x["def_scale"]

    x["weight_target"] = (x["off_target"] + x["def_target"]).clip(lower=0.0, upper=1.0)
    x["weight_exec"] = x["weight_target"].shift(1).fillna(0.0)
    x["exec_return"] = x["open"].shift(-1) / x["open"] - 1.0

    prev_w = x["weight_exec"].shift(1).fillna(0.0)
    if str(cost_mode).lower() == "entry_exit":
        # Legacy/optimistic: charge only when crossing flat <-> non-flat.
        crossed = ((x["weight_exec"] > 0).astype(int) != (prev_w > 0).astype(int))
        x["turnover"] = np.where(crossed, (x["weight_exec"] - prev_w).abs(), 0.0)
    else:
        # Realistic/default: charge on any weight change.
        x["turnover"] = (x["weight_exec"] - prev_w).abs()
    x["cost"] = x["turnover"] * (float(cost_bps) / 10000.0)
    x["strategy_return"] = x["weight_exec"] * x["exec_return"].fillna(0.0) - x["cost"]
    x = x.dropna(subset=["exec_return"]).copy()
    x["eq"] = (1.0 + x["strategy_return"]).cumprod()
    x["spot_eq"] = (1.0 + x["exec_return"]).cumprod()
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH+BTC parallel sleeve portfolio backtest (lag-1 realistic)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--eth-symbol", default="ETHUSDC")
    ap.add_argument("--btc-symbol", default="BTCUSDC")
    ap.add_argument("--eth-confirm-days", type=int, default=3)
    ap.add_argument("--btc-confirm-days", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--cost-mode", choices=["entry_exit", "weight_change"], default="weight_change")
    ap.add_argument("--allocation-mode", choices=["fixed", "signal_weighted"], default="fixed")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--derisking", action="store_true")
    ap.add_argument("--pup-fallback", action="store_true")
    ap.add_argument("--include-gold", action="store_true")
    ap.add_argument("--gold-symbol", default="PAXG-USD")
    ap.add_argument("--gold-cap", type=float, default=0.3)
    ap.add_argument("--gold-cost-bps", type=float, default=10.0)
    ap.add_argument("--eth-defensive-csv", default="artifacts/backtest/direction_event_model_v1_flat_defensive_6y_gapfilled.csv")
    ap.add_argument("--eth-perp-csv", default="data/backtest/ETH_perp_features_5m_6y_gapfilled.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/eth_btc_portfolio_summary.csv")
    ap.add_argument("--out-daily-csv", default="artifacts/backtest/eth_btc_portfolio_daily.csv")
    args = ap.parse_args()

    def_proxy = load_eth_defensive_proxy(Path(args.eth_defensive_csv), Path(args.eth_perp_csv), bool(args.pup_fallback))

    eth = run_sleeve(
        symbol=args.eth_symbol,
        confirm_days=int(args.eth_confirm_days),
        def_proxy=def_proxy,
        cost_bps=float(args.cost_bps),
        cost_mode=str(args.cost_mode),
        derisking=bool(args.derisking),
        start=args.start,
        end=args.end,
        out_price_csv=Path("data/eth_daily.csv"),
    )
    btc = run_sleeve(
        symbol=args.btc_symbol,
        confirm_days=int(args.btc_confirm_days),
        def_proxy=def_proxy,
        cost_bps=float(args.cost_bps),
        cost_mode=str(args.cost_mode),
        derisking=bool(args.derisking),
        start=args.start,
        end=args.end,
        out_price_csv=Path("data/btc_daily.csv"),
    )

    keep = [
        "timestamp",
        "day",
        "open",
        "close",
        "regime_v2",
        "off_active",
        "off_target",
        "weight_exec",
        "exec_return",
        "strategy_return",
        "turnover",
        "p_up_eff",
        "p_up_low_days",
        "derisk_multiplier",
        "pup_fallback_used",
    ]
    e = eth[keep].copy().rename(
        columns={
            "open": "eth_open",
            "close": "eth_close",
            "regime_v2": "eth_regime",
            "off_active": "eth_off_active",
            "off_target": "eth_off_target",
            "weight_exec": "eth_weight_exec",
            "exec_return": "eth_spot_return",
            "strategy_return": "eth_strategy_return",
            "turnover": "eth_turnover",
            "p_up_eff": "eth_p_up_eff",
            "p_up_low_days": "eth_p_up_low_days",
            "derisk_multiplier": "eth_derisk_multiplier",
            "pup_fallback_used": "eth_pup_fallback_used",
        }
    )
    b = btc[keep].copy().rename(
        columns={
            "open": "btc_open",
            "close": "btc_close",
            "regime_v2": "btc_regime",
            "off_active": "btc_off_active",
            "off_target": "btc_off_target",
            "weight_exec": "btc_weight_exec",
            "exec_return": "btc_spot_return",
            "strategy_return": "btc_strategy_return",
            "turnover": "btc_turnover",
            "p_up_eff": "btc_p_up_eff",
            "p_up_low_days": "btc_p_up_low_days",
            "derisk_multiplier": "btc_derisk_multiplier",
            "pup_fallback_used": "btc_pup_fallback_used",
        }
    )

    merged = e.merge(b, on=["timestamp", "day"], how="inner").sort_values("timestamp").copy()
    if str(args.allocation_mode) == "signal_weighted":
        score_sum = merged["eth_weight_exec"] + merged["btc_weight_exec"]
        active = score_sum > 0
        gross = np.where(active, float(args.gross_cap), 0.0)
        merged["alloc_eth"] = np.where(active, gross * (merged["eth_weight_exec"] / score_sum), 0.0)
        merged["alloc_btc"] = np.where(active, gross * (merged["btc_weight_exec"] / score_sum), 0.0)
    else:
        merged["alloc_eth"] = 0.5
        merged["alloc_btc"] = 0.5

    merged["gold_symbol_used"] = ""
    merged["gold_weight_exec"] = 0.0
    merged["gold_strategy_return"] = 0.0
    merged["gold_spot_return"] = 0.0
    if bool(args.include_gold):
        gold_px, gold_used = _download_gold_daily(args.start, args.end, args.gold_symbol)
        g = pd.DataFrame({"day": pd.to_datetime(merged["day"], errors="coerce")}).drop_duplicates().sort_values("day")
        g["day"] = g["day"].dt.tz_localize(None)
        gold_px = gold_px.copy()
        gold_px["day"] = pd.to_datetime(gold_px["day"], errors="coerce")
        if getattr(gold_px["day"].dt, "tz", None) is not None:
            gold_px["day"] = gold_px["day"].dt.tz_localize(None)
        g = g.merge(gold_px, on="day", how="left").sort_values("day")
        g["open"] = g["open"].ffill()
        g["close"] = g["close"].ffill()
        g["gold_ema21"] = g["close"].ewm(span=21, adjust=False).mean()
        g["gold_ema55"] = g["close"].ewm(span=55, adjust=False).mean()
        g["gold_ema144"] = g["close"].ewm(span=144, adjust=False).mean()
        g["gold_aligned"] = (g["gold_ema21"] > g["gold_ema55"]) & (g["gold_ema55"] > g["gold_ema144"])
        g["gold_ret_exec"] = g["open"].pct_change().fillna(0.0)

        merged_tmp = merged.copy()
        merged_tmp["day_merge"] = pd.to_datetime(merged_tmp["day"], errors="coerce")
        if getattr(merged_tmp["day_merge"].dt, "tz", None) is not None:
            merged_tmp["day_merge"] = merged_tmp["day_merge"].dt.tz_localize(None)
        gm = merged_tmp.merge(g[["day", "gold_aligned", "gold_ret_exec"]], left_on="day_merge", right_on="day", how="left")
        gm = gm.drop(columns=["day_y", "day_merge"], errors="ignore").rename(columns={"day_x": "day"})
        gm["gold_aligned"] = gm["gold_aligned"].fillna(False)
        gm["gold_ret_exec"] = pd.to_numeric(gm["gold_ret_exec"], errors="coerce").fillna(0.0)
        idle_cap = (1.0 - (gm["alloc_eth"] + gm["alloc_btc"])).clip(lower=0.0)
        both_flat = (gm["eth_off_active"] <= 0) & (gm["btc_off_active"] <= 0)
        bear_gate = gm["eth_regime"].astype(str).eq("BEAR")
        gold_gate = both_flat & bear_gate & gm["gold_aligned"]
        gm["gold_weight_target"] = np.where(gold_gate, np.minimum(idle_cap, float(args.gold_cap)), 0.0)
        gm["gold_weight_exec"] = gm["gold_weight_target"].shift(1).fillna(0.0)
        gprev = gm["gold_weight_exec"].shift(1).fillna(0.0)
        if str(args.cost_mode).lower() == "entry_exit":
            crossed = ((gm["gold_weight_exec"] > 0).astype(int) != (gprev > 0).astype(int))
            gm["gold_turnover"] = np.where(crossed, (gm["gold_weight_exec"] - gprev).abs(), 0.0)
        else:
            gm["gold_turnover"] = (gm["gold_weight_exec"] - gprev).abs()
        gm["gold_cost"] = gm["gold_turnover"] * (float(args.gold_cost_bps) / 10000.0)
        gm["gold_strategy_return"] = gm["gold_weight_exec"] * gm["gold_ret_exec"] - gm["gold_cost"]
        gm["gold_spot_return"] = gm["gold_ret_exec"]
        gm["gold_symbol_used"] = gold_used
        merged = gm

    merged["combined_return"] = (
        merged["alloc_eth"] * merged["eth_strategy_return"]
        + merged["alloc_btc"] * merged["btc_strategy_return"]
        + merged["gold_strategy_return"]
    )
    merged["combined_spot_return"] = 0.5 * merged["eth_spot_return"] + 0.5 * merged["btc_spot_return"]
    merged["combined_eq"] = (1.0 + merged["combined_return"]).cumprod()
    merged["combined_spot_eq"] = (1.0 + merged["combined_spot_return"]).cumprod()
    merged["combined_time_in_market"] = ((merged["eth_weight_exec"] > 0).astype(int) + (merged["btc_weight_exec"] > 0).astype(int)) / 2.0

    m_eth = perf(merged["eth_strategy_return"])
    m_btc = perf(merged["btc_strategy_return"])
    m_comb = perf(merged["combined_return"])
    eth_trades = int(((merged["eth_weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())
    btc_trades = int(((merged["btc_weight_exec"] > 0).astype(int).diff().abs().fillna(0.0) > 0).sum())

    corr_strategy = float(merged["eth_strategy_return"].corr(merged["btc_strategy_return"]))
    corr_spot = float(merged["eth_spot_return"].corr(merged["btc_spot_return"]))

    out_daily = Path(args.out_daily_csv)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_daily, index=False)

    summary = pd.DataFrame(
        [
            {
                "start": str(pd.to_datetime(merged["timestamp"].iloc[0]).date()),
                "end": str(pd.to_datetime(merged["timestamp"].iloc[-1]).date()),
                "eth_confirm_days": int(args.eth_confirm_days),
                "btc_confirm_days": int(args.btc_confirm_days),
                "cost_bps": float(args.cost_bps),
                "cost_mode": str(args.cost_mode),
                "allocation_mode": str(args.allocation_mode),
                "gross_cap": float(args.gross_cap),
                "derisking": bool(args.derisking),
                "pup_fallback": bool(args.pup_fallback),
                "include_gold": bool(args.include_gold),
                "gold_symbol": str(merged["gold_symbol_used"].iloc[-1]) if bool(args.include_gold) else "",
                "gold_cap": float(args.gold_cap),
                "combined_cagr": m_comb["cagr"],
                "combined_sharpe": m_comb["sharpe"],
                "combined_max_dd": m_comb["max_dd"],
                "combined_ann_vol": m_comb["ann_vol"],
                "combined_time_in_market_pct": float(merged["combined_time_in_market"].mean() * 100.0),
                "eth_cagr": m_eth["cagr"],
                "eth_sharpe": m_eth["sharpe"],
                "eth_max_dd": m_eth["max_dd"],
                "eth_trades": eth_trades,
                "btc_cagr": m_btc["cagr"],
                "btc_sharpe": m_btc["sharpe"],
                "btc_max_dd": m_btc["max_dd"],
                "btc_trades": btc_trades,
                "corr_strategy_returns": corr_strategy,
                "corr_spot_returns": corr_spot,
                "gold_time_in_market_pct": float((merged["gold_weight_exec"] > 0).mean() * 100.0),
                "derisk_days": int(((merged["eth_derisk_multiplier"] < 1.0) | (merged["btc_derisk_multiplier"] < 1.0)).sum()),
                "avg_derisk_multiplier_when_fired": float(
                    pd.concat(
                        [
                            merged.loc[merged["eth_derisk_multiplier"] < 1.0, "eth_derisk_multiplier"],
                            merged.loc[merged["btc_derisk_multiplier"] < 1.0, "btc_derisk_multiplier"],
                        ]
                    ).mean()
                )
                if int(((merged["eth_derisk_multiplier"] < 1.0) | (merged["btc_derisk_multiplier"] < 1.0)).sum()) > 0
                else 1.0,
                "pup_fallback_days_replaced": int((merged["eth_pup_fallback_used"] | merged["btc_pup_fallback_used"]).sum()),
            }
        ]
    )

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print("===============================================")
    print("ETH+BTC PORTFOLIO vs STANDALONE")
    print("===============================================")
    print("Metric        ETH only  BTC only  Combined")
    print("-----------------------------------------------")
    print(f"CAGR          {m_eth['cagr']*100:>6.2f}%   {m_btc['cagr']*100:>6.2f}%   {m_comb['cagr']*100:>6.2f}%")
    print(f"Sharpe        {m_eth['sharpe']:>6.3f}   {m_btc['sharpe']:>6.3f}   {m_comb['sharpe']:>6.3f}")
    print(f"MaxDD         {m_eth['max_dd']*100:>6.2f}%  {m_btc['max_dd']*100:>6.2f}%  {m_comb['max_dd']*100:>6.2f}%")
    print("-----------------------------------------------")
    print(f"Combined ann vol:      {m_comb['ann_vol']*100:.2f}%")
    print(f"Combined time in mkt:  {float(merged['combined_time_in_market'].mean()*100.0):.2f}%")
    print(f"ETH trades: {eth_trades} | BTC trades: {btc_trades}")
    print(f"Corr strategy returns: {corr_strategy:.3f}")
    print(f"Corr spot returns:     {corr_spot:.3f}")
    print(f"Cost mode:             {args.cost_mode}")
    print(f"Allocation mode:       {args.allocation_mode} (gross cap={float(args.gross_cap):.2f})")
    print(f"Derisking:             {'ON' if bool(args.derisking) else 'OFF'}")
    print(f"P(up) fallback:        {'ON' if bool(args.pup_fallback) else 'OFF'}")
    print(f"Derisk days:           {int(((merged['eth_derisk_multiplier'] < 1.0) | (merged['btc_derisk_multiplier'] < 1.0)).sum())}")
    print(f"P(up) fallback days:   {int((merged['eth_pup_fallback_used'] | merged['btc_pup_fallback_used']).sum())}")
    if bool(args.include_gold):
        print(f"Gold sleeve:           ON ({str(merged['gold_symbol_used'].iloc[-1])}) cap={float(args.gold_cap):.2f}")
        print(f"Gold time in market:   {float((merged['gold_weight_exec']>0).mean()*100.0):.2f}%")
    else:
        print("Gold sleeve:           OFF")
    print("===============================================")
    print(f"Saved: {args.out_summary_csv}")
    print(f"Saved: {args.out_daily_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
