from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_fear_greed import _recompute_main_return, _segment_stats
from backtest_forex_optimised import _stats


COINMETRICS_URLS = {"ETH": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv", "BTC": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"}


def _download_onchain(asset: str, out_flow_csv: Path, out_addr_csv: Path | None, refresh: bool) -> pd.DataFrame:
    """CoinMetrics community data (github.com/coinmetrics/data) is public, no API key
    required. Glassnode's endpoints in the task spec returned 401 without a paid key
    (verified directly); CoinMetrics' free CSVs include exchange flow AND active
    addresses for both ETH and BTC, so that's used instead, per the task's own
    fallback instruction."""
    need_addr = out_addr_csv is not None
    if out_flow_csv.exists() and (not need_addr or out_addr_csv.exists()) and not refresh:
        flow = pd.read_csv(out_flow_csv, parse_dates=["date"])
        flow["date"] = flow["date"].dt.tz_localize("UTC") if flow["date"].dt.tz is None else flow["date"]
        if need_addr:
            addr = pd.read_csv(out_addr_csv, parse_dates=["date"])
            addr["date"] = addr["date"].dt.tz_localize("UTC") if addr["date"].dt.tz is None else addr["date"]
            return flow.merge(addr, on="date", how="outer").sort_values("date").reset_index(drop=True)
        return flow

    cols = ["time", "FlowInExUSD", "FlowOutExUSD"] + (["AdrActCnt"] if need_addr else [])
    raw = pd.read_csv(COINMETRICS_URLS[asset], usecols=cols)
    raw["date"] = pd.to_datetime(raw["time"], utc=True).dt.floor("D")

    flow_df = raw[["date", "FlowInExUSD", "FlowOutExUSD"]].copy()
    flow_df["net_flow_usd"] = flow_df["FlowInExUSD"] - flow_df["FlowOutExUSD"]
    out_flow_csv.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_csv(out_flow_csv, index=False)

    if need_addr:
        addr_df = raw[["date", "AdrActCnt"]].copy()
        out_addr_csv.parent.mkdir(parents=True, exist_ok=True)
        addr_df.to_csv(out_addr_csv, index=False)
        return flow_df.merge(addr_df, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return flow_df


def _zscore(s: pd.Series, window: int = 30) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def _apply_onchain_gate(base: pd.DataFrame, eth_ok: pd.Series, btc_ok: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = base.copy()
    blocked_rows = []
    for leg, off_col, alloc_col, ret_col, ok in [("eth", "eth_off_active", "alloc_eth", "eth_strategy_return", eth_ok), ("btc", "btc_off_active", "alloc_btc", "btc_strategy_return", btc_ok)]:
        active = x[off_col].fillna(0).to_numpy().astype(bool)
        starts, ends, trade_returns = _segment_stats(active, x[ret_col].fillna(0.0).to_numpy())
        alloc = x[alloc_col].to_numpy().copy()
        ok_arr = ok.fillna(False).to_numpy()
        for s, e, r in zip(starts, ends, trade_returns):
            if not ok_arr[s]:
                alloc[s:e + 1] = 0.0
                blocked_rows.append({"leg": leg, "entry_date": pd.Timestamp(x["day"].iloc[s]).date().isoformat(), "exit_date": pd.Timestamp(x["day"].iloc[e]).date().isoformat(), "days_held": int(e - s + 1), "would_be_return_pct": float(r * 100.0)})
        x[alloc_col] = alloc
    return x, pd.DataFrame(blocked_rows)


def _apply_onchain_conviction(base: pd.DataFrame, eth_flow_z: pd.Series, btc_flow_z: pd.Series, gross_cap: float) -> pd.DataFrame:
    x = base.copy()
    def mult(z: pd.Series) -> pd.Series:
        return pd.Series(np.select([z < -1.5, z > 1.5], [1.2, 0.7], default=1.0), index=z.index)
    eth_mult, btc_mult = mult(eth_flow_z), mult(btc_flow_z)
    eth_active, btc_active = x["alloc_eth"] > 0, x["alloc_btc"] > 0
    x.loc[eth_active, "alloc_eth"] = x.loc[eth_active, "alloc_eth"] * eth_mult.loc[eth_active]
    x.loc[btc_active, "alloc_btc"] = x.loc[btc_active, "alloc_btc"] * btc_mult.loc[btc_active]
    gross = x["alloc_eth"] + x["alloc_btc"]
    over = gross > gross_cap
    x.loc[over, "alloc_eth"] = x.loc[over, "alloc_eth"] * gross_cap / gross.loc[over]
    x.loc[over, "alloc_btc"] = x.loc[over, "alloc_btc"] * gross_cap / gross.loc[over]
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="On-chain (CoinMetrics free community data) signal tests: entry filter, conviction modifier, standalone predictive check.")
    ap.add_argument("--work-dir", default="artifacts/backtest/forex_optimised")
    ap.add_argument("--data-dir", default="data/onchain")
    ap.add_argument("--out-summary", default="artifacts/backtest/onchain_summary.csv")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    eth_oc = _download_onchain("ETH", data_dir / "ETH_exchange_flow.csv", data_dir / "ETH_active_addresses.csv", bool(args.refresh_data))
    btc_oc = _download_onchain("BTC", data_dir / "BTC_exchange_flow.csv", None, bool(args.refresh_data))
    eth_oc["flow_z"] = _zscore(eth_oc["net_flow_usd"])
    eth_oc["addr_z"] = _zscore(eth_oc["AdrActCnt"])
    btc_oc["flow_z"] = _zscore(btc_oc["net_flow_usd"])

    base = pd.read_csv(Path(args.work_dir) / "validated_crypto_daily.csv", low_memory=False)
    base["day"] = pd.to_datetime(base["day"], utc=True, errors="coerce").dt.floor("D")
    base = base.merge(eth_oc[["date", "flow_z", "addr_z", "net_flow_usd"]].rename(columns={"date": "day", "flow_z": "eth_flow_z", "addr_z": "eth_addr_z", "net_flow_usd": "eth_net_flow"}), on="day", how="left")
    base = base.merge(btc_oc[["date", "flow_z"]].rename(columns={"date": "day", "flow_z": "btc_flow_z"}), on="day", how="left")
    base = base.dropna(subset=["eth_flow_z", "eth_addr_z", "btc_flow_z"]).reset_index(drop=True)
    print(f"Data coverage: {len(base)} days ({base['day'].min().date()} to {base['day'].max().date()})")
    print("(ETH flow+addr and BTC flow are used per-leg; BTC active addresses weren't in the requested data list, so BTC's gate uses flow_z alone.)")

    main_return_baseline = _recompute_main_return(base, args.gross_cap, args.cost_bps)
    main_stats = _stats(main_return_baseline)

    # ---- Test 2C: standalone predictive check (do this first -- it's diagnostic, not a backtest) ----
    eth_fwd5 = base["eth_close"].shift(-5) / base["eth_close"] - 1.0
    signal_days = base["eth_flow_z"] < -1.0
    base_rate = float((eth_fwd5.dropna() > 0).mean())
    cond_rate = float((eth_fwd5[signal_days].dropna() > 0).mean()) if signal_days.sum() else np.nan
    n_signal_days = int(signal_days.sum())
    avg_fwd_ret_signal = float(eth_fwd5[signal_days].mean() * 100) if signal_days.sum() else np.nan
    avg_fwd_ret_base = float(eth_fwd5.mean() * 100)
    print("=" * 80); print("TEST 2C: STANDALONE PREDICTIVE CHECK (flow_z < -1 -> ETH rises next 5d?)"); print("=" * 80)
    print(f"Signal days (flow_z < -1): {n_signal_days} / {len(base)}")
    print(f"P(ETH up in 5d | flow_z<-1): {cond_rate*100:.1f}%   avg fwd 5d return: {avg_fwd_ret_signal:.2f}%")
    print(f"P(ETH up in 5d | unconditional): {base_rate*100:.1f}%   avg fwd 5d return: {avg_fwd_ret_base:.2f}%")
    print(f"Edge over base rate: {(cond_rate-base_rate)*100:+.1f}pp")

    # ---- Test 2A: entry filter ----
    eth_ok = (base["eth_flow_z"] < 0) & (base["eth_addr_z"] > -0.5)
    btc_ok = base["btc_flow_z"] < 0
    base_a, blocked_a = _apply_onchain_gate(base, eth_ok, btc_ok)
    main_return_a = _recompute_main_return(base_a, args.gross_cap, args.cost_bps)
    stats_a = _stats(main_return_a)
    print("=" * 80); print("TEST 2A: ON-CHAIN ENTRY FILTER"); print("=" * 80)
    print(f"Main alone:         Sharpe {main_stats['sharpe']:.3f}")
    print(f"Main + flow filter: Sharpe {stats_a['sharpe']:.3f}")
    total_entries = int((base["eth_off_active"].diff() == 1).sum() + (base["btc_off_active"].diff() == 1).sum())
    print(f"Entries blocked: {len(blocked_a)} / {total_entries} total")
    if len(blocked_a):
        print(blocked_a.to_string(index=False))

    # ---- Test 2B: conviction modifier ----
    base_b = _apply_onchain_conviction(base, base["eth_flow_z"], base["btc_flow_z"], args.gross_cap)
    main_return_b = _recompute_main_return(base_b, args.gross_cap, args.cost_bps)
    stats_b = _stats(main_return_b)
    print("=" * 80); print("TEST 2B: ON-CHAIN CONVICTION MODIFIER"); print("=" * 80)
    print(f"Main alone:      Sharpe {main_stats['sharpe']:.3f}")
    print(f"Main + flow mod: Sharpe {stats_b['sharpe']:.3f}")
    print(f"Improvement: {stats_b['sharpe']-main_stats['sharpe']:+.3f}")

    print("=" * 80)
    candidates = {"A": stats_a["sharpe"] - main_stats["sharpe"], "B": stats_b["sharpe"] - main_stats["sharpe"], "C": (cond_rate - base_rate)}
    best_app = max(candidates, key=lambda k: candidates[k])
    if candidates[best_app] <= 0:
        best_app = "NONE"
    decision = "IMPLEMENT" if best_app in {"A", "B"} and candidates.get(best_app, 0) > 0.1 else "RESEARCH" if best_app != "NONE" else "SKIP"
    print(f"Best on-chain application: {best_app}")
    print(f"Decision: {decision}")

    pd.DataFrame([
        {"test": "C_predictive", "signal_days": n_signal_days, "predictive_accuracy": cond_rate, "base_rate": base_rate, "edge_pp": (cond_rate - base_rate) * 100},
        {"test": "A_entry_filter", "main_only_sharpe": main_stats["sharpe"], "with_filter_sharpe": stats_a["sharpe"], "improvement": stats_a["sharpe"] - main_stats["sharpe"], "entries_blocked": len(blocked_a), "entries_total": total_entries},
        {"test": "B_conviction_modifier", "main_only_sharpe": main_stats["sharpe"], "with_mod_sharpe": stats_b["sharpe"], "improvement": stats_b["sharpe"] - main_stats["sharpe"]},
    ]).to_csv(args.out_summary, index=False)
    print(f"\nSaved: {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
