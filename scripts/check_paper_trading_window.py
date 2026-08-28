from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_mode_a_validation import SHRINK_PANIC
from backtest_continuous_regime_integrations import _apply_state_conviction
from continuous_regime_score import _compute_states


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity/curiosity check: how would the continuous regime score and Mode A have read the live paper-trading window? Not a validation test -- 5 months proves nothing statistically.")
    ap.add_argument("--fresh-daily", default="artifacts/backtest/paper_window_fresh/validated_crypto_daily.csv")
    ap.add_argument("--window-start", default="2026-03-21")
    ap.add_argument("--window-end", default="2026-08-27")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--out", default="artifacts/backtest/paper_trading_window_check.csv")
    args = ap.parse_args()

    raw = pd.read_csv(args.fresh_daily, low_memory=False)
    raw["day"] = pd.to_datetime(raw["day"], utc=True, errors="coerce").dt.floor("D")
    raw = raw.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    for col in ["alloc_eth", "alloc_btc", "eth_close", "btc_close"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # continuous_regime_score's _compute_states just needs a "date"/"btc_close" (+ optional
    # "eth_close") frame -- reuse it directly on the fresh live data instead of re-running
    # the standalone script, so the state definitions are byte-identical to what's already
    # been validated, just fed 2019-2026 history (plenty of warm-up for the slowest primitive,
    # the 365-day vol percentile) instead of the full 2012+ history.
    scored = _compute_states(raw.rename(columns={"day": "date"})[["date", "btc_close", "eth_close"]])
    d = raw.merge(scored[["date", "state", "regime_score"]], left_on="day", right_on="date", how="left")

    window = d[(d["day"] >= pd.to_datetime(args.window_start, utc=True)) & (d["day"] <= pd.to_datetime(args.window_end, utc=True))].reset_index(drop=True)
    if window.empty:
        raise SystemExit(f"No rows in fresh data for {args.window_start}..{args.window_end} -- check --fresh-daily covers this range.")

    # Mode A applied to this window's actual alloc_eth/alloc_btc (whatever they were).
    modeA_window = _apply_state_conviction(window, SHRINK_PANIC, args.gross_cap)

    window_out = pd.DataFrame({
        "day": window["day"].dt.date.astype(str),
        "state": window["state"],
        "regime_score": window["regime_score"],
        "eth_regime_ema": window["eth_regime"],
        "btc_regime_ema": window["btc_regime"],
        "actual_alloc_eth": window["alloc_eth"],
        "actual_alloc_btc": window["alloc_btc"],
        "mode_a_alloc_eth": modeA_window["alloc_eth"],
        "mode_a_alloc_btc": modeA_window["alloc_btc"],
    })
    window_out["mode_a_would_change_sizing"] = (
        (window_out["mode_a_alloc_eth"] - window_out["actual_alloc_eth"]).abs() > 1e-9
    ) | ((window_out["mode_a_alloc_btc"] - window_out["actual_alloc_btc"]).abs() > 1e-9)

    print("=" * 110)
    print(f"PAPER-TRADING WINDOW CHECK: {args.window_start} to {args.window_end}")
    print("Not a validation test -- 5 months proves nothing statistically. Sanity/curiosity check only.")
    print("=" * 110)
    print(window_out.to_string(index=False))
    print()

    n_days = len(window_out)
    was_flat = bool((window_out["actual_alloc_eth"].abs() < 1e-9).all() and (window_out["actual_alloc_btc"].abs() < 1e-9).all())
    n_would_change = int(window_out["mode_a_would_change_sizing"].sum())
    state_counts = window_out["state"].value_counts()
    ema_regime_combo = (window_out["eth_regime_ema"].astype(str) + "/" + window_out["btc_regime_ema"].astype(str)).value_counts()

    print(f"Days in window: {n_days}")
    print(f"Bot was flat (alloc_eth == alloc_btc == 0) every day: {was_flat}")
    print(f"Days Mode A would have changed sizing vs actual: {n_would_change}")
    if was_flat and n_would_change == 0:
        print("-> Mode A would NOT have changed anything during this window. Sizing multipliers only matter when there's")
        print("   a position to size, and the bot held zero position the entire time -- 0.5x or 1.2x of zero is still zero.")
    elif n_would_change > 0:
        print("-> Mode A WOULD have changed sizing on some days -- see mode_a_would_change_sizing column for which ones.")
    print()

    print("Continuous regime score state distribution during the window:")
    print(state_counts.to_string())
    print()
    print("EMA-based regime (eth/btc) distribution during the window:")
    print(ema_regime_combo.to_string())
    print()

    non_chop_states = window_out[~window_out["state"].isin(["weakening"])]
    ema_all_chop = bool((window_out["eth_regime_ema"].astype(str) == "CHOP").all() and (window_out["btc_regime_ema"].astype(str) == "CHOP").all())
    print(f"EMA regime was CHOP for both ETH and BTC every day: {ema_all_chop}")
    if not state_counts.reindex(["risk_on", "risk_off", "panic"], fill_value=0).eq(0).all():
        print("Cross-check: the continuous score registered risk_on/risk_off/panic days during a period the EMA-based")
        print("classifier called CHOP the whole time -- i.e. the two signals read this period differently. Specific days:")
        print(non_chop_states[["day", "state", "regime_score", "eth_regime_ema", "btc_regime_ema"]].to_string(index=False))
    else:
        print("Cross-check: the continuous score stayed in 'weakening' (its closest analogue to CHOP) throughout too --")
        print("no meaningful divergence from the EMA-based classifier for this specific window.")

    window_out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
