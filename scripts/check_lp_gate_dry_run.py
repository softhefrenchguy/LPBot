from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns  # noqa: E402


def _segments(on: np.ndarray) -> list[tuple[int, int]]:
    n = len(on)
    if not on.any():
        return []
    diff = np.diff(on.astype(np.int8), prepend=np.int8(0))
    starts = np.flatnonzero(diff == 1)
    end_candidates = np.flatnonzero(diff == -1) - 1
    ends = np.append(end_candidates, n - 1) if on[-1] else end_candidates
    return list(zip(starts.tolist(), ends.tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-run: does lp_overlay_v1's existing _build_lp_on gate, driven by the REAL production eth_off_active/btc_off_active signals, produce sensible on/off timing?")
    ap.add_argument("--daily", default="artifacts/backtest/paper_window_fresh/validated_crypto_daily.csv")
    ap.add_argument("--lp-vol-on-env", type=float, default=0.25, help="The literal LP_VOL_ON value stored in .env")
    ap.add_argument("--lp-vol-on-calibrated", type=float, default=None, help="Defaults to the median of rolling_vol_20d if not given")
    ap.add_argument("--lp-min-on-bars", type=int, default=3)
    ap.add_argument("--lp-cooldown-bars", type=int, default=5)
    ap.add_argument("--out", default="artifacts/backtest/lp_gate_dry_run.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.daily, low_memory=False)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    eth_active = d["eth_off_active"].fillna(0).astype(bool)
    btc_active = d["btc_off_active"].fillna(0).astype(bool)
    gate = (eth_active | btc_active)  # True = a real trend is confirmed in either sleeve -> LP should be blocked

    lp_input = pd.DataFrame({
        "timestamp": d["day"],
        "close": d["eth_close"],
        "weight": 1.0,  # constant: isolates the on/off decision to gate + vol, per the dry-run's purpose
        "sigma_ann_smooth": pd.to_numeric(d["rolling_vol_20d"], errors="coerce"),
        "gate": gate,
    })

    calibrated_threshold = args.lp_vol_on_calibrated if args.lp_vol_on_calibrated is not None else float(lp_input["sigma_ann_smooth"].median())

    print("=" * 110)
    print("LP GATE DRY-RUN: _build_lp_on driven by real production eth_off_active/btc_off_active")
    print("=" * 110)
    print(f"gate=True on {int(gate.sum())}/{len(gate)} days ({gate.mean()*100:.1f}%) -- days a real trend is confirmed in either sleeve")
    print(f"rolling_vol_20d: median={lp_input['sigma_ann_smooth'].median():.3f}  25th pct={lp_input['sigma_ann_smooth'].quantile(0.25):.3f}  75th pct={lp_input['sigma_ann_smooth'].quantile(0.75):.3f}")
    print()

    results = {}
    for label, vol_on in [(".env literal (LP_VOL_ON=0.25)", args.lp_vol_on_env), (f"calibrated (median vol, {calibrated_threshold:.2f})", calibrated_threshold)]:
        out = compute_lp_overlay_returns(
            lp_input, bar_minutes=1440, fee_rate_ann=0.15, il_k=0.5,
            lp_vol_on=vol_on, lp_scale=1.0, lp_weight_max=1.0,
            lp_min_on_bars=args.lp_min_on_bars, lp_cooldown_bars=args.lp_cooldown_bars,
        )
        out["day"] = d["day"].values
        results[label] = out
        lp_on_pct = float(out["lp_on"].mean() * 100)
        print(f"{label}: LP on {lp_on_pct:.1f}% of days")

    print()
    calibrated = results[f"calibrated (median vol, {calibrated_threshold:.2f})"]
    lp_on = calibrated["lp_on"].to_numpy()

    # Check: does LP turn off at/near the same time a real trend gets confirmed?
    entry_days_eth = np.flatnonzero(eth_active.to_numpy() & ~eth_active.shift(1, fill_value=False).to_numpy())
    entry_days_btc = np.flatnonzero(btc_active.to_numpy() & ~btc_active.shift(1, fill_value=False).to_numpy())
    all_entries = sorted(set(entry_days_eth.tolist()) | set(entry_days_btc.tolist()))

    print(f"Checking {len(all_entries)} real trend-entry days: was LP already off (or did it turn off within 3 days)?")
    n_correctly_off = 0
    for idx in all_entries:
        window = lp_on[max(0, idx - 1): idx + 4]
        was_off = not bool(window.any()) if len(window) else True
        n_correctly_off += int(was_off)
    print(f"LP was off at/around {n_correctly_off}/{len(all_entries)} real trend entries (checked +/- a few days)")
    print()

    print("LP-on segments under the calibrated threshold (first 15):")
    segs = _segments(lp_on)
    for s, e in segs[:15]:
        length = e - s + 1
        start_d, end_d = d["day"].iloc[s].date(), d["day"].iloc[e].date()
        gate_during = gate.iloc[s:e + 1]
        print(f"  {start_d} -> {end_d} ({length}d), gate True during this window: {int(gate_during.sum())}/{length} days")
    print(f"Total LP-on segments: {len(segs)}")

    calibrated.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
