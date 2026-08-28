from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _walkforward_runner import EMA_CANDIDATES, WINDOWS, optimise_window, run_oos_warm  # noqa: E402

SIMPLIFIED_AVG_OOS_SHARPE = -0.010  # from the earlier (vol-filter + transition-momentum only) walk-forward
PRODUCTION_FIXED_SHARPE = 1.109  # from the fixed-production-config walk-forward (no re-optimisation)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-stack walk-forward: re-optimise EMA per window, but with asymmetric sizing, gold sleeve, and the mean-reversion CHOP overlay all enabled (matching the actual production feature set).")
    ap.add_argument("--work-dir", default="artifacts/backtest/walkforward_fullstack")
    ap.add_argument("--out-summary", default="artifacts/backtest/walkforward_fullstack_summary.csv")
    ap.add_argument("--out-grid", default="artifacts/backtest/walkforward_fullstack_grid.csv")
    args = ap.parse_args()

    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90); print("FULL-STACK WALK-FORWARD VALIDATION"); print("=" * 90)
    print(f"EMA candidate pool ({len(EMA_CANDIDATES)}): {', '.join(EMA_CANDIDATES.keys())}")
    print("Full stack: vol filter + transition momentum + asymmetric sizing + gold (PAXG) sleeve + mean-reversion CHOP overlay")
    print("Note: there is no --meanrev-overlay flag on backtest_eth_btc_portfolio.py -- the overlay only exists")
    print("as post-hoc logic in _crypto_main. Reconstructed identically here (see _apply_meanrev_overlay) so")
    print("'full stack' actually includes it rather than silently running without it.")
    print("Confirm days fixed at production defaults (ETH=3, BTC=5); EMA re-optimised per window")
    print("=" * 90)

    window_rows = []
    all_grid_rows = []
    for w in WINDOWS:
        print(f"\nWindow {w['n']}: Train {w['train_start']} to {w['train_end']}, Test {w['test_start']} to {w['test_end']}")
        best_eth, best_btc, in_sample_sharpe, grid_df = optimise_window(w["train_start"], w["train_end"], work_dir, w["n"], full_stack=True)
        grid_df["window"] = w["n"]
        all_grid_rows.append(grid_df)
        matches_production = best_eth == "50/120/300" and best_btc == "15/40/120"
        print(f"  Best train params: ETH {best_eth}  BTC {best_btc}  (in-sample Sharpe {in_sample_sharpe:.3f})  {'[MATCHES production config]' if matches_production else '[differs from production]'}")

        test_stats = run_oos_warm(w["test_start"], w["test_end"], EMA_CANDIDATES[best_eth], EMA_CANDIDATES[best_btc], work_dir, f"w{w['n']}_ooswarm", full_stack=True)
        oos_sharpe = float(test_stats["sharpe"])
        oos_cagr = float(test_stats["cagr"])
        oos_maxdd = float(test_stats["maxdd"])
        print(f"  Out-of-sample Sharpe: {oos_sharpe:.3f}  CAGR: {oos_cagr*100:.1f}%  MaxDD: {oos_maxdd*100:.1f}%")

        window_rows.append({
            "window": w["n"], "train_start": w["train_start"], "train_end": w["train_end"],
            "test_start": w["test_start"], "test_end": w["test_end"],
            "best_eth_ema": best_eth, "best_btc_ema": best_btc, "matches_production_config": matches_production,
            "in_sample_sharpe": in_sample_sharpe, "oos_sharpe": oos_sharpe, "oos_cagr": oos_cagr, "oos_maxdd": oos_maxdd,
        })

    wf_df = pd.DataFrame(window_rows)
    wf_df.to_csv(args.out_summary, index=False)
    pd.concat(all_grid_rows, ignore_index=True).to_csv(args.out_grid, index=False)

    print("\n" + "=" * 90); print("SUMMARY"); print("=" * 90)
    for _, r in wf_df.iterrows():
        tag = "MATCHES production" if r["matches_production_config"] else "differs from production"
        print(f"Window {int(r['window'])}: Best ETH {r['best_eth_ema']}  BTC {r['best_btc_ema']}  ({tag})")
        print(f"  In-sample Sharpe: {r['in_sample_sharpe']:.3f}   Out-of-sample Sharpe: {r['oos_sharpe']:.3f}")

    avg_oos = float(wf_df["oos_sharpe"].mean())
    n_positive = int((wf_df["oos_sharpe"] > 0).sum())
    n_matches = int(wf_df["matches_production_config"].sum())
    eth_choices = sorted(set(wf_df["best_eth_ema"]))
    btc_choices = sorted(set(wf_df["best_btc_ema"]))

    print(f"\nEMA chosen each window -- ETH: {list(wf_df['best_eth_ema'])}")
    print(f"EMA chosen each window -- BTC: {list(wf_df['best_btc_ema'])}")
    print(f"Windows matching production config (ETH 50/120/300, BTC 15/40/120): {n_matches}/4")
    print(f"\nAverage OOS Sharpe: {avg_oos:.3f}")
    print(f"Positive OOS windows: {n_positive}/4")

    print("\nComparison:")
    print(f"  Simplified walk-forward (no asym/gold/overlay), avg OOS Sharpe: {SIMPLIFIED_AVG_OOS_SHARPE:.3f}")
    print(f"  Full-stack walk-forward (this run),              avg OOS Sharpe: {avg_oos:.3f}")
    print(f"  Production fixed config (no re-optimisation),     avg OOS Sharpe: {PRODUCTION_FIXED_SHARPE:.3f}")

    if n_matches == 4:
        verdict_line = "Full-stack walk-forward CONSISTENTLY chose the current production config (50/120/300 / 15/40/120) -- current config CONFIRMED."
    elif n_matches == 0:
        verdict_line = "Full-stack walk-forward NEVER chose the current production config -- it consistently preferred different EMAs. Worth investigating a switch."
    else:
        verdict_line = f"Full-stack walk-forward chose the current production config in {n_matches}/4 windows -- mixed signal, not a clean confirmation either way."
    print(f"\n{verdict_line}")
    print(f"Are the full-stack optimal params the same as what we're running? {'YES' if n_matches == 4 else 'NO' if n_matches == 0 else 'PARTIALLY'}")

    print(f"\nSaved: {args.out_summary}")
    print(f"Saved: {args.out_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
