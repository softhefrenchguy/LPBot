from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _walkforward_runner import EMA_CANDIDATES, WINDOWS, optimise_window, run_backtest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="True walk-forward validation: re-optimise EMA per rolling training window, test out-of-sample.")
    ap.add_argument("--work-dir", default="artifacts/backtest/walkforward")
    ap.add_argument("--out-summary", default="artifacts/backtest/walkforward_summary.csv")
    ap.add_argument("--out-grid", default="artifacts/backtest/walkforward_grid.csv")
    args = ap.parse_args()

    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90); print("WALK-FORWARD VALIDATION"); print("=" * 90)
    print(f"EMA candidate pool ({len(EMA_CANDIDATES)}): {', '.join(EMA_CANDIDATES.keys())}")
    print("Confirm days held fixed at production defaults (ETH=3, BTC=5); EMA re-optimised per window")
    print("Search: coordinate descent -- optimise ETH EMA holding BTC fixed, then BTC EMA holding ETH fixed")
    print("=" * 90)

    window_rows = []
    all_grid_rows = []
    for w in WINDOWS:
        print(f"\nWindow {w['n']}: Train {w['train_start']} to {w['train_end']}, Test {w['test_start']} to {w['test_end']}")
        best_eth, best_btc, in_sample_sharpe, grid_df = optimise_window(w["train_start"], w["train_end"], work_dir, w["n"])
        grid_df["window"] = w["n"]
        all_grid_rows.append(grid_df)
        print(f"  Best train params: ETH {best_eth}  BTC {best_btc}  (in-sample Sharpe {in_sample_sharpe:.3f})")

        test_row = run_backtest(w["test_start"], w["test_end"], EMA_CANDIDATES[best_eth], EMA_CANDIDATES[best_btc], work_dir, f"w{w['n']}_oos")
        oos_sharpe = float(test_row["combined_sharpe"])
        oos_cagr = float(test_row["combined_cagr"])
        oos_maxdd = float(test_row["combined_max_dd"])
        print(f"  Out-of-sample Sharpe: {oos_sharpe:.3f}  CAGR: {oos_cagr*100:.1f}%  MaxDD: {oos_maxdd*100:.1f}%")

        window_rows.append({
            "window": w["n"], "train_start": w["train_start"], "train_end": w["train_end"],
            "test_start": w["test_start"], "test_end": w["test_end"],
            "best_eth_ema": best_eth, "best_btc_ema": best_btc,
            "in_sample_sharpe": in_sample_sharpe, "oos_sharpe": oos_sharpe, "oos_cagr": oos_cagr, "oos_maxdd": oos_maxdd,
        })

    wf_df = pd.DataFrame(window_rows)
    wf_df.to_csv(args.out_summary, index=False)
    pd.concat(all_grid_rows, ignore_index=True).to_csv(args.out_grid, index=False)

    print("\n" + "=" * 90); print("SUMMARY"); print("=" * 90)
    for _, r in wf_df.iterrows():
        print(f"Window {int(r['window'])}: Train {r['train_start'][:4]}-{r['train_end'][:4]}, Test {r['test_start'][:4]}")
        print(f"  Best train params: ETH {r['best_eth_ema']}  BTC {r['best_btc_ema']}")
        print(f"  In-sample Sharpe: {r['in_sample_sharpe']:.3f}   Out-of-sample Sharpe: {r['oos_sharpe']:.3f}")

    avg_oos = float(wf_df["oos_sharpe"].mean())
    n_positive = int((wf_df["oos_sharpe"] > 0).sum())
    eth_choices = sorted(set(wf_df["best_eth_ema"]))
    btc_choices = sorted(set(wf_df["best_btc_ema"]))
    same_eth = len(eth_choices) == 1
    same_btc = len(btc_choices) == 1
    print("\nParameter stability:")
    print(f"  ETH EMA chosen across windows: {eth_choices}")
    print(f"  BTC EMA chosen across windows: {btc_choices}")
    print(f"Average OOS Sharpe: {avg_oos:.3f}")
    print(f"Positive OOS windows: {n_positive}/4")

    stable_params = same_eth and same_btc
    robust_performance = n_positive >= 3 and avg_oos > 0.5
    verdict = "ROBUST" if (stable_params and robust_performance) else "FRAGILE" if (not stable_params and not robust_performance) else "MIXED"
    print(f"\nFinal verdict: strategy is {verdict} based on walk-forward evidence")
    print(f"  Parameter stability: {'HIGH -- same/near-same params each window' if stable_params else 'LOW -- params vary across windows'}")
    print(f"  OOS performance: {'consistently positive' if robust_performance else 'inconsistent'}")

    print(f"\nSaved: {args.out_summary}")
    print(f"Saved: {args.out_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
