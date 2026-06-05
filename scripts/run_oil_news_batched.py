from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _year_bounds(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = pd.read_csv(path)
    except Exception:
        return False
    return bool(len(d) and ("status" not in d.columns))


def main() -> int:
    p = argparse.ArgumentParser(description="Run oil news backtest in yearly batches with resume checkpoints.")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--ticker", default="USO")
    p.add_argument("--summary-out", default="artifacts/backtest/oil_news_summary_batched_yearly.csv")
    p.add_argument("--events-merged-out", default="data/gdelt_oil_events_batched.csv")
    args = p.parse_args()

    years = list(range(int(args.start_year), int(args.end_year) + 1))
    scripts_dir = Path(__file__).resolve().parent
    backtest_script = scripts_dir / "backtest_oil_news.py"

    summaries: list[pd.DataFrame] = []
    event_files: list[Path] = []

    for y in years:
        s, e = _year_bounds(y)
        ev = Path(f"data/gdelt_oil_events_{y}.csv")
        sm = Path(f"artifacts/backtest/oil_news_summary_{y}.csv")
        dy = Path(f"artifacts/backtest/oil_news_daily_{y}.csv")

        print(f"[BATCH] Year {y}: start")
        if _is_done(sm):
            print(f"[BATCH] Year {y}: already complete, skipping")
        else:
            cmd = [
                sys.executable,
                str(backtest_script),
                "--start",
                s,
                "--end",
                e,
                "--ticker",
                str(args.ticker),
                "--cost-bps",
                str(float(args.cost_bps)),
                "--workers",
                str(int(args.workers)),
                "--checkpoint-every",
                str(int(args.checkpoint_every)),
                "--events-out",
                str(ev),
                "--summary-out",
                str(sm),
                "--daily-out",
                str(dy),
            ]
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                print(f"[BATCH] Year {y}: FAILED rc={proc.returncode}")
                return proc.returncode
            print(f"[BATCH] Year {y}: complete")

        if sm.exists():
            try:
                d = pd.read_csv(sm)
                if len(d):
                    d["year"] = y
                    summaries.append(d)
            except Exception:
                pass
        if ev.exists():
            event_files.append(ev)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"[BATCH] Wrote yearly summary: {out_path}")

    if event_files:
        merged_events: list[pd.DataFrame] = []
        for f in event_files:
            try:
                d = pd.read_csv(f, low_memory=False)
                if len(d):
                    merged_events.append(d)
            except Exception:
                continue
        if merged_events:
            ed = pd.concat(merged_events, ignore_index=True)
            # Deduplicate exact rows if overlap exists.
            ed = ed.drop_duplicates()
            out_ev = Path(args.events_merged_out)
            out_ev.parent.mkdir(parents=True, exist_ok=True)
            ed.to_csv(out_ev, index=False)
            print(f"[BATCH] Wrote merged events: {out_ev} rows={len(ed)}")

    print("[BATCH] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
