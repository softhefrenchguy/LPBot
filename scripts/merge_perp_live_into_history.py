from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, low_memory=False)
    if "timestamp" not in d.columns:
        return pd.DataFrame()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    return d


def main() -> int:
    p = argparse.ArgumentParser(description="Merge live perp features into long history CSV.")
    p.add_argument("--history-csv", default="data/backtest/ETH_perp_features_5m_6y.csv")
    p.add_argument("--live-csv", default="data/backtest/ETH_perp_features_5m_live.csv")
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    hist = _load(Path(args.history_csv))
    live = _load(Path(args.live_csv))
    if hist.empty and live.empty:
        raise SystemExit("both inputs empty/missing")

    if hist.empty:
        out = live
    elif live.empty:
        out = hist
    else:
        common = [c for c in hist.columns if c in live.columns]
        if "timestamp" not in common:
            common = ["timestamp"] + common
        out = pd.concat([hist[common], live[common]], ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    target = Path(args.out_csv) if args.out_csv else Path(args.history_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"wrote {target} rows={len(out)} first={out['timestamp'].iloc[0]} last={out['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
