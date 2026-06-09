from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "regime_v2", "price", "recorded_at", "source"])
    snap = pd.read_csv(path, low_memory=False)
    if snap.empty:
        return pd.DataFrame(columns=["date", "regime_v2", "price", "recorded_at", "source"])
    if "date" in snap.columns:
        snap["date"] = pd.to_datetime(snap["date"], utc=True, errors="coerce").dt.floor("D")
    else:
        snap["date"] = pd.NaT
    return snap


def _append_if_new(snap: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    d = pd.to_datetime(row["date"], utc=True, errors="coerce")
    if pd.isna(d):
        return snap
    if (snap["date"] == d.floor("D")).any():
        return snap
    out = pd.concat([snap, pd.DataFrame([row])], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.floor("D")
    return out.sort_values("date").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Create immutable daily regime snapshots (point-in-time labels).")
    p.add_argument("--classifier-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    p.add_argument("--snapshot-csv", default="artifacts/paper_trade/regime_snapshot_immutable.csv")
    p.add_argument("--regime-col", default="regime_v2")
    p.add_argument("--checks-log-csv", default="artifacts/paper_trade/daily_checks_log.csv")
    p.add_argument("--backfill-from-checks", action="store_true")
    p.add_argument("--today-only", action="store_true")
    args = p.parse_args()

    classifier_path = Path(args.classifier_csv)
    snapshot_path = Path(args.snapshot_csv)
    checks_log_path = Path(args.checks_log_csv)

    if not classifier_path.exists():
        raise FileNotFoundError(f"classifier csv missing: {classifier_path}")

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snap = _load_snapshot(snapshot_path)

    # Optional backfill from actual live daily checks (source of truth for prior days)
    if args.backfill_from_checks and checks_log_path.exists():
        checks = pd.read_csv(checks_log_path, low_memory=False)
        if not checks.empty and {"date", "regime"}.issubset(set(checks.columns)):
            checks["date"] = pd.to_datetime(checks["date"], utc=True, errors="coerce").dt.floor("D")
            price_col = _pick_col(checks, ["eth_price", "close", "spot_close"])
            rec_col = _pick_col(checks, ["timestamp_utc", "recorded_at"])
            for _, r in checks.sort_values("date").iterrows():
                d = r["date"]
                if pd.isna(d):
                    continue
                row = {
                    "date": d,
                    "regime_v2": str(r["regime"]),
                    "price": float(r[price_col]) if price_col and pd.notna(r[price_col]) else float("nan"),
                    "recorded_at": str(r[rec_col]) if rec_col and pd.notna(r[rec_col]) else "",
                    "source": "daily_checks_log",
                }
                snap = _append_if_new(snap, row)

    clf = pd.read_csv(classifier_path, low_memory=False)
    if clf.empty:
        snap.to_csv(snapshot_path, index=False)
        print(f"wrote {snapshot_path} (empty classifier)")
        return 0

    date_col = _pick_col(clf, ["day", "date", "timestamp", "ts"])
    if date_col is None:
        raise ValueError("classifier csv has no date/day/timestamp column")
    if args.regime_col not in clf.columns:
        raise ValueError(f"classifier csv missing regime col: {args.regime_col}")
    price_col = _pick_col(clf, ["eth_close", "close", "spot_close"])

    clf["date"] = pd.to_datetime(clf[date_col], utc=True, errors="coerce").dt.floor("D")
    clf = clf.dropna(subset=["date"]).sort_values("date")
    if args.today_only:
        today = pd.Timestamp.now("UTC").floor("D")
        clf = clf[clf["date"] == today]

    added = 0
    for _, r in clf.iterrows():
        row = {
            "date": r["date"],
            "regime_v2": str(r[args.regime_col]),
            "price": float(r[price_col]) if price_col and pd.notna(r[price_col]) else float("nan"),
            "recorded_at": str(pd.Timestamp.now("UTC")),
            "source": "classifier_daily_snapshot",
        }
        before = len(snap)
        snap = _append_if_new(snap, row)
        if len(snap) > before:
            added += 1

    snap = snap.sort_values("date").reset_index(drop=True)
    snap.to_csv(snapshot_path, index=False)
    print(f"wrote {snapshot_path} rows={len(snap)} added={added}")
    if len(snap):
        print(
            f"last_snapshot date={snap['date'].iloc[-1]} regime={snap['regime_v2'].iloc[-1]} source={snap.get('source', pd.Series([''])).iloc[-1]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
