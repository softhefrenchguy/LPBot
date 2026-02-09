from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def read_last_timestamp(log_csv: str) -> Optional[pd.Timestamp]:
    path = Path(log_csv).expanduser().resolve()
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return None
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return ts.iloc[-1]


def append_rows(log_csv: str, df_rows: pd.DataFrame) -> None:
    if df_rows is None or df_rows.empty:
        return
    ensure_parent_dir(log_csv)
    path = Path(log_csv).expanduser().resolve()
    write_header = not path.exists() or path.stat().st_size == 0
    df_rows.to_csv(path, mode="a", index=False, header=write_header)
