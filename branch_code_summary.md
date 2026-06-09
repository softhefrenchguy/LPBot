# LPBot Frozen Branch Code Summary

Generated: 2026-06-09 13:13:48 +01:00


---

## Branch: feat/hmm-v1

### git log --oneline -10

```text
fa35cbb Freeze regime detector v1 (1h micro, 8h macro, BTC/ETH)
93a5eb8 Add OHLCV resampling from 1m to 5m/15m/1h
9aac30e Finalize Binance 1m downloader
0d1f0bb Add Binance 1m downloader
```

### Python files found: 8


#### scripts/download_binance_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
LIMIT = 1000
SLEEP_MIN = 0.2
SLEEP_MAX = 0.5
MAX_RETRIES = 5
TIMEOUT_SEC = 20

SYMBOLS = ["ETHUSDC", "BTCUSDC", "SOLUSDC"]
START_UTC = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def utc_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def request_with_retries(session: requests.Session, params: dict) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                else:
                    time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            if resp.status_code >= 500:
                time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.5 * attempt + random.uniform(0, 0.5))
```

#### scripts/run_health_checks.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Summarize data health reports across regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes-dir", default="data/regimes")
    args = p.parse_args()

    base = Path(args.regimes_dir)
    rows = []

    for tf_dir in base.iterdir():
        if not tf_dir.is_dir():
            continue
        timeframe = tf_dir.name
        for profile_dir in tf_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            profile = profile_dir.name
            for sym_dir in profile_dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                health = sym_dir / f"data_health_{timeframe}.csv"
                if not health.exists():
                    continue
                df = pd.read_csv(health)
                metrics = (
                    df[df["type"] == "metric"].set_index("name")["value"].to_dict()
                )
                rows.append(
                    {
                        "timeframe": timeframe,
                        "profile": profile,
                        "symbol": sym_dir.name,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False) if not out.empty else "No data health files found.")


```

#### scripts/run_macro_8h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 8h macro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-2)
    p.add_argument("--min-count", type=int, default=240)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "8h",
            "--min-count",
            str(args.min_count),
            *sym_args,
```

#### scripts/run_micro_1h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 1h micro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-3)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "1h",
            *sym_args,
        ]
    )

```

#### src/hmm/fit_hmm.py

Description: HMM regime modelling or diagnostics script.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Fit Gaussian HMMs (K=2,3,4 by default) on 1h observation datasets and export regimes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


FEATURES = ["r", "abs_r", "vol20", "vol_z"]
R0_EPS = 1e-12


def _timeframe_to_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("h") and tf[:-1].isdigit():
        return float(tf[:-1])
    if tf.endswith("m") and tf[:-1].isdigit():
        return float(tf[:-1]) / 60.0
    if tf.endswith("d") and tf[:-1].isdigit():
        return float(tf[:-1]) * 24.0
    return 1.0


def _load_symbols_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Symbols file not found: {path.resolve()}")
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)
    return symbols


def _print_symbol_summary(requested: List[str], found: List[str], missing: List[str]) -> None:
    print(f"Symbols requested: {', '.join(requested) if requested else '(none)'}")
    print(f"Symbols found: {', '.join(found) if found else '(none)'}")
    if missing:
```

#### src/lp/simulate_lp.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""LP simulation placeholder (v1)."""


def main() -> None:
    raise SystemExit("simulate_lp.py is a placeholder in v1.")


if __name__ == "__main__":
    main()
```

#### src/obs/build_obs.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Build HMM observation datasets from 1h OHLCV CSVs.

Rows with close <= 0 are dropped (log return invalid).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")


def _timestamp_sort_key(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().any():
        return dt
    return s.astype(str)


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ts_col = _timestamp_column(df)
    df = df.copy()
    df["_ts_key"] = _timestamp_sort_key(df[ts_col])
    df = df.dropna(subset=["_ts_key"])

    df = df.sort_values("_ts_key")
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.reset_index(drop=True)

    df = df.rename(columns={ts_col: "timestamp"})

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]

```

#### src/resample/resample_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Resample Binance 1-minute OHLCV CSVs into higher timeframes.

Input:  data/{SYMBOL}_1m.csv
Output: data/{SYMBOL}_{TF}.csv  where TF in {5m,15m,1h}

Assumptions about input columns (common Binance klines export):
- timestamp in milliseconds OR ISO datetime. We try to detect.
- Must include: open, high, low, close, volume
Optional: quote_volume, trades, taker_base_volume, taker_quote_volume

This script:
- parses timestamps
- sorts + dedupes
- resamples with OHLC rules and volume sums
- drops empty bars
- reports missing-bar gaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_TFS = ["5m", "15m", "1h"]


def _normalize_tf(tf: str) -> str:
    """
    Normalize short timeframes like '5m' into pandas-compatible offsets.
    """
    tf = tf.strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{tf[:-1]}min"
    return tf


def _default_min_count(tf: str) -> int:
    tf = tf.strip().lower()
    if tf == "8h":
        return 240
    return 1


def _infer_timestamp_series(df: pd.DataFrame) -> pd.Series:
```

---

## Branch: feat/elastic-net-v1

### git log --oneline -10

```text
3fe44fc ElasticNet vol v1.0 balanced exposure + backtest
22e0abd Add elasticnet v1 scaffold
6a952df Add 1m OHLCV via LFS
6e9368e Track CSVs with Git LFS
00e7c57 Add data installer + helper scripts
8f45b57 Add data installer script (copy or zip URL)
fa35cbb Freeze regime detector v1 (1h micro, 8h macro, BTC/ETH)
93a5eb8 Add OHLCV resampling from 1m to 5m/15m/1h
9aac30e Finalize Binance 1m downloader
0d1f0bb Add Binance 1m downloader
```

### Python files found: 23


#### lpbot/__init__.py

Description: LPBot package.

First 50 lines:
```python
"""LPBot package."""
```

#### lpbot/models/__init__.py

Description: Model package.

First 50 lines:
```python
"""Model package."""
```

#### lpbot/models/elasticnet_v1/__init__.py

Description: ElasticNet v1.0 model utilities.

First 50 lines:
```python
"""ElasticNet v1.0 model utilities."""
```

#### lpbot/models/elasticnet_v1/cli_backtest_exposure.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _annualization_factor(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return np.sqrt(bars_per_year)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _cagr_approx(equity: pd.Series, bar_minutes: int) -> float:
    if equity.empty:
        return float("nan")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    years = len(equity) / bars_per_year
    if years <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exposure = pd.read_csv(Path(args.exposure_csv))
    prices = pd.read_csv(Path(args.price_csv))

    if "timestamp" not in exposure.columns or "weight" not in exposure.columns:
        raise SystemExit("exposure-csv must contain columns: timestamp, weight")
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        raise SystemExit("price-csv must contain columns: timestamp, close")

```

#### lpbot/models/elasticnet_v1/cli_build_dataset.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.dataset import build_dataset, save_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")

    args = p.parse_args()

    dataset = build_dataset(
        path_1m=args.input_1m,
        horizon_min=args.horizon_min,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        regime_path=args.regime_path,
    )

    save_dataset(dataset, Path(args.out))
    print(f"rows={len(dataset)} cols={len(dataset.columns)} out={args.out}")


if __name__ == "__main__":
    main()
```

#### lpbot/models/elasticnet_v1/cli_evaluate.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.features import load_ohlcv_1m
from lpbot.models.elasticnet_v1.walk_forward import _time_windows


def _series_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return {"mse": float("nan"), "mae": float("nan"), "corr": float("nan"), "spearman": float("nan")}
    y = aligned.iloc[:, 0].to_numpy()
    yhat = aligned.iloc[:, 1].to_numpy()
    mse = float(np.mean((y - yhat) ** 2))
    mae = float(np.mean(np.abs(y - yhat)))
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    spearman = float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))
    return {"mse": mse, "mae": mae, "corr": corr, "spearman": spearman}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--input-1m", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    args = p.parse_args()

    meta_path = Path(args.artifacts_dir) / "metadata.json"
    target_type = "return"
    horizon_min = None
    train_days = 60
    val_days = 7
    step_days = 7
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target_type = meta.get("target_type", "return")
        horizon_min = meta.get("horizon_min", None)
        train_days = meta.get("train_days", train_days)
        val_days = meta.get("val_days", val_days)
        step_days = meta.get("step_days", step_days)

```

#### lpbot/models/elasticnet_v1/cli_exposure.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m
from lpbot.models.elasticnet_v1.inference import load_artifacts
from lpbot.models.elasticnet_v1.vol_mapping import sigma_ann_from_sigma_fwd, sigma_fwd_from_yhat


def _parse_riskoff_values(raw: str) -> tuple[set[str], set[float]]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    str_vals = {t.lower() for t in tokens}
    num_vals: set[float] = set()
    for token in tokens:
        try:
            num_vals.add(float(token))
        except ValueError:
            continue
    return str_vals, num_vals


def _ema_span_bars(ema_span_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    span = ema_span_minutes / bar_minutes
    span_int = int(round(span))
    return max(1, span_int)


def _horizon_bars(horizon_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars = horizon_minutes / bar_minutes
    return max(1, int(round(bars)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--model-dir", default="models/elasticnet-v1.0")
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--ema-span-minutes", type=float, default=None)
```

#### lpbot/models/elasticnet_v1/cli_infer.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.inference import predict_latest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--n-latest", type=int, default=1)
    p.add_argument("--out", default=None)

    args = p.parse_args()

    out_df = predict_latest(
        path_1m=args.input_1m,
        artifacts_dir=args.artifacts_dir,
        regime_path=args.regime_path,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        n_latest=args.n_latest,
    )

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

#### lpbot/models/elasticnet_v1/cli_train.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.walk_forward import walk_forward_train


def _parse_float_list(val: str) -> list[float]:
    return [float(x.strip()) for x in val.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--val-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--alpha-grid", default="1e-4,1e-3,1e-2,1e-1,1.0")
    p.add_argument("--l1-grid", default="0.1,0.5,0.9")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument(
        "--target-type",
        choices=["return", "vol"],
        default="return",
        help="Training target type (default: return).",
    )

    args = p.parse_args()

    if args.dataset is None and args.input_1m is None:
        raise SystemExit("Provide --dataset or --input-1m")

    if args.dataset is not None:
        dataset = pd.read_csv(Path(args.dataset))
    else:
        dataset = build_supervised_dataset(
            path_1m=args.input_1m,
            horizon_min=args.horizon_min,
            include_time_features=not args.no_time_features,
            atr_window=args.atr_window,
```

#### lpbot/models/elasticnet_v1/dataset.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def _validate_vol_target_alignment(
    r1: pd.Series,
    sigma_fwd: pd.Series,
    horizon_min: int,
    sample_size: int = 5,
) -> None:
    if not __debug__:
        return
    if len(sigma_fwd) == 0:
        return
    if np.nanstd(r1.values) == 0:
        return

    valid_idx = np.where(~np.isnan(sigma_fwd.values))[0]
    if len(valid_idx) == 0:
        return

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        valid_idx, size=min(sample_size, len(valid_idx)), replace=False
    )

    for idx in sample_idx:
        manual = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        vec = sigma_fwd.iloc[idx]
        if not np.isclose(manual, vec, rtol=1e-6, atol=1e-12, equal_nan=True):
            raise ValueError(
                "Vol target alignment mismatch: forward sigma does not match "
                "vectorized computation. Check shift direction and window."
            )

    past_candidates = [i for i in sample_idx if i >= horizon_min - 1]
    if not past_candidates:
        return

    all_past_match = True
    for idx in past_candidates:
        forward = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        past = r1.iloc[idx - horizon_min + 1 : idx + 1].std()
        if not np.isclose(forward, past, rtol=1e-6, atol=1e-12, equal_nan=True):
```

#### lpbot/models/elasticnet_v1/features.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Missing timestamp column. Columns: {list(df.columns)}")


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        median = int(s_num.dropna().median()) if s_num.dropna().empty is False else 0
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(s_num, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_ohlcv_1m(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    ts_col = _timestamp_column(df)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.reset_index(drop=True)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
```

#### lpbot/models/elasticnet_v1/inference.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def load_artifacts(artifacts_dir: str | Path):
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / "model.joblib")
    scaler = joblib.load(artifacts_path / "scaler.joblib")
    meta = json.loads((artifacts_path / "metadata.json").read_text(encoding="utf-8"))
    feature_list = meta.get("feature_list", [])
    return model, scaler, feature_list, meta


def predict_latest(
    path_1m: str | Path,
    artifacts_dir: str | Path,
    regime_path: str | Path | None = None,
    include_time_features: bool = True,
    atr_window: int = 14,
    n_latest: int = 1,
) -> pd.DataFrame:
    df_1m = load_ohlcv_1m(path_1m)
    regime_df = None
    if regime_path is not None:
        regime_df = pd.read_csv(Path(regime_path))

    features = build_features(
        df_1m=df_1m,
        include_time_features=include_time_features,
        atr_window=atr_window,
        regime_df=regime_df,
    )

    model, scaler, feature_list, _ = load_artifacts(artifacts_dir)
    features = features.set_index("timestamp")
    features = features[feature_list]
    latest = features.tail(n_latest)
    X = scaler.transform(latest.values)
    y_hat = model.predict(X)

    out = pd.DataFrame(
        {
```

#### lpbot/models/elasticnet_v1/vol_mapping.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import numpy as np


def horizon_bars(horizon_minutes: int, bar_minutes: int) -> float:
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive.")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    return horizon_minutes / bar_minutes


def sigma_fwd_from_yhat(yhat: float | np.ndarray) -> float | np.ndarray:
    yhat_arr = np.asarray(yhat, dtype=float)
    sigma_fwd = np.expm1(yhat_arr)
    if np.ndim(yhat_arr) == 0:
        return float(sigma_fwd)
    return sigma_fwd


def sigma_ann_from_sigma_fwd(
    sigma_fwd: float | np.ndarray,
    horizon_bars: float,
    bar_minutes: int,
) -> float | np.ndarray:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")

    sigma_fwd_arr = np.asarray(sigma_fwd, dtype=float)
    if horizon_bars <= 0:
        sigma_ann = np.full_like(sigma_fwd_arr, np.nan, dtype=float)
    else:
        sigma_per_bar = sigma_fwd_arr / np.sqrt(horizon_bars)
        bars_per_year = 365 * 24 * (60 / bar_minutes)
        sigma_ann = sigma_per_bar * np.sqrt(bars_per_year)

    sigma_ann = np.where(np.isfinite(sigma_ann), sigma_ann, np.nan)
    sigma_ann = np.where(sigma_ann == 0, np.nan, sigma_ann)
    if np.ndim(sigma_fwd_arr) == 0:
        return float(sigma_ann)
    return sigma_ann
```

#### lpbot/models/elasticnet_v1/walk_forward.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowResult:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    best_alpha: float
    best_l1_ratio: float
    mse: float
    mae: float
    corr: float
    spearman: float
    directional_acc: float
    nonzero: int


def _time_windows(
    timestamps: pd.Series, train_days: int, val_days: int, step_days: int
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = timestamps.min()
    end = timestamps.max()
    step = pd.Timedelta(days=step_days)
    train_delta = pd.Timedelta(days=train_days)
    val_delta = pd.Timedelta(days=val_days)

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_delta
        val_start = train_end
        val_end = val_start + val_delta
        if val_end > end:
```

#### scripts/download_binance_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
LIMIT = 1000
SLEEP_MIN = 0.2
SLEEP_MAX = 0.5
MAX_RETRIES = 5
TIMEOUT_SEC = 20

SYMBOLS = ["ETHUSDC", "BTCUSDC", "SOLUSDC"]
START_UTC = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def utc_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def request_with_retries(session: requests.Session, params: dict) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                else:
                    time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            if resp.status_code >= 500:
                time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.5 * attempt + random.uniform(0, 0.5))
```

#### scripts/get_data.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REQUIRED = ["ETHUSDC_1m.csv", "BTCUSDC_1m.csv"]


def copy_mode(src_dir: Path, out_dir: Path) -> None:
    src_dir = src_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in REQUIRED:
        src = src_dir / name
        dst = out_dir / name
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    if missing:
        print("\nMissing files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)


def url_mode(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_dir / "_market_data.zip"

    print(f"Downloading zip: {url}")
    urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as z:
        names = set(z.namelist())
        # allow zip to contain nested paths; extract only the required files
        extracted = 0
        for req in REQUIRED:
            match = next((n for n in names if n.endswith(req)), None)
            if not match:
                raise FileNotFoundError(f"{req} not found inside zip")
            z.extract(match, out_dir)
```

#### scripts/run_health_checks.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Summarize data health reports across regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes-dir", default="data/regimes")
    args = p.parse_args()

    base = Path(args.regimes_dir)
    rows = []

    for tf_dir in base.iterdir():
        if not tf_dir.is_dir():
            continue
        timeframe = tf_dir.name
        for profile_dir in tf_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            profile = profile_dir.name
            for sym_dir in profile_dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                health = sym_dir / f"data_health_{timeframe}.csv"
                if not health.exists():
                    continue
                df = pd.read_csv(health)
                metrics = (
                    df[df["type"] == "metric"].set_index("name")["value"].to_dict()
                )
                rows.append(
                    {
                        "timeframe": timeframe,
                        "profile": profile,
                        "symbol": sym_dir.name,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False) if not out.empty else "No data health files found.")


```

#### scripts/run_macro_8h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 8h macro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-2)
    p.add_argument("--min-count", type=int, default=240)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "8h",
            "--min-count",
            str(args.min_count),
            *sym_args,
```

#### scripts/run_micro_1h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 1h micro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-3)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "1h",
            *sym_args,
        ]
    )

```

#### src/hmm/fit_hmm.py

Description: HMM regime modelling or diagnostics script.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Fit Gaussian HMMs (K=2,3,4 by default) on 1h observation datasets and export regimes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


FEATURES = ["r", "abs_r", "vol20", "vol_z"]
R0_EPS = 1e-12


def _timeframe_to_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("h") and tf[:-1].isdigit():
        return float(tf[:-1])
    if tf.endswith("m") and tf[:-1].isdigit():
        return float(tf[:-1]) / 60.0
    if tf.endswith("d") and tf[:-1].isdigit():
        return float(tf[:-1]) * 24.0
    return 1.0


def _load_symbols_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Symbols file not found: {path.resolve()}")
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)
    return symbols


def _print_symbol_summary(requested: List[str], found: List[str], missing: List[str]) -> None:
    print(f"Symbols requested: {', '.join(requested) if requested else '(none)'}")
    print(f"Symbols found: {', '.join(found) if found else '(none)'}")
    if missing:
```

#### src/lp/simulate_lp.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""LP simulation placeholder (v1)."""


def main() -> None:
    raise SystemExit("simulate_lp.py is a placeholder in v1.")


if __name__ == "__main__":
    main()
```

#### src/obs/build_obs.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Build HMM observation datasets from 1h OHLCV CSVs.

Rows with close <= 0 are dropped (log return invalid).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")


def _timestamp_sort_key(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().any():
        return dt
    return s.astype(str)


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ts_col = _timestamp_column(df)
    df = df.copy()
    df["_ts_key"] = _timestamp_sort_key(df[ts_col])
    df = df.dropna(subset=["_ts_key"])

    df = df.sort_values("_ts_key")
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.reset_index(drop=True)

    df = df.rename(columns={ts_col: "timestamp"})

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]

```

#### src/resample/resample_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Resample Binance 1-minute OHLCV CSVs into higher timeframes.

Input:  data/{SYMBOL}_1m.csv
Output: data/{SYMBOL}_{TF}.csv  where TF in {5m,15m,1h}

Assumptions about input columns (common Binance klines export):
- timestamp in milliseconds OR ISO datetime. We try to detect.
- Must include: open, high, low, close, volume
Optional: quote_volume, trades, taker_base_volume, taker_quote_volume

This script:
- parses timestamps
- sorts + dedupes
- resamples with OHLC rules and volume sums
- drops empty bars
- reports missing-bar gaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_TFS = ["5m", "15m", "1h"]


def _normalize_tf(tf: str) -> str:
    """
    Normalize short timeframes like '5m' into pandas-compatible offsets.
    """
    tf = tf.strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{tf[:-1]}min"
    return tf


def _default_min_count(tf: str) -> int:
    tf = tf.strip().lower()
    if tf == "8h":
        return 240
    return 1


def _infer_timestamp_series(df: pd.DataFrame) -> pd.Series:
```

---

## Branch: feat/lp-overlay-v1

### git log --oneline -10

```text
8136b91 Add trend EMA hysteresis option
eaba165 Add attribution fields and harden report refresh
1ea8af7 Fix report server directory
1c16077 Add auto-updating HTML report server
0d8177f Add price/vol/drawdown to report
c0dd2bf Inline volume fetch in paper runner
2539fa3 Handle LFS pointer price files in runner
c942e42 Add price start/lookback for paper runner
e1725d8 Add paper trade v1 modules
e209f9f Add paper package init
```

### Python files found: 37


#### lpbot/__init__.py

Description: LPBot package.

First 50 lines:
```python
"""LPBot package."""
```

#### lpbot/models/__init__.py

Description: Model package.

First 50 lines:
```python
"""Model package."""
```

#### lpbot/models/elasticnet_v1/__init__.py

Description: ElasticNet v1.0 model utilities.

First 50 lines:
```python
"""ElasticNet v1.0 model utilities."""
```

#### lpbot/models/elasticnet_v1/cli_backtest_exposure.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _annualization_factor(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return np.sqrt(bars_per_year)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _cagr_approx(equity: pd.Series, bar_minutes: int) -> float:
    if equity.empty:
        return float("nan")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    years = len(equity) / bars_per_year
    if years <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exposure = pd.read_csv(Path(args.exposure_csv))
    prices = pd.read_csv(Path(args.price_csv))

    if "timestamp" not in exposure.columns or "weight" not in exposure.columns:
        raise SystemExit("exposure-csv must contain columns: timestamp, weight")
    if "timestamp" not in prices.columns or "close" not in prices.columns:
        raise SystemExit("price-csv must contain columns: timestamp, close")

```

#### lpbot/models/elasticnet_v1/cli_build_dataset.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.dataset import build_dataset, save_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")

    args = p.parse_args()

    dataset = build_dataset(
        path_1m=args.input_1m,
        horizon_min=args.horizon_min,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        regime_path=args.regime_path,
    )

    save_dataset(dataset, Path(args.out))
    print(f"rows={len(dataset)} cols={len(dataset.columns)} out={args.out}")


if __name__ == "__main__":
    main()
```

#### lpbot/models/elasticnet_v1/cli_evaluate.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.features import load_ohlcv_1m
from lpbot.models.elasticnet_v1.walk_forward import _time_windows


def _series_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return {"mse": float("nan"), "mae": float("nan"), "corr": float("nan"), "spearman": float("nan")}
    y = aligned.iloc[:, 0].to_numpy()
    yhat = aligned.iloc[:, 1].to_numpy()
    mse = float(np.mean((y - yhat) ** 2))
    mae = float(np.mean(np.abs(y - yhat)))
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    spearman = float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))
    return {"mse": mse, "mae": mae, "corr": corr, "spearman": spearman}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--input-1m", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    args = p.parse_args()

    meta_path = Path(args.artifacts_dir) / "metadata.json"
    target_type = "return"
    horizon_min = None
    train_days = 60
    val_days = 7
    step_days = 7
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target_type = meta.get("target_type", "return")
        horizon_min = meta.get("horizon_min", None)
        train_days = meta.get("train_days", train_days)
        val_days = meta.get("val_days", val_days)
        step_days = meta.get("step_days", step_days)

```

#### lpbot/models/elasticnet_v1/cli_exposure.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m
from lpbot.models.elasticnet_v1.inference import load_artifacts
from lpbot.models.elasticnet_v1.vol_mapping import sigma_ann_from_sigma_fwd, sigma_fwd_from_yhat


def _parse_riskoff_values(raw: str) -> tuple[set[str], set[float]]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    str_vals = {t.lower() for t in tokens}
    num_vals: set[float] = set()
    for token in tokens:
        try:
            num_vals.add(float(token))
        except ValueError:
            continue
    return str_vals, num_vals


def _ema_span_bars(ema_span_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    span = ema_span_minutes / bar_minutes
    span_int = int(round(span))
    return max(1, span_int)


def _horizon_bars(horizon_minutes: float, bar_minutes: int) -> int:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    bars = horizon_minutes / bar_minutes
    return max(1, int(round(bars)))


def _trend_riskoff_with_hyst(
    close: pd.Series, ema: pd.Series, hyst: float
) -> pd.Series:
    if hyst <= 0:
        return close < ema
    upper = ema * (1.0 + hyst)
    lower = ema * (1.0 - hyst)
    out = np.zeros(len(close), dtype=bool)
    is_riskoff = False
    for i in range(len(close)):
```

#### lpbot/models/elasticnet_v1/cli_infer.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from lpbot.models.elasticnet_v1.inference import predict_latest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", required=True)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--regime-path", default=None)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--n-latest", type=int, default=1)
    p.add_argument("--out", default=None)

    args = p.parse_args()

    out_df = predict_latest(
        path_1m=args.input_1m,
        artifacts_dir=args.artifacts_dir,
        regime_path=args.regime_path,
        include_time_features=not args.no_time_features,
        atr_window=args.atr_window,
        n_latest=args.n_latest,
    )

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"out={args.out}")

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

#### lpbot/models/elasticnet_v1/cli_train.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lpbot.models.elasticnet_v1.dataset import build_supervised_dataset
from lpbot.models.elasticnet_v1.walk_forward import walk_forward_train


def _parse_float_list(val: str) -> list[float]:
    return [float(x.strip()) for x in val.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-1m", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--regime-path", default=None)
    p.add_argument("--artifacts-dir", default="models/elasticnet-v1.0")
    p.add_argument("--horizon-min", type=int, default=15)
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--val-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--alpha-grid", default="1e-4,1e-3,1e-2,1e-1,1.0")
    p.add_argument("--l1-grid", default="0.1,0.5,0.9")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument(
        "--target-type",
        choices=["return", "vol"],
        default="return",
        help="Training target type (default: return).",
    )

    args = p.parse_args()

    if args.dataset is None and args.input_1m is None:
        raise SystemExit("Provide --dataset or --input-1m")

    if args.dataset is not None:
        dataset = pd.read_csv(Path(args.dataset))
    else:
        dataset = build_supervised_dataset(
            path_1m=args.input_1m,
            horizon_min=args.horizon_min,
            include_time_features=not args.no_time_features,
            atr_window=args.atr_window,
```

#### lpbot/models/elasticnet_v1/dataset.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def _validate_vol_target_alignment(
    r1: pd.Series,
    sigma_fwd: pd.Series,
    horizon_min: int,
    sample_size: int = 5,
) -> None:
    if not __debug__:
        return
    if len(sigma_fwd) == 0:
        return
    if np.nanstd(r1.values) == 0:
        return

    valid_idx = np.where(~np.isnan(sigma_fwd.values))[0]
    if len(valid_idx) == 0:
        return

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        valid_idx, size=min(sample_size, len(valid_idx)), replace=False
    )

    for idx in sample_idx:
        manual = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        vec = sigma_fwd.iloc[idx]
        if not np.isclose(manual, vec, rtol=1e-6, atol=1e-12, equal_nan=True):
            raise ValueError(
                "Vol target alignment mismatch: forward sigma does not match "
                "vectorized computation. Check shift direction and window."
            )

    past_candidates = [i for i in sample_idx if i >= horizon_min - 1]
    if not past_candidates:
        return

    all_past_match = True
    for idx in past_candidates:
        forward = r1.iloc[idx + 1 : idx + 1 + horizon_min].std()
        past = r1.iloc[idx - horizon_min + 1 : idx + 1].std()
        if not np.isclose(forward, past, rtol=1e-6, atol=1e-12, equal_nan=True):
```

#### lpbot/models/elasticnet_v1/features.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Missing timestamp column. Columns: {list(df.columns)}")


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        median = int(s_num.dropna().median()) if s_num.dropna().empty is False else 0
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(s_num, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_ohlcv_1m(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    ts_col = _timestamp_column(df)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.reset_index(drop=True)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
```

#### lpbot/models/elasticnet_v1/inference.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from lpbot.models.elasticnet_v1.features import build_features, load_ohlcv_1m


def load_artifacts(artifacts_dir: str | Path):
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / "model.joblib")
    scaler = joblib.load(artifacts_path / "scaler.joblib")
    meta = json.loads((artifacts_path / "metadata.json").read_text(encoding="utf-8"))
    feature_list = meta.get("feature_list", [])
    return model, scaler, feature_list, meta


def predict_latest(
    path_1m: str | Path,
    artifacts_dir: str | Path,
    regime_path: str | Path | None = None,
    include_time_features: bool = True,
    atr_window: int = 14,
    n_latest: int = 1,
) -> pd.DataFrame:
    df_1m = load_ohlcv_1m(path_1m)
    regime_df = None
    if regime_path is not None:
        regime_df = pd.read_csv(Path(regime_path))

    features = build_features(
        df_1m=df_1m,
        include_time_features=include_time_features,
        atr_window=atr_window,
        regime_df=regime_df,
    )

    model, scaler, feature_list, _ = load_artifacts(artifacts_dir)
    features = features.set_index("timestamp")
    features = features[feature_list]
    latest = features.tail(n_latest)
    X = scaler.transform(latest.values)
    y_hat = model.predict(X)

    out = pd.DataFrame(
        {
```

#### lpbot/models/elasticnet_v1/vol_mapping.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import numpy as np


def horizon_bars(horizon_minutes: int, bar_minutes: int) -> float:
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive.")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    return horizon_minutes / bar_minutes


def sigma_fwd_from_yhat(yhat: float | np.ndarray) -> float | np.ndarray:
    yhat_arr = np.asarray(yhat, dtype=float)
    sigma_fwd = np.expm1(yhat_arr)
    if np.ndim(yhat_arr) == 0:
        return float(sigma_fwd)
    return sigma_fwd


def sigma_ann_from_sigma_fwd(
    sigma_fwd: float | np.ndarray,
    horizon_bars: float,
    bar_minutes: int,
) -> float | np.ndarray:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")

    sigma_fwd_arr = np.asarray(sigma_fwd, dtype=float)
    if horizon_bars <= 0:
        sigma_ann = np.full_like(sigma_fwd_arr, np.nan, dtype=float)
    else:
        sigma_per_bar = sigma_fwd_arr / np.sqrt(horizon_bars)
        bars_per_year = 365 * 24 * (60 / bar_minutes)
        sigma_ann = sigma_per_bar * np.sqrt(bars_per_year)

    sigma_ann = np.where(np.isfinite(sigma_ann), sigma_ann, np.nan)
    sigma_ann = np.where(sigma_ann == 0, np.nan, sigma_ann)
    if np.ndim(sigma_fwd_arr) == 0:
        return float(sigma_ann)
    return sigma_ann
```

#### lpbot/models/elasticnet_v1/walk_forward.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowResult:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    best_alpha: float
    best_l1_ratio: float
    mse: float
    mae: float
    corr: float
    spearman: float
    directional_acc: float
    nonzero: int


def _time_windows(
    timestamps: pd.Series, train_days: int, val_days: int, step_days: int
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = timestamps.min()
    end = timestamps.max()
    step = pd.Timedelta(days=step_days)
    train_delta = pd.Timedelta(days=train_days)
    val_delta = pd.Timedelta(days=val_days)

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_delta
        val_start = train_end
        val_end = val_start + val_delta
        if val_end > end:
```

#### lpbot/overlays/lp_overlay_v1/__init__.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python

```

#### lpbot/overlays/lp_overlay_v1/cli_lp_overlay_backtest.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns


def _bars_per_year(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    return 365 * 24 * (60 / bar_minutes)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _stats(returns: pd.Series, bar_minutes: int) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    bars_per_year = _bars_per_year(bar_minutes)
    equity = np.exp(r.cumsum())
    years = len(r) / bars_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    std = float(r.std())
    ann_vol = std * np.sqrt(bars_per_year)
    sharpe = float(r.mean() / std) * np.sqrt(bars_per_year) if std > 0 else float("nan")
    mdd = _max_drawdown(equity)
    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


```

#### lpbot/overlays/lp_overlay_v1/cli_lp_overlay_sweep.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _bars_per_year(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    return 365 * 24 * (60 / bar_minutes)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _stats(returns: pd.Series, bar_minutes: int) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"cagr": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    bars_per_year = _bars_per_year(bar_minutes)
    n_bars = len(r)
    equity = np.exp(r.cumsum())
    cagr = float(equity.iloc[-1] ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else float("nan")
    ann_vol = float(r.std()) * np.sqrt(bars_per_year)
    sharpe = cagr / ann_vol if ann_vol > 0 else float("nan")
    mdd = _max_drawdown(equity)
    return {"cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


def _sanitize_float(val: float) -> str:
    s = f"{val:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def main() -> None:
```

#### lpbot/overlays/lp_overlay_v1/cli_lp_overlay_volume_backtest.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lpbot.overlays.lp_overlay_v1.overlay import compute_lp_overlay_returns


def _bars_per_year(bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise SystemExit("bar_minutes must be positive.")
    return 365 * 24 * (60 / bar_minutes)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _stats(returns: pd.Series, bar_minutes: int) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"cagr": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    bars_per_year = _bars_per_year(bar_minutes)
    n_bars = len(r)
    equity = np.exp(r.cumsum())
    cagr = float(equity.iloc[-1] ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else float("nan")
    ann_vol = float(r.std()) * np.sqrt(bars_per_year)
    sharpe = cagr / ann_vol if ann_vol > 0 else float("nan")
    mdd = _max_drawdown(equity)
    return {"cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--volume-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=1)
    p.add_argument("--fee-tier", type=float, required=True)
    p.add_argument("--in-range-frac", type=float, required=True)
    p.add_argument("--range-sigma-ref", type=float, default=2.0)
    p.add_argument("--min-in-range-frac", type=float, default=0.05)
    p.add_argument("--max-in-range-frac", type=float, default=0.25)
```

#### lpbot/overlays/lp_overlay_v1/overlay.py

Description: Overlay logic or helper module for LP overlay experiments.

First 50 lines:
```python
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def per_bar_rate_from_annual(rate_ann: float, bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return float(rate_ann) / bars_per_year


def _build_lp_on(
    raw_on: pd.Series, stable_on: pd.Series, lp_cooldown_bars: int
) -> pd.Series:
    lp_on = np.zeros(len(raw_on), dtype=bool)
    cooldown = 0
    is_on = False
    for i in range(len(raw_on)):
        raw = bool(raw_on.iat[i])
        stable = bool(stable_on.iat[i])
        if is_on and not raw:
            is_on = False
            cooldown = lp_cooldown_bars
        if cooldown > 0:
            cooldown -= 1
            is_on = False
        elif not is_on and stable:
            is_on = True
        lp_on[i] = is_on
    return pd.Series(lp_on, index=raw_on.index)


def compute_lp_overlay_returns(
    df: pd.DataFrame,
    bar_minutes: int,
    fee_rate_ann: float,
    il_k: float,
    lp_vol_on: float,
    lp_scale: float,
    lp_weight_max: float,
    lp_min_on_bars: int = 6,
    lp_cooldown_bars: int = 12,
) -> pd.DataFrame:
    required = {"timestamp", "close", "weight", "sigma_ann_smooth", "gate"}
    missing = required - set(df.columns)
    if missing:
```

#### lpbot/paper/__init__.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
```

#### lpbot/paper/paper_trade_v1/__init__.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
```

#### lpbot/paper/paper_trade_v1/cli_paper_append.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse

import pandas as pd

from lpbot.paper.paper_trade_v1.paper_log import append_rows, read_last_timestamp
from lpbot.paper.paper_trade_v1.paper_tick import compute_paper_rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append paper-trade diagnostics log.")
    p.add_argument("--exposure-csv", required=True)
    p.add_argument("--price-csv", required=True)
    p.add_argument("--volume-csv", required=True)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--fee-tier", type=float, required=True)
    p.add_argument("--pool-tvl-usd", type=float, required=True)
    p.add_argument("--in-range-frac", type=float, required=True)
    p.add_argument("--range-sigma", type=float, required=True)
    p.add_argument("--range-sigma-ref", type=float, required=True)
    p.add_argument("--min-in-range-frac", type=float, required=True)
    p.add_argument("--max-in-range-frac", type=float, required=True)
    p.add_argument("--churn-k", type=float, required=True)
    p.add_argument("--il-k", type=float, required=True)
    p.add_argument("--lp-vol-on", type=float, required=True)
    p.add_argument("--lp-scale", type=float, required=True)
    p.add_argument("--lp-weight-max", type=float, required=True)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)
    p.add_argument("--log-csv", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exposure = pd.read_csv(args.exposure_csv)
    price = pd.read_csv(args.price_csv)
    volume = pd.read_csv(args.volume_csv)

    rows = compute_paper_rows(
        exposure,
        price,
        volume,
        bar_minutes=args.bar_minutes,
        fee_tier=args.fee_tier,
        pool_tvl_usd=args.pool_tvl_usd,
        in_range_frac=args.in_range_frac,
        range_sigma=args.range_sigma,
```

#### lpbot/paper/paper_trade_v1/cli_paper_report.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate HTML report from paper log.")
    p.add_argument("--log-csv", default="artifacts/paper/paper_log.csv")
    p.add_argument("--out", default="artifacts/paper/report.html")
    p.add_argument("--max-rows", type=int, default=5000, help="Max rows to plot (tail).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_csv)
    if not log_path.exists():
        raise SystemExit(f"Missing log csv: {log_path}")

    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if args.max_rows and len(df) > args.max_rows:
        df = df.tail(args.max_rows)

    df["core_r"] = pd.to_numeric(df.get("core_r"), errors="coerce").fillna(0.0)
    df["combined_r"] = pd.to_numeric(df.get("combined_r"), errors="coerce").fillna(0.0)
    df["r"] = pd.to_numeric(df.get("r"), errors="coerce").fillna(0.0)

    close = pd.to_numeric(df.get("close"), errors="coerce").ffill()
    df = df[close.notna()].copy()
    close = close.loc[df.index]

    df["core_eq"] = np.exp(df["core_r"].cumsum())
    df["combined_eq"] = np.exp(df["combined_r"].cumsum())
    spot_r = np.log(close / close.shift(1)).fillna(0.0)
    df["spot_eq"] = np.exp(spot_r.cumsum())

    peak = np.maximum.accumulate(df["combined_eq"].values)
    drawdown = df["combined_eq"].values / peak - 1.0
    spot_peak = np.maximum.accumulate(df["spot_eq"].values)
    spot_drawdown = df["spot_eq"].values / spot_peak - 1.0

    bar_minutes = 5
```

#### lpbot/paper/paper_trade_v1/cli_paper_report_server.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from lpbot.paper.paper_trade_v1.cli_paper_report import main as generate_report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_log(event: str, **fields: object) -> None:
    payload = {"ts": _utc_now(), "event": event, **fields}
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve auto-updating paper report.")
    p.add_argument("--log-csv", default="artifacts/paper/paper_log.csv")
    p.add_argument("--out", default="artifacts/paper/report.html")
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--interval-seconds", type=int, default=3600)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bind", default="0.0.0.0")
    return p.parse_args()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _serve(directory: Path, bind: str, port: int) -> None:
    directory = directory.resolve()
    # Ensure the handler serves from the report directory
    import os
    os.chdir(directory)
    handler = _QuietHandler
    server = ThreadingHTTPServer((bind, port), handler)
    _json_log("report_server_start", bind=bind, port=port, dir=str(directory))
    server.serve_forever()


```

#### lpbot/paper/paper_trade_v1/cli_paper_runner.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVAL = "1m"
BINANCE_LIMIT = 1000
BINANCE_TIMEOUT = 20


def _utc_now_minute() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _read_last_timestamp_ms(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"]).tail(1)
    except Exception:
        return None
    if df.empty:
        return None
    ts = df["timestamp"].iloc[0]
    if pd.isna(ts):
        return None
    try:
        ts_val = int(float(ts))
    except Exception:
        return None
    return ts_val


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
```

#### lpbot/paper/paper_trade_v1/cli_paper_service.py

Description: Command-line or service entry point for this branch component.

First 50 lines:
```python
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_log(event: str, **fields: object) -> None:
    payload = {"ts": _utc_now().isoformat(), "event": event, **fields}
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _write_heartbeat(path: Path) -> None:
    path.write_text(str(int(_utc_now().timestamp())), encoding="utf-8")


def _read_last_log_timestamp(log_csv: Path) -> Optional[str]:
    if not log_csv.exists() or log_csv.stat().st_size == 0:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(log_csv, usecols=["timestamp"]).tail(1)
        if df.empty:
            return None
        ts = df["timestamp"].iloc[0]
        return str(ts)
    except Exception:
        return None


def _write_state(state_path: Path, status: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp": _utc_now().isoformat(),
    }
```

#### lpbot/paper/paper_trade_v1/paper_log.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
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
```

#### lpbot/paper/paper_trade_v1/paper_tick.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def per_bar_rate_from_annual(rate_ann: float, bar_minutes: int) -> float:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    return float(rate_ann) / bars_per_year


def compute_in_range_frac_eff(
    in_range_frac: float,
    range_sigma: float,
    range_sigma_ref: float,
    min_frac: float,
    max_frac: float,
) -> float:
    if range_sigma <= 0:
        raise ValueError("range_sigma must be positive.")
    raw = float(in_range_frac) * (float(range_sigma_ref) / float(range_sigma))
    return float(min(max(raw, min_frac), max_frac))


def _build_lp_on(
    raw_on: pd.Series, stable_on: pd.Series, lp_cooldown_bars: int
) -> pd.Series:
    lp_on = np.zeros(len(raw_on), dtype=bool)
    cooldown = 0
    is_on = False
    for i in range(len(raw_on)):
        raw = bool(raw_on.iat[i])
        stable = bool(stable_on.iat[i])
        if is_on and not raw:
            is_on = False
            cooldown = lp_cooldown_bars
        if cooldown > 0:
            cooldown -= 1
            is_on = False
        elif not is_on and stable:
            is_on = True
        lp_on[i] = is_on
    return pd.Series(lp_on, index=raw_on.index)


```

#### scripts/download_binance_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
LIMIT = 1000
SLEEP_MIN = 0.2
SLEEP_MAX = 0.5
MAX_RETRIES = 5
TIMEOUT_SEC = 20

SYMBOLS = ["ETHUSDC", "BTCUSDC", "SOLUSDC"]
START_UTC = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def utc_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def request_with_retries(session: requests.Session, params: dict) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                else:
                    time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            if resp.status_code >= 500:
                time.sleep(1.5 * attempt + random.uniform(0, 0.5))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.5 * attempt + random.uniform(0, 0.5))
```

#### scripts/get_data.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REQUIRED = ["ETHUSDC_1m.csv", "BTCUSDC_1m.csv"]


def copy_mode(src_dir: Path, out_dir: Path) -> None:
    src_dir = src_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in REQUIRED:
        src = src_dir / name
        dst = out_dir / name
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    if missing:
        print("\nMissing files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)


def url_mode(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_dir / "_market_data.zip"

    print(f"Downloading zip: {url}")
    urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as z:
        names = set(z.namelist())
        # allow zip to contain nested paths; extract only the required files
        extracted = 0
        for req in REQUIRED:
            match = next((n for n in names if n.endswith(req)), None)
            if not match:
                raise FileNotFoundError(f"{req} not found inside zip")
            z.extract(match, out_dir)
```

#### scripts/run_health_checks.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Summarize data health reports across regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes-dir", default="data/regimes")
    args = p.parse_args()

    base = Path(args.regimes_dir)
    rows = []

    for tf_dir in base.iterdir():
        if not tf_dir.is_dir():
            continue
        timeframe = tf_dir.name
        for profile_dir in tf_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            profile = profile_dir.name
            for sym_dir in profile_dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                health = sym_dir / f"data_health_{timeframe}.csv"
                if not health.exists():
                    continue
                df = pd.read_csv(health)
                metrics = (
                    df[df["type"] == "metric"].set_index("name")["value"].to_dict()
                )
                rows.append(
                    {
                        "timeframe": timeframe,
                        "profile": profile,
                        "symbol": sym_dir.name,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False) if not out.empty else "No data health files found.")


```

#### scripts/run_macro_8h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 8h macro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-2)
    p.add_argument("--min-count", type=int, default=240)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "8h",
            "--min-count",
            str(args.min_count),
            *sym_args,
```

#### scripts/run_micro_1h.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Run 1h micro pipeline: resample -> obs -> HMM (BTC/ETH only by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--out-dir", default="data/regimes")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--symbols-file", default="config/symbols.txt")
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-3)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    obs_dir = Path(args.obs_dir)

    sym_args = []
    if args.symbols:
        sym_args = ["--symbols", *args.symbols]
    elif args.symbols_file:
        sym_args = ["--symbols-file", args.symbols_file]

    _run(
        [
            sys.executable,
            "src/resample/resample_1m.py",
            "--data-dir",
            str(data_dir),
            "--tfs",
            "1h",
            *sym_args,
        ]
    )

```

#### src/hmm/fit_hmm.py

Description: HMM regime modelling or diagnostics script.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Fit Gaussian HMMs (K=2,3,4 by default) on 1h observation datasets and export regimes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


FEATURES = ["r", "abs_r", "vol20", "vol_z"]
R0_EPS = 1e-12


def _timeframe_to_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("h") and tf[:-1].isdigit():
        return float(tf[:-1])
    if tf.endswith("m") and tf[:-1].isdigit():
        return float(tf[:-1]) / 60.0
    if tf.endswith("d") and tf[:-1].isdigit():
        return float(tf[:-1]) * 24.0
    return 1.0


def _load_symbols_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Symbols file not found: {path.resolve()}")
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)
    return symbols


def _print_symbol_summary(requested: List[str], found: List[str], missing: List[str]) -> None:
    print(f"Symbols requested: {', '.join(requested) if requested else '(none)'}")
    print(f"Symbols found: {', '.join(found) if found else '(none)'}")
    if missing:
```

#### src/lp/simulate_lp.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""LP simulation placeholder (v1)."""


def main() -> None:
    raise SystemExit("simulate_lp.py is a placeholder in v1.")


if __name__ == "__main__":
    main()
```

#### src/obs/build_obs.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""Build HMM observation datasets from 1h OHLCV CSVs.

Rows with close <= 0 are dropped (log return invalid).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")


def _timestamp_sort_key(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().any():
        return dt
    return s.astype(str)


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ts_col = _timestamp_column(df)
    df = df.copy()
    df["_ts_key"] = _timestamp_sort_key(df[ts_col])
    df = df.dropna(subset=["_ts_key"])

    df = df.sort_values("_ts_key")
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.reset_index(drop=True)

    df = df.rename(columns={ts_col: "timestamp"})

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]

```

#### src/resample/resample_1m.py

Description: Python module/script in this branch; purpose inferred from filename and imports.

First 50 lines:
```python
#!/usr/bin/env python3
"""
Resample Binance 1-minute OHLCV CSVs into higher timeframes.

Input:  data/{SYMBOL}_1m.csv
Output: data/{SYMBOL}_{TF}.csv  where TF in {5m,15m,1h}

Assumptions about input columns (common Binance klines export):
- timestamp in milliseconds OR ISO datetime. We try to detect.
- Must include: open, high, low, close, volume
Optional: quote_volume, trades, taker_base_volume, taker_quote_volume

This script:
- parses timestamps
- sorts + dedupes
- resamples with OHLC rules and volume sums
- drops empty bars
- reports missing-bar gaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_TFS = ["5m", "15m", "1h"]


def _normalize_tf(tf: str) -> str:
    """
    Normalize short timeframes like '5m' into pandas-compatible offsets.
    """
    tf = tf.strip()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{tf[:-1]}min"
    return tf


def _default_min_count(tf: str) -> int:
    tf = tf.strip().lower()
    if tf == "8h":
        return 240
    return 1


def _infer_timestamp_series(df: pd.DataFrame) -> pd.Series:
```
