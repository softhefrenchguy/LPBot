from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _parse_riskoff_values(raw: str) -> Tuple[set[str], set[float]]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    str_vals = set()
    num_vals = set()
    for p in parts:
        str_vals.add(p.lower())
        try:
            num_vals.add(float(p))
        except Exception:
            pass
    return str_vals, num_vals


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must contain timestamp, close")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def _load_regimes(path: Path, regime_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("regime csv must contain timestamp")
    if regime_col is None:
        if "state" in df.columns:
            regime_col = "state"
        elif "regime_label" in df.columns:
            regime_col = "regime_label"
        else:
            raise ValueError("regime csv must contain state or regime_label")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df[["timestamp", regime_col]].rename(columns={regime_col: "regime"})
    return df


def _merge_asof(price_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        price_df.sort_values("timestamp"),
        regime_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.dropna(subset=["regime"])
    return merged


def _compute_metrics(log_r: np.ndarray, bar_minutes: int) -> tuple[float, float, float, float]:
    log_r = np.asarray(log_r, float)
    log_r = log_r[np.isfinite(log_r)]
    if len(log_r) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(log_r))
    cagr = eq[-1] ** (bars_per_year / len(log_r)) - 1
    ann_vol = np.std(log_r) * np.sqrt(bars_per_year)
    sharpe = (np.mean(log_r) * bars_per_year) / (ann_vol + 1e-12)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return float(cagr), float(ann_vol), float(sharpe), float(mdd)


def main() -> None:
    p = argparse.ArgumentParser(description="HMM regime-gated backtest (gate-only).")
    p.add_argument("--price-csv", required=True)
    p.add_argument("--regime-csv", required=True)
    p.add_argument("--regime-col", default=None)
    p.add_argument("--riskoff-values", default="bear,0,2")
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    price_df = _load_price(Path(args.price_csv))
    regime_df = _load_regimes(Path(args.regime_csv), args.regime_col)

    merged = _merge_asof(price_df, regime_df)

    r = np.log(merged["close"] / merged["close"].shift(1))

    str_vals, num_vals = _parse_riskoff_values(args.riskoff_values)
    reg = merged["regime"]
    riskoff = pd.Series(False, index=merged.index)
    riskoff = riskoff | reg.astype(str).str.lower().isin(str_vals)
    reg_num = pd.to_numeric(reg, errors="coerce")
    riskoff = riskoff | reg_num.isin(num_vals)

    weight = (~riskoff).astype(float)
    strat_r = weight.shift(1) * r

    out_df = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "close": merged["close"],
            "r": r,
            "regime": reg,
            "riskoff": riskoff,
            "weight": weight,
            "strat_r": strat_r,
        }
    )
    out_df = out_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    cagr, ann_vol, sharpe, mdd = _compute_metrics(out_df["strat_r"].to_numpy(), args.bar_minutes)
    print(f"CAGR={cagr:.4f} | ann_vol={ann_vol:.4f} | Sharpe={sharpe:.4f} | max_drawdown={mdd:.4f}")
    print(f"rows={len(out_df)} riskoff_pct={float(riskoff.mean()*100.0):.2f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"out={out_path}")


if __name__ == "__main__":
    main()
