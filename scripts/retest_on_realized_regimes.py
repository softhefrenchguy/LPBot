from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ModelSpec:
    name: str
    path: str
    ret_col: str


def load_price(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("price csv must have timestamp, close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df["spot_r"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    return df[["timestamp", "spot_r"]]


def load_model(spec: ModelSpec) -> pd.DataFrame | None:
    p = Path(spec.path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "timestamp" not in df.columns or spec.ret_col not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"], utc=True, errors="coerce"),
            spec.name: pd.to_numeric(df[spec.ret_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out[spec.name] = out[spec.name].fillna(0.0)
    return out


def forward_window_returns_from_arr(x: np.ndarray, horizon_bars: int, valid: np.ndarray | None = None) -> np.ndarray:
    h = int(horizon_bars)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n <= h:
        return out

    if valid is None:
        arr = np.nan_to_num(x.astype(float), nan=0.0)
        cs = np.cumsum(arr)
        sums = cs[h:] - cs[:-h]
        out[:-h] = np.exp(sums) - 1.0
        return out

    arr = np.where(valid, x.astype(float), 0.0)
    cs = np.cumsum(arr)
    sums = cs[h:] - cs[:-h]
    vc = np.cumsum(valid.astype(int))
    cnts = vc[h:] - vc[:-h]
    ok = cnts >= int(0.95 * h)
    f = np.full(n - h, np.nan, dtype=float)
    f[ok] = np.exp(sums[ok]) - 1.0
    out[:-h] = f
    return out


def non_overlapping_top(starts: pd.DataFrame, n: int, days: int, ascending: bool, sort_col: str = "spot_fwd") -> pd.DataFrame:
    s = starts.sort_values(sort_col, ascending=ascending)
    picked = []
    for _, r in s.iterrows():
        t = r["timestamp"]
        if all(abs((t - x).total_seconds()) >= days * 24 * 3600 for x in picked):
            picked.append(t)
        if len(picked) >= n:
            break
    return starts[starts["timestamp"].isin(picked)].sort_values("timestamp")


def window_return(log_r: np.ndarray) -> float:
    return float(np.exp(np.nansum(log_r)) - 1.0)


def evaluate_windows(
    df: pd.DataFrame,
    model_cols: list[str],
    starts: pd.DataFrame,
    days: int,
    horizon_bars: int,
    model_fwd: dict[str, np.ndarray],
    spot_fwd: np.ndarray,
) -> pd.DataFrame:
    rows = []
    idx = starts["idx"].to_numpy(dtype=int)
    for m in model_cols:
        mm = model_fwd[m]
        ss = spot_fwd
        mvals = mm[idx]
        svals = ss[idx]
        mask = np.isfinite(mvals) & np.isfinite(svals)
        if mask.sum() == 0:
            continue
        arr_m = mvals[mask]
        arr_s = svals[mask]
        rows.append(
            {
                "model": m,
                "n_windows": len(arr_m),
                "avg_model_ret": float(arr_m.mean()),
                "med_model_ret": float(np.median(arr_m)),
                "avg_spot_ret": float(arr_s.mean()),
                "avg_excess": float((arr_m - arr_s).mean()),
                "capture_ratio": float(arr_m.mean() / (arr_s.mean() + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retest all candidate models on realized bull/bear/chop windows.")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--bull-threshold", type=float, default=0.20)
    ap.add_argument("--bear-threshold", type=float, default=-0.20)
    ap.add_argument("--chop-abs-threshold", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out-summary-csv", default="artifacts/paper/retest_realized_regimes_summary.csv")
    ap.add_argument("--out-windows-csv", default="artifacts/paper/retest_realized_regimes_windows.csv")
    args = ap.parse_args()

    specs = [
        ModelSpec("breakout", "artifacts/paper/breakout_paper.csv", "core_r"),
        ModelSpec("v5", "artifacts/paper/hmm_enet_scaled_v5.csv", "strat_r"),
        ModelSpec("v5_prob_best", "artifacts/paper/hmm_enet_prob_best.csv", "strat_r"),
        ModelSpec("funding_basis_db", "artifacts/paper/funding_basis_mr_365d_db.csv", "strat_r"),
        ModelSpec("momentum_v4", "artifacts/paper/momentum_riskon_365d_v4.csv", "strat_r"),
        ModelSpec("bull_model_v2", "artifacts/paper/bull_model_strategy_v2.csv", "strat_r"),
        ModelSpec("base_plus_overlay", "artifacts/paper/base_plus_overlay_365d.csv", "ret_combo"),
    ]

    base = load_price(args.price_csv)
    merged = base.copy()
    model_cols = []
    for s in specs:
        d = load_model(s)
        if d is None:
            continue
        merged = merged.merge(d, on="timestamp", how="left")
        model_cols.append(s.name)

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    h = int(args.window_days * 24 * 12)
    spot_arr = merged["spot_r"].to_numpy(dtype=float)
    spot_fwd = forward_window_returns_from_arr(spot_arr, h)
    merged["spot_fwd"] = spot_fwd
    merged["idx"] = np.arange(len(merged))
    starts = merged[["idx", "timestamp", "spot_fwd"]].dropna().copy()

    model_fwd = {}
    for m in model_cols:
        arr = merged[m].to_numpy(dtype=float)
        valid = np.isfinite(arr)
        model_fwd[m] = forward_window_returns_from_arr(arr, h, valid=valid)

    bull = starts[starts["spot_fwd"] >= float(args.bull_threshold)].copy()
    bear = starts[starts["spot_fwd"] <= float(args.bear_threshold)].copy()
    chop = starts[starts["spot_fwd"].abs() <= float(args.chop_abs_threshold)].copy()

    top_bull = non_overlapping_top(bull, int(args.top_n), int(args.window_days), ascending=False)
    top_bear = non_overlapping_top(bear, int(args.top_n), int(args.window_days), ascending=True)
    chop2 = chop.assign(absret=chop["spot_fwd"].abs())
    top_chop = non_overlapping_top(chop2, int(args.top_n), int(args.window_days), ascending=True, sort_col="absret")

    out_windows = []
    for name, w in [("bull_top", top_bull), ("bear_top", top_bear), ("chop_top", top_chop)]:
        ww = w.copy()
        ww["bucket"] = name
        out_windows.append(ww[["bucket", "timestamp", "spot_fwd"]])
    windows_df = pd.concat(out_windows, ignore_index=True)

    sum_rows = []
    for bucket_name, w in [
        ("bull_top", top_bull),
        ("bear_top", top_bear),
        ("chop_top", top_chop),
        ("bull_all", bull),
        ("bear_all", bear),
        ("chop_all", chop),
    ]:
        ev = evaluate_windows(
            merged,
            model_cols,
            w,
            int(args.window_days),
            h,
            model_fwd=model_fwd,
            spot_fwd=spot_fwd,
        )
        if ev.empty:
            continue
        ev["bucket"] = bucket_name
        sum_rows.append(ev)

    summary = pd.concat(sum_rows, ignore_index=True)
    summary = summary[
        ["bucket", "model", "n_windows", "avg_model_ret", "med_model_ret", "avg_spot_ret", "avg_excess", "capture_ratio"]
    ].sort_values(["bucket", "avg_excess"], ascending=[True, False])

    out_sum = Path(args.out_summary_csv)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)
    out_win = Path(args.out_windows_csv)
    out_win.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(out_win, index=False)

    print("wrote", out_sum)
    print("wrote", out_win)
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
