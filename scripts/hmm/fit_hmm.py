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
        print(f"Symbols skipped (missing files): {', '.join(missing)}")


def _normalize_out_root(out_dir: Path, timeframe: str) -> Path:
    if out_dir.name == timeframe:
        return out_dir
    return out_dir / timeframe


def _longest_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns: {list(df.columns)}")


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        median = int(s_num.dropna().median()) if s_num.dropna().empty is False else 0
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(s_num, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def _load_obs(path_in: Path) -> pd.DataFrame:
    df = pd.read_csv(path_in)
    ts_col = _timestamp_column(df)

    df = df.copy()
    df = df.rename(columns={ts_col: "timestamp"})
    df["_ts_sort"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["_ts_sort"])
    df = df.sort_values("_ts_sort")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.reset_index(drop=True)
    df = df.drop(columns=["_ts_sort"])

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path_in.name}")

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[FEATURES].isna().any().any():
        raise ValueError(f"NaNs found in features for {path_in.name}")

    return df


def _train_mask(
    ts: pd.Series, train_start: str | None, train_end: str | None
) -> pd.Series:
    if train_start is None and train_end is None:
        return pd.Series([True] * len(ts), index=ts.index)
    ts_dt = _parse_timestamp_series(ts)
    if ts_dt.isna().any():
        raise ValueError("Timestamps could not be parsed for train range filtering.")
    mask = pd.Series([True] * len(ts_dt), index=ts_dt.index)
    if train_start is not None:
        start_dt = pd.to_datetime(train_start, utc=True)
        mask &= ts_dt >= start_dt
    if train_end is not None:
        end_dt = pd.to_datetime(train_end, utc=True)
        mask &= ts_dt <= end_dt
    return mask


def _n_params(K: int, D: int, covariance_type: str) -> int:
    base = (K - 1) + K * (K - 1) + K * D
    if covariance_type == "diag":
        cov = K * D
    else:
        cov = K * D * (D + 1) // 2
    return base + cov


def _covars_valid(covars: np.ndarray, covariance_type: str) -> bool:
    if covars is None:
        return False
    if not np.isfinite(covars).all():
        return False
    if covariance_type == "diag":
        return covars.ndim == 2 and np.all(covars > 0)
    if covars.ndim != 3:
        return False
    for i in range(covars.shape[0]):
        try:
            np.linalg.cholesky(covars[i])
        except np.linalg.LinAlgError:
            return False
    return True


def _regularize_covars(
    model: GaussianHMM, covariance_type: str, min_covar: float
) -> bool:
    covars = model.covars_
    if covars is None:
        return False
    if covariance_type == "diag":
        if covars.ndim == 3:
            covars = np.array([np.diag(c) for c in covars])
        if covars.ndim == 1:
            covars = covars.reshape(1, -1)
        covars = np.nan_to_num(covars, nan=min_covar, posinf=min_covar, neginf=min_covar)
        covars = np.maximum(covars, min_covar)
        model.covars_ = covars
        return _covars_valid(covars, covariance_type)

    dim = covars.shape[1]
    if covars.ndim == 2:
        covars = np.array([np.diag(c) for c in covars])
    covars = np.nan_to_num(covars, nan=0.0, posinf=0.0, neginf=0.0)
    eye = np.eye(dim) * min_covar
    covars = covars.copy()
    for i in range(covars.shape[0]):
        covars[i] = covars[i] + eye
    model.covars_ = covars
    return _covars_valid(covars, covariance_type)


def _data_health_report(
    symbol: str,
    profile: str,
    timeframe: str,
    df: pd.DataFrame,
    X_std: np.ndarray,
    out_dir: Path,
) -> None:
    ts = _parse_timestamp_series(df["timestamp"])
    gaps = ts.diff() > pd.Timedelta(hours=_timeframe_to_hours(timeframe))
    gap_count = int(gaps.sum()) if len(gaps) else 0

    r = df["r"].to_numpy()
    r0_mask = np.abs(r) < R0_EPS
    pct_r0 = float(np.mean(r0_mask) * 100.0)
    longest_r0 = _longest_run(r0_mask)

    vol0_pct = np.nan
    if "volume" in df.columns:
        vol0_pct = float(np.mean(df["volume"].to_numpy() == 0) * 100.0)

    vol_z = df["vol_z"].to_numpy()
    vol_z0_mask = np.abs(vol_z) < R0_EPS
    longest_vol_z0 = _longest_run(vol_z0_mask)

    rows: List[Dict[str, object]] = []
    rows.append({"type": "metric", "name": "pct_r0", "value": pct_r0})
    rows.append({"type": "metric", "name": "longest_r0_run_bars", "value": longest_r0})
    rows.append({"type": "metric", "name": "pct_volume_zero", "value": vol0_pct})
    rows.append({"type": "metric", "name": "longest_vol_z0_run_bars", "value": longest_vol_z0})
    rows.append(
        {"type": "metric", "name": f"timestamp_gaps_gt_{timeframe}", "value": gap_count}
    )

    for idx, feat in enumerate(FEATURES):
        raw = df[feat].to_numpy()
        rows.append(
            {
                "type": "feature_raw",
                "name": feat,
                "min": float(np.min(raw)),
                "max": float(np.max(raw)),
                "mean": float(np.mean(raw)),
                "std": float(np.std(raw, ddof=0)),
            }
        )
        std_vals = X_std[:, idx]
        rows.append(
            {
                "type": "feature_std",
                "name": feat,
                "min": float(np.min(std_vals)),
                "max": float(np.max(std_vals)),
                "mean": float(np.mean(std_vals)),
                "std": float(np.std(std_vals, ddof=0)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    health_path = out_dir / f"data_health_{timeframe}.csv"
    pd.DataFrame(rows).to_csv(health_path, index=False)

    if longest_r0 * _timeframe_to_hours(timeframe) > 48 or pct_r0 > 2.0 or gap_count > 0:
        print(
            f"[{symbol}][{profile}] WARNING: data health flags detected; "
            "HMM regimes may be unreliable."
        )


def _fit_best_hmm(
    X: np.ndarray,
    K: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    seed: int,
    restarts: int,
    min_covar: float,
) -> Tuple[GaussianHMM | None, float, bool, List[Dict[str, object]]]:
    best_model = None
    best_loglik = -np.inf
    best_converged = False
    failures: List[Dict[str, object]] = []

    for i in range(restarts):
        rs = seed + i
        model = GaussianHMM(
            n_components=K,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=rs,
            min_covar=min_covar,
        )
        try:
            model.fit(X)
            if not _regularize_covars(model, covariance_type, min_covar):
                raise ValueError("invalid covariances")

            if not np.allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError("transmat rows not ~1")
            if not np.allclose(model.startprob_.sum(), 1.0, atol=1e-4):
                raise ValueError("startprob not ~1")

            loglik = model.score(X)
            if not np.isfinite(loglik):
                raise ValueError("non-finite loglik")

            converged = bool(getattr(model, "monitor_", None) and model.monitor_.converged)
        except Exception as exc:
            failures.append({"seed": rs, "error": str(exc)})
            print(f"[K={K}] restart seed={rs} failed: {exc}")
            continue

        if loglik > best_loglik:
            best_loglik = loglik
            best_model = model
            best_converged = converged

    return best_model, best_loglik, best_converged, failures


def _state_durations(states: np.ndarray) -> Dict[int, List[int]]:
    durations: Dict[int, List[int]] = {}
    if len(states) == 0:
        return durations

    current_state = int(states[0])
    run = 1
    for s in states[1:]:
        s = int(s)
        if s == current_state:
            run += 1
        else:
            durations.setdefault(current_state, []).append(run)
            current_state = s
            run = 1
    durations.setdefault(current_state, []).append(run)
    return durations


def _write_metrics(out_dir: Path, timeframe: str, metrics_rows: List[Dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"hmm_metrics_{timeframe}.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)


def _write_outputs(
    symbol: str,
    timeframe: str,
    profile: str,
    df: pd.DataFrame,
    X_std: np.ndarray,
    scaler: StandardScaler,
    train_start: str | None,
    train_end: str | None,
    out_dir: Path,
    metrics_rows: List[Dict[str, object]],
    best: Dict[str, object],
) -> None:
    _write_metrics(out_dir, timeframe, metrics_rows)

    scaler_path = out_dir / f"scaler_{timeframe}.json"
    scaler_payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "profile": profile,
        "features": FEATURES,
        "means": scaler.mean_.tolist(),
        "stds": np.sqrt(scaler.var_).tolist(),
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    selected_path = out_dir / f"hmm_selected_{timeframe}.json"
    selected_payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "profile": profile,
        "K_best": best["K"],
        "features": FEATURES,
        "scaling": {
            "method": "StandardScaler",
            "params_file": scaler_path.name,
            "train_start": train_start,
            "train_end": train_end,
        },
        "random_seed": best["seed"],
        "covariance_type": best["covariance_type"],
        "min_covar": best.get("min_covar"),
        "n_iter": best["n_iter"],
        "tol": best["tol"],
        "restarts": best["restarts"],
        "converged": best["converged"],
        "valid_model": best.get("valid_model"),
        "selected_valid": best.get("valid_model"),
        "min_expected_dur": best.get("min_expected_dur"),
        "max_expected_dur": best.get("max_expected_dur"),
        "dominant_state_pct": best.get("dominant_state_pct"),
        "training_range": {
            "start": str(df["timestamp"].iloc[0]) if len(df) else None,
            "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        },
    }
    selected_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    model: GaussianHMM = best["model"]
    post = model.predict_proba(X_std)
    viterbi = model.predict(X_std)

    if np.isnan(post).any():
        raise ValueError("Posterior contains NaNs.")
    row_sums = post.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        print(f"[{symbol}] Warning: posterior row sums deviate by {max_dev:.6f}")

    regimes = pd.DataFrame({"timestamp": df["timestamp"], "state": viterbi})
    regimes["profile"] = profile
    for i in range(post.shape[1]):
        regimes[f"p_state_{i}"] = post[:, i]
    for c in FEATURES:
        regimes[c] = df[c].values

    regimes_path = out_dir / f"regimes_{timeframe}.csv"
    regimes.to_csv(regimes_path, index=False)

    A = model.transmat_
    if not np.allclose(A.sum(axis=1), 1.0, atol=1e-6):
        max_dev = float(np.max(np.abs(A.sum(axis=1) - 1.0)))
        print(f"[{symbol}] Warning: transition rows deviate by {max_dev:.6f}")

    expected_durations = 1.0 / (1.0 - np.diag(A))
    trans_df = pd.DataFrame(A, columns=[f"to_{i}" for i in range(A.shape[1])])
    trans_df.insert(0, "from_state", [f"{i}" for i in range(A.shape[0])])
    trans_df["expected_duration"] = expected_durations
    trans_path = out_dir / f"transition_matrix_{timeframe}.csv"
    trans_df.to_csv(trans_path, index=False)

    durations = _state_durations(viterbi)
    rows = []
    for i in range(A.shape[0]):
        runs = durations.get(i, [])
        avg_dur = float(np.mean(runs)) if runs else 0.0
        med_dur = float(np.median(runs)) if runs else 0.0

        state_mask = viterbi == i
        count = int(state_mask.sum())
        pct_time = count / len(viterbi) if len(viterbi) else 0.0

        feature_stats = {}
        for c in FEATURES:
            vals = df.loc[state_mask, c].values
            feature_stats[f"mean_{c}"] = float(np.mean(vals)) if len(vals) else 0.0
            feature_stats[f"std_{c}"] = float(np.std(vals, ddof=0)) if len(vals) else 0.0

        rows.append(
            {
                "state": i,
                "count": count,
                "pct_time": pct_time,
                "avg_duration": avg_dur,
                "median_duration": med_dur,
                "expected_duration": float(expected_durations[i]),
                **feature_stats,
            }
        )

    summary_path = out_dir / f"state_summary_{timeframe}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    if all(r["median_duration"] <= 2 for r in rows):
        print(f"[{symbol}] Warning: regimes look too choppy (median duration <= 2).")


def process_symbol(
    path_in: Path,
    out_root: Path,
    timeframe: str,
    profile: str,
    K_list: List[int],
    covariance_type: str,
    min_covar: float,
    n_iter: int,
    tol: float,
    seed: int,
    restarts: int,
    train_start: str | None,
    train_end: str | None,
    min_expected_duration_hours: float,
    max_expected_duration_hours: float | None,
    max_dominant_state_pct: float,
) -> bool:
    symbol = path_in.name.replace(f"_obs_{timeframe}.csv", "")
    df = _load_obs(path_in)

    n_obs = len(df)
    print(
        f"[{symbol}][{profile}] n_obs={n_obs:,} range={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
    )

    train_mask = _train_mask(df["timestamp"], train_start, train_end)
    if not train_mask.any():
        raise ValueError(f"[{symbol}] train range produced 0 rows.")

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURES].values)
    X_std = scaler.transform(df[FEATURES].values)

    if np.isnan(X_std).any():
        raise ValueError(f"[{symbol}] NaNs found in standardized matrix.")

    out_dir = out_root / profile / symbol
    _data_health_report(symbol, profile, timeframe, df, X_std, out_dir)

    metrics_rows = []
    failures_by_k: Dict[int, List[Dict[str, object]]] = {}
    results: List[Dict[str, object]] = []

    D = len(FEATURES)
    tf_hours = _timeframe_to_hours(timeframe)
    enforce_max_dur = max_expected_duration_hours is not None
    if not enforce_max_dur:
        print(f"[{symbol}][{profile}] NOTE: max-duration gate disabled when not set")
    for K in K_list:
        model, loglik, converged, failures = _fit_best_hmm(
            X_std, K, covariance_type, n_iter, tol, seed, restarts, min_covar
        )
        n_params = _n_params(K, D, covariance_type)
        if model is None or not np.isfinite(loglik):
            metrics_rows.append(
                {
                    "symbol": symbol,
                    "profile": profile,
                    "timeframe": timeframe,
                    "K": K,
                    "n_obs": n_obs,
                    "n_features": D,
                    "loglik": np.nan,
                    "n_params": n_params,
                    "bic": np.nan,
                    "converged": False,
                    "status": "failed",
                    "min_expected_dur": np.nan,
                    "max_expected_dur": np.nan,
                    "dominant_state_pct": np.nan,
                    "valid_model": False,
                    "n_failed_restarts": len(failures),
                }
            )
            failures_by_k[K] = failures
            print(f"[{symbol}][{profile}] K={K} failed (all restarts).")
            continue

        loglik = float(loglik)
        bic = float(-2.0 * loglik + n_params * np.log(n_obs))
        A = model.transmat_
        expected_dur_bars = 1.0 / (1.0 - np.diag(A))
        expected_dur_hours = expected_dur_bars * tf_hours
        min_dur = float(np.min(expected_dur_hours))
        max_dur = float(np.max(expected_dur_hours))

        viterbi = model.predict(X_std)
        dominant_pct = (
            float(np.max(np.bincount(viterbi)) / len(viterbi)) if len(viterbi) else 1.0
        )

        valid_model = min_dur >= min_expected_duration_hours and dominant_pct <= max_dominant_state_pct
        if enforce_max_dur:
            valid_model = valid_model and max_dur <= max_expected_duration_hours

        metrics_rows.append(
            {
                "symbol": symbol,
                "profile": profile,
                "timeframe": timeframe,
                "K": K,
                "n_obs": n_obs,
                "n_features": D,
                "loglik": loglik,
                "n_params": n_params,
                "bic": bic,
                "converged": converged,
                "status": "ok",
                "min_expected_dur": min_dur,
                "max_expected_dur": max_dur,
                "dominant_state_pct": dominant_pct,
                "valid_model": valid_model,
                "n_failed_restarts": len(failures),
            }
        )

        dur_list = ", ".join([f"{v:.2f}" for v in expected_dur_hours])
        print(
            f"[{symbol}][{profile}] K={K} loglik={loglik:.2f} bic={bic:.2f} "
            f"converged={converged} durations_hours=[{dur_list}] "
            f"min_dur={min_dur:.2f} max_dur={max_dur:.2f} "
            f"dom={dominant_pct:.3f} valid={valid_model}"
        )

        results.append(
            {
                "bic": bic,
                "K": K,
                "model": model,
                "loglik": loglik,
                "converged": converged,
                "seed": seed,
                "covariance_type": covariance_type,
                "min_covar": min_covar,
                "n_iter": n_iter,
                "tol": tol,
                "restarts": restarts,
                "valid_model": valid_model,
                "min_expected_dur": min_dur,
                "max_expected_dur": max_dur,
                "dominant_state_pct": dominant_pct,
            }
        )

    if results:
        valid_results = [r for r in results if r["valid_model"]]
        if valid_results:
            best = min(valid_results, key=lambda r: r["bic"])
        else:
            best = min(results, key=lambda r: r["bic"])
            print(f"[{symbol}][{profile}] WARNING: all models failed sanity gates; using lowest BIC.")
    else:
        best = None

    best_model = best["model"] if best is not None else None
    if failures_by_k:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure_path = out_dir / f"failures_{timeframe}.txt"
        lines = []
        for K, failures in failures_by_k.items():
            seeds = [f.get("seed") for f in failures]
            lines.append(f"K={K} seeds={seeds}")
            for f in failures:
                lines.append(f"  seed={f.get('seed')} error={f.get('error')}")
        failure_path.write_text("\n".join(lines), encoding="utf-8")

    if best_model is None:
        _write_metrics(out_dir, timeframe, metrics_rows)
        print(f"[{symbol}][{profile}] No successful K; skipping outputs.")
        return False

    if not best.get("valid_model", False):
        invalid_path = out_dir / "INVALID_MODEL.txt"
        invalid_path.write_text(
            "All models failed sanity gates; using lowest BIC fallback.\n",
            encoding="utf-8",
        )

    A = best_model.transmat_
    tl = A[0, 0] if A.size else np.nan
    expected_dur = 1.0 / (1.0 - np.diag(A)) if A.size else np.array([])
    expected_dur_hours = expected_dur * tf_hours
    ed_full = ", ".join([f"{v:.2f}" for v in expected_dur_hours])
    dominant_pct = best.get("dominant_state_pct", np.nan)

    print(
        f"[{symbol}][{profile}] selected K={best['K']} bic={best['bic']:.2f} "
        f"valid={best.get('valid_model')} min_dur={best.get('min_expected_dur'):.2f} "
        f"max_dur={best.get('max_expected_dur'):.2f} dom={best.get('dominant_state_pct'):.3f}"
    )
    if A.size:
        print(
            f"[{symbol}][{profile}] A00={tl:.4f} durations_hours=[{ed_full}] "
            f"dominant_pct={dominant_pct:.3f} valid={best.get('valid_model')}"
        )
    else:
        print(
            f"[{symbol}][{profile}] durations_hours=[{ed_full}] "
            f"dominant_pct={dominant_pct:.3f} valid={best.get('valid_model')}"
        )

    _write_outputs(
        symbol=symbol,
        timeframe=timeframe,
        profile=profile,
        df=df,
        X_std=X_std,
        scaler=scaler,
        train_start=train_start,
        train_end=train_end,
        out_dir=out_dir,
        metrics_rows=metrics_rows,
        best=best,
    )
    return bool(best.get("valid_model", False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obs-dir", default="data/obs", help="Directory of obs CSVs.")
    p.add_argument("--out-dir", default="data/regimes/1h", help="Output root directory.")
    p.add_argument("--timeframe", default="1h", help="Timeframe suffix (default: 1h).")
    p.add_argument("--symbols", nargs="*", default=None, help="Symbols to process.")
    p.add_argument(
        "--symbols-file",
        default=None,
        help="Path to a newline-delimited symbols file (e.g. config/symbols.txt).",
    )
    p.add_argument("--K", nargs="*", type=int, default=[2, 3, 4], help="K values.")
    p.add_argument("--covariance-type", default="diag", choices=["full", "diag"])
    p.add_argument("--min-covar", type=float, default=1e-4)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--train-start", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--profile", default="both", choices=["micro", "macro", "both"])
    p.add_argument("--min-expected-duration-hours", type=float, default=None)
    p.add_argument("--max-expected-duration-hours", type=float, default=None)
    p.add_argument("--max-dominant-state-pct", type=float, default=None)
    p.add_argument("--auto-fallback", default="none", choices=["none", "macro"])

    args = p.parse_args()

    obs_dir = Path(args.obs_dir)
    if not obs_dir.exists():
        raise SystemExit(f"Obs dir not found: {obs_dir.resolve()}")

    print(f"Scanning obs directory: {obs_dir.resolve()}")

    suffix = f"_obs_{args.timeframe}.csv"
    pattern = re.compile(rf"^(?P<symbol>.+){re.escape(suffix)}$")

    if args.symbols_file or args.symbols:
        requested = args.symbols or _load_symbols_file(Path(args.symbols_file))
        candidates = [obs_dir / f"{s}{suffix}" for s in requested]
        found = [p for p in candidates if p.exists()]
        missing = [p.name.replace(suffix, "") for p in candidates if not p.exists()]
        _print_symbol_summary(
            requested,
            [p.name.replace(suffix, "") for p in found],
            missing,
        )
        paths = []
        for path_in in found:
            match = pattern.fullmatch(path_in.name)
            if not match:
                continue
            symbol = match.group("symbol")
            if "_obs" in symbol:
                continue
            paths.append(path_in)
    else:
        files = [p for p in obs_dir.iterdir() if p.is_file()]
        paths = []
        for path_in in files:
            match = pattern.fullmatch(path_in.name)
            if not match:
                continue
            symbol = match.group("symbol")
            if "_obs" in symbol:
                continue
            paths.append(path_in)
        print(f"Found {len(paths)} candidate obs files for timeframe {args.timeframe}.")
        print(f"Accepted {len(paths)} obs files after filtering.")

    if not paths:
        raise SystemExit(f"No *_obs_{args.timeframe}.csv files found in {obs_dir.resolve()}")

    macro_min_default = 12.0 if args.timeframe == "4h" else 24.0
    profile_settings = {
        "micro": {
            "min_expected_duration_hours": 2.0,
            "max_expected_duration_hours": 24 * 14,
            "max_dominant_state_pct": 0.95,
        },
        "macro": {
            "min_expected_duration_hours": macro_min_default,
            "max_expected_duration_hours": None,
            "max_dominant_state_pct": 0.95,
        },
    }

    out_root = _normalize_out_root(Path(args.out_dir), args.timeframe)
    profiles = ["micro", "macro"] if args.profile == "both" else [args.profile]
    for path_in in paths:
        ran_macro = False
        micro_valid = None
        for profile in profiles:
            settings = profile_settings[profile]
            min_dur = (
                args.min_expected_duration_hours
                if args.min_expected_duration_hours is not None
                else settings["min_expected_duration_hours"]
            )
            max_dur = (
                args.max_expected_duration_hours
                if args.max_expected_duration_hours is not None
                else settings["max_expected_duration_hours"]
            )
            max_dom = (
                args.max_dominant_state_pct
                if args.max_dominant_state_pct is not None
                else settings["max_dominant_state_pct"]
            )

            valid = process_symbol(
                path_in=path_in,
                out_root=out_root,
                timeframe=args.timeframe,
                profile=profile,
                K_list=args.K,
                covariance_type=args.covariance_type,
                min_covar=args.min_covar,
                n_iter=args.n_iter,
                tol=args.tol,
                seed=args.seed,
                restarts=args.restarts,
                train_start=args.train_start,
                train_end=args.train_end,
                min_expected_duration_hours=min_dur,
                max_expected_duration_hours=max_dur,
                max_dominant_state_pct=max_dom,
            )
            if profile == "macro":
                ran_macro = True
            if profile == "micro":
                micro_valid = valid

        if args.auto_fallback == "macro" and micro_valid is False and not ran_macro:
            symbol = path_in.name.split("_obs_")[0]
            print(f"[{symbol}] micro invalid; ran macro fallback")
            settings = profile_settings["macro"]
            process_symbol(
                path_in=path_in,
                out_root=out_root,
                timeframe=args.timeframe,
                profile="macro",
                K_list=args.K,
                covariance_type=args.covariance_type,
                min_covar=args.min_covar,
                n_iter=args.n_iter,
                tol=args.tol,
                seed=args.seed,
                restarts=args.restarts,
                train_start=args.train_start,
                train_end=args.train_end,
                min_expected_duration_hours=(
                    args.min_expected_duration_hours
                    if args.min_expected_duration_hours is not None
                    else settings["min_expected_duration_hours"]
                ),
                max_expected_duration_hours=(
                    args.max_expected_duration_hours
                    if args.max_expected_duration_hours is not None
                    else settings["max_expected_duration_hours"]
                ),
                max_dominant_state_pct=(
                    args.max_dominant_state_pct
                    if args.max_dominant_state_pct is not None
                    else settings["max_dominant_state_pct"]
                ),
            )


if __name__ == "__main__":
    main()
