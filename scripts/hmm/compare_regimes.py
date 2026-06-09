from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

FEATURES = ["r", "abs_r", "vol20", "vol_z"]
REGIMES = ["BULL", "CHOP", "BEAR"]


def _read_obs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", *FEATURES]).sort_values("timestamp")
    return df


def _read_ema(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    dcol = "day" if "day" in df.columns else "date"
    rcol = "regime_v2" if "regime_v2" in df.columns else "regime"
    df["day"] = pd.to_datetime(df[dcol], utc=True, errors="coerce").dt.floor("D")
    df = df.dropna(subset=["day"]).sort_values("day")
    return df[["day", rcol]].rename(columns={rcol: "ema_regime"})


def _read_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    tcol = next((c for c in ["timestamp", "open_time", "date", "datetime", "time"] if c in df.columns), None)
    if tcol is None:
        raise SystemExit(f"No timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    daily = df.assign(day=df["timestamp"].dt.floor("D")).groupby("day", as_index=False)["close"].last()
    daily["fwd_5d_ret"] = daily["close"].shift(-5) / daily["close"] - 1.0
    return daily


def _map_states(daily: pd.DataFrame, k: int) -> dict[int, str]:
    stats = daily.groupby("state")["fwd_5d_ret"].mean().sort_values(ascending=False)
    ordered = list(stats.index)
    mapping: dict[int, str] = {}
    if k == 2:
        mapping[int(ordered[0])] = "BULL"
        mapping[int(ordered[-1])] = "BEAR"
    else:
        mapping[int(ordered[0])] = "BULL"
        mapping[int(ordered[-1])] = "BEAR"
        for s in ordered[1:-1]:
            mapping[int(s)] = "CHOP"
    return mapping


def _fit_hmm(obs: pd.DataFrame, k: int, covariance_type: str, seed: int) -> pd.DataFrame:
    x = obs[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = GaussianHMM(
        n_components=k,
        covariance_type=covariance_type,
        n_iter=300,
        tol=1e-4,
        min_covar=1e-3,
        random_state=seed,
    )
    model.fit(xs)
    states = model.predict(xs)
    out = obs[["timestamp", *FEATURES]].copy()
    out["state"] = states
    return out


def _daily_hmm(hmm: pd.DataFrame, price_daily: pd.DataFrame, k: int) -> pd.DataFrame:
    h = hmm.copy()
    h["day"] = h["timestamp"].dt.floor("D")
    # Daily state = modal intraday state; last mode wins only on exact tie.
    daily = h.groupby("day")["state"].agg(lambda s: int(s.value_counts().sort_values(ascending=False).index[0])).reset_index()
    daily = daily.merge(price_daily, on="day", how="left")
    mapping = _map_states(daily.dropna(subset=["fwd_5d_ret"]), k)
    daily["hmm_regime"] = daily["state"].map(mapping).fillna("CHOP")
    return daily


def _agreement_report(merged: pd.DataFrame, timeframe: str, k: int) -> dict[str, float | str | int]:
    m = merged.dropna(subset=["ema_regime", "hmm_regime"]).copy()
    m = m[m["ema_regime"].isin(REGIMES)]
    if m.empty:
        return {"timeframe": timeframe, "K": k, "n_days": 0}
    out: dict[str, float | str | int] = {
        "timeframe": timeframe,
        "K": k,
        "n_days": int(len(m)),
        "agreement_rate": float((m["ema_regime"] == m["hmm_regime"]).mean()),
    }
    for regime in REGIMES:
        sub = m[m["ema_regime"] == regime]
        out[f"ema_{regime.lower()}_days"] = int(len(sub))
        out[f"ema_{regime.lower()}_hmm_agree"] = float((sub["hmm_regime"] == regime).mean()) if len(sub) else np.nan
    cases = {
        "hmm_bear_ema_bull_fwd5": (m["hmm_regime"].eq("BEAR") & m["ema_regime"].eq("BULL")),
        "hmm_bull_ema_bear_fwd5": (m["hmm_regime"].eq("BULL") & m["ema_regime"].eq("BEAR")),
        "hmm_bear_ema_chop_fwd5": (m["hmm_regime"].eq("BEAR") & m["ema_regime"].eq("CHOP")),
        "hmm_bull_ema_chop_fwd5": (m["hmm_regime"].eq("BULL") & m["ema_regime"].eq("CHOP")),
    }
    for name, mask in cases.items():
        out[name] = float(m.loc[mask, "fwd_5d_ret"].mean()) if mask.any() else np.nan
        out[name.replace("_fwd5", "_n")] = int(mask.sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ema-regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    p.add_argument("--price-csv", default="data/ETHUSDC_1h.csv")
    p.add_argument("--obs-dir", default="data/obs")
    p.add_argument("--symbol", default="ETHUSDC")
    p.add_argument("--timeframes", nargs="*", default=["1h", "8h"])
    p.add_argument("--k-values", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--covariance-type", default="diag")
    p.add_argument("--out", default="artifacts/backtest/hmm_comparison.csv")
    args = p.parse_args()

    ema = _read_ema(Path(args.ema_regime_csv))
    price_daily = _read_price(Path(args.price_csv))
    rows = []
    detail_frames = []
    for tf in args.timeframes:
        obs_path = Path(args.obs_dir) / f"{args.symbol}_{tf}_obs.csv"
        if not obs_path.exists():
            obs_path = Path(args.obs_dir) / f"{args.symbol}_obs_{tf}.csv"
        obs = _read_obs(obs_path)
        for k in args.k_values:
            hmm = _fit_hmm(obs, k, args.covariance_type, seed=42 + k)
            daily = _daily_hmm(hmm, price_daily, k)
            merged = daily.merge(ema, on="day", how="inner")
            row = _agreement_report(merged, tf, k)
            rows.append(row)
            detail = merged[["day", "state", "hmm_regime", "ema_regime", "close", "fwd_5d_ret"]].copy()
            detail["timeframe"] = tf
            detail["K"] = k
            detail_frames.append(detail)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    details.to_csv(out_path.with_name(out_path.stem + "_daily.csv"), index=False)

    print("=" * 72)
    print("REGIME COMPARISON: HMM vs EMA")
    print("=" * 72)
    show_cols = ["timeframe", "K", "n_days", "agreement_rate", "ema_bull_hmm_agree", "ema_chop_hmm_agree", "ema_bear_hmm_agree", "hmm_bear_ema_bull_fwd5", "hmm_bull_ema_bear_fwd5"]
    print(summary[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_name(out_path.stem + '_daily.csv')}")


if __name__ == "__main__":
    main()
