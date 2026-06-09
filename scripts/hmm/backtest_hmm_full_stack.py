from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


BASIC_FEATURES = ["r", "abs_r", "vol20", "vol_z"]
ENHANCED_FEATURES = ["r", "abs_r", "vol20", "vol_z", "funding_z", "volume_z", "price_vs_ema"]


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "open_time", "time", "date", "datetime"]:
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")


def _stats(r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
    eq = (1.0 + x).cumprod()
    years = len(x) / 252.0 if len(x) else np.nan
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else np.nan
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def _load_ohlcv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    ts = _timestamp_column(d)
    d = d.rename(columns={ts: "timestamp"}).copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            raise ValueError(f"{path} missing required column: {col}")
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=["timestamp", "close", "volume"]).sort_values("timestamp")


def _load_funding(path: Path, timeframe: str) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False)
    ts = _timestamp_column(p)
    p = p.rename(columns={ts: "timestamp"}).copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    if "funding_rate" not in p.columns:
        return pd.DataFrame(columns=["timestamp", "funding_z"])
    p["funding_rate"] = pd.to_numeric(p["funding_rate"], errors="coerce")
    p = p.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    f = p["funding_rate"].resample(timeframe).mean().to_frame("funding_rate")
    mu = f["funding_rate"].rolling(60, min_periods=20).mean()
    sd = f["funding_rate"].rolling(60, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    f["funding_z"] = ((f["funding_rate"] - mu) / sd).replace([np.inf, -np.inf], np.nan)
    return f[["funding_z"]].reset_index()


def build_enhanced_obs(price_csv: Path, funding_csv: Path, timeframe: str, out_csv: Path) -> pd.DataFrame:
    d = _load_ohlcv(price_csv).set_index("timestamp")
    log_close = np.log(d["close"])
    obs = pd.DataFrame(index=d.index)
    obs["r"] = log_close.diff()
    obs["abs_r"] = obs["r"].abs()
    obs["vol20"] = obs["r"].rolling(20, min_periods=20).std()
    vol_mu = obs["vol20"].rolling(252, min_periods=50).mean()
    vol_sd = obs["vol20"].rolling(252, min_periods=50).std(ddof=0).replace(0.0, np.nan)
    obs["vol_z"] = ((obs["vol20"] - vol_mu) / vol_sd).replace([np.inf, -np.inf], np.nan)
    vol_mean = d["volume"].rolling(480 if timeframe == "1h" else 60, min_periods=20).mean()
    vol_std = d["volume"].rolling(480 if timeframe == "1h" else 60, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    obs["volume_z"] = ((d["volume"] - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)
    ema21 = d["close"].ewm(span=21, adjust=False).mean()
    obs["price_vs_ema"] = ((d["close"] - ema21) / ema21).replace([np.inf, -np.inf], np.nan)

    funding = _load_funding(funding_csv, timeframe).set_index("timestamp")
    obs = obs.merge(funding, left_index=True, right_index=True, how="left")
    obs["funding_z"] = obs["funding_z"].ffill().fillna(0.0)
    obs = obs.reset_index().dropna().sort_values("timestamp")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    obs[["timestamp", *ENHANCED_FEATURES]].to_csv(out_csv, index=False)
    return obs[["timestamp", *ENHANCED_FEATURES]].copy()


def _fit_hmm(obs: pd.DataFrame, features: list[str], k: int) -> pd.DataFrame:
    xdf = obs.dropna(subset=features).copy().sort_values("timestamp")
    x = xdf[features].to_numpy(dtype=float)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    best = None
    best_score = -np.inf
    for seed in [11, 23, 42]:
        model = GaussianHMM(
            n_components=int(k),
            covariance_type="diag",
            n_iter=500,
            min_covar=1e-3,
            random_state=seed,
        )
        model.fit(xs)
        score = float(model.score(xs))
        if score > best_score:
            best = model
            best_score = score
    assert best is not None
    xdf["state"] = best.predict(xs)
    return xdf


def _map_states_to_regimes(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_stats = (
        df.groupby("state")
        .agg(avg_ret=("r", "mean"), avg_vol=("vol20", "mean"), count=("state", "size"))
        .reset_index()
        .sort_values("avg_ret")
    )
    labels: dict[int, str] = {}
    labels[int(state_stats.iloc[0]["state"])] = "BEAR"
    labels[int(state_stats.iloc[-1]["state"])] = "BULL"
    for s in state_stats["state"]:
        labels.setdefault(int(s), "CHOP")
    if int(k) == 2:
        # no CHOP state; keep highest-return BULL and lowest-return BEAR.
        pass
    out = df.copy()
    out["hmm_regime"] = out["state"].map(labels)
    state_stats["mapped_regime"] = state_stats["state"].map(lambda s: labels[int(s)])
    return out, state_stats


def build_hmm_daily_regime(obs_csv: Path, features: list[str], k: int, out_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(obs_csv)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True, errors="coerce")
    hmm = _fit_hmm(obs, features, k)
    hmm, state_stats = _map_states_to_regimes(hmm, k)
    hmm["day"] = hmm["timestamp"].dt.floor("D")
    daily = (
        hmm.groupby("day")["hmm_regime"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .sort_values("day")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    return daily, state_stats


def _run_backtest(repo: Path, name: str, start: str, end: str, out_dir: Path, hmm_csv: Path | None = None) -> dict[str, object]:
    summary = out_dir / f"{name}_summary.csv"
    daily = out_dir / f"{name}_daily.csv"
    cmd = [
        sys.executable,
        "scripts/backtest_eth_btc_portfolio.py",
        "--start",
        start,
        "--end",
        end,
        "--cost-bps",
        "20",
        "--cost-mode",
        "weight_change",
        "--allocation-mode",
        "signal_weighted",
        "--gross-cap",
        "0.8",
        "--vol-filter",
        "--transition-momentum",
        "--asymmetric-sizing",
        "--out-summary-csv",
        str(summary),
        "--out-daily-csv",
        str(daily),
    ]
    if hmm_csv is not None:
        cmd += ["--regime-source", "hmm", "--hmm-regime-csv", str(hmm_csv)]
    subprocess.run(cmd, cwd=repo, check=True)
    row = pd.read_csv(summary).iloc[0].to_dict()
    return {
        "configuration": name,
        "cagr": float(row["combined_cagr"]),
        "sharpe": float(row["combined_sharpe"]),
        "maxdd": float(row["combined_max_dd"]),
        "summary_csv": str(summary),
        "daily_csv": str(daily),
    }


def _disagreement(best_daily_csv: Path, ema_daily_csv: Path, out_csv: Path) -> pd.DataFrame:
    hmm = pd.read_csv(best_daily_csv)
    hmm["day"] = pd.to_datetime(hmm["day"], utc=True, errors="coerce").dt.floor("D")
    ema = pd.read_csv(ema_daily_csv)
    day_col = "day" if "day" in ema.columns else "timestamp"
    ema["day"] = pd.to_datetime(ema[day_col], utc=True, errors="coerce").dt.floor("D")
    regime_col = "regime_v2" if "regime_v2" in ema.columns else "regime"
    price_col = "close" if "close" in ema.columns else "eth_close" if "eth_close" in ema.columns else None
    if price_col is None:
        return pd.DataFrame()
    ema["ema_regime"] = ema[regime_col].astype(str).str.upper()
    ema["close"] = pd.to_numeric(ema[price_col], errors="coerce")
    d = ema[["day", "ema_regime", "close"]].merge(hmm[["day", "hmm_regime"]], on="day", how="inner")
    d["fwd_5d_ret"] = d["close"].shift(-5) / d["close"] - 1.0
    rows = []
    for label, mask in {
        "agree": d["hmm_regime"].eq(d["ema_regime"]),
        "hmm_bull_ema_bear": d["hmm_regime"].eq("BULL") & d["ema_regime"].eq("BEAR"),
        "hmm_bear_ema_bull": d["hmm_regime"].eq("BEAR") & d["ema_regime"].eq("BULL"),
        "hmm_chop_ema_directional": d["hmm_regime"].eq("CHOP") & d["ema_regime"].isin(["BULL", "BEAR"]),
    }.items():
        sub = d.loc[mask, "fwd_5d_ret"].dropna()
        rows.append({"case": label, "n": int(len(sub)), "avg_eth_next_5d": float(sub.mean()) if len(sub) else np.nan})
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fair HMM regime replacement test for full ETH+BTC stack.")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-csv", default="data/ETHUSDC_1h.csv")
    ap.add_argument("--basic-obs-csv", default="data/obs/ETHUSDC_obs_1h.csv")
    ap.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_6y_gapfilled.csv")
    ap.add_argument("--ema-regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--out-dir", default="artifacts/backtest/hmm_fair")
    ap.add_argument("--out-summary", default="artifacts/backtest/hmm_fair_comparison.csv")
    args = ap.parse_args()

    repo = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enhanced_obs_csv = Path("data/obs/ETHUSDC_1h_enhanced_obs.csv")
    build_enhanced_obs(Path(args.price_csv), Path(args.funding_csv), args.timeframe, enhanced_obs_csv)

    regime_rows = []
    state_rows = []
    regime_csvs: dict[str, Path] = {}
    for feature_set, obs_csv, features in [
        ("basic", Path(args.basic_obs_csv), BASIC_FEATURES),
        ("enhanced", enhanced_obs_csv, ENHANCED_FEATURES),
    ]:
        for k in [2, 3, 4]:
            name = f"hmm_K{k}_{feature_set}"
            regime_csv = out_dir / f"{name}_daily_regime.csv"
            _, state_stats = build_hmm_daily_regime(obs_csv, features, k, regime_csv)
            regime_csvs[name] = regime_csv
            for _, row in state_stats.iterrows():
                state_rows.append(
                    {
                        "configuration": name,
                        "state": int(row["state"]),
                        "avg_ret": float(row["avg_ret"]),
                        "avg_vol": float(row["avg_vol"]),
                        "count": int(row["count"]),
                        "mapped_regime": str(row["mapped_regime"]),
                    }
                )
            print(f"\n{name} state characterisation")
            print(state_stats.to_string(index=False))

    results = [_run_backtest(repo, "ema_full_stack", args.start, args.end, out_dir)]
    for name, regime_csv in regime_csvs.items():
        results.append(_run_backtest(repo, name, args.start, args.end, out_dir, regime_csv))

    summary = pd.DataFrame(results)
    ema_sharpe = float(summary.loc[summary["configuration"].eq("ema_full_stack"), "sharpe"].iloc[0])
    hmm_only = summary[~summary["configuration"].eq("ema_full_stack")].copy()
    best = hmm_only.sort_values(["sharpe", "maxdd"], ascending=[False, False]).iloc[0]
    summary["vs_ema_sharpe"] = summary["sharpe"] - ema_sharpe
    summary["decision"] = np.where(
        (summary["configuration"] != "ema_full_stack")
        & (summary["vs_ema_sharpe"] > 0.05)
        & (summary["maxdd"] >= float(summary.loc[summary["configuration"].eq("ema_full_stack"), "maxdd"].iloc[0])),
        "SWITCH_CANDIDATE",
        "KEEP_EMA",
    )
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    pd.DataFrame(state_rows).to_csv(out_dir / "hmm_state_characterisation.csv", index=False)
    disag = _disagreement(
        regime_csvs[str(best["configuration"])],
        Path(args.ema_regime_csv),
        out_dir / "best_hmm_disagreement.csv",
    )

    print("\n====================================================")
    print("FAIR HMM vs EMA COMPARISON")
    print("Full strategy stack, 20bps")
    print("====================================================")
    print(summary[["configuration", "sharpe", "maxdd", "cagr", "vs_ema_sharpe", "decision"]].to_string(index=False))
    print("----------------------------------------------------")
    print(f"Best HMM config: {best['configuration']} Sharpe {float(best['sharpe']):.3f}")
    print(f"vs EMA baseline: {float(best['sharpe']) - ema_sharpe:+.3f}")
    if not disag.empty:
        print("\nBest HMM disagreement analysis:")
        print(disag.to_string(index=False))
    print("====================================================")
    print(f"Saved: {args.out_summary}")
    print(f"Saved state mapping: {out_dir / 'hmm_state_characterisation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
