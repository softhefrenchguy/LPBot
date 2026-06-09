from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = ["BULL", "CHOP", "BEAR"]


def perf(log_r: pd.Series, bar_minutes: int = 5) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(x) == 0:
        return {"ret": np.nan, "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return {
        "ret": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (bars_per_year / len(eq)) - 1.0),
        "ann_vol": float(np.std(x) * np.sqrt(bars_per_year)),
        "sharpe": float((np.mean(x) * bars_per_year) / (np.std(x) * np.sqrt(bars_per_year) + 1e-12)),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def _load_sleeve(path: Path, prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "timestamp" not in d.columns or "weight" not in d.columns or "spot_r" not in d.columns:
        raise ValueError(f"{path} must contain timestamp, weight, spot_r")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d[f"{prefix}_weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    d[f"{prefix}_spot_r"] = pd.to_numeric(d["spot_r"], errors="coerce").fillna(0.0)
    keep = ["timestamp", f"{prefix}_weight", f"{prefix}_spot_r"]
    if "close" in d.columns:
        d[f"{prefix}_close"] = pd.to_numeric(d["close"], errors="coerce")
        keep.append(f"{prefix}_close")
    if "eth_close" in d.columns:
        d[f"{prefix}_eth_close"] = pd.to_numeric(d["eth_close"], errors="coerce")
        keep.append(f"{prefix}_eth_close")
    return d[keep].drop_duplicates(subset=["timestamp"], keep="last")


def _attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bars_per_year = 365 * 24 * 12
    for source in ["regime_v2", "regime_fwd_label"]:
        for rg in REGIMES:
            d = df[df[source].astype(str) == rg].copy()
            if len(d) == 0:
                rows.append(
                    {
                        "label_source": source,
                        "regime": rg,
                        "rows": 0,
                        "strat_ret": np.nan,
                        "spot_ret": np.nan,
                        "excess_ret": np.nan,
                        "mean_excess_bps_per_bar": np.nan,
                        "strat_sharpe": np.nan,
                        "avg_weight": np.nan,
                        "time_in_market_pct": np.nan,
                    }
                )
                continue
            sr = pd.to_numeric(d["strat_r"], errors="coerce").fillna(0.0)
            br = pd.to_numeric(d["spot_r"], errors="coerce").fillna(0.0)
            ex = sr - br
            vol = float(sr.std(ddof=0) * np.sqrt(bars_per_year))
            rows.append(
                {
                    "label_source": source,
                    "regime": rg,
                    "rows": int(len(d)),
                    "strat_ret": float(np.expm1(sr.sum())),
                    "spot_ret": float(np.expm1(br.sum())),
                    "excess_ret": float(np.expm1(sr.sum()) - np.expm1(br.sum())),
                    "mean_excess_bps_per_bar": float(ex.mean() * 1e4),
                    "strat_sharpe": float((sr.mean() * bars_per_year) / (vol + 1e-12)),
                    "avg_weight": float(pd.to_numeric(d["weight_target"], errors="coerce").fillna(0.0).mean()),
                    "time_in_market_pct": float((pd.to_numeric(d["weight_target"], errors="coerce").fillna(0.0) > 1e-9).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Option-B router: continuous momentum sizing + defensive complement")
    ap.add_argument("--momentum-csv", default="artifacts/backtest/offense_momentum_only_v1_5y.csv")
    ap.add_argument("--defensive-csv", default="artifacts/backtest/direction_event_model_v1_5y_1d_hiconv.csv")
    ap.add_argument("--regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)

    ap.add_argument("--size-bull", type=float, default=1.0)
    ap.add_argument("--size-chop", type=float, default=0.5)
    ap.add_argument("--size-bear", type=float, default=0.2)
    ap.add_argument("--def-bull", type=float, default=0.0)
    ap.add_argument("--def-chop", type=float, default=0.5)
    ap.add_argument("--def-bear", type=float, default=0.8)
    ap.add_argument("--mom-bear-filter", action="store_true", help="Suppress momentum sleeve in BEAR when rolling 20d return is below threshold")
    ap.add_argument("--mom-bear-filter-th", type=float, default=-0.10)

    ap.add_argument("--out-csv", default="artifacts/backtest/router_option_b_5y.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/router_option_b_5y_summary.csv")
    ap.add_argument("--out-attrib-csv", default="artifacts/backtest/router_option_b_5y_regime_attribution.csv")
    args = ap.parse_args()

    mom = _load_sleeve(Path(args.momentum_csv), "mom")
    deff = _load_sleeve(Path(args.defensive_csv), "def")
    reg = pd.read_csv(args.regime_csv)

    reg["day"] = pd.to_datetime(reg["day"], utc=True, errors="coerce").dt.floor("D")
    reg = reg.dropna(subset=["day"])[["day", "regime_v2", "regime_fwd_label"]].drop_duplicates(subset=["day"], keep="last")

    # Strict common timestamp window.
    df = mom.merge(deff, on="timestamp", how="inner")
    df = df.sort_values("timestamp").copy()
    df["day"] = df["timestamp"].dt.floor("D")
    df = df.merge(reg, on="day", how="left")
    df["regime_v2"] = df["regime_v2"].fillna("CHOP")
    df["regime_fwd_label"] = df["regime_fwd_label"].fillna("CHOP")

    # Spot return consistency check.
    spot_diff = (df["mom_spot_r"] - df["def_spot_r"]).abs().max()

    size_map = {"BULL": float(args.size_bull), "CHOP": float(args.size_chop), "BEAR": float(args.size_bear)}
    def_map = {"BULL": float(args.def_bull), "CHOP": float(args.def_chop), "BEAR": float(args.def_bear)}
    df["mom_size"] = df["regime_v2"].map(size_map).astype(float)
    df["def_size"] = df["regime_v2"].map(def_map).astype(float)
    denom = (df["mom_size"] + df["def_size"]).replace(0.0, np.nan)

    # Canonical close field for protocol checker and signal-level filters.
    if "mom_close" in df.columns and df["mom_close"].notna().any():
        df["eth_close"] = df["mom_close"]
    elif "def_eth_close" in df.columns and df["def_eth_close"].notna().any():
        df["eth_close"] = df["def_eth_close"]
    elif "def_close" in df.columns and df["def_close"].notna().any():
        df["eth_close"] = df["def_close"]
    else:
        df["eth_close"] = np.nan

    daily_close = df.groupby("day", as_index=True)["eth_close"].last().sort_index()
    daily_ret20 = daily_close.pct_change(20)
    df["rolling_20d_return"] = df["day"].map(daily_ret20).astype(float)

    df["mom_weight_eff"] = df["mom_weight"].astype(float)
    if bool(args.mom_bear_filter):
        bear_filter = (df["regime_v2"] == "BEAR") & (df["rolling_20d_return"] < float(args.mom_bear_filter_th))
        df.loc[bear_filter, "mom_weight_eff"] = 0.0
        df["mom_bear_filter_on"] = bear_filter.astype(int)
    else:
        df["mom_bear_filter_on"] = 0

    num = df["mom_weight_eff"] * df["mom_size"] + df["def_weight"] * df["def_size"]
    df["weight_target"] = (num / denom).fillna(0.0).clip(0.0, 1.0)

    # 1x notional safeguard.
    df["total_size"] = (df["mom_size"] + df["def_size"]).astype(float)
    if float(df["weight_target"].max()) > 1.000001:
        raise ValueError("weight_target exceeded 1.0x notional")

    # Router execution and costs.
    df["spot_r"] = df["mom_spot_r"]
    df["turnover"] = df["weight_target"].diff().abs().fillna(0.0)
    cost = df["turnover"] * (float(args.trade_cost_bps) / 10000.0)
    df["cost_r"] = -cost
    df["strat_r"] = df["weight_target"].shift(1).fillna(0.0) * df["spot_r"] - cost
    df["eq"] = np.exp(np.cumsum(df["strat_r"].to_numpy()))
    df["spot_eq"] = np.exp(np.cumsum(df["spot_r"].to_numpy()))

    m = perf(df["strat_r"], 5)
    s = perf(df["spot_r"], 5)
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "start": str(df["timestamp"].iloc[0]) if len(df) else "",
                "end": str(df["timestamp"].iloc[-1]) if len(df) else "",
                "ret": m["ret"],
                "cagr": m["cagr"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "spot_ret": s["ret"],
                "excess_vs_spot": m["ret"] - s["ret"],
                "avg_weight": float(df["weight_target"].mean()),
                "time_in_market_pct": float((df["weight_target"] > 1e-9).mean() * 100.0),
                "turnover": float(df["turnover"].sum()),
                "spot_diff_max_abs": float(spot_diff),
                "max_notional_weight": float(df["weight_target"].max()),
                "mom_bear_filter_enabled": bool(args.mom_bear_filter),
                "mom_bear_filter_th": float(args.mom_bear_filter_th),
                "mom_bear_filter_pct": float(df["mom_bear_filter_on"].mean() * 100.0),
            }
        ]
    )

    attrib = _attribution(df)

    out_cols = [
        "timestamp",
        "day",
        "eth_close",
        "spot_r",
        "regime_v2",
        "regime_fwd_label",
        "mom_weight",
        "mom_weight_eff",
        "def_weight",
        "mom_size",
        "def_size",
        "total_size",
        "rolling_20d_return",
        "mom_bear_filter_on",
        "weight_target",
        "turnover",
        "cost_r",
        "strat_r",
        "eq",
        "spot_eq",
    ]
    out = df[out_cols].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    out_attr = Path(args.out_attrib_csv)
    out_attr.parent.mkdir(parents=True, exist_ok=True)
    attrib.to_csv(out_attr, index=False)

    print(f"wrote {out_csv}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_attr}")
    print(summary.round(6).to_string(index=False))
    print("\nRegime attribution:")
    print(attrib.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
