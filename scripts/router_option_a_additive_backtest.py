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


def load_sleeve(path: Path, prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {"timestamp", "weight", "spot_r"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path} missing required columns: {sorted(list(need - set(d.columns)))}")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d[f"{prefix}_weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(0.0)
    d[f"{prefix}_spot_r"] = pd.to_numeric(d["spot_r"], errors="coerce").fillna(0.0)
    keep = ["timestamp", f"{prefix}_weight", f"{prefix}_spot_r"]
    if "close" in d.columns:
        d[f"{prefix}_close"] = pd.to_numeric(d["close"], errors="coerce")
        keep.append(f"{prefix}_close")
    if "eth_close" in d.columns:
        d[f"{prefix}_eth_close"] = pd.to_numeric(d["eth_close"], errors="coerce")
        keep.append(f"{prefix}_eth_close")
    return d.dropna(subset=["timestamp"]).sort_values("timestamp")[keep].drop_duplicates(subset=["timestamp"], keep="last")


def regime_attribution(df: pd.DataFrame) -> pd.DataFrame:
    bars_per_year = 365 * 24 * 12
    rows = []
    for source in ["regime_v2", "regime_fwd_label"]:
        for rg in REGIMES:
            d = df[df[source].astype(str) == rg].copy()
            if d.empty:
                rows.append(
                    {
                        "label_source": source,
                        "regime": rg,
                        "rows": 0,
                        "strategy_ret": np.nan,
                        "spot_ret": np.nan,
                        "excess": np.nan,
                        "sharpe": np.nan,
                        "bps_per_bar": np.nan,
                        "cap_bind_pct": np.nan,
                    }
                )
                continue
            sr = d["strat_r"]
            br = d["spot_r"]
            vol = float(sr.std(ddof=0) * np.sqrt(bars_per_year))
            rows.append(
                {
                    "label_source": source,
                    "regime": rg,
                    "rows": int(len(d)),
                    "strategy_ret": float(np.expm1(sr.sum())),
                    "spot_ret": float(np.expm1(br.sum())),
                    "excess": float(np.expm1(sr.sum()) - np.expm1(br.sum())),
                    "sharpe": float((sr.mean() * bars_per_year) / (vol + 1e-12)),
                    "bps_per_bar": float((sr - br).mean() * 1e4),
                    "cap_bind_pct": float(d["cap_bind"].mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Option A additive router: ungated momentum + regime-scaled defensive sleeve")
    ap.add_argument("--momentum-csv", default="artifacts/backtest/offense_momentum_only_v1_5y.csv")
    ap.add_argument("--defensive-csv", default="artifacts/backtest/direction_event_model_v1_5y_1d_hiconv.csv")
    ap.add_argument("--regime-csv", default="artifacts/backtest/regime_classifier_v2_daily.csv")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)

    ap.add_argument("--mom-base-size", type=float, default=1.0)
    ap.add_argument("--off-bull", type=float, default=np.nan, help="Optional offensive sleeve scale in BULL")
    ap.add_argument("--off-chop", type=float, default=np.nan, help="Optional offensive sleeve scale in CHOP")
    ap.add_argument("--off-bear", type=float, default=np.nan, help="Optional offensive sleeve scale in BEAR")
    ap.add_argument("--def-bull", type=float, default=0.0)
    ap.add_argument("--def-chop", type=float, default=0.4)
    ap.add_argument("--def-bear", type=float, default=1.0)

    ap.add_argument("--out-csv", default="artifacts/backtest/router_option_a_additive_5y.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/router_option_a_additive_5y_summary.csv")
    ap.add_argument("--out-attrib-csv", default="artifacts/backtest/router_option_a_additive_5y_regime_attribution.csv")
    args = ap.parse_args()

    mom = load_sleeve(Path(args.momentum_csv), "mom")
    deff = load_sleeve(Path(args.defensive_csv), "def")
    reg = pd.read_csv(args.regime_csv)
    reg["day"] = pd.to_datetime(reg["day"], utc=True, errors="coerce").dt.floor("D")
    reg = reg.dropna(subset=["day"])[["day", "regime_v2", "regime_fwd_label"]].drop_duplicates(subset=["day"], keep="last")

    df = mom.merge(deff, on="timestamp", how="inner").sort_values("timestamp").copy()
    df["day"] = df["timestamp"].dt.floor("D")
    df = df.merge(reg, on="day", how="left")
    df["regime_v2"] = df["regime_v2"].fillna("CHOP")
    df["regime_fwd_label"] = df["regime_fwd_label"].fillna("CHOP")

    spot_diff = float((df["mom_spot_r"] - df["def_spot_r"]).abs().max())
    df["spot_r"] = df["mom_spot_r"]

    def_map = {"BULL": float(args.def_bull), "CHOP": float(args.def_chop), "BEAR": float(args.def_bear)}
    if np.isfinite(args.off_bull) and np.isfinite(args.off_chop) and np.isfinite(args.off_bear):
        off_map = {"BULL": float(args.off_bull), "CHOP": float(args.off_chop), "BEAR": float(args.off_bear)}
        df["off_scale"] = df["regime_v2"].map(off_map).astype(float)
    else:
        df["off_scale"] = float(args.mom_base_size)
    df["mom_component"] = df["mom_weight"] * df["off_scale"]
    df["def_scale"] = df["regime_v2"].map(def_map).astype(float)
    df["def_component"] = df["def_weight"] * df["def_scale"]
    df["weight_raw"] = df["mom_component"] + df["def_component"]
    df["cap_bind"] = (df["weight_raw"] > 1.0).astype(int)
    df["weight_target"] = df["weight_raw"].clip(0.0, 1.0)

    if "mom_close" in df.columns and df["mom_close"].notna().any():
        df["eth_close"] = df["mom_close"]
    elif "def_eth_close" in df.columns and df["def_eth_close"].notna().any():
        df["eth_close"] = df["def_eth_close"]
    elif "def_close" in df.columns and df["def_close"].notna().any():
        df["eth_close"] = df["def_close"]
    else:
        df["eth_close"] = np.nan

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
                "start": str(df["timestamp"].iloc[0]),
                "end": str(df["timestamp"].iloc[-1]),
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
                "cap_bind_pct": float(df["cap_bind"].mean() * 100.0),
                "spot_diff_max_abs": spot_diff,
                "max_notional_weight": float(df["weight_target"].max()),
            }
        ]
    )

    attrib = regime_attribution(df)

    out_cols = [
        "timestamp",
        "day",
        "eth_close",
        "spot_r",
        "regime_v2",
        "regime_fwd_label",
        "mom_weight",
        "def_weight",
        "off_scale",
        "def_scale",
        "mom_component",
        "def_component",
        "weight_raw",
        "cap_bind",
        "weight_target",
        "turnover",
        "cost_r",
        "strat_r",
        "eq",
        "spot_eq",
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out_csv, index=False)

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
