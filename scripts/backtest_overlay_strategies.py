from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _stats(r: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if x.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan}
    eq = (1.0 + x).cumprod()
    years = len(x) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    ex = x - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "ann_vol": ann_vol}


def _load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    required = {"combined_return", "eth_regime", "eth_close", "eth_spot_return", "alloc_eth"}
    missing = required - set(d.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
    for col in ["combined_return", "eth_close", "eth_spot_return", "alloc_eth", "alloc_btc"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    d["eth_regime"] = d["eth_regime"].astype(str).str.upper()
    d["alloc_btc"] = pd.to_numeric(d.get("alloc_btc", 0.0), errors="coerce").fillna(0.0)
    d["gold_weight_exec"] = pd.to_numeric(d.get("gold_weight_exec", 0.0), errors="coerce").fillna(0.0)
    return d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


def _load_funding(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False)
    if "timestamp" not in p.columns or "funding_rate" not in p.columns:
        raise SystemExit(f"{path} must contain timestamp and funding_rate")
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p["funding_rate"] = pd.to_numeric(p["funding_rate"], errors="coerce")
    p["basis"] = pd.to_numeric(p.get("basis", np.nan), errors="coerce")
    p = p.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    daily = p.resample("1D").agg(funding_rate=("funding_rate", "mean"), basis=("basis", "last"))
    daily["funding_daily"] = daily["funding_rate"] * 3.0
    mu = daily["funding_rate"].rolling(60, min_periods=20).mean()
    sd = daily["funding_rate"].rolling(60, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    daily["funding_z"] = ((daily["funding_rate"] - mu) / sd).replace([np.inf, -np.inf], np.nan)
    daily["basis_slippage"] = daily["basis"].diff().abs().fillna(0.0)
    daily.index.name = "day"
    return daily.reset_index()


def _mean_reversion_overlay(
    d: pd.DataFrame,
    gross_cap: float,
    cost_bps: float,
    z_entry: float,
    z_exit: float,
    ret_entry: float,
    max_hold_days: int,
) -> pd.DataFrame:
    x = d.copy()
    close = pd.to_numeric(x["eth_close"], errors="coerce")
    roll_mean = close.rolling(20, min_periods=20).mean()
    roll_std = close.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    x["mr_z_score"] = ((close - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan)
    x["mr_daily_ret"] = close.pct_change()
    used_cap = (x["alloc_eth"] + x["alloc_btc"] + x["gold_weight_exec"]).clip(lower=0.0)
    x["mr_remaining_cap"] = (float(gross_cap) - used_cap).clip(lower=0.0)

    active = False
    held = 0
    target = []
    entry_flags = []
    exit_flags = []
    for _, row in x.iterrows():
        is_chop = str(row["eth_regime"]) == "CHOP"
        z = float(row["mr_z_score"]) if pd.notna(row["mr_z_score"]) else np.nan
        r = float(row["mr_daily_ret"]) if pd.notna(row["mr_daily_ret"]) else np.nan
        enter = (not active) and is_chop and np.isfinite(z) and np.isfinite(r) and z < float(z_entry) and r < float(ret_entry)
        exit_ = active and ((not is_chop) or (np.isfinite(z) and z > float(z_exit)) or held >= int(max_hold_days))
        if enter:
            active = True
            held = 0
        elif exit_:
            active = False
            held = 0
        target.append(float(row["mr_remaining_cap"]) if active and is_chop else 0.0)
        entry_flags.append(bool(enter))
        exit_flags.append(bool(exit_))
        if active:
            held += 1

    x["mr_weight_target"] = target
    x["mr_weight_exec"] = x["mr_weight_target"].shift(1).fillna(0.0)
    x["mr_turnover"] = (x["mr_weight_exec"] - x["mr_weight_exec"].shift(1).fillna(0.0)).abs()
    x["mr_cost"] = x["mr_turnover"] * (float(cost_bps) / 10000.0)
    x["mr_return"] = x["mr_weight_exec"] * x["eth_spot_return"].fillna(0.0) - x["mr_cost"]
    x["mr_entry"] = entry_flags
    x["mr_exit"] = exit_flags
    return x


def _funding_overlay(
    d: pd.DataFrame,
    funding: pd.DataFrame,
    cost_bps: float,
    funding_z_threshold: float,
    basis_slippage_k: float,
) -> pd.DataFrame:
    x = d.merge(funding[["day", "funding_daily", "funding_z", "basis_slippage"]], on="day", how="left")
    x["funding_daily"] = pd.to_numeric(x["funding_daily"], errors="coerce").fillna(0.0)
    x["funding_z"] = pd.to_numeric(x["funding_z"], errors="coerce")
    x["basis_slippage"] = pd.to_numeric(x["basis_slippage"], errors="coerce").fillna(0.0)

    long_spot = pd.to_numeric(x["alloc_eth"], errors="coerce").fillna(0.0) > 0.0
    regime_ok = x["eth_regime"].isin(["BULL", "CHOP"])
    high_funding = x["funding_z"] > float(funding_z_threshold)
    x["funding_weight_target"] = np.where(long_spot & regime_ok & high_funding, x["alloc_eth"], 0.0)
    x["funding_weight_exec"] = x["funding_weight_target"].shift(1).fillna(0.0)
    x["funding_turnover"] = (x["funding_weight_exec"] - x["funding_weight_exec"].shift(1).fillna(0.0)).abs()
    x["funding_cost"] = x["funding_turnover"] * (float(cost_bps) / 10000.0)
    carry = x["funding_daily"] - (float(basis_slippage_k) * x["basis_slippage"])
    x["funding_return"] = x["funding_weight_exec"] * carry - x["funding_cost"]
    return x


def _summarize(name: str, returns: pd.Series, extra: dict[str, float | int | str] | None = None) -> dict[str, float | int | str]:
    st = _stats(returns)
    row: dict[str, float | int | str] = {"configuration": name, **st}
    if extra:
        row.update(extra)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest overlays on top of the validated full-stack strategy.")
    ap.add_argument("--baseline-daily", default="artifacts/backtest/hmm_fair/ema_full_stack_daily.csv")
    ap.add_argument("--funding-csv", default="data/backtest/ETH_perp_features_5m_6y_gapfilled.csv")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--z-entry", type=float, default=-1.5)
    ap.add_argument("--z-exit", type=float, default=-0.5)
    ap.add_argument("--ret-entry", type=float, default=-0.03)
    ap.add_argument("--max-hold-days", type=int, default=10)
    ap.add_argument("--funding-z-threshold", type=float, default=1.0)
    ap.add_argument("--basis-slippage-k", type=float, default=0.25)
    ap.add_argument("--out-summary", default="artifacts/backtest/overlay_strategy_summary.csv")
    ap.add_argument("--out-daily", default="artifacts/backtest/overlay_strategy_daily.csv")
    args = ap.parse_args()

    base = _load_daily(Path(args.baseline_daily))
    funding = _load_funding(Path(args.funding_csv))

    mr = _mean_reversion_overlay(
        base,
        gross_cap=float(args.gross_cap),
        cost_bps=float(args.cost_bps),
        z_entry=float(args.z_entry),
        z_exit=float(args.z_exit),
        ret_entry=float(args.ret_entry),
        max_hold_days=int(args.max_hold_days),
    )
    fund = _funding_overlay(
        base,
        funding,
        cost_bps=float(args.cost_bps),
        funding_z_threshold=float(args.funding_z_threshold),
        basis_slippage_k=float(args.basis_slippage_k),
    )
    both = _funding_overlay(
        mr,
        funding,
        cost_bps=float(args.cost_bps),
        funding_z_threshold=float(args.funding_z_threshold),
        basis_slippage_k=float(args.basis_slippage_k),
    )

    out = base.copy()
    out["baseline_return"] = base["combined_return"]
    out["mean_reversion_overlay_return"] = mr["mr_return"]
    out["funding_overlay_return"] = fund["funding_return"]
    out["both_overlay_return"] = mr["mr_return"] + both["funding_return"]
    out["return_mean_reversion"] = out["baseline_return"] + out["mean_reversion_overlay_return"]
    out["return_funding"] = out["baseline_return"] + out["funding_overlay_return"]
    out["return_both"] = out["baseline_return"] + out["both_overlay_return"]

    rows = [
        _summarize("Full stack baseline", out["baseline_return"]),
        _summarize(
            "+ Mean-rev in CHOP",
            out["return_mean_reversion"],
            {
                "overlay_days": int((mr["mr_weight_exec"] > 0).sum()),
                "overlay_entries": int(pd.Series(mr["mr_entry"]).sum()),
                "avg_overlay_weight": float(pd.to_numeric(mr["mr_weight_exec"], errors="coerce").mean()),
            },
        ),
        _summarize(
            "+ Funding in BULL/CHOP",
            out["return_funding"],
            {
                "overlay_days": int((fund["funding_weight_exec"] > 0).sum()),
                "overlay_entries": int(((fund["funding_weight_exec"] > 0).astype(int).diff().fillna(0) > 0).sum()),
                "avg_overlay_weight": float(pd.to_numeric(fund["funding_weight_exec"], errors="coerce").mean()),
            },
        ),
        _summarize(
            "+ Both overlays",
            out["return_both"],
            {
                "overlay_days": int(((mr["mr_weight_exec"] > 0) | (both["funding_weight_exec"] > 0)).sum()),
                "overlay_entries": int(pd.Series(mr["mr_entry"]).sum() + ((both["funding_weight_exec"] > 0).astype(int).diff().fillna(0) > 0).sum()),
                "avg_overlay_weight": float((pd.to_numeric(mr["mr_weight_exec"], errors="coerce").fillna(0.0) + pd.to_numeric(both["funding_weight_exec"], errors="coerce").fillna(0.0)).mean()),
            },
        ),
    ]
    summary = pd.DataFrame(rows)
    best = summary.sort_values(["sharpe", "maxdd"], ascending=[False, False]).iloc[0]

    out_path = Path(args.out_daily)
    summary_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("================================================")
    print("OVERLAY STRATEGY RESULTS")
    print("Full stack baseline + overlays")
    print("================================================")
    print("Config                    Sharpe  MaxDD    CAGR")
    print("------------------------------------------------")
    for _, row in summary.iterrows():
        print(
            f"{str(row['configuration'])[:24]:<24}"
            f"{float(row['sharpe']):>7.3f} "
            f"{float(row['maxdd']) * 100:>7.2f}% "
            f"{float(row['cagr']) * 100:>6.2f}%"
        )
    print("------------------------------------------------")
    print(f"Best config: {best['configuration']} (Sharpe {float(best['sharpe']):.3f})")
    print("================================================")
    print(f"Saved: {summary_path}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
