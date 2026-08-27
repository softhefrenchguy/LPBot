from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_overlay_strategies import _mean_reversion_overlay


VARIANTS = [
    ("A current", "current"),
    ("B relax_z", "relax_z"),
    ("C z_only", "z_only"),
    ("D RSI", "rsi"),
    ("E combined", "combined"),
]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _stats(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ann_vol": np.nan, "return": np.nan}
    eq = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=0) * np.sqrt(252.0))
    ex = r - (0.05 / 252.0)
    sd = float(ex.std(ddof=0))
    sharpe = float(ex.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else np.nan
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "ann_vol": ann_vol, "return": float(eq.iloc[-1] - 1.0)}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _trade_metrics(d: pd.DataFrame, weight_col: str, ret_col: str) -> dict[str, float | int]:
    active = pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0) > 0
    returns = pd.to_numeric(d[ret_col], errors="coerce").fillna(0.0)
    trades: list[float] = []
    holds: list[int] = []
    in_trade = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_trade:
            start = i
            in_trade = True
        if in_trade and ((not on) or i == len(active) - 1):
            end = i - 1 if not on else i
            tr = float((1.0 + returns.iloc[start : end + 1]).prod() - 1.0)
            trades.append(tr)
            holds.append(end - start + 1)
            in_trade = False
    if not trades:
        return {"trades": 0, "win_rate": np.nan, "avg_return": np.nan, "avg_hold": np.nan}
    arr = np.array(trades, dtype=float)
    return {
        "trades": int(len(arr)),
        "win_rate": float((arr > 0).mean()),
        "avg_return": float(arr.mean()),
        "avg_hold": float(np.mean(holds)),
    }


def _load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["day"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    for col in [
        "combined_return",
        "eth_close",
        "eth_spot_return",
        "alloc_eth",
        "alloc_btc",
        "gold_weight_exec",
        "gold_strategy_return",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    d["alloc_eth"] = pd.to_numeric(d.get("alloc_eth", 0.0), errors="coerce").fillna(0.0)
    d["alloc_btc"] = pd.to_numeric(d.get("alloc_btc", 0.0), errors="coerce").fillna(0.0)
    d["gold_weight_exec"] = pd.to_numeric(d.get("gold_weight_exec", 0.0), errors="coerce").fillna(0.0)
    return d


def _prepare_main(daily_path: Path, gross_cap: float, cost_bps: float) -> pd.DataFrame:
    base = _load_daily(daily_path)
    mr = _mean_reversion_overlay(
        base,
        gross_cap=float(gross_cap),
        cost_bps=float(cost_bps),
        z_entry=-1.5,
        z_exit=-0.5,
        ret_entry=-0.03,
        max_hold_days=10,
    )
    out = mr.copy()
    out["current_mr_return"] = pd.to_numeric(out["mr_return"], errors="coerce").fillna(0.0)
    out["main_return"] = pd.to_numeric(out["combined_return"], errors="coerce").fillna(0.0) + out["current_mr_return"]
    out["main_used_cap"] = (
        pd.to_numeric(out["alloc_eth"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["alloc_btc"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["gold_weight_exec"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["mr_weight_exec"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    return out


def _parallel_signal(d: pd.DataFrame, variant: str, gross_cap: float, max_alloc: float, cost_bps: float) -> pd.DataFrame:
    x = d.copy()
    close = pd.to_numeric(x["eth_close"], errors="coerce")
    mu = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    x["parallel_z"] = ((close - mu) / sd).replace([np.inf, -np.inf], np.nan)
    x["parallel_rsi"] = _rsi(close, 14)
    x["parallel_daily_ret"] = close.pct_change()
    x["parallel_room"] = (float(gross_cap) - pd.to_numeric(x["main_used_cap"], errors="coerce").fillna(0.0)).clip(lower=0.0)

    active = False
    held = 0
    target: list[float] = []
    entries: list[bool] = []
    exits: list[bool] = []
    for _, row in x.iterrows():
        z = float(row["parallel_z"]) if pd.notna(row["parallel_z"]) else np.nan
        rsi = float(row["parallel_rsi"]) if pd.notna(row["parallel_rsi"]) else np.nan
        dret = float(row["parallel_daily_ret"]) if pd.notna(row["parallel_daily_ret"]) else np.nan
        main_active = float(row["main_used_cap"]) > 1e-9

        if variant == "current":
            enter_signal = np.isfinite(z) and np.isfinite(dret) and z < -1.5 and dret < -0.03
        elif variant == "relax_z":
            enter_signal = np.isfinite(z) and np.isfinite(dret) and z < -1.2 and dret < -0.02
        elif variant == "z_only":
            enter_signal = np.isfinite(z) and z < -1.2
        elif variant == "rsi":
            enter_signal = np.isfinite(rsi) and rsi < 35.0
        elif variant == "combined":
            enter_signal = (np.isfinite(z) and z < -1.0) or (np.isfinite(rsi) and rsi < 35.0)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        exit_signal = (
            main_active
            or (np.isfinite(z) and z > -0.3)
            or (np.isfinite(rsi) and rsi > 60.0)
            or held >= 7
        )

        enter = (not active) and (not main_active) and enter_signal
        exit_ = active and exit_signal
        if enter:
            active = True
            held = 0
        elif exit_:
            active = False
            held = 0

        room = float(row["parallel_room"]) if np.isfinite(float(row["parallel_room"])) else 0.0
        target.append(min(float(max_alloc), max(room, 0.0)) if active else 0.0)
        entries.append(bool(enter))
        exits.append(bool(exit_))
        if active:
            held += 1

    x["parallel_weight_target"] = target
    x["parallel_weight_exec"] = x["parallel_weight_target"].shift(1).fillna(0.0)
    prev = x["parallel_weight_exec"].shift(1).fillna(0.0)
    x["parallel_turnover"] = (x["parallel_weight_exec"] - prev).abs()
    x["parallel_cost"] = x["parallel_turnover"] * (float(cost_bps) / 10000.0)
    x["parallel_return"] = x["parallel_weight_exec"] * pd.to_numeric(x["eth_spot_return"], errors="coerce").fillna(0.0) - x["parallel_cost"]
    x["combined_with_parallel"] = pd.to_numeric(x["main_return"], errors="coerce").fillna(0.0) + x["parallel_return"]
    x["parallel_entry"] = entries
    x["parallel_exit"] = exits
    return x


def _run_base_portfolio(
    start: str,
    end: str,
    daily_out: Path,
    summary_out: Path,
    eth_data: str = "",
    btc_data: str = "",
    cost_bps: float = 20.0,
) -> None:
    cmd = [
        sys.executable,
        "scripts/backtest_eth_btc_portfolio.py",
        "--start",
        start,
        "--end",
        end,
        "--vol-filter",
        "--transition-momentum",
        "--asymmetric-sizing",
        "--allocation-mode",
        "signal_weighted",
        "--cost-mode",
        "weight_change",
        "--cost-bps",
        str(float(cost_bps)),
        "--gross-cap",
        "0.8",
        "--eth-confirm-days",
        "3",
        "--btc-confirm-days",
        "5",
        "--eth-ema",
        "50,120,300",
        "--btc-ema",
        "15,40,120",
        "--include-gold",
        "--gold-symbol",
        "PAXG-USD",
        "--gold-ema",
        "25,65,180",
        "--gold-cap",
        "0.3",
        "--gold-cost-bps",
        str(float(cost_bps)),
        "--out-summary-csv",
        str(summary_out),
        "--out-daily-csv",
        str(daily_out),
    ]
    if eth_data:
        cmd.extend(["--eth-data", eth_data])
    if btc_data:
        cmd.extend(["--btc-data", btc_data])
    _run(cmd)


def _evaluate_period(
    label: str,
    main: pd.DataFrame,
    start: str,
    end: str,
    gross_cap: float,
    max_alloc: float,
    cost_bps: float,
    daily_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    d = main[(main["day"] >= s) & (main["day"] <= e)].copy().reset_index(drop=True)
    main_stats = _stats(d["main_return"])
    rows: list[dict[str, object]] = []
    daily_by_variant: dict[str, pd.DataFrame] = {}

    for variant_label, slug in VARIANTS:
        x = _parallel_signal(d, slug, gross_cap=gross_cap, max_alloc=max_alloc, cost_bps=cost_bps)
        combined = _stats(x["combined_with_parallel"])
        parallel = _stats(x["parallel_return"])
        tm = _trade_metrics(x, "parallel_weight_exec", "parallel_return")
        corr = float(pd.to_numeric(x["parallel_return"], errors="coerce").fillna(0.0).corr(pd.to_numeric(x["main_return"], errors="coerce").fillna(0.0)))
        rows.append(
            {
                "period": label,
                "variant": variant_label,
                "slug": slug,
                "trades": tm["trades"],
                "win_rate": tm["win_rate"],
                "avg_return": tm["avg_return"],
                "avg_hold": tm["avg_hold"],
                "parallel_sharpe": parallel["sharpe"],
                "parallel_return": parallel["return"],
                "main_sharpe": main_stats["sharpe"],
                "main_return": main_stats["return"],
                "combined_sharpe": combined["sharpe"],
                "combined_maxdd": combined["maxdd"],
                "combined_cagr": combined["cagr"],
                "combined_return": combined["return"],
                "sharpe_improvement": combined["sharpe"] - main_stats["sharpe"],
                "corr_vs_main": corr,
                "active_days": int((pd.to_numeric(x["parallel_weight_exec"], errors="coerce").fillna(0.0) > 0).sum()),
            }
        )
        daily_by_variant[slug] = x
        x.to_csv(daily_dir / f"{label}_{slug}_daily.csv", index=False)
    return pd.DataFrame(rows), daily_by_variant


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest relaxed ETH parallel mean-reversion overlays.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--paper-warmup-start", default="2025-04-14")
    ap.add_argument("--paper-start", default="2026-03-21")
    ap.add_argument("--paper-end", default="2026-08-27")
    ap.add_argument("--paper-eth-data", default="artifacts/tmp_server_compare/data/eth_daily_from_5m_live.csv")
    ap.add_argument("--paper-btc-data", default="artifacts/tmp_server_compare/data/btc_daily.csv")
    ap.add_argument("--gross-cap", type=float, default=0.8)
    ap.add_argument("--parallel-max-alloc", type=float, default=0.15)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out-summary", default="artifacts/backtest/parallel_signal_summary.csv")
    ap.add_argument("--out-paper-summary", default="artifacts/backtest/parallel_signal_paper_window.csv")
    ap.add_argument("--work-dir", default="artifacts/backtest/parallel_signal")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    base_daily = work / "validated_base_2019_2024_daily.csv"
    base_summary = work / "validated_base_2019_2024_summary.csv"
    _run_base_portfolio(
        start=args.start,
        end=args.end,
        daily_out=base_daily,
        summary_out=base_summary,
        cost_bps=float(args.cost_bps),
    )
    main = _prepare_main(base_daily, gross_cap=float(args.gross_cap), cost_bps=float(args.cost_bps))
    summary, _ = _evaluate_period(
        "2019_2024",
        main,
        args.start,
        args.end,
        gross_cap=float(args.gross_cap),
        max_alloc=float(args.parallel_max_alloc),
        cost_bps=float(args.cost_bps),
        daily_dir=work,
    )

    paper_daily = work / "validated_base_paper_warm_daily.csv"
    paper_summary = work / "validated_base_paper_warm_summary.csv"
    _run_base_portfolio(
        start=args.paper_warmup_start,
        end=args.paper_end,
        daily_out=paper_daily,
        summary_out=paper_summary,
        eth_data=args.paper_eth_data if Path(args.paper_eth_data).exists() else "",
        btc_data=args.paper_btc_data if Path(args.paper_btc_data).exists() else "",
        cost_bps=float(args.cost_bps),
    )
    paper_main = _prepare_main(paper_daily, gross_cap=float(args.gross_cap), cost_bps=float(args.cost_bps))
    paper_summary_df, _ = _evaluate_period(
        "paper_window",
        paper_main,
        args.paper_start,
        args.paper_end,
        gross_cap=float(args.gross_cap),
        max_alloc=float(args.parallel_max_alloc),
        cost_bps=float(args.cost_bps),
        daily_dir=work,
    )

    out_summary = Path(args.out_summary)
    out_paper = Path(args.out_paper_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    paper_summary_df.to_csv(out_paper, index=False)

    best = summary.sort_values(["combined_sharpe", "combined_maxdd"], ascending=[False, False]).iloc[0]
    main_sharpe = float(summary["main_sharpe"].iloc[0])
    decision = "YES" if float(best["combined_sharpe"]) - main_sharpe > 0.03 else "NO"

    print("")
    print("=" * 56)
    print("PARALLEL SIGNAL RESULTS")
    print("2019-2024 | 20bps | 15% max allocation")
    print("=" * 56)
    print("Variant     Trades WinRate AvgRet  Sharpe_combined Corr")
    print("-" * 56)
    for _, r in summary.iterrows():
        win = "n/a" if pd.isna(r["win_rate"]) else f"{float(r['win_rate'])*100:5.1f}%"
        avg = "n/a" if pd.isna(r["avg_return"]) else f"{float(r['avg_return'])*100:5.2f}%"
        corr = "n/a" if pd.isna(r["corr_vs_main"]) else f"{float(r['corr_vs_main']):5.3f}"
        print(
            f"{str(r['variant'])[:10]:<10}"
            f"{int(r['trades']):>6} "
            f"{win:>7} "
            f"{avg:>6} "
            f"{float(r['combined_sharpe']):>8.3f}        "
            f"{corr:>5}"
        )
    print("=" * 56)
    print(f"Main strategy Sharpe: {main_sharpe:.3f}")
    print(f"Best variant: {best['variant']}")
    print(f"Sharpe improvement: {float(best['combined_sharpe']) - main_sharpe:+.3f}")
    print(f"Implement: {decision}")
    print("")
    print("Paper window Mar 21 - Aug 27 2026:")
    print("Variant     Trades Return  Captured basket rally?")
    print("-" * 56)
    for _, r in paper_summary_df.iterrows():
        captured = "YES" if float(r["combined_return"]) > float(r["main_return"]) and int(r["trades"]) > 0 else "NO"
        print(f"{str(r['variant'])[:10]:<10}{int(r['trades']):>6} {float(r['combined_return'])*100:>6.2f}%  {captured}")
    print("=" * 56)
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_paper}")
    print(f"Daily files: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
