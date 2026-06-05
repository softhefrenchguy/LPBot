from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def parse_case(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"invalid case format: {value}")
    k, v = value.split("=", 1)
    return k.strip(), v.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply daily bull-floor to event model weights and sweep floors.")
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[
            "12h_base=artifacts/backtest/direction_event_model_v1_12h_2021bull.csv",
            "1d_base=artifacts/backtest/direction_event_model_v1_1d_2021bull.csv",
            "12h_hiconv=artifacts/backtest/direction_event_model_v1_12h_2021bull_hiconv.csv",
            "1d_hiconv=artifacts/backtest/direction_event_model_v1_1d_2021bull_hiconv.csv",
        ],
    )
    ap.add_argument("--floors", default="0,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--daily-ema", type=int, default=200)
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--out-summary-csv", default="artifacts/backtest/event_model_bull_floor_2021bull_summary.csv")
    args = ap.parse_args()

    floors = [float(x) for x in str(args.floors).split(",") if str(x).strip()]
    out_rows: list[dict[str, float | str | int]] = []

    for item in args.cases:
        case_name, path_str = parse_case(item)
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["eth_close"] = pd.to_numeric(df["eth_close"], errors="coerce")
        df["r"] = pd.to_numeric(df["r"], errors="coerce")
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["timestamp", "eth_close", "r", "weight"]).sort_values("timestamp").reset_index(drop=True)

        day = (
            df.set_index("timestamp")["eth_close"]
            .resample("1D")
            .last()
            .dropna()
            .to_frame("close")
            .reset_index()
            .rename(columns={"timestamp": "day"})
        )
        day["ema"] = day["close"].ewm(span=int(args.daily_ema), adjust=False).mean()
        day["bull"] = (day["close"] > day["ema"]).astype(int)

        d = df.copy()
        d["day"] = d["timestamp"].dt.floor("D")
        d = d.merge(day[["day", "bull"]], on="day", how="left")
        d["bull"] = d["bull"].fillna(0).astype(int)
        bull_mask = d["bull"].to_numpy(dtype=bool)

        for f in floors:
            w_base = d["weight"].to_numpy(dtype=float)
            # In bull regime, force a minimum long exposure.
            w_adj = np.where(bull_mask, np.maximum(w_base, float(f)), w_base)
            turnover = np.abs(np.diff(w_adj, prepend=w_adj[0]))
            cost = turnover * (float(args.trade_cost_bps) / 10000.0)
            strat_r = pd.Series(np.r_[0.0, w_adj[:-1]] * d["r"].to_numpy(dtype=float) - cost, index=d.index)

            m = perf(strat_r, 5)
            spot_m = perf(d["r"], 5)
            bull_r = d.loc[bull_mask, "r"].to_numpy(dtype=float)
            bull_sr = strat_r.loc[bull_mask].to_numpy(dtype=float)
            bull_capture = np.nan
            if len(bull_r) > 0:
                bull_capture = float(np.nansum(bull_sr) / (np.nansum(bull_r) + 1e-12))

            out_rows.append(
                {
                    "case": case_name,
                    "floor": float(f),
                    "rows": int(len(d)),
                    "bull_bar_pct": float(bull_mask.mean() * 100.0),
                    "time_in_market_pct": float((np.abs(w_adj) > 1e-12).mean() * 100.0),
                    "avg_abs_weight": float(np.mean(np.abs(w_adj))),
                    "turnover": float(np.nansum(turnover)),
                    "ret": m["ret"],
                    "cagr": m["cagr"],
                    "ann_vol": m["ann_vol"],
                    "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"],
                    "spot_ret": spot_m["ret"],
                    "excess_vs_spot": m["ret"] - spot_m["ret"],
                    "bull_capture_log": bull_capture,
                }
            )

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["case", "ret"], ascending=[True, False]).reset_index(drop=True)
    out_path = Path(args.out_summary_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(out.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
