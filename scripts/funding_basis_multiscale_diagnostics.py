from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rolling_z(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(20, window // 4)).std()
    return (x - mu) / (sd + 1e-12)


def eval_signal(signal: pd.Series, close: pd.Series, horizon_bars: int) -> dict[str, float]:
    d = pd.DataFrame({"signal": signal, "close": close}).dropna().copy()
    d["fwd"] = np.log(d["close"].shift(-horizon_bars) / d["close"])
    d = d.dropna(subset=["fwd", "signal"])
    if len(d) < 200:
        return {"n": 0, "ic": np.nan, "hit": np.nan, "spread": np.nan}
    ic = float(d["signal"].corr(d["fwd"]))
    hit = float((np.sign(d["signal"]) == np.sign(d["fwd"])).mean())
    d["decile"] = pd.qcut(d["signal"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile")["fwd"].mean()
    spread = float(g.iloc[-1] - g.iloc[0]) if len(g) >= 2 else np.nan
    return {"n": int(len(d)), "ic": ic, "hit": hit, "spread": spread}


def main() -> None:
    ap = argparse.ArgumentParser(description="Funding+basis diagnostics across intraday and day/week horizons.")
    ap.add_argument("--input-csv", default="data/backtest/ETH_perp_features_5m_400d.csv")
    ap.add_argument("--window-days", type=int, default=400)
    ap.add_argument("--z-window-bars", type=int, default=288)
    ap.add_argument("--out-csv", default="artifacts/backtest/funding_basis_multiscale_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("spot_close", "basis", "funding_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "spot_close", "basis", "funding_rate"]).sort_values("timestamp")
    end = df["timestamp"].iloc[-1]
    start = end - pd.Timedelta(days=int(args.window_days))
    df = df[df["timestamp"] >= start].copy()

    # 5m signal
    w = int(args.z_window_bars)
    df["basis_z"] = rolling_z(df["basis"], w)
    df["funding_z"] = rolling_z(df["funding_rate"].ffill(), w)
    df["signal_5m"] = -(0.6 * df["basis_z"] + 0.4 * df["funding_z"])

    rows: list[dict[str, float | str | int]] = []
    intraday_h = [("1h", 12), ("4h", 48), ("12h", 144), ("1d", 288), ("3d", 864), ("7d", 2016)]
    for name, bars in intraday_h:
        m = eval_signal(df["signal_5m"], df["spot_close"], bars)
        rows.append({"scale": "5m_signal", "horizon": name, **m})

    # Daily aggregated signal (mean of 5m signal by day)
    daily = (
        df.assign(day=df["timestamp"].dt.floor("D"))
        .groupby("day", as_index=False)
        .agg(close=("spot_close", "last"), signal=("signal_5m", "mean"))
        .sort_values("day")
    )
    day_h = [("1d", 1), ("3d", 3), ("7d", 7), ("14d", 14), ("28d", 28)]
    for name, bars in day_h:
        m = eval_signal(daily["signal"], daily["close"], bars)
        rows.append({"scale": "daily_signal", "horizon": name, **m})

    out = pd.DataFrame(rows)
    out["window_start"] = str(df["timestamp"].iloc[0])
    out["window_end"] = str(df["timestamp"].iloc[-1])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
