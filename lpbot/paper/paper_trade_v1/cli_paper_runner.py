from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVAL = "1m"
BINANCE_LIMIT = 1000
BINANCE_TIMEOUT = 20


def _utc_now_minute() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _read_last_timestamp_ms(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"]).tail(1)
    except Exception:
        return None
    if df.empty:
        return None
    ts = df["timestamp"].iloc[0]
    if pd.isna(ts):
        return None
    try:
        ts_val = int(float(ts))
    except Exception:
        return None
    return ts_val


def _fetch_binance_1m(symbol: str, start_ms: int, end_ms: int) -> list:
    session = requests.Session()
    rows = []
    current = start_ms
    while current <= end_ms:
        params = {
            "symbol": symbol,
            "interval": BINANCE_INTERVAL,
            "limit": BINANCE_LIMIT,
            "startTime": current,
            "endTime": end_ms,
        }
        resp = session.get(BINANCE_URL, params=params, timeout=BINANCE_TIMEOUT)
        if resp.status_code == 429:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open >= end_ms:
            break
        next_start = last_open + 60_000
        if next_start <= current:
            break
        current = next_start
        time.sleep(0.2)
    return rows


def _rows_to_df(rows: list) -> pd.DataFrame:
    trimmed = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]
    df = pd.DataFrame(trimmed, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def _fill_missing_minutes(df: pd.DataFrame, start_ms: int, end_ms: int, last_close: Optional[float]) -> pd.DataFrame:
    full_index = pd.RangeIndex(start_ms, end_ms + 1, 60_000)
    reindexed = df.set_index("timestamp").reindex(full_index)
    is_gap = reindexed["open"].isna()

    if last_close is not None:
        reindexed["close"] = reindexed["close"].ffill()
        reindexed["close"] = reindexed["close"].fillna(last_close)
    else:
        reindexed["close"] = reindexed["close"].ffill()

    for col in ["open", "high", "low"]:
        reindexed[col] = reindexed[col].fillna(reindexed["close"])
    reindexed["volume"] = reindexed["volume"].fillna(0)
    reindexed["is_gap"] = is_gap.astype("int64")

    out = reindexed.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].astype("int64")
    return out


def _append_rows(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=write_header)


def _parse_price_start(price_start: str | None, lookback_days: int | None) -> datetime:
    if price_start:
        try:
            dt = datetime.strptime(price_start, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError("price_start must be YYYY-MM-DD") from exc
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("price_lookback_days must be positive.")
        return _utc_now_minute() - timedelta(days=lookback_days)
    return datetime(2021, 1, 1, tzinfo=timezone.utc)


def update_binance_1m(
    symbol: str,
    out_csv: Path,
    price_start: str | None,
    price_lookback_days: int | None,
) -> bool:
    last_ts = _read_last_timestamp_ms(out_csv)
    now = _utc_now_minute()
    end_ms = _dt_to_ms(now)

    if last_ts is None:
        start_dt = _parse_price_start(price_start, price_lookback_days)
        start_ms = _dt_to_ms(start_dt)
    else:
        start_ms = last_ts + 60_000

    if start_ms > end_ms:
        print("No new 1m bars.")
        return False

    rows = _fetch_binance_1m(symbol, start_ms, end_ms)
    if not rows:
        print("No new rows fetched from Binance.")
        return False

    df_new = _rows_to_df(rows)
    if df_new.empty:
        print("No valid rows after parsing.")
        return False

    last_close = None
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            last_close = pd.read_csv(out_csv, usecols=["close"]).tail(1)["close"].iloc[0]
        except Exception:
            last_close = None

    filled = _fill_missing_minutes(df_new, int(df_new["timestamp"].iloc[0]), int(df_new["timestamp"].iloc[-1]), last_close)

    if out_csv.exists() and out_csv.stat().st_size > 0:
        header_cols = list(pd.read_csv(out_csv, nrows=0).columns)
        for col in header_cols:
            if col not in filled.columns:
                filled[col] = 0
        filled = filled[header_cols]
    else:
        cols = ["timestamp", "open", "high", "low", "close", "volume", "is_gap"]
        filled = filled[cols]

    _append_rows(out_csv, filled)
    print(f"Appended 1m rows: {len(filled)}")
    return True


def resample_to_5m(symbol: str) -> None:
    cmd = [
        "python",
        "src/resample/resample_1m.py",
        "--data-dir",
        "data",
        "--symbols",
        symbol,
        "--tfs",
        "5m",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def update_volume_5m(
    out_csv: Path,
    pool: str,
    subgraph_id: str,
    graph_api_key: str,
    bar_minutes: int,
    start_date: Optional[str],
) -> None:
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            last_ts = pd.read_csv(out_csv, usecols=["timestamp"]).tail(1)["timestamp"].iloc[0]
            last_dt = pd.to_datetime(last_ts, utc=True, errors="coerce")
            if pd.notna(last_dt):
                start_date = last_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    end_date = (_utc_now_minute() + timedelta(days=1)).strftime("%Y-%m-%d")

    cmd = [
        "python",
        "-m",
        "lpbot.data_sources.uniswap_v3_subgraph.cli_fetch_pool_volume_5m",
        "--graph-api-key",
        graph_api_key,
        "--subgraph-id",
        subgraph_id,
        "--pool",
        pool,
        "--start",
        start_date,
        "--end",
        end_date,
        "--bar-minutes",
        str(bar_minutes),
        "--out",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def update_exposure(
    model_dir: str,
    input_1m: str,
    bar_minutes: int,
    target_vol: float,
    w_max: float,
    ema_span_minutes: int,
    panic_vol: float,
    riskoff_mode: str,
    trend_filter: str,
    trend_timeframe: str,
    trend_ema: int,
    out_csv: str,
) -> None:
    cmd = [
        "python",
        "-m",
        "lpbot.models.elasticnet_v1.cli_exposure",
        "--model-dir",
        model_dir,
        "--input-1m",
        input_1m,
        "--bar-minutes",
        str(bar_minutes),
        "--target-vol",
        str(target_vol),
        "--w-max",
        str(w_max),
        "--ema-span-minutes",
        str(ema_span_minutes),
        "--panic-vol",
        str(panic_vol),
        "--riskoff-mode",
        riskoff_mode,
        "--trend-filter",
        trend_filter,
        "--trend-timeframe",
        trend_timeframe,
        "--trend-ema",
        str(trend_ema),
        "--out",
        out_csv,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def append_paper_log(
    exposure_csv: str,
    price_csv: str,
    volume_csv: str,
    bar_minutes: int,
    fee_tier: float,
    pool_tvl_usd: float,
    in_range_frac: float,
    range_sigma: float,
    range_sigma_ref: float,
    min_in_range_frac: float,
    max_in_range_frac: float,
    churn_k: float,
    il_k: float,
    lp_vol_on: float,
    lp_scale: float,
    lp_weight_max: float,
    lp_min_on_bars: int,
    lp_cooldown_bars: int,
    log_csv: str,
) -> None:
    cmd = [
        "python",
        "-m",
        "lpbot.paper.paper_trade_v1.cli_paper_append",
        "--exposure-csv",
        exposure_csv,
        "--price-csv",
        price_csv,
        "--volume-csv",
        volume_csv,
        "--bar-minutes",
        str(bar_minutes),
        "--fee-tier",
        str(fee_tier),
        "--pool-tvl-usd",
        str(pool_tvl_usd),
        "--in-range-frac",
        str(in_range_frac),
        "--range-sigma",
        str(range_sigma),
        "--range-sigma-ref",
        str(range_sigma_ref),
        "--min-in-range-frac",
        str(min_in_range_frac),
        "--max-in-range-frac",
        str(max_in_range_frac),
        "--churn-k",
        str(churn_k),
        "--il-k",
        str(il_k),
        "--lp-vol-on",
        str(lp_vol_on),
        "--lp-scale",
        str(lp_scale),
        "--lp-weight-max",
        str(lp_weight_max),
        "--lp-min-on-bars",
        str(lp_min_on_bars),
        "--lp-cooldown-bars",
        str(lp_cooldown_bars),
        "--log-csv",
        log_csv,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end paper trade update.")
    p.add_argument("--symbol", default="ETHUSDC")
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--price-start", default=os.environ.get("PRICE_START"))
    p.add_argument(
        "--price-lookback-days",
        type=int,
        default=int(os.environ.get("PRICE_LOOKBACK_DAYS"))
        if os.environ.get("PRICE_LOOKBACK_DAYS")
        else None,
    )

    p.add_argument("--graph-api-key-env", default="GRAPH_KEY")
    p.add_argument("--subgraph-id", default="FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM")
    p.add_argument("--pool", default="0xC6962004f452bE9203591991D15f6b388e09E8D0")

    p.add_argument("--price-1m-csv", default="data/ETHUSDC_1m.csv")
    p.add_argument("--price-5m-csv", default="data/ETHUSDC_5m.csv")
    p.add_argument("--volume-csv", default="data/ETHUSDC_pool_volume_5m.csv")
    p.add_argument("--exposure-csv", default="artifacts/exposure.csv")
    p.add_argument("--paper-log", default="artifacts/paper/paper_log.csv")

    p.add_argument("--model-dir", default="models/elasticnet-v1.0-5m-vol")
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--ema-span-minutes", type=int, default=180)
    p.add_argument("--panic-vol", type=float, default=0.9)
    p.add_argument("--riskoff-mode", default="both")
    p.add_argument("--trend-filter", default="ema")
    p.add_argument("--trend-timeframe", default="1h")
    p.add_argument("--trend-ema", type=int, default=200)

    p.add_argument("--fee-tier", type=float, default=0.003)
    p.add_argument("--pool-tvl-usd", type=float, default=20000000)
    p.add_argument("--in-range-frac", type=float, default=0.25)
    p.add_argument("--range-sigma", type=float, default=2.0)
    p.add_argument("--range-sigma-ref", type=float, default=2.0)
    p.add_argument("--min-in-range-frac", type=float, default=0.05)
    p.add_argument("--max-in-range-frac", type=float, default=0.25)
    p.add_argument("--churn-k", type=float, default=1e-5)
    p.add_argument("--il-k", type=float, default=0.5)
    p.add_argument("--lp-vol-on", type=float, default=0.25)
    p.add_argument("--lp-scale", type=float, default=0.5)
    p.add_argument("--lp-weight-max", type=float, default=0.5)
    p.add_argument("--lp-min-on-bars", type=int, default=6)
    p.add_argument("--lp-cooldown-bars", type=int, default=12)

    p.add_argument("--volume-start", default="2021-01-01")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    graph_key = os.environ.get(args.graph_api_key_env)
    if not graph_key:
        raise SystemExit(f"Missing API key in env var {args.graph_api_key_env}")

    price_1m = Path(args.price_1m_csv)
    price_5m = Path(args.price_5m_csv)
    volume_csv = Path(args.volume_csv)

    print("Updating 1m price data...")
    updated = update_binance_1m(
        args.symbol,
        price_1m,
        price_start=args.price_start,
        price_lookback_days=args.price_lookback_days,
    )

    if updated:
        print("Resampling to 5m...")
        resample_to_5m(args.symbol)

    print("Updating pool volume 5m...")
    update_volume_5m(
        out_csv=volume_csv,
        pool=args.pool,
        subgraph_id=args.subgraph_id,
        graph_api_key=graph_key,
        bar_minutes=args.bar_minutes,
        start_date=args.volume_start,
    )

    print("Updating exposure...")
    update_exposure(
        model_dir=args.model_dir,
        input_1m=str(price_5m),
        bar_minutes=args.bar_minutes,
        target_vol=args.target_vol,
        w_max=args.w_max,
        ema_span_minutes=args.ema_span_minutes,
        panic_vol=args.panic_vol,
        riskoff_mode=args.riskoff_mode,
        trend_filter=args.trend_filter,
        trend_timeframe=args.trend_timeframe,
        trend_ema=args.trend_ema,
        out_csv=args.exposure_csv,
    )

    print("Appending paper log...")
    append_paper_log(
        exposure_csv=args.exposure_csv,
        price_csv=str(price_5m),
        volume_csv=str(volume_csv),
        bar_minutes=args.bar_minutes,
        fee_tier=args.fee_tier,
        pool_tvl_usd=args.pool_tvl_usd,
        in_range_frac=args.in_range_frac,
        range_sigma=args.range_sigma,
        range_sigma_ref=args.range_sigma_ref,
        min_in_range_frac=args.min_in_range_frac,
        max_in_range_frac=args.max_in_range_frac,
        churn_k=args.churn_k,
        il_k=args.il_k,
        lp_vol_on=args.lp_vol_on,
        lp_scale=args.lp_scale,
        lp_weight_max=args.lp_weight_max,
        lp_min_on_bars=args.lp_min_on_bars,
        lp_cooldown_bars=args.lp_cooldown_bars,
        log_csv=args.paper_log,
    )


if __name__ == "__main__":
    main()
