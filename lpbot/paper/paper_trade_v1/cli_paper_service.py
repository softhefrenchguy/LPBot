from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_log(event: str, **fields: object) -> None:
    payload = {"ts": _utc_now().isoformat(), "event": event, **fields}
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _write_heartbeat(path: Path) -> None:
    path.write_text(str(int(_utc_now().timestamp())), encoding="utf-8")


def _read_last_log_timestamp(log_csv: Path) -> Optional[str]:
    if not log_csv.exists() or log_csv.stat().st_size == 0:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(log_csv, usecols=["timestamp"]).tail(1)
        if df.empty:
            return None
        ts = df["timestamp"].iloc[0]
        return str(ts)
    except Exception:
        return None


def _write_state(state_path: Path, status: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp": _utc_now().isoformat(),
    }
    log_csv = Path(os.environ.get("PAPER_LOG", "artifacts/paper/paper_log.csv"))
    last_ts = _read_last_log_timestamp(log_csv)
    if last_ts:
        payload["last_log_timestamp"] = last_ts
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _heartbeat_loop(stop_event: Event, seconds: int, heartbeat_path: Path) -> None:
    while not stop_event.is_set():
        try:
            _write_heartbeat(heartbeat_path)
            _json_log("heartbeat")
        except Exception as exc:
            _json_log("heartbeat_error", error=str(exc))
        stop_event.wait(seconds)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _bool_env(name: str, default: bool = False) -> bool:
    val = _env(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _build_runner_cmd() -> list[str]:
    cmd = ["python", "-m", "lpbot.paper.paper_trade_v1.cli_paper_runner"]
    mapping = {
        "SYMBOL": "--symbol",
        "BAR_MINUTES": "--bar-minutes",
        "GRAPH_KEY_ENV": "--graph-api-key-env",
        "SUBGRAPH_ID": "--subgraph-id",
        "POOL_ADDRESS": "--pool",
        "PRICE_1M_CSV": "--price-1m-csv",
        "PRICE_5M_CSV": "--price-5m-csv",
        "PRICE_START": "--price-start",
        "PRICE_LOOKBACK_DAYS": "--price-lookback-days",
        "VOLUME_CSV": "--volume-csv",
        "EXPOSURE_CSV": "--exposure-csv",
        "PAPER_LOG": "--paper-log",
        "MODEL_DIR": "--model-dir",
        "TARGET_VOL": "--target-vol",
        "W_MAX": "--w-max",
        "EMA_SPAN_MINUTES": "--ema-span-minutes",
        "PANIC_VOL": "--panic-vol",
        "RISKOFF_MODE": "--riskoff-mode",
        "TREND_FILTER": "--trend-filter",
        "TREND_TIMEFRAME": "--trend-timeframe",
        "TREND_EMA": "--trend-ema",
        "TREND_HYST": "--trend-hyst",
        "FEE_TIER": "--fee-tier",
        "POOL_TVL_USD": "--pool-tvl-usd",
        "IN_RANGE_FRAC": "--in-range-frac",
        "RANGE_SIGMA": "--range-sigma",
        "RANGE_SIGMA_REF": "--range-sigma-ref",
        "MIN_IN_RANGE_FRAC": "--min-in-range-frac",
        "MAX_IN_RANGE_FRAC": "--max-in-range-frac",
        "CHURN_K": "--churn-k",
        "IL_K": "--il-k",
        "LP_VOL_ON": "--lp-vol-on",
        "LP_SCALE": "--lp-scale",
        "LP_WEIGHT_MAX": "--lp-weight-max",
        "LP_MIN_ON_BARS": "--lp-min-on-bars",
        "LP_COOLDOWN_BARS": "--lp-cooldown-bars",
        "VOLUME_START": "--volume-start",
    }
    for env_name, arg in mapping.items():
        val = _env(env_name)
        if val is not None:
            cmd.extend([arg, str(val)])
    return cmd


def _run_once(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        _json_log("run_failed", returncode=proc.returncode, stderr=proc.stderr[-4000:])
        raise RuntimeError("paper runner failed")


def _s3_sync(artifacts_dir: str, logs_dir: str | None, bucket: str, prefix: str) -> None:
    if not shutil.which("aws"):
        raise RuntimeError("aws cli not found in PATH")
    date_key = _utc_now().strftime("%Y-%m-%d")
    base = f"s3://{bucket}/{prefix}/{date_key}"
    subprocess.run(
        ["aws", "s3", "sync", artifacts_dir, f"{base}/artifacts", "--only-show-errors"],
        check=True,
    )
    if logs_dir:
        subprocess.run(
            ["aws", "s3", "sync", logs_dir, f"{base}/logs", "--only-show-errors"],
            check=True,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-running LPBot paper trade service.")
    p.add_argument("--sleep-seconds", type=int, default=int(_env("SLEEP_SECONDS", "300")))
    p.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=int(_env("HEARTBEAT_SECONDS", "60")),
    )
    p.add_argument(
        "--state-file",
        default=_env("STATE_FILE", "artifacts/paper/state.json"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    stop_event = Event()
    heartbeat_path = Path("/tmp/heartbeat.txt")

    def _handle_signal(signum: int, _frame: object) -> None:
        _json_log("signal", signum=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    hb_thread = Thread(
        target=_heartbeat_loop,
        args=(stop_event, args.heartbeat_seconds, heartbeat_path),
        daemon=True,
    )
    hb_thread.start()

    s3_bucket = _env("S3_BUCKET")
    s3_prefix = _env("S3_PREFIX", "lpbot")
    s3_logs = _bool_env("S3_SYNC_LOGS", False)
    last_s3_date: Optional[str] = None

    cmd = _build_runner_cmd()
    _json_log("service_start", cmd=" ".join(cmd))

    try:
        while not stop_event.is_set():
            start = time.time()
            _json_log("run_start")
            _run_once(cmd)
            _json_log("run_complete", duration_sec=round(time.time() - start, 2))

            if s3_bucket:
                today = _utc_now().strftime("%Y-%m-%d")
                if today != last_s3_date:
                    _json_log("s3_sync_start", bucket=s3_bucket, prefix=s3_prefix)
                    _s3_sync(
                        artifacts_dir=_env("ARTIFACTS_DIR", "artifacts"),
                        logs_dir=_env("LOGS_DIR", "logs") if s3_logs else None,
                        bucket=s3_bucket,
                        prefix=s3_prefix,
                    )
                    last_s3_date = today
                    _json_log("s3_sync_complete", date=today)

            stop_event.wait(args.sleep_seconds)
    except Exception as exc:
        _json_log("fatal", error=str(exc))
        _write_state(Path(args.state_file), status="fatal")
        sys.exit(1)

    _write_state(Path(args.state_file), status="stopped")
    _json_log("service_stop")


if __name__ == "__main__":
    main()
