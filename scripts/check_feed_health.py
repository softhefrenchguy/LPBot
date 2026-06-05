from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

import pandas as pd


def _send_discord_alert(webhook_url: str, title: str, message: str, color: int, timeout_sec: float) -> tuple[bool, str]:
    payload = {
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": int(color),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "LPBot Feed Monitor | Hetzner"},
            }
        ]
    }
    req = request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "LPBot-FeedHealth/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            code = int(getattr(resp, "status", 0))
            if code in (200, 204):
                return True, "discord_sent"
            return False, f"discord_http_{code}"
    except error.HTTPError as e:
        return False, f"discord_http_{int(e.code)}"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"discord_error_{type(e).__name__}"


def _find_timestamp_col(df: pd.DataFrame) -> str | None:
    for c in ["timestamp", "ts", "date", "day"]:
        if c in df.columns:
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Check funding feed freshness and send Discord alert on failure.")
    p.add_argument("--funding-csv", default=os.environ.get("FUNDING_CSV", "data/backtest/ETH_perp_features_5m_live.csv"))
    p.add_argument("--discord-webhook", default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    p.add_argument("--max-age-hours", type=float, default=float(os.environ.get("FUNDING_MAX_AGE_HOURS", "9")))
    p.add_argument("--discord-timeout-sec", type=float, default=10.0)
    args = p.parse_args()

    funding_path = Path(args.funding_csv)
    webhook = str(args.discord_webhook).strip()

    def alert(title: str, message: str, color: int) -> int:
        if not webhook:
            print(f"ALERT (no webhook configured): {title} | {message}")
            return 1
        ok, msg = _send_discord_alert(webhook, title, message, color, float(args.discord_timeout_sec))
        print(msg if ok else f"WARNING: {msg}")
        return 1

    try:
        if not funding_path.exists():
            return alert("🚨 LPBot Feed FAILED", f"Funding CSV missing: `{funding_path}`", 0xFF0000)

        df = pd.read_csv(funding_path, low_memory=False)
        if df.empty:
            return alert("🚨 LPBot Feed FAILED", f"Funding CSV is empty: `{funding_path}`", 0xFF0000)

        ts_col = _find_timestamp_col(df)
        if ts_col is None:
            return alert("🚨 LPBot Feed FAILED", f"No timestamp column in `{funding_path}`", 0xFF0000)

        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
        if ts.empty:
            return alert("🚨 LPBot Feed FAILED", f"No parseable timestamps in `{funding_path}`", 0xFF0000)

        last_ts = ts.iloc[-1].to_pydatetime()
        age_h = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600.0

        if age_h > float(args.max_age_hours):
            return alert(
                "⚠️ LPBot Feed Alert",
                f"Funding data stale: `{age_h:.2f}h` old (max `{args.max_age_hours:.2f}h`). Last ts: `{last_ts.isoformat()}`",
                0xFFA500,
            )

        # Healthy feed: intentionally silent to Discord.
        print(f"feed_ok age_h={age_h:.2f} last_ts={last_ts.isoformat()}")
        return 0
    except Exception as e:  # pragma: no cover - defensive
        return alert("🚨 LPBot Feed FAILED", f"Exception reading funding feed: `{type(e).__name__}: {e}`", 0xFF0000)


if __name__ == "__main__":
    raise SystemExit(main())
