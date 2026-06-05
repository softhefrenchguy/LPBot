from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib import request
from xml.etree import ElementTree as ET
import pandas as pd


DEFAULT_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://decrypt.co/feed",
]
MAX_AGE_HOURS = 12.0

SYSTEM_PROMPT = (
    "You are a macro trading analyst. "
    "Identify only significant events that could move ETH, gold, oil, or broader markets. "
    "Be conservative - ignore routine news."
)

USER_PROMPT_TEMPLATE = """Given these headlines, output JSON:
{{
  "major_event": true/false,
  "severity": "low/medium/high",
  "affected": [assets],
  "direction": "bullish/bearish/mixed",
  "summary": "one sentence max",
  "confidence": "low/medium/high"
}}

Headlines:
{headlines_text}
"""


def _clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _child_text(node: ET.Element, tags: list[str]) -> str:
    for tag in tags:
        v = node.findtext(tag)
        if v:
            return _clean_text(v)
    return ""


def _parse_dt(raw: str) -> dt.datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        d = parsedate_to_datetime(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            d = dt.datetime.strptime(raw, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc)
        except Exception:
            continue
    return None


def _parse_rss_items(
    xml_bytes: bytes,
    feed_url: str,
    per_feed: int,
    scan_limit: int,
    today_utc: dt.date,
    only_today: bool,
) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    out: list[dict[str, str]] = []

    rss_items = root.findall(".//item")
    if rss_items:
        for n in rss_items[:scan_limit]:
            title = _child_text(n, ["title"])
            summary = _child_text(n, ["description", "summary", "{http://purl.org/rss/1.0/modules/content/}encoded"])
            pub_raw = _child_text(n, ["pubDate", "published", "date", "{http://purl.org/dc/elements/1.1/}date"])
            pub_dt = _parse_dt(pub_raw)
            if only_today and (pub_dt is None or pub_dt.date() != today_utc):
                continue
            if not title and not summary:
                continue
            out.append(
                {
                    "feed": feed_url,
                    "published": pub_dt.date().isoformat() if pub_dt is not None else "",
                    "published_dt": pub_dt.isoformat() if pub_dt is not None else "",
                    "title": title,
                    "summary": summary,
                }
            )
            if len(out) >= per_feed:
                break
        return out

    atom_items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for n in atom_items[:scan_limit]:
        title = _child_text(n, ["{http://www.w3.org/2005/Atom}title", "title"])
        summary = _child_text(
            n,
            [
                "{http://www.w3.org/2005/Atom}summary",
                "{http://www.w3.org/2005/Atom}content",
                "summary",
                "content",
            ],
        )
        pub_raw = _child_text(
            n,
            [
                "{http://www.w3.org/2005/Atom}published",
                "{http://www.w3.org/2005/Atom}updated",
                "published",
                "updated",
            ],
        )
        pub_dt = _parse_dt(pub_raw)
        if only_today and (pub_dt is None or pub_dt.date() != today_utc):
            continue
        if not title and not summary:
            continue
        out.append(
            {
                "feed": feed_url,
                "published": pub_dt.date().isoformat() if pub_dt is not None else "",
                "published_dt": pub_dt.isoformat() if pub_dt is not None else "",
                "title": title,
                "summary": summary,
            }
        )
        if len(out) >= per_feed:
            break
    return out


def fetch_feed(
    feed_url: str,
    per_feed: int,
    scan_limit: int,
    timeout_sec: float,
    today_utc: dt.date,
    only_today: bool,
) -> list[dict[str, str]]:
    req = request.Request(feed_url, headers={"User-Agent": "LPBot-NewsSentiment/1.0"})
    with request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read()
    return _parse_rss_items(payload, feed_url, per_feed, scan_limit, today_utc, only_today)


def _parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Claude response did not contain valid JSON")
    parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Claude response JSON is not an object")
    return parsed


def call_claude(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": 400,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.0,
    }
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "user-agent": "LPBot-NewsSentiment/1.0",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = raw.get("content", [])
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
    return _parse_model_json("\n".join(text_parts).strip())


def normalize_result(d: dict[str, Any]) -> dict[str, Any]:
    major = bool(d.get("major_event", False))
    severity = str(d.get("severity", "low")).strip().lower() or "low"
    direction = str(d.get("direction", "mixed")).strip().lower() or "mixed"
    summary = str(d.get("summary", "")).strip()
    confidence = str(d.get("confidence", "low")).strip().lower() or "low"
    affected_raw = d.get("affected", [])
    if isinstance(affected_raw, list):
        affected = [str(x).strip() for x in affected_raw if str(x).strip()]
    elif isinstance(affected_raw, str):
        affected = [x.strip() for x in affected_raw.split(",") if x.strip()]
    else:
        affected = []
    if not major:
        severity = "low"
        direction = "mixed"
        if summary.strip().lower().startswith("no recent articles"):
            summary = "No recent articles (last 12h)"
        else:
            summary = "No major macro events"
        affected = []
    return {
        "major_event": major,
        "severity": severity,
        "affected": affected,
        "direction": direction,
        "summary": summary[:280],
        "confidence": confidence,
    }


def append_log(path: Path, result: dict[str, Any], now_utc: dt.datetime, articles_scanned: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "major_event", "severity", "affected", "direction", "summary", "articles_scanned"]
    row = {
        "date": now_utc.date().isoformat(),
        "major_event": bool(result["major_event"]),
        "severity": str(result["severity"]),
        "affected": "|".join(result["affected"]),
        "direction": str(result["direction"]),
        "summary": str(result["summary"]),
        "articles_scanned": int(articles_scanned),
    }
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    for c in fieldnames:
        if c not in d.columns:
            d[c] = "" if c not in ("major_event", "articles_scanned") else (False if c == "major_event" else 0)
    d = d[fieldnames]
    d = pd.concat([d, pd.DataFrame([row], columns=fieldnames)], ignore_index=True)
    d.to_csv(path, index=False)


def format_alert(result: dict[str, Any]) -> str:
    if not bool(result["major_event"]):
        return "\U0001F4F0 News: No major macro events"
    assets = ", ".join(result["affected"]) if result["affected"] else "broad markets"
    return f"\u26A0\uFE0F NEWS ALERT: {result['summary']}\nAffected: {assets} | {result['direction']}"


def _default_result() -> dict[str, Any]:
    return normalize_result(
        {
            "major_event": False,
            "severity": "low",
            "affected": [],
            "direction": "mixed",
            "summary": "No major macro events",
            "confidence": "low",
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Daily macro-news sentiment scan via RSS + Claude")
    p.add_argument("--feeds", default=",".join(DEFAULT_FEEDS))
    p.add_argument("--per-feed", type=int, default=5)
    p.add_argument("--scan-limit", type=int, default=20)
    p.add_argument("--timeout-sec", type=float, default=15.0)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--api-key", default="")
    p.add_argument("--log-csv", default="artifacts/news/news_log.csv")
    p.add_argument("--debug-out", default="artifacts/news/news_last_input.json")
    args = p.parse_args()

    now_utc = dt.datetime.now(dt.timezone.utc)
    today_utc = now_utc.date()
    api_key = str(args.api_key).strip() or str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    feeds = [x.strip() for x in str(args.feeds).split(",") if x.strip()]
    only_today = True

    headlines_today: list[dict[str, str]] = []
    for url in feeds:
        try:
            rows = fetch_feed(
                url,
                int(args.per_feed),
                int(args.scan_limit),
                float(args.timeout_sec),
                today_utc=today_utc,
                only_today=only_today,
            )
        except Exception:
            rows = []
        headlines_today.extend(rows[: int(args.per_feed)])
    headlines_today = headlines_today[: int(args.per_feed) * len(feeds)]

    # Hard freshness gate: keep only articles from the last 12 hours.
    fresh_headlines: list[dict[str, str]] = []
    for h in headlines_today:
        pub_dt = _parse_dt(str(h.get("published_dt", "")))
        if pub_dt is None:
            continue
        age_h = (now_utc - pub_dt).total_seconds() / 3600.0
        if 0.0 <= age_h <= MAX_AGE_HOURS:
            h2 = dict(h)
            h2["age_hours"] = f"{age_h:.2f}"
            fresh_headlines.append(h2)

    debug_path = Path(args.debug_out)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(
            {
                "generated_utc": now_utc.isoformat(),
                "date_filter": "today_utc",
                "max_age_hours": MAX_AGE_HOURS,
                "headlines_today_count": len(headlines_today),
                "headlines_fresh_count": len(fresh_headlines),
                "headlines_today": headlines_today,
                "headlines_fresh": fresh_headlines,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not fresh_headlines:
        result = normalize_result(
            {
                "major_event": False,
                "severity": "low",
                "affected": [],
                "direction": "mixed",
                "summary": "No recent articles (last 12h)",
                "confidence": "low",
            }
        )
        out_line = "\U0001F4F0 News: No recent articles (last 12h)"
    elif not api_key:
        result = _default_result()
        out_line = format_alert(result)
    else:
        headlines_text = "\n\n".join(
            [
                f"DATE: {h.get('published', '')}\nSOURCE: {h.get('feed', '')}\nAGE_HOURS: {h.get('age_hours', '')}\nTITLE: {h.get('title', '')}\nSUMMARY: {h.get('summary', '')}"
                for h in fresh_headlines
            ]
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(headlines_text=headlines_text)
        try:
            raw = call_claude(
                api_key=api_key,
                model=str(args.model),
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_sec=float(args.timeout_sec),
            )
            result = normalize_result(raw)
        except Exception:
            result = _default_result()
        out_line = format_alert(result)

    append_log(Path(args.log_csv), result, now_utc, articles_scanned=len(fresh_headlines))
    try:
        print(out_line)
    except UnicodeEncodeError:
        safe = out_line.encode("cp1252", errors="ignore").decode("cp1252").strip()
        print(safe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
