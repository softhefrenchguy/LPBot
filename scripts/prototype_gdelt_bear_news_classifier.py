from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


"""
Free GDELT prototype for major-bear news classification.

The script fetches a bear_start +/- N day article window from GDELT DOC 2.0,
but by default scores only articles whose GDELT seendate is <= bear_start
23:59:59 UTC. That keeps the actual news_based_type no-look-ahead while still
saving the requested post-start context for inspection.

This is not production wiring and does not spend on Benzinga/Massive.
"""


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TERM_SETS = {
    "PANIC_BEAR": [
        "panic",
        "crash",
        "selloff",
        "sell-off",
        "rout",
        "meltdown",
        "turmoil",
        "volatility",
        "risk off",
        "risk-off",
        "recession",
        "liquidation",
        "plunge",
    ],
    "INFLATION_BEAR": [
        "inflation",
        "cpi",
        "rate hike",
        "rate hikes",
        "interest rates",
        "fed",
        "federal reserve",
        "central bank",
        "bond yields",
        "treasury yields",
        "hawkish",
        "tightening",
    ],
    "GEOPOLITICAL_BEAR": [
        "war",
        "sanctions",
        "conflict",
        "attack",
        "missile",
        "military",
        "iran",
        "russia",
        "ukraine",
        "china",
        "tariff",
        "trade war",
        "geopolitical",
    ],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototype GDELT news-based bear classifier.")
    p.add_argument("--classifications", default="artifacts/bear_classifier/bear_period_classifications.csv")
    p.add_argument("--pre-days", type=int, default=3)
    p.add_argument("--post-days", type=int, default=3)
    p.add_argument("--maxrecords", type=int, default=25)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="artifacts/bear_classifier/gdelt_cache")
    p.add_argument("--out", default="artifacts/bear_classifier/bear_period_news_classifications.csv")
    return p.parse_args()


def _query_for_category(category: str) -> str:
    terms = TERM_SETS[category]
    quoted = []
    for term in terms:
        if " " in term or "-" in term:
            quoted.append(f'"{term}"')
        else:
            quoted.append(term)
    return f"({' OR '.join(quoted)}) sourcelang:english"


def _query_all_terms() -> str:
    seen: list[str] = []
    for terms in TERM_SETS.values():
        for term in terms:
            if term not in seen:
                seen.append(term)
    quoted = []
    for term in seen:
        if " " in term or "-" in term:
            quoted.append(f'"{term}"')
        else:
            quoted.append(term)
    return f"({' OR '.join(quoted)}) sourcelang:english"


def _dt_arg(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def _cache_name(start: pd.Timestamp, category: str, pre_days: int, post_days: int) -> str:
    return f"{start.strftime('%Y%m%d')}_{category}_m{pre_days}_p{post_days}.json"


def _fetch_gdelt(
    bear_start: pd.Timestamp,
    category: str,
    pre_days: int,
    post_days: int,
    maxrecords: int,
    cache_dir: Path,
    refresh: bool,
    timeout: int,
    retries: int,
) -> list[dict]:
    cache = cache_dir / _cache_name(bear_start, category, pre_days, post_days)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8")).get("articles", [])

    start = (bear_start - pd.Timedelta(days=pre_days)).floor("D")
    end = (bear_start + pd.Timedelta(days=post_days)).floor("D") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    params = {
        "query": _query_for_category(category),
        "mode": "artlist",
        "format": "json",
        "startdatetime": _dt_arg(start),
        "enddatetime": _dt_arg(end),
        "maxrecords": str(maxrecords),
        "sort": "hybridrel",
    }
    payload = {"articles": []}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                GDELT_DOC_URL,
                params=params,
                headers={"User-Agent": "LPBot research; no-lookahead classifier prototype"},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2.0 * attempt)
    if last_error is not None:
        payload = {"articles": [], "error": str(last_error)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(0.5)
    return payload.get("articles", [])


def _fetch_gdelt_period(
    bear_start: pd.Timestamp,
    pre_days: int,
    post_days: int,
    maxrecords: int,
    cache_dir: Path,
    refresh: bool,
    timeout: int,
    retries: int,
) -> list[dict]:
    cache = cache_dir / _cache_name(bear_start, "ALL_TERMS", pre_days, post_days)
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return payload.get("articles", [])

    start = (bear_start - pd.Timedelta(days=pre_days)).floor("D")
    end = (bear_start + pd.Timedelta(days=post_days)).floor("D") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    params = {
        "query": _query_all_terms(),
        "mode": "artlist",
        "format": "json",
        "startdatetime": _dt_arg(start),
        "enddatetime": _dt_arg(end),
        "maxrecords": str(maxrecords),
        "sort": "hybridrel",
    }
    payload = {"articles": []}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                GDELT_DOC_URL,
                params=params,
                headers={"User-Agent": "LPBot research; no-lookahead classifier prototype"},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2.0 * attempt)
    if last_error is not None:
        payload = {"articles": [], "error": str(last_error)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(0.5)
    return payload.get("articles", [])


def _article_text(article: dict) -> str:
    return f"{article.get('title', '')} {article.get('domain', '')}".lower()


def _term_hits(articles: list[dict], category: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for term in TERM_SETS[category]:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")
        count = sum(len(pat.findall(_article_text(a))) for a in articles)
        if count:
            hits[term] = count
    return hits


def _seen_ts(article: dict) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(article.get("seendate"), utc=True, errors="coerce")


def _score_period(row: pd.Series, pre_days: int, post_days: int, maxrecords: int, cache_dir: Path, refresh: bool, timeout: int, retries: int) -> dict:
    bear_start = pd.to_datetime(row["bear_start"], utc=True)
    cutoff = bear_start.floor("D") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    category_rows = []
    score: dict[str, int] = {}
    term_evidence: dict[str, dict[str, int]] = {}
    top_titles: dict[str, list[str]] = {}
    articles = _fetch_gdelt_period(bear_start, pre_days, post_days, maxrecords, cache_dir, refresh, timeout, retries)
    pre_all = [a for a in articles if pd.notna(_seen_ts(a)) and _seen_ts(a) <= cutoff]
    post_all = [a for a in articles if pd.notna(_seen_ts(a)) and _seen_ts(a) > cutoff]

    for category in TERM_SETS:
        pre_articles = [a for a in pre_all if _term_hits([a], category)]
        hits = _term_hits(pre_articles, category)
        score[category] = int(sum(hits.values()))
        term_evidence[category] = hits
        top_titles[category] = [str(a.get("title", ""))[:180] for a in pre_articles[:5]]
        category_rows.append(
            {
                "category": category,
                "articles_fetched_pm_window": len(articles),
                "articles_used_no_lookahead": len(pre_articles),
                "articles_after_start_context_only": len(post_all),
                "term_hits_used": score[category],
            }
        )

    sorted_scores = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_scores[0][1] == 0:
        news_type = "UNCLASSIFIED"
        borderline = True
    elif len(sorted_scores) > 1 and sorted_scores[0][1] - sorted_scores[1][1] <= max(2, int(sorted_scores[0][1] * 0.15)):
        news_type = "UNCLASSIFIED"
        borderline = True
    else:
        news_type = sorted_scores[0][0]
        borderline = False

    existing = str(row["assigned_bear_type"])
    return {
        "bear_start": row["bear_start"],
        "bear_end": row["bear_end"],
        "existing_rules_type": existing,
        "news_based_type": news_type,
        "news_agrees_with_rules": news_type == existing,
        "news_borderline": borderline,
        "panic_news_score": score["PANIC_BEAR"],
        "inflation_news_score": score["INFLATION_BEAR"],
        "geopolitical_news_score": score["GEOPOLITICAL_BEAR"],
        "raw_term_evidence_json": json.dumps(term_evidence, sort_keys=True),
        "top_titles_json": json.dumps(top_titles, ensure_ascii=False),
        "category_counts_json": json.dumps(category_rows, sort_keys=True),
        "best_hindsight_asset_return_over_bear_period": row["best_hindsight_asset_return_over_bear_period"],
        "all_inputs_no_lookahead": row["all_inputs_no_lookahead"],
        "gdelt_window_start": (bear_start - pd.Timedelta(days=pre_days)).date().isoformat(),
        "gdelt_window_end": (bear_start + pd.Timedelta(days=post_days)).date().isoformat(),
        "scored_articles_cutoff_utc": cutoff.isoformat(),
        "post_start_articles_context_only": True,
    }


def main() -> int:
    args = _parse_args()
    inp = Path(args.classifications)
    if not inp.exists():
        raise SystemExit(f"Missing classifications file: {inp}. Run scripts/classify_bear_periods.py first.")
    d = pd.read_csv(inp)
    rows = []
    for _, row in d.iterrows():
        rows.append(
            _score_period(
                row,
                args.pre_days,
                args.post_days,
                args.maxrecords,
                Path(args.cache_dir),
                args.refresh,
                args.timeout,
                args.retries,
            )
        )
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    unclassified = out[out["existing_rules_type"].eq("UNCLASSIFIED")]
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    print("=" * 120)
    print("GDELT NEWS-BASED BEAR CLASSIFIER PROTOTYPE")
    print("=" * 120)
    print(f"Rows: {len(out)}")
    print(f"Saved: {out_path}")
    print("Fetched window: bear_start -3d to bear_start +3d by default.")
    print("Scoring: no-look-ahead articles only, seendate <= bear_start 23:59:59 UTC.")
    print()
    print(
        out[
            [
                "bear_start",
                "existing_rules_type",
                "news_based_type",
                "news_agrees_with_rules",
                "news_borderline",
                "panic_news_score",
                "inflation_news_score",
                "geopolitical_news_score",
                "best_hindsight_asset_return_over_bear_period",
            ]
        ].to_string(index=False)
    )
    print()
    print("Existing UNCLASSIFIED rows:")
    if unclassified.empty:
        print("  none")
    else:
        print(
            unclassified[
                [
                    "bear_start",
                    "news_based_type",
                    "news_borderline",
                    "panic_news_score",
                    "inflation_news_score",
                    "geopolitical_news_score",
                    "raw_term_evidence_json",
                ]
            ].to_string(index=False)
        )
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
