# Operations

This documents the current daily paper-trading workflow.

## Runtime

- Primary host: Hetzner server.
- Runtime: Docker Compose service `lpbot`.
- Secrets/config: `.env` on the host, not committed.
- Main output: Discord daily summary plus CSVs under `artifacts/paper_trade/`.

## Required Environment

Copy `.env.example` to `.env` and fill values.

Important keys:

- `DISCORD_WEBHOOK_URL`: Discord webhook for daily summaries.
- `ANTHROPIC_API_KEY`: Claude API key for `news_sentiment.py`.
- `CLAUDE_MODEL`: defaults to `claude-sonnet-4-6`.
- `PAPER_START_DATE`: paper tracking start date.
- `FUNDING_CSV`: ETH funding/perp feature CSV.
- `PERP_CSV`: ETH perp CSV for defensive refresh.
- `REGIME_CSV`: immutable regime snapshot path.
- `PAXG_CSV`: PAXG daily price CSV.
- `BTC_DAILY_CSV`: BTC daily price CSV.
- `BTC_PERP_CSV`: BTC perp/funding feature CSV.

## Daily Schedule

Recommended cron order:

```cron
0 7 * * * cd ~/LPBot && python scripts/download_btc_daily.py >> logs/btc_feed.log 2>&1
10 */4 * * * cd ~/LPBot && python scripts/download_btc_perp_features.py >> logs/btc_perp_feed.log 2>&1
45 7 * * * cd ~/LPBot && scripts/run_market_tracker_daily.sh >> logs/market_tracker_cron.log 2>&1
50 7 * * * cd ~/LPBot && scripts/run_news_sentiment_daily.sh >> logs/news_sentiment_cron.log 2>&1
0 8 * * * cd ~/LPBot && scripts/run_paper_check_daily.sh >> logs/paper_check_cron.log 2>&1
```

Exact timing can vary, but market/news/defensive refreshes should complete before the paper checklist.

## Manual Runs

Paper checklist:

```bash
scripts/run_paper_check_daily.sh
```

Market tracker:

```bash
scripts/run_market_tracker_daily.sh
```

News watcher:

```bash
scripts/run_news_sentiment_daily.sh
```

Defensive refresh:

```bash
scripts/run_defensive_live_refresh.sh
```

## Active Paper Outputs

- `artifacts/paper_trade/daily_check_YYYYMMDD.csv`: daily status row.
- `artifacts/paper_trade/daily_checks_log.csv`: daily history.
- `artifacts/paper_trade/completed_trades.csv`: closed offensive trades.
- `artifacts/paper_trade/btc_daily_checks_log.csv`: BTC paper state/history.
- `artifacts/paper_trade/defensive_live.csv`: latest defensive model output.
- `artifacts/news/news_log.csv`: news classifier daily log.
- `artifacts/markets/market_tracker.csv`: latest market tracker rows.

These are generated operational records and are ignored by git.

## Current Strategy Notes

- BEAR regime forces the ETH offensive trade closed immediately.
- EMA 3-day break is still tracked as diagnostic, but BEAR close takes priority.
- BTC uses 5-day confirmation.
- PAXG is paper-tracked as a gold rotation candidate; it does not execute real trades.
- News watcher only reads recent RSS headlines, not full articles.

## Funding Feed Stability

Checked on 2026-07-18 after the incremental feed fix (`d7ba02a`):

- ETH perp feed cron is active through `scripts/run_eth_perp_feed.sh`.
- BTC funding feed cron is active through `scripts/download_btc_perp_features.py`.
- Latest inspected files on Hetzner were fresh at `2026-07-18 12:10 UTC`.
- Current BTC funding age was `5.49h`, below the `9.00h` alert threshold.
- Recent logs show continuous successful writes. Older timeout/container-restart traces remain in logs but are not current.

Verification command:

```bash
cd ~/LPBot
grep -E "(OK|FAIL|age_h)" logs/perp_feed.log | tail -50
tail -50 logs/btc_perp_feed.log
```
