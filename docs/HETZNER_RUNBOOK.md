# Hetzner Runbook

This is the safe public runbook for operating LPBot on Hetzner. It intentionally does not include the server IP, SSH private key, passwords, Discord webhook, or API keys.

## What Lives On Hetzner

Expected repo path:

```bash
~/LPBot
```

Expected runtime:

```bash
docker-compose up -d
```

Host-local files that must exist but are not committed:

- `~/LPBot/.env`
- `~/LPBot/data/`
- `~/LPBot/artifacts/`
- `~/LPBot/logs/`

## SSH

Use your local SSH config rather than committing connection details:

```sshconfig
Host lpbot-hetzner
  HostName <server-ip>
  User <server-user>
  IdentityFile <path-to-private-key>
```

Then connect with:

```bash
ssh lpbot-hetzner
```

Do not commit the SSH config if it contains the real IP or key path.

## Deploy Code

On the server:

```bash
cd ~/LPBot
git fetch origin
git checkout main
git pull --ff-only origin main
docker-compose build
docker-compose up -d
```

The production branch is `main`. (This section used to say `feat/low-risk-investr`, which no longer
exists on origin, so following it would fail the pull and silently leave the server on old code.)
If the production branch changes again, replace `main` above with the deployed branch.

### What updates on `git pull` vs. what needs a rebuild

`scripts/run_paper_check_daily.sh` defaults to `USE_DOCKER=1`: it refreshes the 5m price file and runs
the daily paper check inside the `lpbot` container, piping `scripts/paper_trade_checklist.py` in from
the host checkout. `docker-compose.yml` mounts `./scripts`, `./config`, `./data`, `./artifacts` and
`./logs` from the host. So:

- **`git pull` on the host is what updates the checklist script and config** that produce the daily
  Discord summary. No rebuild is needed for those.
- **`docker-compose build` is only needed for the Python environment** (`requirements.txt`, the
  Dockerfile). The pins need the Python 3.13 base the Dockerfile uses. If a rebuild fails, the old
  image keeps running, with the newly pulled scripts on top of the old libraries.
- Host-cron jobs such as `python scripts/download_btc_daily.py` use the host's `python3`, not the
  container.

### Verify what the server is actually running

```bash
cd ~/LPBot
git rev-parse --abbrev-ref HEAD && git log -1 --format='%h %ad %s'
docker-compose exec -T lpbot python -c "import sys, pandas, numpy; print(sys.version.split()[0], pandas.__version__, numpy.__version__)"
```

Expect branch `main` at the latest commit you pushed, and (once the pinned image is running) Python
3.13.x with pandas 3.0.5 and numpy 2.5.2. A different branch or an old commit date means the daily
Discord summaries are still coming from old code; older library versions mean the image was never
rebuilt with the pins.

## Required `.env`

Create `~/LPBot/.env` from `.env.example` and fill real values on the server:

```bash
cp .env.example .env
nano .env
```

Required operational keys:

- `DISCORD_WEBHOOK_URL`
- `ANTHROPIC_API_KEY`
- `PAPER_START_DATE`
- `FUNDING_CSV`
- `PERP_CSV`
- `REGIME_CSV`
- `PAXG_CSV`
- `BTC_DAILY_CSV`
- `BTC_PERP_CSV`

Optional research key:

- `EIA_API_KEY`

## Cron

Inspect current cron:

```bash
crontab -l
```

Recommended schedule:

```cron
0 7 * * * cd ~/LPBot && python scripts/download_btc_daily.py >> logs/btc_feed.log 2>&1
10 */4 * * * cd ~/LPBot && python scripts/download_btc_perp_features.py >> logs/btc_perp_feed.log 2>&1
45 7 * * * cd ~/LPBot && scripts/run_market_tracker_daily.sh >> logs/market_tracker_cron.log 2>&1
50 7 * * * cd ~/LPBot && scripts/run_news_sentiment_daily.sh >> logs/news_sentiment_cron.log 2>&1
0 8 * * * cd ~/LPBot && scripts/run_paper_check_daily.sh >> logs/paper_check_cron.log 2>&1
```

## Manual Checks

Run the daily paper checklist:

```bash
cd ~/LPBot
scripts/run_paper_check_daily.sh
```

Check recent logs:

```bash
tail -100 logs/paper_check_cron.log
tail -100 logs/news_sentiment.log
tail -100 logs/market_tracker.log
tail -100 logs/defensive_live_refresh.log
```

Check Docker:

```bash
docker-compose ps
docker-compose logs --tail=100 lpbot
```

## Pull Operational Artifacts Locally

From your PC:

```bash
scp lpbot-hetzner:~/LPBot/artifacts/paper_trade/daily_checks_log.csv artifacts/paper_trade/hetzner/
scp lpbot-hetzner:~/LPBot/artifacts/paper_trade/completed_trades.csv artifacts/paper_trade/hetzner/
scp lpbot-hetzner:~/LPBot/artifacts/paper_trade/btc_daily_checks_log.csv artifacts/paper_trade/hetzner/
```

These files are ignored locally and should not be committed.

## What Not To Commit

- Real `.env`
- Server IP/user/SSH private key
- Discord webhook URL
- Claude/EIA/API keys
- Live CSV logs
- `artifacts/`, `data/`, `logs/`, `models/`
