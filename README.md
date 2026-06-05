# LPBot

Research, backtesting, and paper-trading infrastructure for a regime-routed ETH/BTC strategy stack.

LPBot currently runs as a paper-trading system. It produces a daily Discord summary covering ETH, BTC, gold rotation, macro markets, news, risk checks, and completed trade records. It does not execute real-money trades.

## What This Repo Contains

| Area | Purpose |
|---|---|
| Live paper system | Daily ETH/BTC paper-trading checklist, Discord report, completed trade log, feed checks. |
| Strategy research | Backtests, stress tests, diagnostics, and rejected strategy experiments. |
| Market intelligence | RSS macro news classifier, market tracker, PAXG/gold rotation diagnostics. |
| Operations | Hetzner cron/Docker runbooks, environment examples, artifact policy. |

## Current Production Path

The live paper workflow is:

1. Refresh market/news/funding/regime data.
2. Run defensive model refresh.
3. Run `scripts/paper_trade_checklist.py`.
4. Write daily CSV artifacts locally.
5. Send Discord summary.

Main runner:

```bash
scripts/run_paper_check_daily.sh
```

Windows/local runner:

```powershell
.\scripts\run_paper_check_daily.ps1
```

## Current Strategy Components

| Component | Status | Notes |
|---|---:|---|
| ETH offensive sleeve | Live paper | EMA 21/55/144, 3-day confirmation, regime-routed sizing. |
| ETH defensive sleeve | Live paper | Funding/perp direction-event model. |
| BTC sleeve | Live paper | EMA 21/55/144, 5-day confirmation, BTC funding features. |
| Gold/PAXG sleeve | Paper tracked | BEAR + flat ETH + PAXG EMA alignment gate. |
| Market tracker | Live info | BTC, oil, gas, metals, equities, bonds, dollar, wheat. |
| News watcher | Live info | 12-hour RSS filter plus Claude macro-event classifier. |

## Repository Layout

```text
LPBot/
  scripts/       Live runners, backtests, diagnostics, research scripts
  lpbot/         Package modules and older retained components
  src/           Original regime/LP infrastructure
  config/        Static configuration
  deploy/        Deployment helpers
  docs/          Runbooks, inventories, operating policy
```

Generated/local directories are intentionally ignored:

```text
data/        downloaded market/news/funding data
artifacts/   paper logs, backtest outputs, charts
logs/        cron/runtime logs
models/      local fitted model files
```

## Documentation

Start here:

1. [Documentation Index](docs/README.md)
2. [Project Map](docs/PROJECT_MAP.md)
3. [Script Inventory](docs/SCRIPT_INVENTORY.md)
4. [Operations](docs/OPERATIONS.md)
5. [Hetzner Runbook](docs/HETZNER_RUNBOOK.md)
6. [Artifact Policy](docs/ARTIFACT_POLICY.md)
7. [GitHub Setup](docs/GITHUB_SETUP.md)

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with local secrets and host-specific paths. Never commit `.env`.

## Basic Checks

Compile the active paper/live scripts:

```powershell
python -m py_compile `
  scripts/paper_trade_checklist.py `
  scripts/market_tracker.py `
  scripts/news_sentiment.py `
  scripts/direction_event_model_v1.py `
  scripts/download_btc_daily.py `
  scripts/download_btc_perp_features.py `
  scripts/download_paxg_daily.py `
  scripts/download_perp_features.py `
  scripts/check_feed_health.py `
  scripts/regime_classifier_daily_snapshot.py `
  scripts/regime_classifier_v2.py
```

## Data And Secrets Policy

This repo should contain source code, docs, config examples, and deployment instructions only.

Do not commit:

- `.env`
- API keys
- Discord webhooks
- SSH keys
- Hetzner server IPs or private host details
- live CSV artifacts
- generated model files

See [Artifact Policy](docs/ARTIFACT_POLICY.md) and [Security](SECURITY.md).

## Current GitHub Branch

Current organised branch:

```text
feat/low-risk-investr
```

Use this branch for review before merging to the main production branch.
