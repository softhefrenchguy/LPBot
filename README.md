# LPBot

LPBot is the research and paper-trading repo for the ETH/BTC strategy stack.

The current live system is a paper-trading bot: it builds daily regime, offensive, defensive, BTC, gold, market, and news diagnostics, then posts a Discord summary. It does not execute real-money trades.

## Current Live Stack

- ETH sleeve: EMA 21/55/144 offensive signal, regime routing, defensive funding model, paper trade tracking.
- BTC sleeve: EMA 21/55/144 with 5-day confirmation, BTC funding features, regime routing, paper trade tracking.
- Gold sleeve: PAXG paper rotation gate for BEAR + flat ETH + PAXG EMA alignment.
- Market tracker: daily macro/commodity/crypto snapshot.
- News watcher: 12-hour RSS filter plus Claude macro event classifier.
- Operations target: Hetzner cron + Docker Compose.

## Repository Map

- `scripts/`: live runners, backtests, diagnostics, and research scripts.
- `lpbot/`: older package modules and retained strategy components.
- `config/`, `deploy/`, `Dockerfile`, `docker-compose.yml`: deployment and runtime config.
- `docs/`: project map, script inventory, operating instructions, and artifact policy.
- `data/`, `artifacts/`, `logs/`, `models/`: local/generated only, intentionally ignored by git.

## Start Here

Read these in order:

1. [Project Map](docs/PROJECT_MAP.md)
2. [Script Inventory](docs/SCRIPT_INVENTORY.md)
3. [Operations](docs/OPERATIONS.md)
4. [Artifact Policy](docs/ARTIFACT_POLICY.md)
5. [Hetzner Runbook](docs/HETZNER_RUNBOOK.md)
6. [GitHub Setup](docs/GITHUB_SETUP.md)

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with local secrets and paths. Never commit `.env`.

## Daily Paper Check

Linux/Hetzner:

```bash
scripts/run_paper_check_daily.sh
```

Windows/local:

```powershell
.\scripts\run_paper_check_daily.ps1
```

## Important Rule

Generated data and paper-trading artifacts are not committed. The repo should contain code, configuration examples, and documentation only. Pull live CSVs from Hetzner or rebuild them locally when needed.
