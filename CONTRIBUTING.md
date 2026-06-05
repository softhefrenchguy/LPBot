# Contributing

This repo is research-heavy, so changes should be easy to audit later.

## Change Rules

- Keep live-system changes separate from research experiments where practical.
- Do not commit generated data, artifacts, logs, charts, fitted models, or secrets.
- Add new generated output paths under `data/`, `artifacts/`, or `logs/`.
- Update docs when adding scripts or changing the live workflow.

## Script Classification

Every new script should fit one category:

- `ACTIVE`: used by daily paper trading or cron.
- `VALIDATED`: reusable backtest or diagnostic with a meaningful result.
- `RESEARCH`: exploratory, not relied on for operations.
- `LEGACY`: old code retained for context only.

Update [docs/SCRIPT_INVENTORY.md](docs/SCRIPT_INVENTORY.md) when adding or promoting scripts.

## Before Commit

Run:

```bash
git status --short
git diff --cached --check
```

For active Python scripts, run:

```bash
python -m py_compile scripts/paper_trade_checklist.py scripts/market_tracker.py scripts/news_sentiment.py
```

## Staging

Prefer explicit staging:

```bash
git add README.md docs scripts lpbot src config deploy
```

Avoid:

```bash
git add .
```

unless you have checked ignored and untracked files carefully.
