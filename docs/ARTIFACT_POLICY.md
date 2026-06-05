# Artifact Policy

The GitHub repo should contain source code, docs, config examples, and deployment files. It should not contain local market data, model artifacts, paper logs, charts, server backups, or secrets.

## Ignored By Design

- `.env`: secrets and host-specific paths.
- `data/`: downloaded price/funding/news/market data.
- `artifacts/`: backtest outputs, paper logs, charts, Discord debug payloads.
- `logs/`: cron/runtime logs.
- `models/`: fitted model binaries.
- `ec2_backup/`: server backup material.
- `.vscode/`: local editor state.
- `*.csv`, `*.html`, `*.joblib`, `*.pkl`: generated files unless explicitly moved into docs as a small example.

## Why

Generated files are large, change daily, and can accidentally contain sensitive operational details. Keeping them out of git makes the repo safe to push and easier to review.

## How To Inspect Live Artifacts

Pull artifacts from Hetzner when needed:

```bash
scp user@hetzner:~/LPBot/artifacts/paper_trade/daily_checks_log.csv artifacts/paper_trade/hetzner/
scp user@hetzner:~/LPBot/artifacts/paper_trade/completed_trades.csv artifacts/paper_trade/hetzner/
```

These files remain local and ignored.

## If An Artifact Must Be Shared

Use one of these instead of committing raw live files:

- A short excerpt in documentation.
- A regenerated chart with no secrets or webhook/debug payloads.
- A small fixture under a dedicated `tests/fixtures/` path, with synthetic or sanitized data.

Do not commit `.env`, Discord webhook URLs, API keys, or full live logs.
