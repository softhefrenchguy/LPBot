# Documentation Index

Use this page as the navigation point for the repo.

## Operational Docs

| Document | Use |
|---|---|
| [Operations](OPERATIONS.md) | Daily paper-trading workflow, env vars, cron, manual runs. |
| [Hetzner Runbook](HETZNER_RUNBOOK.md) | Safe server operations guide with placeholders only. |
| [Artifact Policy](ARTIFACT_POLICY.md) | What is ignored by git and why. |
| [GitHub Setup](GITHUB_SETUP.md) | How to stage, commit, and push safely. |

## Codebase Docs

| Document | Use |
|---|---|
| [Project Map](PROJECT_MAP.md) | High-level map of live, research, legacy, and generated areas. |
| [Script Inventory](SCRIPT_INVENTORY.md) | Script-by-script status: active, validated, research, legacy. |

## Repo Root Docs

| Document | Use |
|---|---|
| [README](../README.md) | GitHub landing page. |
| [Security](../SECURITY.md) | Secret handling and incident response. |
| [Contributing](../CONTRIBUTING.md) | Rules for changing code and adding scripts. |

## Status Labels

- `ACTIVE`: used by the current daily paper-trading workflow.
- `VALIDATED`: kept because it produced a useful/reusable backtest or diagnostic result.
- `RESEARCH`: experimental, diagnostic, or exploratory.
- `LEGACY`: old workflow retained for context.

## Maintenance Rule

When a new script is added, update [Script Inventory](SCRIPT_INVENTORY.md). When a new artifact path is added, update [Artifact Policy](ARTIFACT_POLICY.md).
