# Security

LPBot handles API keys, Discord webhooks, SSH keys, and operational trading logs. Treat the repo as source-code-only. Secrets belong outside git.

## Never Commit

- `.env`
- Claude/Anthropic API keys
- EIA or other data-provider API keys
- Discord webhook URLs
- GitHub personal access tokens
- SSH private keys
- Hetzner server IPs, usernames, or private host config
- live operational CSVs or logs

## Accepted Secret Locations

- Local `.env`, ignored by git.
- Windows Credential Manager / Git Credential Manager for GitHub authentication.
- Password manager such as 1Password or Bitwarden.
- GitHub Actions Secrets for CI-only usage.
- SSH agent / local `~/.ssh` directory for server access.

## If A Secret Is Accidentally Committed

1. Rotate/revoke the secret immediately.
2. Remove it from the current tree.
3. Assume git history is compromised.
4. If necessary, rewrite history with a tool such as `git filter-repo`.
5. Force-push only after confirming collaborators understand the impact.

Deleting the line in a later commit is not enough. Git history still contains it.

## Public Vs Private Repo

A private GitHub repo is not a secrets manager. Private repos can still leak through:

- collaborators,
- compromised GitHub sessions,
- CI logs,
- accidental repo visibility changes,
- local clone backups,
- copied patches or screenshots.

Keep secrets out of git regardless of repo visibility.
