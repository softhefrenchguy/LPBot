# GitHub Setup

The repo remote is expected to be:

```bash
git remote -v
```

For this workspace it is currently `origin` pointing at `softhefrenchguy/LPBot`.

## Before Pushing

Run:

```bash
git status --short
git diff -- .gitignore README.md .env.example docs
```

Check that generated directories are not staged:

```bash
git status --short data artifacts logs models ec2_backup .vscode
```

## Stop Tracking Old Raw CSVs

If raw CSVs were previously committed, remove them from the current index while keeping local copies:

```bash
git rm --cached data/BTCUSDC_1m.csv data/ETHUSDC_1m.csv
```

This does not delete local files. It only prevents future commits from carrying them.

## Stage Source And Docs

Use explicit paths rather than `git add .`:

```bash
git add .gitignore .env.example README.md docs
git add scripts lpbot src config deploy Dockerfile docker-compose.yml requirements.txt README_DEPLOY_AWS.md
```

Then inspect:

```bash
git status --short
```

## Commit And Push

```bash
git commit -m "Organise LPBot repo for GitHub"
git push origin main
```

If the branch is not `main`, check with:

```bash
git branch --show-current
```
