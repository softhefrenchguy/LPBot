# LPBot

LPBot is a modular systematic trading research and paper-trading framework. It separates core strategy logic from optional overlays, while keeping live paper-trading operations, dashboarding, and research diagnostics in one organised repo.

The current production workflow is a paper-trading system for a regime-routed ETH/BTC strategy stack. It produces a daily Discord summary covering ETH, BTC, gold rotation, macro markets, news, risk checks, and completed trade records. It does not execute real-money trades unless explicitly configured separately.

## What This Repo Contains

| Area | Purpose |
|---|---|
| Live paper system | Daily ETH/BTC paper-trading checklist, Discord report, completed trade log, feed checks. |
| Web dashboard | FastAPI + React dashboard for live state, benchmark comparison, trades, markets, and news. |
| Strategy research | Backtests, stress tests, diagnostics, and rejected strategy experiments. |
| Market intelligence | RSS macro news classifier, market tracker, PAXG/gold rotation diagnostics. |
| Operations | Hetzner cron/Docker runbooks, environment examples, artifact policy. |

## Design Principles

- Core strategy must be robust standalone.
- Overlays are optional, gated, and reversible.
- Backtests are treated as research until stress-tested and paper-tracked.
- Generated data, live artifacts, secrets, and model files stay out of Git.
- Branches/tags are used to freeze independently developed components.

## Current Production Path

The live paper workflow is:

1. Refresh market/news/funding/regime data.
2. Run defensive model refresh.
3. Run `scripts/paper_trade_checklist.py`.
4. Write daily CSV artifacts locally.
5. Send Discord summary.
6. Serve the dashboard from the latest CSV artifacts.

Main runner:

```bash
scripts/run_paper_check_daily.sh
```

Windows/local runner:

```powershell
.\scripts\run_paper_check_daily.ps1
```

## Dashboard

Dashboard services:

- API: FastAPI on port `8000`
- Frontend: React/Tailwind/Recharts on port `3000`

Docker:

```bash
docker-compose up -d --build dashboard-api dashboard
```

Local backend:

```bash
uvicorn scripts.dashboard_api:app --host 0.0.0.0 --port 8000
```

Frontend source lives in `dashboard/`.

## Current Strategy Components

| Component | Status | Notes |
|---|---:|---|
| ETH offensive sleeve | Live paper | EMA 21/55/144, 3-day confirmation, regime-routed sizing. |
| ETH defensive sleeve | Live paper | Funding/perp direction-event model. |
| BTC sleeve | Live paper | EMA 15/40/120, 5-day confirmation, BTC funding diagnostics. |
| Gold/PAXG sleeve | Paper tracked | BEAR + flat ETH + PAXG EMA alignment gate. |
| Market tracker | Live info | BTC, oil, gas, metals, equities, bonds, dollar, wheat. |
| News watcher | Live info | 12-hour RSS filter plus Claude macro-event classifier. |
| Mode A (state-based sizing) | Validated, not yet promoted to live | Scales `alloc_eth`/`alloc_btc` by regime conviction. Re-validated on corrected data: +0.246 Sharpe over the 2021-2024 OOS walk-forward average vs. the same-period baseline. See [Research Log](#research-log-data-correction--lp-overlay-investigation) below. |
| LP overlay (Uniswap V3) | Researched, execution code built, **on hold** | Full-stack portfolio integration came back negligible-to-negative (+0.002 Sharpe standalone, -0.008 once sharing idle capital with the CHOP overlay). Not deployed. See [Research Log](#research-log-data-correction--lp-overlay-investigation) and [`execution/README.md`](execution/README.md). |

## Repository Layout

```text
LPBot/
  scripts/       Live runners, backtests, diagnostics, research scripts
  dashboard/     React dashboard frontend
  lpbot/         Package modules and retained components
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
8. [Dashboard README](dashboard/README.md)

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
  scripts/dashboard_api.py `
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
- private host credentials
- live CSV artifacts
- generated model files

See [Artifact Policy](docs/ARTIFACT_POLICY.md) and [Security](SECURITY.md).

## Research Log: Data Correction & LP Overlay Investigation

A multi-day research thread (Aug 2026) that (1) found and fixed a silent data gap affecting the
production reference numbers, (2) re-validated the existing Mode A sizing overlay on corrected
data, and (3) investigated adding a Uniswap V3 LP overlay end-to-end — concluding, honestly, that
it isn't worth deploying. Kept here as a record, per the project's design principle of tracking
what was tried and didn't pan out, not just what shipped.

### 1. Data gap fix and corrected reference Sharpe

`scripts/backtest_btc_full_stack.py`'s Binance downloader was silently skipping a data
unavailability window: Binance's paginated API returns the *next available* candle past a gap
instead of erroring, so a genuine 164-day hole in `*USDC` pair data (2022-09-30 to 2023-03-11,
around the FTX collapse) was going undetected and quietly compressing that stretch out of the
backtest. Fixed at the source with `_fill_symbol_gaps()`, which detects any missing calendar day
for a `*USDC` symbol and backfills it from the equivalent `*USDT` pair. A full-history rescan after
the fix also caught a second, larger, unrelated gap in `SOLUSDC`.

This changes the numbers: the production reference Sharpe used throughout the codebase was
**1.762 (uncosted) / 1.633 (costed)**, computed before the gap fix. Corrected values are **1.660
(uncosted) / 1.528 (costed)**. All references to the old figures — `check_baseline_rebalance_cost.py`,
`backtest_continuous_regime_integrations.py`, `backtest_mode_a_validation.py`, `analyse_daily.py` —
were updated to the corrected numbers; a repo-wide grep confirmed no remaining "1.762"/"1.633"
mentions.

### 2. Mode A re-validated on corrected data

`scripts/backtest_mode_a_validation.py` re-ran its full 4-step validation (walk-forward,
attribution, vol-filter double-counting check, rebalancing cost) against the corrected dataset.
Mode A holds up: baseline 2021-2024 OOS walk-forward average is 1.086, Mode A's is 1.332 — a
genuine **+0.246 Sharpe** improvement, like-for-like on the same period. (Comparing against the
full 2019-2024 baseline Sharpe of 1.660 instead would be an apples-to-oranges mistake — that
period includes 2020's outlier 2.796 Sharpe, which isn't part of Mode A's walk-forward test.)

### 3. LP overlay investigation (Uniswap V3), concluding "not worth it"

Chain of research, each step feeding the next:

- **Whipsaw/gate research** (`scripts/check_lp_gate_dry_run.py`,
  `scripts/check_lp_panic_gate_comparison.py`): the LP overlay's exit gate
  (`eth_off_active`/`btc_off_active`) isn't actually the mechanism protecting against crashes —
  that's the separate `rolling_vol_20d < LP_VOL_ON` volatility filter. Swapping the gate to the
  continuous regime score's panic state outright made things worse (39.5% LP-on during crash
  events vs. 16.5%); augmenting the existing gate with panic-state as an OR condition gave a real
  but small improvement (11 fewer crash-exposure days) — judged not "meaningful" enough to
  implement. Along the way, `LP_VOL_ON` was also found stale at 0.25 and recalibrated to 0.57 in
  `.env`.
- **Full-stack integration** (`scripts/backtest_lp_full_stack.py`): rather than testing LP in
  isolation, integrated it into the actual portfolio backtest across the full corrected 2019-2024
  history, using real Graph-fetched fee/volume data from 2024-04-01 onward (10.30% annualized fee
  rate) and flagging pre-2024 fee assumptions as unverified. Result: baseline 1.660 →
  capacity-aware full-stack 1.652 (**-0.008 Sharpe**). Negative at the full-stack level, and worse
  under a lower-fee-rate sensitivity check.
- **CHOP vs. LP head-to-head** (`scripts/backtest_lp_chop_comparison.py`): both LP and the existing
  mean-reversion/CHOP overlay compete for the same idle capital window. Standalone, CHOP alone
  contributes +0.066 Sharpe vs. LP alone's +0.002 — roughly 33x. No capital-sharing split rule
  tested (50/50, proportional-by-Sharpe, day-by-day severity winner, LP-first/CHOP-leftover, etc.)
  robustly beat just running CHOP alone; the one variant that looked better (+0.012) affected only
  15 days and is judged noise. The two triggers' overlap is small (6% of LP-eligible days,
  concentrated in April 2022) and weakly negatively correlated (-0.134) — not redundant signals,
  but LP's slice of the opportunity is small regardless.

**Conclusion:** the LP overlay does not clear its own bar at the portfolio level. The execution
code to run it live was still built and safety-fixed (see below), so the option stays available if
the underlying economics change (materially better verified fee yields, cheaper execution costs,
or a real answer to the capital-sharing question) — but it is not currently recommended to deploy.

### 4. Execution layer built alongside (on hold)

Two generations of live execution code live under `execution/` — see
[`execution/README.md`](execution/README.md) for full detail:

- **`legacy_LP_Automation/`** — first version, screen-scraping trigger, superseded.
- **`bot_20/`** — second version, on-chain polling + decision layer. This session: added real
  slippage protection (previously hardcoded to zero minimum output on every mint/withdraw — a real
  sandwich-attack exposure) and an explicit max-position-USD cap; migrated the full chain config
  from Arbitrum to Ethereum mainnet, with every contract address independently verified via live
  `eth_call`s rather than trusted from memory; added full-range minting as the default position
  mode given mainnet gas economics.

Both wallets' private keys (found in plaintext in `.env` files during the audit) were confirmed
abandoned/unused and removed rather than rotated. Given the negative full-stack finding above,
neither is currently intended to run live.

## Frozen Research Branches

Historical/frozen components may exist as branches or tags, including:

- `feat/hmm-v1` / `regime-v1.0`
- `feat/elastic-net-v1` / `elasticnet-vol-v1.0-balanced`
- `feat/lp-overlay-v1` / `lp-overlay-v1-freeze`

`main` is the stable organised production/research baseline.
