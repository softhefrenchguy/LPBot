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
| Mode A (state-based sizing) | Researched, **not recommended** | Scales `alloc_eth`/`alloc_btc` by regime conviction. An earlier pass claimed +0.246 Sharpe OOS; after the lookahead-bias fix and the real-cost correction, the true edge is +0.050 uncosted and Mode A doesn't beat baseline even before its own extra rebalancing cost is added (1.509 vs 1.510). See [Research Log](#research-log-data-correction--lp-overlay-investigation) below. |
| LP overlay (Uniswap V3) | Researched, execution code built, **on hold** | Full-stack portfolio integration came back negligible-to-negative (+0.001 Sharpe standalone, -0.008 once sharing idle capital with the CHOP overlay). Not deployed. See [Research Log](#research-log-data-correction--lp-overlay-investigation) and [`execution/README.md`](execution/README.md). |

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
**1.762 (uncosted) / 1.633 (costed)**, computed before the gap fix. The first correction pass
produced **1.660 (uncosted) / 1.528 (costed)**.

### 1b. Lookahead-bias fix (second, later correction layer)

A follow-up full-codebase audit (see below) traced every signal-to-return alignment in
`scripts/backtest_eth_btc_portfolio.py` and found that three of the five multiplier layers applied
to `alloc_eth`/`alloc_btc` — the vol-filter, macro-filter, and asymmetric-sizing multipliers — were
computed from **that same day's own close** and applied with no shift, while the base EMA sleeves
and gold rotation were already correctly lagged (`weight_exec = weight_target.shift(1)`). Since a
position sized at day *t*'s open can't legitimately be scaled using information only available
after day *t*'s close, this was a real, mechanistic lookahead bug — and the vol-filter and
asymmetric-sizing layers are both active in the exact config that produced the reference Sharpe
(vol-filter unconditionally, asymmetric-sizing whenever the full-stack config is used). Mode A's
regime-conviction multiplier (`backtest_continuous_regime_integrations.py`) had the identical bug.

Fixed by shifting each multiplier's underlying stats by one day before merging (`_eth_vol_frame`,
`_sp500_macro_frame` in `backtest_eth_btc_portfolio.py`; the `conviction` column computed inside
`run_sleeve`; `_load_states` in `backtest_continuous_regime_integrations.py`), then rerunning the
full validated pipeline. The honest result: the bias was real but its magnitude was small, not the
dramatic inflation lookahead bugs often produce. Corrected production reference: **1.649 (uncosted)
/ 1.521 (costed)** — full-period CAGR actually moved *up* slightly (36.7% → 37.5%) and the
2021-2024 walk-forward OOS average moved *up* too (1.086 → 1.144), because the bias wasn't
uniformly inflating performance, just misdating a few percentage points of sizing each day.

All references to the reference Sharpe across the codebase — `check_baseline_rebalance_cost.py`,
`backtest_continuous_regime_integrations.py`, `backtest_mode_a_validation.py`,
`backtest_walkforward_fullstack.py`, `analyse_daily.py`, `config/portfolio_config.json` — were
updated to these final, lookahead-bias-corrected numbers.

### 1c. Cost model correction (real Kraken fees at actual trading volume — third correction layer)

The 20bps-per-leg cost assumption used throughout every backtest above was calibrated for roughly
$25k-$50k/month Kraken trading volume. Checked against Kraken's real fee schedule and the actual
expected trading volume for this strategy (~$1k-$10k/month), the realistic taker fee is **0.60%
(60bps) per leg**, not 0.20% — Kraken+ (a $4.99/month subscription some users have) does not apply
here, since it only covers Kraken's simple Web/App interface, not the Kraken Pro API this bot
actually trades through. `cost_bps` was corrected from 20 → 60 everywhere it's used (`--cost-bps`/
`--gold-cost-bps` defaults in `backtest_eth_btc_portfolio.py`, `backtest_forex_optimised.py`'s
`--crypto-cost-bps`, `_walkforward_runner.py`'s hardcoded `_apply_meanrev_overlay_series` calls, and
every validation script's own `--cost-bps` default), and the full pipeline rerun.

This has a real, material impact — noticeably larger than either prior correction:

|  | Sharpe (uncosted) | Sharpe (fully turnover-costed) | CAGR | MaxDD |
|---|---|---|---|---|
| At 20bps/leg (prior) | 1.649 | 1.521 | 37.5% | -14.5% |
| **At 60bps/leg (corrected, current)** | **1.510** | **1.121** | **34.2%** | **-15.6%** |
| 2021-2024 walk-forward OOS average | 1.144 → **0.938** | | | |

Per-year walk-forward OOS at the corrected cost: 2021 Sharpe 1.996 (CAGR 60.6%), 2022 Sharpe -0.496
(CAGR 1.0%), 2023 Sharpe 0.163 (CAGR 6.4%), 2024 Sharpe 2.087 (CAGR 49.5%) — still robust (3/4
positive years) but visibly worse than the 20bps figures, especially 2022 and 2023, the two years
with the most rebalancing activity relative to their return. This is the number that should be used
for any forward-looking expectation now, not the 20bps-based ones above — those remain in this log
only as a record of what was corrected, not as current reference figures.

### 2. Mode A re-validated — smaller edge than first reported, and it evaporates once fully costed

`scripts/backtest_mode_a_validation.py`'s 4-step validation was rerun after both the lookahead-bias
fix and the cost-model correction. The original claim from earlier in this research thread — a
"genuine +0.246 Sharpe" 2021-2024 OOS improvement — **does not hold up** on the corrected pipeline.
Three things changed it, each layering on the last:

1. The lookahead-bias fix by itself shrank the gap (walk-forward OOS: baseline 1.144 vs. Mode A's
   1.212, an edge of +0.068, not +0.246).
2. The cost-model correction shrank the gap further: at 60bps, baseline OOS average is 0.938 vs.
   Mode A's 0.988 — an edge of only **+0.050**.
3. Step 3 (vol-filter double-counting check) shows essentially none of that edge is a genuinely new
   signal: with the vol filter on (production config), Mode A's full-period uplift is **-0.001**
   (1.510 vs. 1.509, i.e. no uplift at all); the +0.265 uplift only appears with the vol filter
   switched off, confirming Mode A is substituting for what the vol filter already does, not adding
   orthogonal information.
4. Step 4 (incremental rebalancing cost): Mode A's own extra state-transition churn (49.8
   transitions/year on average) costs real drag once priced. **Mode A does not beat the production
   baseline even before this extra cost is added (1.509 vs. baseline's 1.510), and is further behind
   once its own incremental rebalancing is costed (1.444 vs. 1.510).**

**Conclusion, corrected: Mode A does not clear its own bar either**, for essentially the same
reason LP doesn't — a real but small standalone effect that a more careful accounting (cost, or in
LP's case capacity-sharing) erases entirely. Not recommended for promotion to live in its current
form.

### 3. LP overlay investigation (Uniswap V3), concluding "not worth it"

Chain of research, each step feeding the next:

- **Whipsaw/gate research** (`scripts/check_lp_gate_dry_run.py`,
  `scripts/check_lp_panic_gate_comparison.py`): the LP overlay's exit gate
  (`eth_off_active`/`btc_off_active`) isn't actually the mechanism protecting against crashes —
  that's the separate `rolling_vol_20d < LP_VOL_ON` volatility filter. Swapping the gate to the
  continuous regime score's panic state outright made things worse (39.5% LP-on over full history
  vs. 16.5% for the current gate); augmenting the existing gate with panic-state as an OR condition
  gave a marginally better full-history net effect (net -7.58% vs -7.71%, 1.00pp less cumulative
  IL) but **zero measurable improvement in the 8 individually detected crash events themselves —
  the augmented gate spent exactly as many days exposed during every single one of them as the
  current gate does** (re-verified by rerunning the script fresh; an earlier "11 fewer
  crash-exposure days" claim from this research thread did not reproduce and has been corrected
  here). Judged not worth implementing — if anything, more clearly not worth it than first thought.
  Along the way, `LP_VOL_ON` was also found stale at 0.25 and recalibrated to 0.57 in `.env`.
- **Full-stack integration** (`scripts/backtest_lp_full_stack.py`): rather than testing LP in
  isolation, integrated it into the actual portfolio backtest across the full corrected 2019-2024
  history, using real Graph-fetched fee/volume data from 2024-04-01 onward (10.30% annualized fee
  rate) and flagging pre-2024 fee assumptions as unverified. Result (at the corrected 60bps cost):
  baseline 1.510 → capacity-aware full-stack 1.502 (**-0.008 Sharpe**, unchanged from the 20bps run
  — this delta is driven by LP's own mechanics, not the trading-cost assumption). Negative at the
  full-stack level, and worse under a lower-fee-rate sensitivity check.
- **CHOP vs. LP head-to-head** (`scripts/backtest_lp_chop_comparison.py`): both LP and the existing
  mean-reversion/CHOP overlay compete for the same idle capital window. Standalone, CHOP alone
  contributes +0.034 Sharpe vs. LP alone's +0.001 (at 60bps cost) — roughly 34x. No capital-sharing
  split rule tested (50/50, proportional-by-Sharpe, day-by-day severity winner, LP-first/CHOP-leftover,
  etc.) robustly beat just running CHOP alone; the one variant that looked better (+0.018) affected
  only 14 days and is judged noise. The two triggers' overlap is small (5.6% of LP-eligible days,
  concentrated in April 2022) and weakly negatively correlated (-0.142) — not redundant signals,
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

### 5. Full-codebase audit and live-vs-backtest reconciliation (Aug 2026)

A follow-up "assume nothing, verify everything" audit covered eight areas: data integrity across
every source (yfinance, FRED, Bitstamp, Graph, GeckoTerminal), numerical consistency (rerunning
every script claiming a "validated" result and diffing against its own claim), the live Kraken
execution path, the cost model, `bot_20`'s execution code, general code health, lookahead-bias
tracing in the core algorithm, and live-vs-backtest parity (does `paper_trade_checklist.py`, the
actual daily live/paper script, implement the same logic as the backtests that produced the
validated numbers?). The lookahead-bias and live-vs-backtest findings are covered above and in 1b;
everything else found and fixed in this pass:

- **Stale yfinance bear-classifier cache**, silently pinned at 2015-02-17 (COVID's real VIX of
  39.16 was reading as a decade-stale 15.8) — fixed the read/write path in
  `scripts/prepare_bear_classifier_data.py` to validate date-range coverage and merge instead of
  overwrite, and refreshed the cache.
- **`bot_20` execution bugs**: `withdraw_liquidity.py`'s `decreaseLiquidity` simulation fallback
  was silently submitting with `amount0Min=amount1Min=0` (full slippage protection disabled)
  whenever the on-chain simulation call failed for any reason — now computes a real fallback
  minimum from liquidity + spot price. `create_position.py`'s position-size cap could be bypassed
  entirely by a zero/negative price read (the WETH side would value at $0) — now hard-aborts the
  mint instead. Both verified with real induced-failure tests, not just code review.
- **Kraken execution reconciliation**: `AddOrder` had no idempotency protection — a lost/timed-out
  response to a request Kraken had actually accepted could cause a retry to submit a second real
  order. Now uses a `userref`-based check before any retry. Separately, a `QueryOrders` failure
  after a successful `AddOrder` was losing the real order_id from the execution log entirely
  (logged as `order_id=""`); now correctly raises `KrakenOrderStateUnknown`, which already carries
  the order_id through to the log and a Discord alert.
- **Live pipeline silent-failure fixes**: a STOP status never changed `paper_trade_checklist.py`'s
  exit code (always returned 0), so nothing downstream could detect it programmatically. And there
  was no top-level exception handler at all — any unhandled crash produced zero notification of
  any kind. Both fixed: STOP now exits 1, and any unhandled exception now sends a Discord alert
  before exiting non-zero.
- **Joint ETH+BTC gross-cap**: each sleeve was independently capped at `gross_cap` (0.8) with
  nothing capping their *sum* — when both were active simultaneously, live could carry up to ~2x
  the exposure the validated backtest ever allowed. Now renormalized jointly, matching the
  backtest's `signal_weighted` allocation mode.
- **Partial-candle read**: Binance's klines endpoint returns the current, still-forming candle as
  the last row when queried without an explicit `endTime` — both `download_btc_daily.py` and
  `paper_trade_checklist.py`'s direct fetch now drop any candle whose `close_time` hasn't passed
  yet, instead of treating a partial (as little as a few hours') daily bar as a finalized close.
- **Smaller fixes**: `joblib` added to `requirements.txt` (was imported but undeclared); the
  `--cost-bps`/`--gold-cost-bps` CLI defaults in `backtest_eth_btc_portfolio.py` corrected from
  10→20 to match the validated config (running the script bare previously understated cost); the
  Bitstamp downloader was silently truncating ~19 weeks of available 2012 history because it made
  a single non-paginated request — now paginates and retrieves the full available range.

**Two more live-vs-backtest divergences found in that audit were resolved this pass** (aligning
live to the validated backtest, since the backtest side can't change without re-validating
everything downstream of it):

- **BTC exit-confirm days**: was 3 live vs. 5 in the validated backtest. `--btc-exit-confirm-days`
  default corrected to 5.
- **Vol-scalar annualization**: was `sqrt(365)` live vs. `sqrt(252)` in the backtest (three sites:
  the shared vol-filter, and the ETH/BTC sleeves' own position-sizing vol-scalars). Corrected to
  `sqrt(252)` at all three. Note: the ETH/BTC sleeves' vol-scalars still differ from the backtest in
  two smaller ways not covered by this fix — log-return vs. pct-change, and `ddof=1` vs. `ddof=0` —
  left as-is since they weren't part of this specific decision.

**Two live-vs-backtest divergences were reviewed and deliberately left as-is**, treated as
intentional extra conservatism in live rather than bugs: the BEAR-regime force-flatten overrides in
the ETH sleeve (live force-closes on `BEAR_REGIME`/`DD_OVERRIDE`; the backtest only scales the
weight down) and the defensive regime scale map (live uses 0.3/0.8 for CHOP/BEAR vs. the backtest's
0.4/1.0 — live derisks harder in both regimes).

**Three items are flagged as separate, real follow-up projects — not attempted in this pass**,
since each requires a design decision beyond a mechanical fix:

- **Gold sleeve portfolio integration.** Live currently tracks gold as an isolated paper account
  with its own entry logic (own confirm-days, ETH-only flatness gate), not wired into
  `eth_target_weight`/`btc_target_weight` at all — the backtest treats it as a real portfolio
  sleeve sized against shared idle capacity. Making these consistent means deciding how gold should
  actually compete for capital against ETH/BTC live, not just copying a parameter.
- **Kraken daily-loss guard rebuild.** `MAX_DAILY_LOSS` can never trip because nothing writes
  `artifacts/live_trades/daily_pnl.csv` — the guard is dead code. Fixing it properly means building
  a real daily PnL computation (mark-to-market vs. starting balance, what counts as "realised" vs.
  "unrealised") rather than a one-line patch.
- **Turnover-cost-gap refactor.** The vol-filter/asymmetric-sizing-driven day-to-day change in
  `alloc_eth`/`alloc_btc` is never costed in the production pipeline (quantified in
  `scripts/check_baseline_rebalance_cost.py`: ~57.7% cumulative drag over 2019-2024 at the corrected
  60bps cost if fully priced, up from ~19.2% at the old 20bps assumption). This affects the
  reference Sharpe itself, not just Mode A — fixing it means re-validating every script built on the
  current uncosted convention, a real refactor.

**The cost assumption is now resolved** (see 1c above): confirmed actual Kraken trading volume is
~$1k-$10k/month, which per Kraken's own published fee schedule means a real taker fee of 0.60% per
leg (Kraken's base $0+ tier is 0.80%, the $2,500+ tier is 0.60% — 60bps was chosen as the more
representative point estimate for this range rather than the worst case). A $4.99/month "Kraken+"
subscription exists but doesn't help here — it only discounts Kraken's simple Web/App interface,
not the Kraken Pro API `execution_kraken.py` actually trades through. `cost_bps` was corrected from
20 → 60 everywhere in the codebase and the full validation pipeline rerun; see 1c for the resulting
numbers.

## Frozen Research Branches

Historical/frozen components may exist as branches or tags, including:

- `feat/hmm-v1` / `regime-v1.0`
- `feat/elastic-net-v1` / `elasticnet-vol-v1.0-balanced`
- `feat/lp-overlay-v1` / `lp-overlay-v1-freeze`

`main` is the stable organised production/research baseline.
