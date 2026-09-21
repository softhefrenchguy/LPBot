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
| Mode A (state-based sizing) | Researched, **not recommended** | Scales `alloc_eth`/`alloc_btc` by regime conviction. An earlier pass claimed +0.246 Sharpe OOS; after the lookahead-bias fix, the real-cost correction, and the alloc-turnover-cost-at-source fix, Mode A's full-period Sharpe (1.093) is simply below baseline's (1.121) — no further adjustment needed to show it loses. See [Research Log](#research-log-data-correction--lp-overlay-investigation) below. |
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
with the most rebalancing activity relative to their return. **This "1.510 uncosted / 1.121
fully-costed" split was itself only a post-hoc adjustment layer at this point — see 1d, which fixes
it at the source and makes 1.121 the actual reference, not an adjustment on top of 1.510.**

### 1d. Alloc-turnover-cost fix at the source (fourth correction layer)

A diagnostic thread investigating what actually drives 2022/2023's cost (see section 5's audit, and
the reversal-cooldown investigation below) traced the "fully turnover-costed" 1.121 figure from 1c
back to its origin: `check_baseline_rebalance_cost.py` was computing it as a **post-hoc adjustment**
— `alloc_eth`/`alloc_btc`'s own day-to-day resizing (driven by the vol-filter and asymmetric-sizing
multipliers, continuously, independent of whether the underlying trend signal is even changing) was
never actually charged a cost inside `backtest_eth_btc_portfolio.py` itself. The 1.121 number was
real, but it lived in a separate verification script, not in the number every other script actually
optimized against or reported as "the" baseline.

Fixed at the source: `backtest_eth_btc_portfolio.py` now computes `alloc_turnover_cost` (same
formula already established, `_alloc_turnover`) and subtracts it directly from `combined_return`,
and `_main_return()` in `backtest_continuous_regime_integrations.py` (used by Mode A validation, the
LP scripts, and this check) recomputes and charges it fresh for whatever alloc series it's given —
important because that function also runs on *modified* alloc series (Mode A's state-conviction
adjustment, LP capital-sharing variants), where a stored baseline-alloc cost would be stale.
Verified: reconstructing the old pre-fix number from the new one exactly reproduces 1.510 (to 3
decimals), confirming the fix's magnitude matches what was previously measured.

One caveat, checked and judged acceptable rather than silently assumed away: `eth_strategy_return`/
`btc_strategy_return` already carry a *sleeve-level* cost from `weight_exec`'s own turnover
(pre-portfolio-normalization), and that gets scaled by `alloc_eth`/`alloc_btc` when combined into
the portfolio return. Charging `alloc_turnover_cost` on top is theoretically a small double-count
at the exact moment a new position is opened (both terms move together). Checked empirically: the
two turnover series have comparable typical magnitude (median day-to-day `alloc_eth` change 0.006
vs median sleeve-level turnover 0.011), and since the sleeve-level term gets *scaled down* by
`alloc_eth` (itself usually well under 1.0) before it reaches the portfolio return, its contribution
is a small, second-order product of two fractions — not large enough to materially distort the
result. Not eliminated, but small enough not to block using this fix as the new reference.

**This changes the numbers again, more than any single correction except the lookahead-bias fix:**

|  | Sharpe | CAGR | MaxDD | Walk-forward OOS avg (2021-2024) |
|---|---|---|---|---|
| 1c (post-hoc adjustment, now superseded) | 1.510 | 34.2% | -15.6% | 0.938 |
| **1d (fixed at source, current reference)** | **1.121** | **25.6%** | **-21.0%** | **0.490** |

Per-year walk-forward OOS at the corrected accounting: 2021 Sharpe 1.610 (CAGR 47.2%), **2022
Sharpe -0.704 (CAGR -0.7%)**, **2023 Sharpe -0.480 (CAGR -1.0%, genuinely negative)**, 2024 Sharpe
1.536 (CAGR 35.7%). **The walk-forward verdict flips from ROBUST (3/4 positive years) to MIXED (2/4
positive years)** — 2023 in particular goes from a marginal +0.163 Sharpe to a real loss once its
own rebalancing is honestly priced, matching the diagnostic finding that ~51% of 2023's cost was
this exact mechanism. This is the current reference; every number below reflects it.

Mode A and the LP overlay were both re-validated against this corrected baseline and **both
conclusions hold** — Mode A still doesn't beat baseline (1.093 vs. 1.121, doesn't even need its own
extra cost added to lose), and LP full-stack integration is still marginally negative (1.121 → 1.113,
-0.008, same magnitude as before). Neither finding flips; both were already robust to this kind of
correction.

### 2. Mode A re-validated — smaller edge than first reported, and it evaporates once fully costed

`scripts/backtest_mode_a_validation.py`'s 4-step validation was rerun after both the lookahead-bias
fix and the cost-model correction. The original claim from earlier in this research thread — a
"genuine +0.246 Sharpe" 2021-2024 OOS improvement — **does not hold up** on the corrected pipeline.
Three things changed it, each layering on the last:

1. The lookahead-bias fix by itself shrank the gap (walk-forward OOS: baseline 1.144 vs. Mode A's
   1.212, an edge of +0.068, not +0.246).
2. The cost-model correction (20→60bps) shrank the gap further: baseline OOS average 0.938 vs.
   Mode A's 0.988 — an edge of only +0.050.
3. Step 3 (vol-filter double-counting check) shows essentially none of that edge was ever a
   genuinely new signal: with the vol filter on (production config), Mode A's full-period uplift is
   effectively zero or negative at every stage of correction; the large positive uplift only ever
   appeared with the vol filter switched off, confirming Mode A is substituting for what the vol
   filter already does, not adding orthogonal information.
4. The alloc-turnover-cost-at-source fix (1d) — Mode A's own state-transition churn moves
   `alloc_eth`/`alloc_btc` every time `state` changes, and that's exactly the kind of resizing 1d
   now charges properly instead of as a post-hoc adjustment. Final result: **Mode A's full-period
   Sharpe is 1.093 against baseline's 1.121 — it loses before any further "incremental cost"
   adjustment is even applied**, and its 2021-2024 walk-forward OOS average (0.530) barely edges
   baseline's (0.490) while flipping sign relationships across individual years (2021/2024 worse,
   2022/2023 better) in a way consistent with in-sample noise, not a real edge.

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
  rate) and flagging pre-2024 fee assumptions as unverified. Result (at the corrected 60bps cost,
  alloc-turnover cost now charged at the source per 1d): baseline 1.121 → capacity-aware full-stack
  1.113 (**-0.008 Sharpe**, the same magnitude across every cost-model revision so far — this delta
  is driven by LP's own mechanics, not the trading-cost assumption). Negative at the full-stack
  level, and worse under a lower-fee-rate sensitivity check.
- **CHOP vs. LP head-to-head** (`scripts/backtest_lp_chop_comparison.py`): both LP and the existing
  mean-reversion/CHOP overlay compete for the same idle capital window. Standalone, CHOP alone
  contributes +0.047 Sharpe vs. LP alone's +0.001 — roughly 47x. No capital-sharing split rule tested
  (50/50, proportional-by-Sharpe, day-by-day severity winner, LP-first/CHOP-leftover, etc.) robustly
  beat just running CHOP alone; the one variant that looked better (+0.015) affected only 14 days and
  is judged noise. The two triggers' overlap is small (5.6% of LP-eligible days, concentrated in
  April 2022) and weakly negatively correlated (-0.142) — not redundant signals, but LP's slice of
  the opportunity is small regardless.

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
  **This first pass was incomplete** — it missed ETH's main path (the EMAs are built from the 5m
  price file, not a klines fetch); see section 8 for the follow-up.
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

**Two items are flagged as separate, real follow-up projects — not attempted in this pass**,
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

**The turnover-cost-gap refactor is done** (see 1d above): `alloc_eth`/`alloc_btc`'s own day-to-day
resizing is now charged at the source in `backtest_eth_btc_portfolio.py`, not measured post-hoc.
This dropped the reference Sharpe from 1.510 to 1.121 and flipped the walk-forward verdict from
ROBUST to MIXED (2023 goes genuinely negative) — a bigger change than the cost-model correction
below. See 1d for the full numbers and the one accepted double-counting caveat.

**The cost assumption is now resolved** (see 1c above): confirmed actual Kraken trading volume is
~$1k-$10k/month, which per Kraken's own published fee schedule means a real taker fee of 0.60% per
leg (Kraken's base $0+ tier is 0.80%, the $2,500+ tier is 0.60% — 60bps was chosen as the more
representative point estimate for this range rather than the worst case). A $4.99/month "Kraken+"
subscription exists but doesn't help here — it only discounts Kraken's simple Web/App interface,
not the Kraken Pro API `execution_kraken.py` actually trades through. `cost_bps` was corrected from
20 → 60 everywhere in the codebase and the full validation pipeline rerun; see 1c for the resulting
numbers.

### 6. Reversal-cooldown investigation: hypothesis rejected

2022 and 2023 are the two weak years driving down the walk-forward average (see 1d). The natural
hypothesis — the strategy is whipsawing, entering and quickly reversing out of positions — was
tested directly rather than assumed. A first diagnostic pass (using `weight_exec > 0` runs to define
trade boundaries) found 5 "quick-reversal" trades in 2022 on both ETH and BTC, 100% of that year's
cost. **That diagnosis was wrong**: `weight_exec` dips briefly to zero *within* a single real trade
(from the derisk multiplier or funding-model fallback), which fragments one real position into
several phantom ones. Redone using the actual entry/exit state machine (`off_active`, what the
strategy really trades on): **ETH and BTC each had zero real trades in 2022**, and every 2023 trade
was held 44-266 days — zero quick reversals in either year.

Decomposing 2022's cost directly instead: ETH's contribution is the tail-end unwind of a trade that
entered in December 2021 (a legitimate exit, not a re-entry); a few days in late January show
identical ETH/BTC turnover from the *defensive* funding-rate proxy activating briefly — and tracing
the code confirmed BTC's "defensive" signal is literally ETH's funding-rate proxy, reused as-is
(`def_proxy = load_eth_defensive_proxy(...)` passed into both `run_sleeve()` calls) — not a bug
necessarily, but a real architectural fact worth knowing. The largest single component in both years
turned out to be the same alloc-level turnover-gap fixed in 1d: 40% of 2022's cost, 51% of 2023's.

A post-reversal cooldown (block re-entry on a sleeve for N days after an exit that followed a short
hold, implemented as `--reversal-cooldown-days`/`--reversal-cooldown-threshold-days` in
`backtest_eth_btc_portfolio.py`, tested at 5/10/15 days) was still built and tested empirically
rather than skipped once the diagnosis came back negative. Result: **it never triggers, at any
length** — confirmed with `reversal_cooldown_blocks` telemetry showing 0 at every setting — and
produces byte-identical Sharpe/CAGR/MaxDD/trade-counts for every year 2019-2024, cooldown-on or off.
This both confirms the diagnosis (there's nothing to block) and proves the feature is safe (zero
impact on the trending years) — it's just solving a problem that doesn't exist in this data. Kept in
the codebase, default off (`--reversal-cooldown-days 0`), as a documented negative finding rather
than reverted, following this repo's convention for tested-and-rejected optional features
(`--stop-loss`, `--dd-aware`).

### 7. Resize deadband: a real but non-surgical improvement, with an overfitting flag

Since 1d found the alloc-level turnover-gap was the dominant cost in both weak years, the natural
follow-up (implemented as `--resize-deadband` in `backtest_eth_btc_portfolio.py`) is to stop
`alloc_eth`/`alloc_btc` from resizing on every small daily wobble in the vol-filter/asymmetric-sizing
multipliers, only updating the executed allocation once the target moves more than a threshold away
from what's currently held — real entries and exits (the `off_active` flag flipping) always execute
immediately regardless, so this only throttles resizing *within* an already-held position.

Tested at 0.03/0.05/0.10/0.15 (and further, see below), full-period + walk-forward OOS:

| Deadband | Sharpe | CAGR | MaxDD | Walk-forward OOS avg | ETH/BTC entries |
|---|---|---|---|---|---|
| 0.00 (off, current reference) | 1.121 | 25.6% | -21.0% | 0.490 | 8 / 18 |
| 0.03 | 1.133 | 25.8% | -20.8% | 0.501 | 8 / 18 |
| 0.05 | 1.138 | 25.9% | -20.9% | 0.505 | 8 / 18 |
| **0.10 (recommended)** | **1.150** | **26.1%** | **-20.9%** | **0.514** | **8 / 18** |
| 0.15 | 1.192 | 26.9% | -19.4% | 0.556 | 8 / 18 |

Entry counts are exactly unchanged at every threshold tested — confirmed directly, not assumed —
so this does not delay or suppress any real entry/exit decision in the trending years or anywhere
else. That's the good news the original question asked for.

**But it does not answer the question the way it was framed.** The premise was that this would be a
surgical 2022/2023 fix. It isn't: **the improvement shows up in every year, including the years that
were already strong** (2019/2020/2021/2024 all improve too), and 2022/2023 improve only modestly and
inconsistently (2022's Sharpe is flat-to-worse at some thresholds before improving at others; 2023
improves more steadily but stays negative at every threshold tested up to 0.15). This is a general
turnover-reduction effect, not a fix targeted at the two weak years specifically.

**Honesty check, not just a recommendation:** the improvement continues monotonically well past any
threshold that's still a "deadband" in spirit — tested to 0.40 (40 percentage points), full-period
Sharpe kept climbing (1.211 at 0.30, 1.342 at 0.40) with walk-forward OOS averages up to 0.754. A
0.40 deadband is no longer "filter out small noise" — at that size the position is close to
static once entered, a materially different and more aggressive behavior than what was actually
being tested. A pattern that keeps improving all the way to "barely resize at all" is a flag for
overfitting to this specific 2019-2024 period, not a green light to pick the largest number tested.
**Recommendation: 0.10 as a moderate, evidence-backed starting point** (clear improvement, still
recognizably the original strategy with noise filtered out) — not currently defaulted on
(`--resize-deadband 0.0` remains the default), and anything beyond ~0.15 should get real
out-of-sample/robustness scrutiny (different assets, different periods) before being trusted, not
just a bigger backtest number.

### 8. Review of the live Discord analyses (Sep 2026): partial-candle follow-up and analysis fixes

Two weeks of real "LPBot Analysis" Discord posts (Sep 7-21) were checked against the code. The bot was
flat throughout, so none of the section-5 live changes (BTC exit days, gross cap, vol-scaled sizing)
were observable in them, and they can't confirm which code version the server runs (see
`docs/HETZNER_RUNBOOK.md` for how to check). They did surface three real problems:

- **Partial-candle fix was incomplete (ETH).** ETH's EMAs, stack alignment, 20d drawdown and the
  mean-reversion z-score are built from `data/ETHUSDC_5m.csv` by taking the last row of each day, so
  the still-forming current-day bar (the cron runs at 08:00 UTC) was counted as a full day. Section 5's
  fix only covered the Binance klines fetch and the BTC downloader. Now `paper_trade_checklist.py`
  drops the in-progress UTC day from the daily series (live spot price and the 24h change still use the
  latest 5m bar), and `download_paxg_daily.py` gets the same filter as the BTC downloader. Simulating
  the 08:00 UTC cron on every day of the local 5m history (1,394 days, 2021-2026): including the partial
  bar changed the ETH entry-confirmation signal on **7 days (0.5%)**, in both directions (3 false
  "entry confirmed", 4 missed), with EMA50 off by a median of $8 (max $50). Rare, but it lands
  exactly on the decision days. Verified end to end by running the real script on a shifted 5m file
  that contains a partial "today" bar: its EMAs match the finalized-only values exactly.
- **Wrong "Next entry trigger" footer.** `analyse_daily.py` had a fixed template that always said
  "EMA50 needs to cross EMA120" (and fed the model a matching wrong-pair gap), while the real entry
  rule is the stack EMA50>EMA120>EMA300 and EMA50 was already above EMA120 for the whole window; the
  actual blocker was EMA120 vs EMA300. The footer is now computed in code: it names the blocking
  leg(s) with gaps, the days already aligned when applicable, and a flat-price EMA projection to
  alignment (Sep 21 numbers give ~37 days + 3 days confirmation, about 6 weeks). The prompt also no
  longer lets the model claim entry needs "N BULL days" — the regime label only scales size.
- **Posts cut off mid-sentence.** The Claude call used `max_tokens=500`; 7 of the 15 posts (Sep 7, 8,
  11, 15, 18, 20, 21) lost the end of the answer, including the footer. Raised to 800, stop reason is
  logged, and the Discord message is composed so that if anything must be shortened it is the analysis,
  never the footer.

## Frozen Research Branches

Historical/frozen components may exist as branches or tags, including:

- `feat/hmm-v1` / `regime-v1.0`
- `feat/elastic-net-v1` / `elasticnet-vol-v1.0-balanced`
- `feat/lp-overlay-v1` / `lp-overlay-v1-freeze`

`main` is the stable organised production/research baseline.
