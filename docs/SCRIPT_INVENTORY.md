# Script Inventory

Status meanings:

- `ACTIVE`: used by the current daily Hetzner/paper workflow.
- `VALIDATED`: useful backtest/research result, not scheduled.
- `RESEARCH`: exploratory or diagnostic, not scheduled.
- `LEGACY`: old workflow retained for reference.

## Active Daily Scripts

| Script | Status | Purpose |
|---|---:|---|
| `scripts/run_paper_check_daily.sh` | ACTIVE | Main daily Linux/Hetzner runner. |
| `scripts/run_paper_check_daily.ps1` | ACTIVE | Windows/local equivalent runner. |
| `scripts/paper_trade_checklist.py` | ACTIVE | Builds ETH/BTC/gold/news/markets Discord summary and paper logs. |
| `scripts/run_defensive_live_refresh.sh` | ACTIVE | Refreshes ETH defensive model artifact. |
| `scripts/direction_event_model_v1.py` | ACTIVE | Defensive event model used by live refresh. |
| `scripts/run_market_tracker_daily.sh` | ACTIVE | Refreshes daily market tracker artifact. |
| `scripts/market_tracker.py` | ACTIVE | Tracks 10 macro/crypto/commodity assets. |
| `scripts/run_news_sentiment_daily.sh` | ACTIVE | Refreshes macro news artifact. |
| `scripts/news_sentiment.py` | ACTIVE | RSS fetch, 12-hour filter, Claude classifier, news log. |
| `scripts/download_btc_daily.py` | ACTIVE | Downloads BTC daily OHLCV. |
| `scripts/download_btc_perp_features.py` | ACTIVE | Downloads BTC perp/funding features. |
| `scripts/download_paxg_daily.py` | ACTIVE | Downloads PAXG daily prices. |
| `scripts/download_perp_features.py` | ACTIVE | Downloads ETH perp/funding features. |
| `scripts/merge_perp_live_into_history.py` | ACTIVE | Merges live perp features into historical data. |
| `scripts/run_regime_classifier_daily.sh` | ACTIVE | Daily regime classifier runner. |
| `scripts/run_regime_snapshot_daily.sh` | ACTIVE | Writes immutable daily regime snapshot. |
| `scripts/regime_classifier_daily_snapshot.py` | ACTIVE | Snapshot helper. |
| `scripts/regime_classifier_v2.py` | ACTIVE | Current regime classifier logic. |
| `scripts/run_feed_health_check.sh` | ACTIVE | Feed health cron runner. |
| `scripts/check_feed_health.py` | ACTIVE | Feed freshness/health checks. |

## Validated Backtests And Diagnostics

| Script | Status | Purpose |
|---|---:|---|
| `scripts/backtest_btc_full_stack.py` | VALIDATED | BTC full-stack daily backtest. |
| `scripts/backtest_eth_btc_portfolio.py` | VALIDATED | ETH+BTC portfolio and allocation-mode test. |
| `scripts/stress_test_btc.py` | VALIDATED | BTC stress test suite. |
| `scripts/stress_test_eth_btc_portfolio.py` | VALIDATED | ETH+BTC stress test suite. |
| `scripts/backfill_btc_paper.py` | VALIDATED | BTC paper backfill and validation. |
| `scripts/backtest_multi_asset.py` | VALIDATED | Simple EMA multi-asset diagnostic. |
| `scripts/backtest_independent_portfolio.py` | VALIDATED | Independent sleeves diagnostic. |
| `scripts/backtest_crypto_rotation.py` | VALIDATED | ETH/BTC rotation test. |
| `scripts/backtest_ranked_rotation.py` | VALIDATED | Failed cross-asset rotation test; retained as negative evidence. |
| `scripts/backtest_oil_gdelt.py` | VALIDATED | GDELT oil news backtest. |
| `scripts/backtest_oil_news.py` | VALIDATED | Oil news strategy diagnostic. |
| `scripts/research_oil_cot.py` | VALIDATED | COT data viability check for oil. |
| `scripts/research_oil_eia.py` | VALIDATED | EIA inventory viability check for oil. |
| `scripts/generate_equity_comparison_charts.py` | VALIDATED | Generates local HTML equity charts. |

## Research Scripts

| Group | Status | Purpose |
|---|---:|---|
| `scripts/offense_*` | RESEARCH | Offensive sleeve variants. |
| `scripts/direction_model_*`, `scripts/direction_event_model_v2_meta.py` | RESEARCH | Non-live defensive/direction variants. |
| `scripts/regime_classifier_v1.py`, `v3.py`, `v4.py` | RESEARCH | Non-current regime classifiers. |
| `scripts/funding_basis_*` | RESEARCH | Funding/basis diagnostics. |
| `scripts/perp_positioning_*` | RESEARCH | Perp positioning diagnostics/backtests. |
| `scripts/commodity_basket_backtest.py` | RESEARCH | Commodity basket tests. |
| `scripts/combine_crypto_etf_*` | RESEARCH | Crypto/ETF combination tests. |
| `scripts/regime_switch_crypto_etf*` | RESEARCH | Regime switching experiments. |
| `scripts/plot_*` | RESEARCH | Local plot utilities. |
| `scripts/*diagnostics.py`, `scripts/*sweep.py`, `scripts/*audit.py` | RESEARCH | Diagnostics, sweeps, and checks. |

## Legacy Scripts

| Script | Status | Purpose |
|---|---:|---|
| `scripts/breakout_backtest.py` | LEGACY | Old breakout backtest. |
| `scripts/breakout_paper_report.py` | LEGACY | Old breakout paper report. |
| `scripts/breakout_paper_service.py` | LEGACY | Old breakout paper service. |
| `scripts/download_binance_1m.py` | LEGACY | Old 1-minute Binance downloader. |
| `scripts/get_data.py` | LEGACY | Old data helper. |
| `scripts/run_macro_8h.py`, `scripts/run_micro_1h.py` | LEGACY | Old HMM/regime runners. |

## Rule For New Scripts

When adding a script, add it to one of these sections and mark whether it is active, validated, research, or legacy. If it writes data or artifacts, those outputs should go under `data/`, `artifacts/`, or `logs/` so git ignores them.
