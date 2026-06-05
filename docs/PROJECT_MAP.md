# Project Map

This repo has three kinds of content: live paper-trading code, reproducible research/backtests, and generated local state.

## Live Paper-Trading System

These files are still used by the daily Hetzner workflow.

- `scripts/run_paper_check_daily.sh`: main daily runner; refreshes market/news/defensive artifacts and runs the Discord checklist.
- `scripts/paper_trade_checklist.py`: main paper-trading report, ETH/BTC/gold state, completed trade logging, Discord output.
- `scripts/run_defensive_live_refresh.sh`: refreshes the ETH defensive model artifact before the checklist.
- `scripts/direction_event_model_v1.py`: defensive direction-event model used by the live refresh.
- `scripts/run_market_tracker_daily.sh`: market snapshot runner.
- `scripts/market_tracker.py`: BTC, oil, gas, metals, equity, bond, dollar, and wheat daily tracker.
- `scripts/run_news_sentiment_daily.sh`: macro news runner.
- `scripts/news_sentiment.py`: 12-hour RSS filter plus Claude event classifier.
- `scripts/download_btc_daily.py`: BTC daily Binance feed.
- `scripts/download_btc_perp_features.py`: BTC funding/perp feature feed.
- `scripts/download_paxg_daily.py`: PAXG daily feed.
- `scripts/download_perp_features.py`: ETH perp/funding feed.
- `scripts/merge_perp_live_into_history.py`: merges live ETH perp features into history.
- `scripts/run_feed_health_check.sh` and `scripts/check_feed_health.py`: feed health checks.
- `scripts/run_regime_classifier_daily.sh`, `scripts/run_regime_snapshot_daily.sh`, `scripts/regime_classifier_daily_snapshot.py`, `scripts/regime_classifier_v2.py`: daily immutable regime snapshot.

## Validated / Reusable Backtests

These are retained because they answer specific research questions and can be rerun.

- `scripts/backtest_btc_full_stack.py`: BTC full-stack daily test.
- `scripts/backtest_eth_btc_portfolio.py`: ETH+BTC portfolio backtest with fixed and signal-weighted allocation.
- `scripts/stress_test_btc.py`: BTC stress tests.
- `scripts/stress_test_eth_btc_portfolio.py`: ETH+BTC portfolio stress tests.
- `scripts/backfill_btc_paper.py`: BTC paper backfill and validation from paper-start date.
- `scripts/backtest_multi_asset.py`: simple EMA multi-asset diagnostic.
- `scripts/backtest_independent_portfolio.py`: independent sleeves diagnostic.
- `scripts/backtest_crypto_rotation.py`: ETH/BTC rotation diagnostic.
- `scripts/backtest_ranked_rotation.py`: cross-asset ranked rotation diagnostic; retained as a negative result.
- `scripts/backtest_oil_gdelt.py`, `scripts/backtest_oil_news.py`: oil news/GDELT research.
- `scripts/research_oil_cot.py`, `scripts/research_oil_eia.py`: oil COT/EIA research.
- `scripts/generate_equity_comparison_charts.py`: Plotly equity chart generator.

## Research / Experimental Scripts

These are not live. They are kept for audit history and future research.

- `scripts/offense_trend_follow_v1.py`, `scripts/offense_*`: offensive sleeve experiments.
- `scripts/regime_classifier_v1.py`, `scripts/regime_classifier_v3.py`, `scripts/regime_classifier_v4.py`: non-live regime variants.
- `scripts/direction_model_v1.py`, `scripts/direction_model_v2.py`, `scripts/direction_event_model_v2_meta.py`: non-live direction model variants.
- `scripts/funding_basis_*`, `scripts/perp_positioning_*`: funding/perp diagnostics.
- `scripts/commodity_basket_backtest.py`, `scripts/combine_crypto_etf_*`, `scripts/regime_switch_crypto_etf*`: cross-asset experiments.
- `scripts/plot_*`: local plotting utilities.
- `scripts/*diagnostics.py`, `scripts/*sweep.py`, `scripts/*audit.py`: diagnostics and parameter sweeps.

## Legacy Package Code

- `lpbot/models/elasticnet_v1/`: older model CLI components.
- `lpbot/paper/paper_trade_v1/`: older paper service components.
- `lpbot/overlays/lp_overlay_v1/`: older LP overlay workstream.
- `lpbot/hmm/`, `lpbot/data_sources/`: retained package modules, not the current daily paper stack unless explicitly wired in.
- `src/`: original regime/LP infrastructure.

## Generated / Local Only

These directories should not be committed.

- `data/`: downloaded prices, funding, GDELT, EIA, COT, yfinance caches.
- `artifacts/`: backtest outputs, paper logs, Discord input/output snapshots, charts.
- `logs/`: cron logs.
- `models/`: local fitted model artifacts.
- `ec2_backup/`: local server backup.
- `.env`: secrets and runtime paths.
