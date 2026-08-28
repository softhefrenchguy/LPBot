# Execution layer — history and status

This folder holds the on-chain execution side of the project: code that actually signs and
sends transactions, as opposed to `scripts/` (research/backtesting, no wallet involved). Kept
here as a record of how this evolved, including the parts that were superseded or shelved —
not just the current state.

## `legacy_LP_Automation/` — first version (Oct–Nov 2025), superseded

The original live LP bot. Screen-scraping trigger (`pyautogui` + HSV color-matching on a fixed
screen region to detect a DEX UI's "out of range" indicator), tight-range (±5–25 tick) WETH-only
positions on Arbitrum. Real wallet, real transactions — ran live 2025-10-30 to 2025-11-03 at
small scale, then stopped. Audited in detail; findings included a plaintext private key in
`.env` (removed before this was committed — the wallet itself was confirmed abandoned/unused,
not rotated because there was nothing left to rotate), zero slippage protection on any
transaction (`amountMin`/`amountOutMinimum` hardcoded to 0 throughout), and no position-size
cap. Superseded entirely by `bot_20/`.

## `bot_20/` — second version (Oct 2025–Feb 2026), current candidate but on hold

A substantially more developed successor: on-chain `slot0()` polling instead of screen-scraping,
an `ai_core/` decision layer (EMA/oscillation breakout detection, a trained panic-exit classifier,
Monte Carlo simulation tooling), KyberSwap aggregator integration with staged rebalancing, and an
emergency `force_close_all_positions.py` kill switch. Also had a live, funded wallet with real
transaction history through Dec 2025 (private key removed here for the same reason as above —
wallet confirmed abandoned).

Work done on this branch of the project (all in this session):
- **Safety fixes**: real slippage protection on `create_position.py`'s mint and
  `withdraw_liquidity.py`'s `decreaseLiquidity`/Uniswap-fallback swap (previously all hardcoded to
  zero minimum output — a real sandwich-attack exposure), plus an explicit `MAX_POSITION_USD` cap
  checked before every mint.
- **Chain migration**: moved from Arbitrum to Ethereum mainnet across all 38 `.py` files. Every
  contract address (WETH, USDC, the target pool) was verified against live on-chain `eth_call`s
  before being hardcoded — not taken from memory. See `../scripts/backtest_lp_full_stack.py` and
  the research thread that produced the addresses used here.
- **Full-range minting** added as the default mode (`POSITION_MODE=full_range`), with the original
  tight-range design kept available (`POSITION_MODE=tight_range`) but no longer default, given
  mainnet gas economics don't favor the original fast-cycling design.

**Current status: on hold, not recommended to proceed.** Once LP was integrated into the actual
portfolio backtest (`../scripts/backtest_lp_full_stack.py`, `../scripts/backtest_lp_chop_comparison.py`)
rather than tested in isolation, its standalone contribution to full-stack risk-adjusted returns
came back negligible to negative (+0.002 Sharpe standalone at best, -0.008 once correctly
accounting for capital competition with the existing mean-reversion/CHOP overlay — see those
scripts' output for the full breakdown). The panic-gate exit-trigger improvement explored earlier
(`../scripts/check_lp_panic_gate_comparison.py`, a genuine but small ~0.13pp crash-exposure
improvement) is documented but deliberately not implemented here, since it isn't worth building on
top of a mode that doesn't clear its own bar at the portfolio level. This folder is kept as a
working, safety-fixed reference in case that full-stack conclusion changes (e.g. materially better
verified fee yields, cheaper execution costs, or a real answer to the capital-sharing question) —
not as something currently intended to run live.
