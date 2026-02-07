# LPBot — Modular Trading Research Framework

## Project Overview
LPBot is a modular systematic trading research framework that separates core strategy logic from optional satellite overlays. The core strategy is designed to stand alone, while overlays are gated and additive.

## High-level Architecture
- Data ingestion (OHLCV, volumes)
- Regime detection (HMM)
- Core exposure (Elastic Net: trend + volatility targeting)
- Optional LP overlay (subordinate, gated)

## Branch Structure
- `main` → stable baseline
- `feat/hmm-v1` → regime detector (frozen, tag: `regime-v1.0`)
- `feat/elastic-net-v1` → core strategy (frozen, tag: `elasticnet-vol-v1.0-balanced`)
- `feat/lp-overlay-v1` → LP overlay satellite (frozen, tag: `lp-overlay-v1-freeze`)

Each branch represents an independently developed and frozen component.

## Design Principles
- Core strategy must be robust standalone
- Overlays are optional and reversible
- Low-vol gating for LP
- Capacity-aware assumptions
- Avoidance of overfitting via freezing

## Reproducibility Notes
- Backtests are research-grade, not production execution
- LP overlay uses simplified but economically constrained proxies
- Real execution would require on-chain simulation

## Minimal Usage
Core exposure can be evaluated independently; regime labels and LP overlays are applied only if explicitly enabled.
