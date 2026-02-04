# LPBot — Regime-First Market Analysis

This project builds a **strategy-neutral market regime detector** and uses it to evaluate whether liquidity provision (LP) strategies are profitable **conditional on regime**.

The core principle is simple:

> **Regimes first. Strategies second. No assumptions.**

---

## 🔒 Current Status — Frozen v1

The regime detector is **frozen at v1.0**.

No further changes to regime logic should be made unless explicitly creating v2.

---

## Assets
- BTCUSDC
- ETHUSDC  
(SOL intentionally excluded due to structural data issues)

---

## Regime Layers

### Micro Regimes
- Timeframe: **1h**
- Model: Gaussian HMM
- Purpose: short-term volatility / activity structure
- Typical duration: **~2–7 hours**

### Macro Regimes
- Timeframe: **8h**
- Model: Gaussian HMM
- Purpose: day-scale market structure
- Typical duration:
  - ETH: **~24–38 hours**
  - BTC: **~20–30 hours**

Macro regimes do **not** reliably appear on 1h or 4h bars with fast features alone.

---

## Features (Frozen)
- `r` — log return
- `abs_r` — absolute return
- `vol20` — rolling volatility of returns
- `vol_z` — rolling volume z-score

No slow features (trend, momentum, moving averages) are included in v1.

---

## Data Rules
- No forward-filling
- Empty resample bins are dropped
- Minimum underlying data enforced per bar
- Health checks run automatically

These rules are non-negotiable.

---

## Repository Structure



src/ Core logic (resampling, obs, HMM, LP)
config/ Frozen configs and symbols
scripts/ Entry-point runners
data/ (local only, ignored by git)
regimes/ (local only, ignored by git)


---

## What This Project Is Not
- ❌ A trading bot
- ❌ A predictive model
- ❌ An LP optimizer

This is **measurement infrastructure**.

---

## Next Workstream
**Regime-conditional LP evaluation**

Questions to answer:
- In which regimes (if any) is LP profitable?
- Are losses regime-specific or structural?
- Does regime awareness improve risk-adjusted outcomes?

LP logic will remain **fixed**.  
Regimes are labels, not signals.

---

## Versioning
- `regime-v1.0` — frozen, reproducible baseline

All future work must reference the regime version used.

---

## Philosophy
Most strategy failures come from assuming regimes.

This project measures them instead.