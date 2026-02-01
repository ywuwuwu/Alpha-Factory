# AlphaFactory-Pro — 1-page Research Summary (Template)

**Author:** YOUR NAME  
**Date:** YYYY-MM-DD  
**Repo:** (link)

## 1. Objective
Build a **market-neutral, medium-frequency** cross-sectional alpha library and show:
- strict validation (purged walk-forward)
- turnover/cost sensitivity
- a signal-combination method that improves stability out-of-sample

## 2. Data
- Source: Yahoo Finance via `yfinance` (cached locally)
- Universe: [N tickers], period: [start]–[end]
- Notes: not survivorship-bias free; results are illustrative.

## 3. Methods
**Signals (factor zoo):**
- Momentum (12-1), reversal (1m), short-term reversal (5d)
- Volatility (20d), volatility change (20d)
- Liquidity: dollar volume, volume z-score

**Validation:**
- Rolling train window: [X years]
- Test window: [1 month]
- Embargo: [Y days]

**Portfolio:**
- Daily long-short: top [q] / bottom [q]
- Gross exposure: 1.0, market-neutral
- Holding: staggered [H]-day with delay [D]

**Combination:**
- Baseline: equal-weight
- Proposed: turnover-aware Online L1-budget allocator

## 4. Results (fill from `results/<timestamp>/report.md`)
- Best single-factor IC / stability:
- Combined (equal) vs combined (online allocator):
- Cost sensitivity (0/5/10/20 bps):
- Failure modes / when it breaks:

## 5. Next Steps
- Better universe construction (liquidity + delisting handling)
- Sector/beta neutralization
- Add fundamentals (WRDS / other provider)
- Add risk attribution (FF5 regression)
