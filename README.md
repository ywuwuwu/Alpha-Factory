# AlphaFactory-Pro (GitHub-ready quant research artifact)

This repo is a **reproducible, end-to-end “alpha research loop”** for **market-neutral, medium-frequency** equity signals:

1) download/cached data (default: Yahoo via `yfinance`)
2) compute factor library (signals)
3) run **purged walk-forward** validation
4) build daily **market-neutral long/short** portfolios
5) apply **turnover-aware transaction costs**
6) optionally **combine signals** using an **online L1-budget allocator** (your differentiator)
7) generate a report (CSV + Markdown + plots)

> ⚠️ Research/education only. Not investment advice.

---

## Motivation

If you only “try random alphas” on platforms (WorldQuant BRAIN etc.), it’s easy to get stuck.
This repo forces a *real* research workflow: **hypothesis → signal → strict evaluation → robustness → library → combination**.

This directly maps to what **alpha-research roles** ask for (build alphas, combine them, handle costs). Trexquant’s GAR program description is a close match to this workflow. (See links in README “References”.)

---

## Quickstart (local machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# run end-to-end pipeline (downloads/caches data, then backtests)
python -m alphafactory.run --config configs/base.yaml
```

Outputs go to: `results/<timestamp>/`

- `factor_ic_summary.csv` — factor IC / ICIR by split
- `portfolio_daily_returns.csv` — daily L/S returns (gross + net of costs)
- `report.md` — human-readable summary + key plots

---

## Data notes (important)

Default data is downloaded with `yfinance` for convenience. This is **not survivorship-bias free** and is not “publication-quality”.
For production-grade research you’d use CRSP/Compustat (WRDS), Norgate, etc.

This repo is built so you can swap in a different data source later:
- see `src/alphafactory/data/`

---

## Repo structure

```
alpha-factory-pro/
  configs/
    base.yaml
  src/alphafactory/
    data/
    features/
    validation/
    portfolio/
    allocator/
    metrics/
    reports/
    run.py
  tests/
  paper/one_pager.md
  results/   # gitignored
```

---

## “Done” definition (what to show recruiters)

You are **done** when your repo has:

- **2–3 factor families** (momentum/reversal/vol/liquidity) implemented cleanly
- purged walk-forward splits + a test proving **no lookahead**
- cost sensitivity sweep (e.g., 0/5/10/20 bps)
- a combined signal that beats equal-weight baseline **out-of-sample**
- a 1-page memo in `paper/one_pager.md` with:
  - your validation design
  - stability results
  - failure modes + next iteration

---

## References

- Qlib (data + ML quant pipeline): https://github.com/microsoft/qlib
- Alphalens (factor analysis): https://github.com/quantopian/alphalens
- vectorbt (fast vectorized research/backtests): https://vectorbt.dev/
- Ken French data library (FF factors for attribution): https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
