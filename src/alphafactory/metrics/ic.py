from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_daily(scores: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Daily Spearman correlation (rank IC) across tickers.

    Returns:
        Series indexed by date (NaN if fewer than 3 valid tickers)
    """
    idx = scores.index.intersection(fwd_returns.index)
    scores = scores.loc[idx]
    fwd_returns = fwd_returns.loc[idx]

    ic = []
    for dt in idx:
        x = scores.loc[dt]
        y = fwd_returns.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < 3:
            ic.append(np.nan)
            continue
        ic.append(x[m].rank().corr(y[m].rank()))
    return pd.Series(ic, index=idx, name="rank_ic")


def summarize_ic(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ir": float("nan"), "n": 0}
    mean = ic.mean()
    std = ic.std(ddof=1)
    ir = mean / std if std and std > 0 else float("nan")
    return {"mean": float(mean), "std": float(std), "ir": float(ir), "n": int(len(ic))}
