from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cs(x: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """Cross-sectional winsorization per date (row-wise)."""
    if pct <= 0:
        return x
    lo = x.quantile(pct, axis=1)
    hi = x.quantile(1 - pct, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def zscore_cs(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0.0, np.nan)
    return (x.sub(mu, axis=0)).div(sd, axis=0)


def rank_cs(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [0,1] per date."""
    return x.rank(axis=1, pct=True)


def ewm_smooth(x: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Time-series EWM smoothing per ticker."""
    if alpha <= 0:
        return x
    return x.ewm(alpha=alpha, adjust=False).mean()
