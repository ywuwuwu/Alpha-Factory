from __future__ import annotations

import pandas as pd


def forward_return(adj_close: pd.DataFrame, delay_days: int, horizon_days: int) -> pd.DataFrame:
    """Forward close-to-close return from t+delay to t+delay+horizon."""
    d = int(delay_days)
    h = int(horizon_days)
    return adj_close.shift(-(d + h)) / adj_close.shift(-d) - 1.0
