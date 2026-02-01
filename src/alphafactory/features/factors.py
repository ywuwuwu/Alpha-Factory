from __future__ import annotations

import pandas as pd


def mom_12_1(adj_close: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum using daily prices (approx):
    return from t- (252+21) to t-21.
    """
    return adj_close.shift(21) / adj_close.shift(252 + 21) - 1.0


def rev_1m(adj_close: pd.DataFrame) -> pd.DataFrame:
    """1-month reversal (negative 21d return)."""
    return -(adj_close / adj_close.shift(21) - 1.0)


def rev_5d(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal (negative 5d return)."""
    return -(adj_close / adj_close.shift(5) - 1.0)


def vol_20d(adj_close: pd.DataFrame) -> pd.DataFrame:
    """20d realized volatility (std of daily returns)."""
    rets = adj_close.pct_change()
    return rets.rolling(20).std()


def vol_change_20d(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Change in 20d volatility over the last month."""
    v = vol_20d(adj_close)
    return v / v.shift(20) - 1.0


def dollar_volume(adj_close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Dollar volume proxy."""
    return adj_close * volume


def volume_z_20d(volume: pd.DataFrame) -> pd.DataFrame:
    """Volume z-score vs 20d mean/std."""
    mu = volume.rolling(20).mean()
    sd = volume.rolling(20).std()
    return (volume - mu) / sd
