from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_weights_from_scores(
    scores: pd.DataFrame,
    long_q: float,
    short_q: float,
    gross_exposure: float,
    max_abs_weight: float,
) -> pd.DataFrame:
    """Convert cross-sectional scores to daily long/short weights.

    Design goals:
      - market-neutral: sum(weights)=0 each day
      - gross exposure up to `gross_exposure`
      - enforce max absolute weight **without scaling back up** (never violate max)

    Mechanics:
      1) pick long/short buckets by quantiles
      2) assign equal weights within each bucket
      3) clip to max_abs_weight
      4) scale DOWN legs so that long_sum == short_sum == leg_exposure <= gross_exposure/2
    """
    long_q = float(long_q)
    short_q = float(short_q)
    gross_exposure = float(gross_exposure)
    max_abs_weight = float(max_abs_weight)

    W = []
    for dt, row in scores.iterrows():
        x = row.dropna()
        w = pd.Series(0.0, index=row.index)
        if len(x) < 10:
            W.append(w)
            continue

        lo = x.quantile(short_q)
        hi = x.quantile(1 - long_q)
        long_names = x[x >= hi].index
        short_names = x[x <= lo].index
        if len(long_names) == 0 or len(short_names) == 0:
            W.append(w)
            continue

        w_long = pd.Series(1.0 / len(long_names), index=long_names)
        w_short = pd.Series(1.0 / len(short_names), index=short_names)

        # clip each leg
        w_long = w_long.clip(upper=max_abs_weight)
        w_short = w_short.clip(upper=max_abs_weight)

        long_sum = float(w_long.sum())
        short_sum = float(w_short.sum())
        if long_sum <= 0 or short_sum <= 0:
            W.append(w)
            continue

        leg_target = gross_exposure / 2.0
        leg_exposure = min(long_sum, short_sum, leg_target)

        w_long = w_long * (leg_exposure / long_sum)
        w_short = w_short * (leg_exposure / short_sum)

        w.loc[w_long.index] = w_long
        w.loc[w_short.index] = -w_short

        # numerical safety
        w = w.clip(lower=-max_abs_weight, upper=max_abs_weight)

        W.append(w)

    return pd.DataFrame(W, index=scores.index, columns=scores.columns)


def staggered_holding_portfolio_returns(
    sub_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    delay_days: int,
    horizon_days: int,
) -> pd.Series:
    """Create a daily return series assuming each day's sub-portfolio is held for `horizon_days`.

    - sub_weights computed at date t using info up to t
    - weights become active after `delay_days`
    - effective weights = average of last `horizon_days` active sub-portfolios
    """
    d = int(delay_days)
    h = int(horizon_days)

    w_active = sub_weights.shift(d)
    W_eff = w_active.rolling(h, min_periods=1).mean()

    idx = W_eff.index.intersection(daily_returns.index)
    W_eff = W_eff.loc[idx]
    R = daily_returns.loc[idx]

    pnl = (W_eff.shift(1) * R).sum(axis=1)
    pnl.name = "portfolio_ret"
    return pnl


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """One-day turnover proxy: 0.5 * sum(|w_t - w_{t-1}|)."""
    dw = weights.diff().abs().sum(axis=1)
    return 0.5 * dw


def apply_linear_costs(
    returns: pd.Series,
    weights: pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """Apply a simple linear transaction cost model.

    cost(t) = (cost_bps / 10000) * turnover(t)
    """
    cost = (cost_bps / 10000.0) * turnover_from_weights(weights)
    net = returns - cost.reindex(returns.index).fillna(0.0)
    net.name = f"net_ret_{int(cost_bps)}bps"
    return net
