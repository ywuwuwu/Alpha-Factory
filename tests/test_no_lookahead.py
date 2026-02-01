import numpy as np
import pandas as pd
from alphafactory.features.factors import mom_12_1


def test_momentum_no_lookahead():
    # create monotonic prices so momentum is deterministic
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    prices = pd.DataFrame({
        "A": np.arange(len(dates), dtype=float) + 100.0
    }, index=dates)

    mom = mom_12_1(prices)

    # If we perturb future prices, momentum up to that point should not change
    prices2 = prices.copy()
    prices2.loc[dates[350]:, "A"] += 1000.0  # huge future shock
    mom2 = mom_12_1(prices2)

    # Compare momentum values before the shock start minus lookback buffer
    # Momentum at date t depends on t-273 and t-21, so dates < 350 should be unaffected.
    chk = dates[300:340]
    assert np.allclose(mom.loc[chk, "A"].values, mom2.loc[chk, "A"].values, equal_nan=True)
