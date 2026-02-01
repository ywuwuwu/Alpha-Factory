import numpy as np
import pandas as pd
from alphafactory.portfolio.longshort import long_short_weights_from_scores


def test_weights_neutral_and_clipped():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = [f"S{i}" for i in range(50)]
    scores = pd.DataFrame(np.random.randn(len(dates), len(cols)), index=dates, columns=cols)

    w = long_short_weights_from_scores(scores, long_q=0.1, short_q=0.1, gross_exposure=1.0, max_abs_weight=0.05)
    # each day should be close to market-neutral
    s = w.sum(axis=1).fillna(0.0)
    assert np.all(np.abs(s.values) < 1e-6)
    # max abs weight constraint
    assert float(w.abs().max().max()) <= 0.05 + 1e-9
