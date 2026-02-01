import pandas as pd
from alphafactory.validation.splits import monthly_walk_forward_splits


def test_monthly_splits_monotone():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    splits = monthly_walk_forward_splits(dates, train_years=2, test_months=1, embargo_days=5)
    assert len(splits) > 10
    for sp in splits:
        assert sp.train_start < sp.train_end < sp.test_start <= sp.test_end
