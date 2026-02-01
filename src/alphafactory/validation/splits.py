from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def monthly_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int,
    test_months: int,
    embargo_days: int,
) -> list[Split]:
    """Generate month-by-month walk-forward splits with an embargo gap.

    - Train window: rolling lookback of `train_years`
    - Test window: `test_months`
    - Embargo: gap between train_end and test_start to reduce leakage via overlapping labels
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    if len(dates) < 400:
        raise ValueError("Not enough dates. Need ~2+ years of daily data for meaningful splits.")

    first = dates.min()
    last = dates.max()

    # Align anchors to month starts
    month_starts = pd.date_range(first.normalize(), last.normalize(), freq="MS")
    splits: list[Split] = []

    for test_start in month_starts:
        train_end = test_start - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.DateOffset(years=train_years)

        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if train_start < first or test_end > last:
            continue

        splits.append(
            Split(
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
            )
        )

    return splits
