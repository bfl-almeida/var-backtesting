"""
test_es.py — unit tests for historical_es().
"""

import numpy as np
import pandas as pd
import pytest

from var_backtest.es import historical_es
from var_backtest.var import historical_var


def test_es_matches_manual_tail_mean():
    """ES equals the manually computed negative mean of the tail returns."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 500))

    es_series = historical_es(returns, alpha=0.99, window=250)

    # Manual computation on the first full window (indices 0–249)
    window = returns.iloc[:250].values
    threshold = np.quantile(window, 0.01)
    tail = window[window <= threshold]
    manual_es = float(-tail.mean())

    np.testing.assert_almost_equal(es_series.iloc[249], manual_es, decimal=12)


def test_es_greater_or_equal_var():
    """ES must be >= VaR at every valid date (coherence property)."""
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, 600))

    var_series = historical_var(returns, alpha=0.99, window=250)
    es_series = historical_es(returns, alpha=0.99, window=250)

    common = var_series.dropna().index.intersection(es_series.dropna().index)
    assert (es_series.loc[common].values >= var_series.loc[common].values - 1e-10).all(), \
        "ES must be >= VaR at every date"


def test_es_nans_before_window():
    """The first window-1 observations must be NaN; index window-1 must not be."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 400))

    es_series = historical_es(returns, alpha=0.99, window=250)

    assert es_series.iloc[:249].isna().all()
    assert pd.notna(es_series.iloc[249])


def test_es_positive():
    """All valid ES values must be positive."""
    rng = np.random.default_rng(99)
    returns = pd.Series(rng.normal(0, 0.01, 600))

    es_series = historical_es(returns, alpha=0.99, window=250)
    valid = es_series.dropna()

    assert (valid > 0).all(), "Historical ES must be positive"
