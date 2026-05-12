"""
test_var.py — unit tests for historical_var().
"""

import numpy as np
import pandas as pd
import pytest

from var_backtest.var import historical_var


def test_historical_var_matches_manual():
    """VaR at window boundary equals manually computed quantile."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 500))

    var_series = historical_var(returns, alpha=0.99, window=250)

    # The first valid VaR is at index 249 (0-based), computed from returns[0:250].
    manual_var = -np.quantile(returns.iloc[:250].values, 0.01)
    np.testing.assert_almost_equal(var_series.iloc[249], manual_var, decimal=12)


def test_historical_var_nans_before_window():
    """The first window-1 observations must be NaN; index window-1 must not be."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 400))

    var_series = historical_var(returns, alpha=0.99, window=250)

    assert var_series.iloc[:249].isna().all(), "Expected NaN before window fills"
    assert pd.notna(var_series.iloc[249]), "Expected valid VaR at first full window"


def test_historical_var_positive():
    """All valid VaR values must be positive for a return series with losses."""
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, 600))

    var_series = historical_var(returns, alpha=0.99, window=250)
    valid = var_series.dropna()

    assert (valid > 0).all(), "Historical VaR must be positive"


def test_historical_var_custom_params():
    """Parameterised alpha and window produce correctly sized output."""
    returns = pd.Series(np.linspace(-0.02, 0.02, 300))

    var_series = historical_var(returns, alpha=0.95, window=100)

    n_valid = var_series.notna().sum()
    assert n_valid == 300 - 100 + 1  # rolling produces value from index 99 onward
