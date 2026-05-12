"""
test_kupiec.py — unit tests for kupiec_test().
"""

import numpy as np
import pytest

from var_backtest.tests import kupiec_test


def test_kupiec_exact_expected_rate():
    """LR statistic is exactly 0 when observed rate equals expected rate."""
    # N/T = 10/1000 = 0.01 = 1 - alpha exactly → numerator equals denominator
    T = 1000
    N = 10
    exceptions = np.zeros(T, dtype=bool)
    exceptions[:N] = True

    result = kupiec_test(exceptions, alpha=0.99)

    assert abs(result.statistic) < 1e-10, "LR_uc must be 0 at the null rate"
    assert result.p_value > 0.99, "p-value must be ~1.0 when null holds exactly"
    assert result.reject is False


def test_kupiec_inflated_exception_rate():
    """Clearly inflated exception rate produces low p-value and rejection."""
    # 5 % observed vs 1 % expected — strongly rejects
    T = 1000
    N = 50
    exceptions = np.zeros(T, dtype=bool)
    exceptions[:N] = True

    result = kupiec_test(exceptions, alpha=0.99)

    assert result.p_value < 0.001
    assert result.reject is True


def test_kupiec_deflated_exception_rate():
    """Clearly deflated exception rate (0 %) also rejects."""
    T = 500
    exceptions = np.zeros(T, dtype=bool)  # zero exceptions

    result = kupiec_test(exceptions, alpha=0.99)

    # LR_uc = -2 * T * log(1 - p) which is large → low p-value
    assert result.p_value < 0.05
    assert result.reject is True


def test_kupiec_statistic_non_negative():
    """LR statistic must be non-negative by definition."""
    rng = np.random.default_rng(99)
    exceptions = rng.random(800) < 0.01

    result = kupiec_test(exceptions, alpha=0.99)

    assert result.statistic >= 0.0


def test_kupiec_returns_test_result_fields():
    """Return value exposes statistic, p_value, and reject."""
    exceptions = np.array([True, False] * 100)
    result = kupiec_test(exceptions, alpha=0.99)

    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "reject")
    assert 0.0 <= result.p_value <= 1.0
