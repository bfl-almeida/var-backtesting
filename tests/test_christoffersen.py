"""
test_christoffersen.py — unit tests for christoffersen_test() and
                         conditional_coverage_test().
"""

import numpy as np
import pytest

from var_backtest.tests import christoffersen_test, conditional_coverage_test


def test_christoffersen_independent_draws():
    """i.i.d. exceptions at 1 % should fail to reject independence."""
    # Large sample with fixed seed → consistent behaviour
    rng = np.random.default_rng(42)
    exceptions = rng.random(5000) < 0.01  # ~50 exceptions, no clustering

    result = christoffersen_test(exceptions)

    assert result.p_value > 0.05, (
        f"Expected p > 0.05 for i.i.d. exceptions, got {result.p_value:.4f}"
    )
    assert result.reject is False


def test_christoffersen_clustered_exceptions():
    """Strongly clustered exceptions should reject independence."""
    # Pattern: 3 consecutive exceptions every 20 days → clear clustering
    T = 1000
    exceptions = np.zeros(T, dtype=bool)
    for i in range(0, T, 20):
        exceptions[i : i + 3] = True  # n_11 will be large

    result = christoffersen_test(exceptions)

    assert result.p_value < 0.001, (
        f"Expected very low p-value for clustered exceptions, got {result.p_value:.4f}"
    )
    assert result.reject is True


def test_christoffersen_zero_exceptions_graceful():
    """Zero exceptions: test must not raise and stat must be non-negative."""
    exceptions = np.zeros(500, dtype=bool)

    result = christoffersen_test(exceptions)

    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_christoffersen_statistic_non_negative():
    """LR_ind must be non-negative (chi-squared support)."""
    rng = np.random.default_rng(7)
    exceptions = rng.random(1000) < 0.02

    result = christoffersen_test(exceptions)

    assert result.statistic >= 0.0


def test_conditional_coverage_lr_is_sum():
    """LR_cc must equal LR_uc + LR_ind to within floating-point tolerance."""
    from var_backtest.tests import kupiec_test

    rng = np.random.default_rng(13)
    exceptions = rng.random(1000) < 0.015

    uc = kupiec_test(exceptions, alpha=0.99)
    ind = christoffersen_test(exceptions)
    cc = conditional_coverage_test(exceptions, alpha=0.99)

    assert abs(cc.statistic - (uc.statistic + ind.statistic)) < 1e-10
