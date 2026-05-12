"""
tests.py — statistical backtesting tests for VaR models.

Implements:
  - Kupiec (1995) unconditional coverage likelihood-ratio test
  - Christoffersen (1998) independence likelihood-ratio test
  - Christoffersen (1998) conditional coverage test (LR_cc = LR_uc + LR_ind)
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a likelihood-ratio backtesting test.

    Attributes
    ----------
    statistic : float
        The likelihood-ratio test statistic.
    p_value : float
        p-value of the test.
    reject : bool
        Whether H0 is rejected at the 5 % significance level.
    """

    statistic: float
    p_value: float
    reject: bool


def _safe_log_term(n: int | float, p: float) -> float:
    """Compute ``n * log(p)``, returning 0 when ``n == 0`` (convention: 0·log 0 := 0)."""
    if n == 0:
        return 0.0
    return float(n) * np.log(p)


def kupiec_test(
    exceptions: np.ndarray,
    alpha: float = 0.99,
) -> TestResult:
    """Kupiec (1995) unconditional coverage likelihood-ratio test.

    Tests whether the observed exception rate matches the expected rate
    ``p = 1 - alpha``.

    .. math::

        LR_{\\text{uc}} = -2 \\ln\\!\\left[
            \\frac{(1-p)^{T-N}\\, p^N}{%
                  \\left(1-\\tfrac{N}{T}\\right)^{T-N} \\left(\\tfrac{N}{T}\\right)^N}
        \\right]

    where *T* is the number of observations and *N* is the number of
    exceptions.  Under H₀ (correct coverage), ``LR_uc ~ chi²(1)``.

    Parameters
    ----------
    exceptions : array-like of bool
        Boolean array where ``True`` indicates a VaR exception.
    alpha : float
        Confidence level used for the VaR estimate (default 0.99).

    Returns
    -------
    TestResult
        ``statistic``, ``p_value``, and ``reject`` (at 5 % level).
    """
    exc = np.asarray(exceptions, dtype=bool)
    T = len(exc)
    N = int(exc.sum())
    p = 1.0 - alpha  # expected exception rate

    if N == 0:
        # MLE is p̂ = 0 → log L₁ = T·log(1); LR = −2·log L₀
        lr_uc = -2.0 * (T * np.log(1.0 - p))
    elif N == T:
        # MLE is p̂ = 1 → log L₁ = T·log(1); LR = −2·log L₀
        lr_uc = -2.0 * (T * np.log(p))
    else:
        p_hat = N / T
        log_l0 = (T - N) * np.log(1.0 - p) + N * np.log(p)
        log_l1 = (T - N) * np.log(1.0 - p_hat) + N * np.log(p_hat)
        lr_uc = -2.0 * (log_l0 - log_l1)

    p_value = float(1.0 - stats.chi2.cdf(lr_uc, df=1))
    return TestResult(statistic=lr_uc, p_value=p_value, reject=p_value < 0.05)


def christoffersen_test(
    exceptions: np.ndarray,
) -> TestResult:
    """Christoffersen (1998) independence likelihood-ratio test.

    Tests whether VaR exceptions are independently distributed over time.
    Clustering of exceptions suggests the model underestimates tail risk in
    volatile periods.

    Let ``n_ij`` be the number of one-step transitions from state *i* to
    state *j* in the exception indicator sequence.  Define conditional
    probabilities

    .. math::

        \\hat{\\pi}_{01} = \\frac{n_{01}}{n_{00}+n_{01}}, \\quad
        \\hat{\\pi}_{11} = \\frac{n_{11}}{n_{10}+n_{11}}, \\quad
        \\hat{\\pi} = \\frac{n_{01}+n_{11}}{n_{00}+n_{01}+n_{10}+n_{11}}

    The test statistic is

    .. math::

        LR_{\\text{ind}} = -2 \\ln\\!\\left[
            \\frac{(1-\\hat{\\pi})^{n_{00}+n_{10}}\\,\\hat{\\pi}^{n_{01}+n_{11}}}{%
                  (1-\\hat{\\pi}_{01})^{n_{00}}\\,\\hat{\\pi}_{01}^{n_{01}}\\,
                  (1-\\hat{\\pi}_{11})^{n_{10}}\\,\\hat{\\pi}_{11}^{n_{11}}}
        \\right]

    Under H₀ (independence), ``LR_ind ~ chi²(1)``.

    Parameters
    ----------
    exceptions : array-like of bool
        Boolean array where ``True`` indicates a VaR exception.

    Returns
    -------
    TestResult
        ``statistic``, ``p_value``, and ``reject`` (at 5 % level).
    """
    exc = np.asarray(exceptions, dtype=int)

    prev = exc[:-1]
    curr = exc[1:]

    n_00 = int(np.sum((prev == 0) & (curr == 0)))
    n_01 = int(np.sum((prev == 0) & (curr == 1)))
    n_10 = int(np.sum((prev == 1) & (curr == 0)))
    n_11 = int(np.sum((prev == 1) & (curr == 1)))

    n_total = n_00 + n_01 + n_10 + n_11

    # Unconditional (independent-model) exception probability
    pi = (n_01 + n_11) / n_total if n_total > 0 else 0.0

    # Conditional probabilities for the Markov (dependent) model
    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0

    # Log-likelihoods (0·log 0 := 0 via _safe_log_term)
    log_l_ind = (
        _safe_log_term(n_00 + n_10, 1.0 - pi)
        + _safe_log_term(n_01 + n_11, pi)
    )
    log_l_dep = (
        _safe_log_term(n_00, 1.0 - pi_01)
        + _safe_log_term(n_01, pi_01)
        + _safe_log_term(n_10, 1.0 - pi_11)
        + _safe_log_term(n_11, pi_11)
    )

    lr_ind = -2.0 * (log_l_ind - log_l_dep)
    # Clamp tiny negative values caused by floating-point arithmetic
    lr_ind = max(lr_ind, 0.0)

    p_value = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    return TestResult(statistic=lr_ind, p_value=p_value, reject=p_value < 0.05)


def conditional_coverage_test(
    exceptions: np.ndarray,
    alpha: float = 0.99,
) -> TestResult:
    """Christoffersen (1998) conditional coverage test.

    Combines the Kupiec and Christoffersen tests into a joint test:

    .. math::

        LR_{\\text{cc}} = LR_{\\text{uc}} + LR_{\\text{ind}} \\sim \\chi^2(2)

    A rejection means the model fails on correct coverage, independence of
    exceptions, or both.

    Parameters
    ----------
    exceptions : array-like of bool
        Boolean array where ``True`` indicates a VaR exception.
    alpha : float
        Confidence level used for the VaR estimate (default 0.99).

    Returns
    -------
    TestResult
        ``statistic``, ``p_value``, and ``reject`` (at 5 % level).
    """
    uc = kupiec_test(exceptions, alpha=alpha)
    ind = christoffersen_test(exceptions)

    lr_cc = uc.statistic + ind.statistic
    p_value = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    return TestResult(statistic=lr_cc, p_value=p_value, reject=p_value < 0.05)
