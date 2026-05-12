"""
var.py — rolling historical Value-at-Risk estimation.
"""

import pandas as pd


def historical_var(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = 250,
) -> pd.Series:
    """Estimate 1-day historical VaR using a rolling empirical quantile.

    For each date *t* the VaR is the negative of the ``(1 - alpha)``-th
    quantile of the preceding ``window`` returns:

    .. math::

        \\text{VaR}_t = -Q_{1-\\alpha}\\bigl(r_{t-\\text{window}+1}, \\ldots, r_t\\bigr)

    **Sign convention:** VaR is returned as a *positive* number representing
    the loss magnitude.  A value of 0.02 means the model estimates a
    ``(1 - alpha)`` probability of losing more than 2 % on the next day.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns indexed by date.
    alpha : float
        Confidence level (e.g. 0.99 for 99 % VaR).  Default 0.99.
    window : int
        Number of historical observations in the rolling window.
        Default 250 (one trading year).

    Returns
    -------
    pd.Series
        Rolling historical VaR (positive values), indexed by date.
        The first ``window - 1`` observations are NaN because the window is
        not yet full.
    """
    var_series = -returns.rolling(window=window).quantile(1.0 - alpha)
    return var_series
