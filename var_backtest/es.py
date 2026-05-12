"""
es.py — rolling historical Expected Shortfall (ES) estimation.
"""

import numpy as np
import pandas as pd


def historical_es(
    returns: pd.Series,
    alpha: float = 0.99,
    window: int = 250,
) -> pd.Series:
    """Estimate 1-day historical Expected Shortfall using a rolling window.

    Expected Shortfall (also called Conditional VaR or CVaR) is the expected
    loss *conditional* on the loss exceeding the VaR threshold.  Unlike VaR,
    it is a coherent risk measure and is the regulatory standard under FRTB.

    For each date *t* the ES is computed from the preceding ``window`` returns:

    .. math::

        \\text{ES}_t = -\\mathbb{E}\\bigl[r \\mid r \\leq -\\text{VaR}_t\\bigr]
                     = -\\frac{1}{|\\mathcal{T}|} \\sum_{r \\in \\mathcal{T}} r

    where :math:`\\mathcal{T} = \\{r \\in \\text{window} : r \\leq Q_{1-\\alpha}\\}`.

    **Sign convention:** ES is returned as a *positive* number representing the
    expected loss magnitude in the tail.  ES ≥ VaR always holds by construction.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns indexed by date.
    alpha : float
        Confidence level (e.g. 0.99 for 99 % ES).  Default 0.99.
    window : int
        Number of historical observations in the rolling window.
        Default 250 (one trading year).

    Returns
    -------
    pd.Series
        Rolling historical ES (positive values), indexed by date.
        Returns NaN for dates where the rolling window is not yet full, and
        also for any window in which no returns fall at or below the VaR
        threshold (an extreme edge case with very short windows or unusual data).
    """

    def _es_from_window(window_arr: np.ndarray) -> float:
        threshold = np.quantile(window_arr, 1.0 - alpha)
        tail = window_arr[window_arr <= threshold]
        if len(tail) == 0:
            return np.nan
        return float(-tail.mean())

    es_series = returns.rolling(window=window).apply(_es_from_window, raw=True)
    return es_series
