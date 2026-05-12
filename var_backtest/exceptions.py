"""
exceptions.py — VaR exception (breach) identification.
"""

import pandas as pd


def identify_exceptions(
    returns: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """Identify days on which the realized return exceeded the VaR estimate.

    An exception is recorded on day *t* when the realized return is worse
    (more negative) than the VaR estimated the *previous* day:

    .. math::

        \\text{exception}_t = \\mathbf{1}\\bigl[r_t < -\\text{VaR}_{t-1}\\bigr]

    Parameters
    ----------
    returns : pd.Series
        Daily realized returns indexed by date.
    var_series : pd.Series
        Rolling VaR estimates (positive numbers) indexed by the *same* dates
        as ``returns``.  The value at date *t* is used to set the threshold
        for date *t + 1*.

    Returns
    -------
    pd.Series
        Boolean series indexed by the date of each realized return.
        ``True`` indicates an exception (the realized loss exceeded the VaR).
        Dates for which the lagged VaR is NaN (i.e. the first ``window``
        observations) are excluded.
    """
    # Shift VaR forward by one day: var_lagged[t] = VaR_{t-1}
    var_lagged = var_series.shift(1)

    # Align on the intersection of both indices
    common_idx = returns.index.intersection(var_lagged.index)
    r = returns.loc[common_idx]
    v = var_lagged.loc[common_idx]

    # Keep only dates where a valid VaR estimate exists
    valid = v.notna()
    exceptions = (r < -v)[valid]
    return exceptions
