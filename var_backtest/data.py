"""
data.py — SPY daily returns via yfinance with simple CSV caching.

On the first call the module downloads SPY adjusted-close prices from Yahoo
Finance and writes them to `cache_path`.  Subsequent calls read the CSV
directly, so no network access is needed for repeated runs.
"""

import os

import pandas as pd
import yfinance as yf

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "output", "spy_prices.csv"
)


def load_spy_returns(
    start: str = "2010-01-01",
    end: str | None = None,
    cache_path: str = _DEFAULT_CACHE,
) -> pd.Series:
    """Download (or load from cache) SPY daily simple returns.

    Parameters
    ----------
    start : str
        Start date in ``'YYYY-MM-DD'`` format.  Ignored when loading from
        the cache file.
    end : str, optional
        End date in ``'YYYY-MM-DD'`` format.  Defaults to today.  Ignored
        when loading from the cache file.
    cache_path : str
        Path to a CSV file used for caching downloaded prices.  The
        directory is created automatically if it does not exist.

    Returns
    -------
    pd.Series
        Daily simple returns (``pct_change``) of SPY, indexed by date, with
        the first NaN row dropped.

    Notes
    -----
    Delete the cache file to force a fresh download.
    """
    cache_path = os.path.abspath(cache_path)

    if os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        if prices.empty:
            # Cache was written from a failed/empty download — discard and re-download
            os.remove(cache_path)
        else:
            returns = prices.pct_change().dropna()
            return returns
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data for SPY. "
            "This is usually a temporary rate-limit. Wait a minute and retry."
        )

    # yfinance may return a MultiIndex frame when multiple tickers are
    # requested; handle both layouts defensively.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[("Close", "SPY")]
    else:
        prices = raw["Close"]

    prices.to_csv(cache_path)

    returns = prices.pct_change().dropna()
    return returns
