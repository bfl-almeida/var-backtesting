"""
plotting.py — backtest visualisation.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_backtest(
    returns: pd.Series,
    var_series: pd.Series,
    exceptions: pd.Series,
    es_series: pd.Series | None = None,
    output_path: str = "examples/output/backtest_plot.png",
) -> None:
    """Plot SPY daily returns, the rolling VaR line, optional ES line, and exception markers.

    The figure contains overlapping elements on a single axes:

    * Daily returns shaded in blue (gains) and red (losses).
    * The rolling 99 % VaR line, negated so it sits below zero in the loss
      territory (``-VaR_t`` is the breach threshold for the next day).
    * The rolling 99 % ES line (dashed, darker), if provided.
    * Red scatter markers on the dates where exceptions occurred, placed at
      the realised return level.

    The plot is saved to ``output_path`` as a PNG file.

    Parameters
    ----------
    returns : pd.Series
        Daily realized returns indexed by date.
    var_series : pd.Series
        Rolling VaR estimates (positive values) indexed by date.
    exceptions : pd.Series
        Boolean series indexed by date where ``True`` indicates an exception.
    es_series : pd.Series, optional
        Rolling ES estimates (positive values) indexed by date.  When provided,
        an additional dashed line is drawn below the VaR line.
    output_path : str
        Filesystem path for the saved PNG.  The directory is created if
        it does not exist.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # --- Returns fill ---
    ax.fill_between(
        returns.index,
        returns.values,
        0,
        where=(returns.values >= 0),
        color="steelblue",
        alpha=0.45,
        label="Returns (gain)",
    )
    ax.fill_between(
        returns.index,
        returns.values,
        0,
        where=(returns.values < 0),
        color="salmon",
        alpha=0.45,
        label="Returns (loss)",
    )

    # --- VaR line (negated → sits below zero) ---
    var_valid = var_series.dropna()
    ax.plot(
        var_valid.index,
        -var_valid.values,
        color="darkred",
        linewidth=1.2,
        label="−VaR (99 %)",
        zorder=3,
    )

    # --- ES line (negated, dashed) ---
    if es_series is not None:
        es_valid = es_series.dropna()
        ax.plot(
            es_valid.index,
            -es_valid.values,
            color="maroon",
            linewidth=1.0,
            linestyle="--",
            label="−ES (99 %)",
            zorder=3,
        )

    # --- Exception markers ---
    exception_dates = exceptions[exceptions].index
    if len(exception_dates) > 0:
        exc_returns = returns.reindex(exception_dates).dropna()
        ax.scatter(
            exc_returns.index,
            exc_returns.values,
            color="red",
            zorder=5,
            s=20,
            label=f"Exceptions (n={len(exception_dates)})",
        )

    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.set_title("SPY Historical VaR Backtest — 99 %, 250-day rolling window")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")
