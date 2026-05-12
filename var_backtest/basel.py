"""
basel.py — Basel traffic-light classification for VaR backtests.
"""


def traffic_light(num_exceptions: int) -> str:
    """Classify a VaR model under the Basel traffic-light framework.

    The Basel II/III internal models approach (originally from the 1996
    Market Risk Amendment) evaluates trading-book VaR models by counting
    exceptions over the most recent 250 trading days at the 99 % confidence
    level.  The exception count determines a capital add-on multiplier:

    * **Green (0–4):** model performance is acceptable; no additional penalty.
    * **Yellow (5–9):** possible model problems; supervisory review required.
    * **Red (≥ 10):** model is presumed inadequate; mandatory capital add-on.

    Parameters
    ----------
    num_exceptions : int
        Number of VaR exceptions over the most recent 250 trading days.

    Returns
    -------
    str
        ``"green"``, ``"yellow"``, or ``"red"``.
    """
    if num_exceptions <= 4:
        return "green"
    elif num_exceptions <= 9:
        return "yellow"
    else:
        return "red"
