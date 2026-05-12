"""
var_backtest — historical VaR backtesting on SPY daily returns.

Public API
----------
load_spy_returns        Download / cache SPY daily returns.
historical_var          Rolling historical VaR estimation.
historical_es           Rolling historical Expected Shortfall estimation.
identify_exceptions     Flag days where realized loss exceeded VaR.
kupiec_test             Kupiec unconditional coverage LR test.
christoffersen_test     Christoffersen independence LR test.
conditional_coverage_test  Combined LR_cc = LR_uc + LR_ind.
traffic_light           Basel traffic-light zone classification.
plot_backtest           Plot returns, VaR line, ES line, and exception markers.
"""

from .data import load_spy_returns
from .var import historical_var
from .es import historical_es
from .exceptions import identify_exceptions
from .tests import kupiec_test, christoffersen_test, conditional_coverage_test
from .basel import traffic_light
from .plotting import plot_backtest

__all__ = [
    "load_spy_returns",
    "historical_var",
    "historical_es",
    "identify_exceptions",
    "kupiec_test",
    "christoffersen_test",
    "conditional_coverage_test",
    "traffic_light",
    "plot_backtest",
]
