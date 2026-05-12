"""
run_backtest.py — end-to-end VaR backtest example for SPY.

Usage
-----
    python examples/run_backtest.py

Downloads SPY daily returns (or loads from CSV cache), computes a rolling
99 % historical VaR with a 250-day window, identifies exceptions, runs
Kupiec and Christoffersen backtests, prints a summary table, and saves the
backtest plot to examples/output/backtest_plot.png.
"""

import os
import sys

# Allow running directly from the repo root or from the examples/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from var_backtest import (
    load_spy_returns,
    historical_var,
    historical_es,
    identify_exceptions,
    kupiec_test,
    christoffersen_test,
    conditional_coverage_test,
    traffic_light,
    plot_backtest,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("Loading SPY returns …")
    returns = load_spy_returns(start="2010-01-01")

    # ------------------------------------------------------------------
    # 2. VaR and exceptions
    # ------------------------------------------------------------------
    print("Computing rolling 99 % historical VaR (250-day window) …")
    var_series = historical_var(returns, alpha=0.99, window=250)

    print("Computing rolling 99 % historical ES (250-day window) …")
    es_series = historical_es(returns, alpha=0.99, window=250)

    print("Identifying exceptions …")
    exceptions = identify_exceptions(returns, var_series)

    N = int(exceptions.sum())
    T = len(exceptions)

    print(
        f"\nBacktest period : {exceptions.index[0].date()} "
        f"to {exceptions.index[-1].date()}"
    )
    print(f"Observations    : {T}")
    print(f"Exceptions      : {N}  (observed rate {N / T:.2%}, expected 1.00 %)\n")

    # ------------------------------------------------------------------
    # 3. Statistical tests
    # ------------------------------------------------------------------
    kupiec = kupiec_test(exceptions, alpha=0.99)
    christoffersen = christoffersen_test(exceptions)
    cc = conditional_coverage_test(exceptions, alpha=0.99)

    # ------------------------------------------------------------------
    # 4. Basel traffic-light (most recent 250 trading days)
    # ------------------------------------------------------------------
    n_last_250 = int(exceptions.iloc[-250:].sum())
    zone = traffic_light(n_last_250)

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    col_w = 38
    val_w = 12
    sep = "-" * (col_w + val_w + 1)

    def row(label: str, value: object) -> str:
        return f"{label:<{col_w}} {str(value):>{val_w}}"

    print(f"{'Metric':<{col_w}} {'Value':>{val_w}}")
    print(sep)
    print(row("Exceptions (full period)", N))
    print(row("Observations (full period)", T))
    print(row("Observed exception rate", f"{N / T:.2%}"))
    print(sep)
    print(row("Kupiec LR statistic", f"{kupiec.statistic:.4f}"))
    print(row("Kupiec p-value", f"{kupiec.p_value:.4f}"))
    print(row("Kupiec — reject H0 at 5 %", kupiec.reject))
    print(sep)
    print(row("Christoffersen LR statistic", f"{christoffersen.statistic:.4f}"))
    print(row("Christoffersen p-value", f"{christoffersen.p_value:.4f}"))
    print(row("Christoffersen — reject H0 at 5 %", christoffersen.reject))
    print(sep)
    print(row("Conditional coverage LR statistic", f"{cc.statistic:.4f}"))
    print(row("Conditional coverage p-value", f"{cc.p_value:.4f}"))
    print(row("Conditional coverage — reject H0 at 5 %", cc.reject))
    print(sep)
    print(row("Basel zone (last 250 days, N={})".format(n_last_250), zone.upper()))
    print(sep)

    # ------------------------------------------------------------------
    # ES summary (most recent estimate + full-period average)
    # ------------------------------------------------------------------
    avg_var = var_series.dropna().mean()
    avg_es  = es_series.dropna().mean()
    last_var = var_series.dropna().iloc[-1]
    last_es  = es_series.dropna().iloc[-1]
    print(row("Latest VaR estimate", f"{last_var:.2%}"))
    print(row("Latest ES  estimate", f"{last_es:.2%}"))
    print(row("ES / VaR ratio (latest)", f"{last_es / last_var:.2f}x"))
    print(sep)
    print(row("Average rolling VaR (full period)", f"{avg_var:.2%}"))
    print(row("Average rolling ES  (full period)", f"{avg_es:.2%}"))
    print(row("ES / VaR ratio (average)", f"{avg_es / avg_var:.2f}x"))
    print(sep)

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    output_path = os.path.join(os.path.dirname(__file__), "output", "backtest_plot.png")
    print("\nGenerating plot …")
    plot_backtest(returns, var_series, exceptions, es_series=es_series, output_path=output_path)
    print("Done.")


if __name__ == "__main__":
    main()
