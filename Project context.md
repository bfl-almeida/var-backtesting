# Project Context — VaR Backtesting Library

## Objective

Build a small, clean, well-tested Python library that performs **historical Value-at-Risk (VaR) backtesting** on SPY daily returns. The deliverable is a public GitHub repository that can be referenced on a CV applying for junior quant / market-risk consultant roles in Paris.

**This is a one-weekend project.** Keep it tight. Do not over-engineer. The goal is a finished, defensible, interview-ready piece of work — not a production risk system.

---

## Deliverable

A public GitHub repository containing:

1. A Python package implementing historical VaR estimation, exception identification, Kupiec test, Christoffersen test, and Basel traffic-light classification.
2. Unit tests covering each statistical function.
3. One example notebook or script that downloads SPY data, runs the full backtest, prints a summary table, and produces one plot (returns + VaR + exception markers).
4. A clear `README.md` explaining what the project does, the math behind each test, how to install, and how to run the example.

---

## Tech Stack — fixed

- Python 3.10+
- Libraries: `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `yfinance` (for SPY data), `pytest` (for tests)
- No other dependencies. No deep-learning libraries. No web framework. No Docker. No CI pipeline.

---

## Scope — MUST INCLUDE

1. **Historical VaR estimation**
   - 1-day horizon
   - 99% confidence level (parameterizable, but default 99%)
   - 250-day rolling window (parameterizable, but default 250)
   - Sign convention: VaR returned as a positive number representing the loss magnitude (i.e. `VaR = -quantile(returns, 1%)`)

2. **Exception identification**
   - At each date `t`, an exception is recorded if the realized return on day `t+1` is less than `-VaR_t`
   - Returns a boolean series aligned with the dates of realized returns

3. **Kupiec unconditional coverage test**
   - Likelihood-ratio test for whether the observed exception rate matches the expected rate (1 − α)
   - Test statistic: `LR_uc = -2 * ln[ ((1-p)^(T-N) * p^N) / ((1-N/T)^(T-N) * (N/T)^N) ]`
     where `T` is sample size, `N` is number of exceptions, `p` is expected exception rate
   - Distribution: chi-squared with 1 degree of freedom
   - Output: test statistic, p-value, reject/fail-to-reject at 5% level

4. **Christoffersen independence test**
   - Likelihood-ratio test for independence of consecutive exceptions (clustering)
   - Compute transition counts `n_00, n_01, n_10, n_11` from the exception series
   - Estimate `π_01 = n_01 / (n_00 + n_01)`, `π_11 = n_11 / (n_10 + n_11)`, `π = (n_01 + n_11) / T`
   - Test statistic: `LR_ind = -2 * ln[ L_independent / L_dependent ]` (standard formula — implement carefully, handle edge cases where any count is zero by using a small epsilon or skipping that term)
   - Distribution: chi-squared with 1 degree of freedom
   - Also expose the **conditional coverage test**: `LR_cc = LR_uc + LR_ind`, chi-squared with 2 degrees of freedom

5. **Basel traffic-light classification**
   - Input: number of exceptions over the most recent 250 trading days at the 99% VaR level
   - Output: zone classification
     - Green: 0 to 4 exceptions
     - Yellow: 5 to 9 exceptions
     - Red: 10 or more exceptions
   - Include a short comment in code referencing the Basel framework rationale


5b. Historical Expected Shortfall (ES) estimation

At each date t, compute ES at the same confidence level α as the VaR (default 99%) using the same rolling window (default 250 days).
Formula: ES is the negative mean of returns in the rolling window that fall at or below the negative VaR threshold.

In code: es_t = -returns_window[returns_window <= -var_t].mean()


Sign convention: ES returned as a positive number, like VaR (magnitude of expected loss in the tail).
Edge case: if no returns in the window fall below -VaR, return NaN for that date and document this in the docstring.
Add ES to the summary table printed by examples/run_backtest.py next to VaR.
Overlay the rolling ES line on the existing plot (different color from VaR, dashed style works).


Add to repository structure
var_backtest/
├── es.py          # historical Expected Shortfall estimation
Add to "Unit tests"


Test ES on a synthetic input with known tail mean. Verify es >= var always (ES is at least as large as VaR by construction).


Update README

Add one short paragraph explaining the math: "Expected Shortfall (also called Conditional VaR) is the expected loss conditional on the loss exceeding the VaR threshold. It is a coherent risk measure (unlike VaR) and is the regulatory standard under FRTB."

6. **Plot**
   - One figure showing: SPY daily returns over time, the rolling VaR line (negated, so it sits below zero), and exception markers (red dots) on the days where breaches occurred
   - Save the plot to `examples/output/backtest_plot.png`

7. **Unit tests** (`tests/`)
   - Test that historical VaR matches a manually computed value on a synthetic input
   - Test Kupiec test: at expected rate, p-value should be high; at clearly wrong rate, p-value should be low
   - Test Christoffersen test: independent draws → high p-value; clustered draws → low p-value
   - Test traffic-light thresholds at boundaries (4, 5, 9, 10)
   - All tests must pass with `pytest`

8. **README.md**
   - Project description (3–4 sentences)
   - Math for each test (one paragraph each, with the formula in LaTeX inline or as a code block)
   - Installation instructions (`pip install -r requirements.txt`)
   - How to run the example
   - Example output (one screenshot of the plot, one snippet of the summary table)
   - Note: "Built as a self-study project for transition to quantitative finance. Not a production system."

---

## Scope — OUT OF SCOPE (do not build)

- Parametric (variance-covariance) VaR
- Monte Carlo VaR
- Expected Shortfall (can be a future extension; mention in README under "Future work" but do not implement)
- Multi-asset portfolios — SPY only
- GARCH or EWMA volatility modeling
- Web UI or dashboard
- Any database or persistent storage beyond CSV caching
- Docker, CI/CD pipelines, type-checking hooks
- Logging frameworks beyond `print` in the example script
- Configuration files (`yaml`, `toml`) — pass parameters as function arguments

---

## Repository Structure

```
var-backtesting/
├── README.md
├── requirements.txt
├── var_backtest/
│   ├── __init__.py
│   ├── data.py          # SPY data loading via yfinance, with simple CSV caching
│   ├── var.py           # historical VaR estimation
│   ├── exceptions.py    # exception identification
│   ├── tests.py         # Kupiec, Christoffersen, conditional coverage
│   ├── basel.py         # traffic-light classification
│   └── plotting.py      # the one plot function
├── examples/
│   ├── run_backtest.py  # the end-to-end example script
│   └── output/
│       └── backtest_plot.png
└── tests/
    ├── test_var.py
    ├── test_kupiec.py
    ├── test_christoffersen.py
    └── test_basel.py
```

---

## Code Quality Requirements

- **Type hints** on every public function signature
- **Docstrings** in NumPy or Google style on every public function — including the math being computed
- **No global state.** Pure functions where possible.
- **No premature abstraction.** No classes unless they clearly help (most of this can be done with functions returning DataFrames or floats).
- **Variable names** match the math: `alpha`, `window`, `var_series`, `exceptions`, `lr_uc`, `p_value`, etc. No cryptic one-letter names except inside a loop.
- **Reproducibility:** if any randomness is introduced in tests, seed it.

---

## Acceptance Criteria

The project is done when **all** of the following are true:

1. `pytest` passes with zero failures.
2. `python examples/run_backtest.py` runs end-to-end, prints a summary table (number of exceptions, Kupiec p-value, Christoffersen p-value, Basel zone), and writes `examples/output/backtest_plot.png`.
3. The README explains the math for each test in plain English plus the formula.
4. The repository is pushed to public GitHub.
5. The code is under 600 lines total (excluding tests, examples, and README). If it exceeds this, simplify.

---

## CV Bullet This Project Will Earn

Once the repo is on public GitHub and the README is in place, the candidate will add this line to the CV:

> *VaR backtesting library (Python, public GitHub) — historical 1-day Value-at-Risk at 99% on SPY returns with a rolling 250-day window; Kupiec unconditional coverage test, Christoffersen independence test, and Basel traffic-light evaluation.*

Build accordingly. The bullet has to be defensible in a 60-second interview probe.

---

## Notes for the Agent

- If any formula in this document is ambiguous, **stop and ask** rather than guessing. The statistical correctness of Kupiec and Christoffersen is the entire point of this library — getting it slightly wrong defeats the purpose.
- If a library version causes friction (for example `yfinance` API changes), pin a working version in `requirements.txt` and move on. Do not spend more than 30 minutes on dependency issues.
- Do not refactor existing CV or GitHub repos. This is a standalone new project.
- Do not invent additional features. If the scope above is finished and tests pass, the project is done.


Ne