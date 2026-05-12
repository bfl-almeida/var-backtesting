# Root-level conftest.py — helps pytest locate the var_backtest package
# when tests are run from the repository root without an installed package.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
