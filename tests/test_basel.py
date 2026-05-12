"""
test_basel.py — unit tests for traffic_light().
"""

import pytest

from var_backtest.basel import traffic_light


def test_green_zone_interior():
    assert traffic_light(0) == "green"
    assert traffic_light(2) == "green"
    assert traffic_light(4) == "green"


def test_yellow_zone_interior():
    assert traffic_light(5) == "yellow"
    assert traffic_light(7) == "yellow"
    assert traffic_light(9) == "yellow"


def test_red_zone():
    assert traffic_light(10) == "red"
    assert traffic_light(15) == "red"
    assert traffic_light(100) == "red"


def test_boundary_green_to_yellow():
    """4 exceptions → green; 5 exceptions → yellow."""
    assert traffic_light(4) == "green"
    assert traffic_light(5) == "yellow"


def test_boundary_yellow_to_red():
    """9 exceptions → yellow; 10 exceptions → red."""
    assert traffic_light(9) == "yellow"
    assert traffic_light(10) == "red"
