"""Pins for quantfinlib.risk.frtb_es, ported from Java
MarketRiskTest.frtbArithmeticMatchesTheRegulationsFormulas."""

import math

import numpy as np
import pytest

from quantfinlib.risk import frtb_es


def test_liquidity_horizon_cascade_pin():
    # Cascade, hand-computed: ES {10, 8, 5} at horizons {10, 20, 60}:
    # sqrt(10^2 + (8*1)^2 + (5*2)^2) = sqrt(100 + 64 + 100) = sqrt(264).
    es = frtb_es.liquidity_horizon_es([10.0, 8.0, 5.0], [10, 20, 60])
    assert es == pytest.approx(math.sqrt(264), abs=1e-12)


def test_stress_calibration_floors_at_one():
    # The stressed ratio scales capital up, never down.
    assert frtb_es.stress_calibrated_es(100, 30, 20) == pytest.approx(
        150, abs=1e-12), "ratio 1.5"
    assert frtb_es.stress_calibrated_es(100, 10, 20) == pytest.approx(
        100, abs=1e-12), "a calm stressed period floors at 1, never discounts"


def test_es975_pin():
    # losses 1..1000; VaR = 975th value, ES = mean of 975..1000
    # (26 values) = 987.5 exactly.
    losses = np.arange(1, 1001, dtype=float)
    assert frtb_es.es975(losses) == pytest.approx(987.5, abs=1e-9), \
        "mean of 975..1000"


def test_traffic_light_boundaries():
    assert frtb_es.TrafficLight.of(4) is frtb_es.TrafficLight.GREEN
    assert frtb_es.TrafficLight.of(5) is frtb_es.TrafficLight.AMBER
    assert frtb_es.TrafficLight.of(9) is frtb_es.TrafficLight.AMBER
    assert frtb_es.TrafficLight.of(10) is frtb_es.TrafficLight.RED
    with pytest.raises(ValueError):
        frtb_es.TrafficLight.of(-1)


def test_cascade_gates():
    with pytest.raises(ValueError):
        frtb_es.liquidity_horizon_es([10.0], [20])          # must start at 10
    with pytest.raises(ValueError):
        frtb_es.liquidity_horizon_es([10.0, 8.0], [10])     # misaligned
    with pytest.raises(ValueError):
        frtb_es.liquidity_horizon_es([10.0, 8.0], [10, 10])  # must ascend
    with pytest.raises(ValueError):
        frtb_es.liquidity_horizon_es([10.0, math.nan], [10, 20])  # NaN-rejecting
    with pytest.raises(ValueError):
        frtb_es.liquidity_horizon_es([10.0, -1.0], [10, 20])
    with pytest.raises(ValueError):
        frtb_es.stress_calibrated_es(100, 30, 0)            # reduced must be > 0
    with pytest.raises(ValueError):
        frtb_es.stress_calibrated_es(math.nan, 30, 20)      # NaN-rejecting
