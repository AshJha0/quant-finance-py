"""Pins for TriangularArbitrage, ForwardCurve and FairValueEngine, ported
from PricingTest.java.
"""

import math

import pytest

from quantfinlib.pricing import (FairValueEngine, ForwardCurve, Quote,
                                 TriangularArbitrage)


def test_consistent_triangle_shows_no_arbitrage():
    eurusd = Quote(1.1000, 1.1002)
    usdjpy = Quote(150.00, 150.02)
    eurjpy = Quote(165.00, 165.06)   # consistent with 1.1001 * 150.01 = 165.03
    assert not TriangularArbitrage.exists(eurusd, usdjpy, eurjpy, 0.0)
    assert TriangularArbitrage.arbitrage_bps(eurusd, usdjpy, eurjpy) < 0


def test_dislocated_cross_shows_executable_arbitrage():
    eurusd = Quote(1.1000, 1.1002)
    usdjpy = Quote(150.00, 150.02)
    # EURJPY bid far above synthetic ask (1.1002 * 150.02 = 165.05):
    # sell direct, buy synthetic.
    eurjpy = Quote(165.40, 165.45)
    bps = TriangularArbitrage.arbitrage_bps(eurusd, usdjpy, eurjpy)
    assert bps > 15
    assert TriangularArbitrage.exists(eurusd, usdjpy, eurjpy, 5)
    assert TriangularArbitrage.implied_cross_mid(eurusd, usdjpy) == pytest.approx(
        1.1001 * 150.01, abs=1e-9)


def test_crossed_quote_is_rejected():
    with pytest.raises(ValueError):
        Quote(1.10, 1.09)


def test_forward_curve_interpolates_and_extrapolates():
    curve = (ForwardCurve(1.1000)
             .add_point(0.25, 1.1050)
             .add_point(1.0, 1.1200))

    assert curve.forward(0) == pytest.approx(1.1000, abs=1e-12)
    assert curve.forward(0.25) == pytest.approx(1.1050, abs=1e-12)
    # Midway between 0.25y and 1.0y pillars.
    halfway = 1.1050 + (1.1200 - 1.1050) * (0.625 - 0.25) / 0.75
    assert curve.forward(0.625) == pytest.approx(halfway, abs=1e-12)
    # Extrapolation continues the last slope.
    slope = (1.1200 - 1.1050) / 0.75
    assert curve.forward(1.5) == pytest.approx(1.1200 + slope * 0.5, abs=1e-9)
    assert curve.forward_points(1.0) == pytest.approx(1.1200 - 1.1000, abs=1e-12)


def test_covered_interest_parity_checks():
    spot, rd, rf, t = 1.10, 0.05, 0.02, 1.0
    fair = ForwardCurve.theoretical_forward(spot, rd, rf, t)
    assert fair == pytest.approx(1.10 * 1.05 / 1.02, abs=1e-12)

    fair_curve = ForwardCurve(spot).add_point(t, fair)
    assert fair_curve.mispricing_bps(t, rd, rf) == pytest.approx(0, abs=1e-9)
    # Implied differential recovers ~ln(F/S)/t.
    assert fair_curve.implied_rate_differential(t) == pytest.approx(
        math.log(fair / spot), abs=1e-12)

    rich_curve = ForwardCurve(spot).add_point(t, fair * 1.001)
    assert rich_curve.mispricing_bps(t, rd, rf) == pytest.approx(10, abs=0.05)


def test_microprice_and_latency_adjusted_fair():
    assert FairValueEngine.microprice(99, 101, 90, 10) == pytest.approx(100.8, abs=1e-12)

    engine = FairValueEngine(64, 1_000_000_000)
    # Mid rising by 1.0 per second: quotes 100ms apart, +0.1 each.
    for i in range(10):
        mid = 100 + 0.1 * i
        engine.on_quote(mid - 0.01, mid + 0.01, 500, 500, i * 100_000_000)
    assert engine.drift_per_second() == pytest.approx(1.0, abs=0.01)
    # Balanced book: microprice = last mid; 50ms latency adds ~0.05 of drift.
    assert engine.latest_microprice() == pytest.approx(100.9, abs=1e-9)
    assert engine.latency_adjusted_fair(50_000_000) == pytest.approx(100.95, abs=0.005)


def test_fair_value_engine_edge_cases():
    engine = FairValueEngine()
    # NaN before the first quote; zero-size book is NaN too.
    assert math.isnan(engine.latest_microprice())
    assert math.isnan(FairValueEngine.microprice(99, 101, 0, 0))
    # Fewer than two samples: no drift estimate.
    engine.on_quote(99, 101, 1, 1, 0)
    assert engine.drift_per_second() == 0
