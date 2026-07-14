"""TWAP/VWAP/POV/IS/WMR scheduler pins. Ported from Java
ExecutionTest / PovAndIsSchedulerTest.
"""

import math

import pytest

from quantfinlib.execution import (allocate_proportionally,
                                   implementation_shortfall_schedule,
                                   risk_aversion_for_front_load,
                                   twap_schedule, twap_schedule_randomized,
                                   vwap_schedule, wmr_fixing_schedule,
                                   WMR_WINDOW_MILLIS)
from quantfinlib.execution.pov_tracker import PovTracker
from quantfinlib.microstructure import AlmgrenChriss

# ------------------------------------------------------------------
# TWAP / VWAP
# ------------------------------------------------------------------


def test_twap_splits_evenly_and_preserves_total():
    slices = twap_schedule(1_000, 60_000, 6)
    assert len(slices) == 6
    total = sum(s.quantity for s in slices)
    for s in slices:
        assert abs(s.quantity - 1_000 / 6) <= 1
    assert total == 1_000
    assert slices[0].offset_millis == 0
    assert slices[-1].offset_millis == 50_000


def test_randomized_twap_preserves_total_and_varies():
    slices = twap_schedule_randomized(10_000, 60_000, 8, 0.3, 42)
    total = sum(s.quantity for s in slices)
    uneven = any(abs(s.quantity - 1_250) > 1 for s in slices)
    assert total == 10_000
    assert uneven, "jitter should produce uneven slices"


def test_vwap_allocates_proportionally_to_profile():
    slices = vwap_schedule(1_000, [1, 3], 20_000)
    assert slices[0].quantity == 250
    assert slices[1].quantity == 750
    assert slices[1].offset_millis == 10_000


def test_proportional_allocation_handles_rounding():
    alloc = allocate_proportionally(100, [1, 1, 1])
    assert alloc[0] + alloc[1] + alloc[2] == 100
    for a in alloc:
        assert 33 <= a <= 34


def test_schedulers_validate_inputs():
    with pytest.raises(ValueError):
        twap_schedule(0, 1000, 5)
    with pytest.raises(ValueError):
        twap_schedule(1000, 1000, 0)
    with pytest.raises(ValueError):
        vwap_schedule(1000, [], 1000)
    with pytest.raises(ValueError):
        vwap_schedule(1000, [-1, 2], 1000)


# ------------------------------------------------------------------
# WMR fixing
# ------------------------------------------------------------------


def test_wmr_delegates_to_twap_over_the_window():
    slices = wmr_fixing_schedule(1_000, 6)
    assert len(slices) == 6
    assert sum(s.quantity for s in slices) == 1_000
    assert slices[0].offset_millis == 0
    assert slices[-1].offset_millis == WMR_WINDOW_MILLIS * 5 // 6


def test_wmr_rejects_more_slices_than_quantity():
    with pytest.raises(ValueError):
        wmr_fixing_schedule(3, 5)


# ------------------------------------------------------------------
# POV
# ------------------------------------------------------------------


def test_pov_chases_target_participation():
    pov = PovTracker(10_000, 0.10, 0, 1_000)
    assert pov.due_quantity() == 0
    pov.on_market_volume(5_000)                  # target = 500
    assert pov.due_quantity() == 500
    pov.on_executed(500)
    assert pov.due_quantity() == 0
    pov.on_market_volume(3_000)                  # target = 800, executed 500
    assert pov.due_quantity() == 300
    assert pov.realized_participation() == pytest.approx(0.0625, abs=1e-12)


def test_pov_respects_slice_bounds_and_parent_remainder():
    pov = PovTracker(1_000, 0.5, 100, 400)
    pov.on_market_volume(150)                    # behind by 75 < minSlice
    assert pov.due_quantity() == 0
    pov.on_market_volume(1_850)                  # behind by 1000 -> capped at 400
    assert pov.due_quantity() == 400
    pov.on_executed(900)
    pov.on_market_volume(10_000)                 # behind, but only 100 left
    assert pov.due_quantity() == 100
    pov.on_executed(100)
    assert pov.done()
    pov.on_market_volume(10_000)
    assert pov.due_quantity() == 0


def test_pov_validates_parameters():
    with pytest.raises(ValueError):
        PovTracker(0, 0.1, 0, 10)
    with pytest.raises(ValueError):
        PovTracker(100, 0, 0, 10)
    with pytest.raises(ValueError):
        PovTracker(100, 1.5, 0, 10)
    with pytest.raises(ValueError):
        PovTracker(100, 0.1, 20, 10)


# ------------------------------------------------------------------
# Implementation shortfall
# ------------------------------------------------------------------


def _params(lam):
    return AlmgrenChriss.Params(100_000, 1.0, 10, 0.5, 1e-5, 1e-6, lam)


def test_is_schedule_sums_exactly_and_front_loads_with_urgency():
    risky = implementation_shortfall_schedule(_params(1e-4), 3_600_000)
    assert len(risky) == 10
    total = sum(s.quantity for s in risky)
    assert total == 100_000
    assert risky[0].quantity > risky[9].quantity, "risk-averse IS must front-load"
    assert risky[0].offset_millis == 0
    assert risky[9].offset_millis == 3_600_000 * 9 // 10


def test_zero_risk_aversion_degrades_to_twap():
    twap = implementation_shortfall_schedule(_params(0), 1_000_000)
    first = twap[0].quantity
    for s in twap:
        assert abs(s.quantity - first) <= 1, "lambda=0 must be a flat schedule"


def test_front_load_calibration_hits_the_requested_fraction():
    base = _params(0)
    lam = risk_aversion_for_front_load(base, 0.30)
    t = AlmgrenChriss.optimal_trajectory(base.with_risk_aversion(lam))
    assert t.trades[0] / 100_000 == pytest.approx(0.30, abs=1e-3)
    with pytest.raises(ValueError):
        risk_aversion_for_front_load(base, 0.05)


def test_sinh_overflow_fails_loudly_instead_of_silently_degrading():
    # A huge lambda overflows sinh in the AC trajectory (NaN holdings):
    # kappa*T ~ 840 here, past sinh's ~710 overflow point. schedule()
    # must raise, not spin out a garbage schedule.
    with pytest.raises(ValueError):
        implementation_shortfall_schedule(_params(1e40), 1_000_000)
    # The calibrator, by contrast, must survive probing overflowing
    # lambdas (NaN reads as "front-loads more than enough") and still
    # land on the requested fraction.
    lam = risk_aversion_for_front_load(_params(0), 0.90)
    t = AlmgrenChriss.optimal_trajectory(_params(0).with_risk_aversion(lam))
    assert t.trades[0] / 100_000 == pytest.approx(0.90, abs=1e-2)
