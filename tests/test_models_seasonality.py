"""Pins for the intraday seasonality curves: VolumeCurve,
VolatilityCurve, SpreadForecaster, DayTypeProfiles.

Java sources: QuantModelsTest.java, QuantModels2Test.java (VolatilityCurve
section), QuantModels3Test.java (DayTypeProfiles section).
"""

import math

import numpy as np
import pytest

from quantfinlib.microstructure.day_type_profiles import DayTypeProfiles
from quantfinlib.microstructure.spread_forecaster import SpreadForecaster
from quantfinlib.microstructure.volatility_curve import VolatilityCurve
from quantfinlib.microstructure.volume_curve import VolumeCurve

SEC = 1_000_000_000


# ------------------------------------------------------------------
# VolumeCurve
# ------------------------------------------------------------------

def test_volume_curve_learns_a_u_shape_and_gives_a_cumulative_fraction():
    vc = VolumeCurve(4, 0.5)
    shape = [40, 10, 10, 40]
    vc.seed_profile(shape)
    assert vc.expected_fraction_elapsed(1, 0.0) == pytest.approx(0.4, abs=1e-9)
    assert vc.expected_fraction_elapsed(2, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert vc.expected_fraction_elapsed(3, 1.0) == pytest.approx(1.0, abs=1e-9)
    assert vc.expected_fraction_elapsed(0, 0.5) == pytest.approx(0.2, abs=1e-9)


def test_volume_curve_rescales_today_to_realized_volume():
    vc = VolumeCurve(4, 0.5)
    vc.seed_profile([40, 10, 10, 40])   # learned total 100
    vc.on_volume(0, 80)
    projected = vc.projected_day_volume(1, 0.0)
    assert projected == pytest.approx(140, abs=1e-6)
    assert vc.expected_volume_remaining(1, 0.0) > 0


def test_volume_curve_degrades_to_linear_without_a_profile():
    vc = VolumeCurve(10, 0.5)
    assert vc.expected_fraction_elapsed(5, 0.0) == pytest.approx(0.5, abs=1e-9)


def test_volume_curve_learns_across_days():
    vc = VolumeCurve(2, 1.0)
    vc.on_volume(0, 30)
    vc.on_volume(1, 70)
    vc.roll_day()
    assert vc.days_learned() == 1
    assert vc.profile_volume(0) == pytest.approx(30, abs=1e-9)
    assert vc.realized_today() == pytest.approx(0, abs=1e-9)


def test_volume_curve_validation():
    with pytest.raises(ValueError):
        VolumeCurve(0, 0.5)
    with pytest.raises(ValueError):
        VolumeCurve(4, 1.5)


# ------------------------------------------------------------------
# VolatilityCurve
# ------------------------------------------------------------------

def test_vol_curve_learns_the_u_shape_across_days():
    vc = VolatilityCurve(3, 0.5)
    vc.on_vol(0, 2e-4)
    vc.on_vol(1, 5e-5)
    vc.on_vol(2, 1.5e-4)
    vc.roll_day()
    assert vc.baseline(0) == pytest.approx(2e-4, abs=1e-12)
    vc.on_vol(0, 4e-4)
    vc.roll_day()
    assert vc.baseline(0) == pytest.approx(3e-4, abs=1e-12)
    assert vc.baseline(1) == pytest.approx(5e-5, abs=1e-12)


def test_regime_is_time_of_day_aware_not_absolute():
    vc = VolatilityCurve(2, 0.5)
    vc.seed_baseline([2e-4, 5e-5])
    assert vc.regime(0, 2e-4) == pytest.approx(0.0, abs=1e-12)
    assert vc.regime(1, 2e-4) == pytest.approx(1.0, abs=1e-12)
    assert vc.regime(1, 7.5e-5) == pytest.approx(0.5, abs=1e-12)
    assert vc.regime(0, math.nan) == pytest.approx(0, abs=1e-12)
    assert VolatilityCurve(2, 0.5).regime(0, 1e-4) == pytest.approx(0, abs=1e-12)


def test_a_bucket_first_observed_later_seeds_instead_of_ramping_from_zero():
    vc = VolatilityCurve(2, 0.1)
    vc.on_vol(0, 2e-4)
    vc.roll_day()                       # bucket 1 unseen on day 1
    vc.on_vol(1, 5e-5)
    vc.roll_day()
    assert vc.baseline(1) == pytest.approx(5e-5, abs=1e-15)
    assert vc.regime(1, 5e-5) == pytest.approx(0.0, abs=1e-12)


def test_vol_curve_ignores_non_finite_readings():
    vc = VolatilityCurve(1, 0.5)
    vc.on_vol(0, 1e-4)
    vc.on_vol(0, math.nan)
    vc.on_vol(0, math.inf)
    vc.on_vol(0, -1)
    vc.roll_day()
    assert vc.baseline(0) == pytest.approx(1e-4, abs=1e-12)


def test_volatility_curve_validation():
    with pytest.raises(ValueError):
        VolatilityCurve(0, 0.5)
    with pytest.raises(ValueError):
        VolatilityCurve(4, 0)
    with pytest.raises(ValueError):
        VolatilityCurve(2, 0.5).seed_baseline(np.zeros(3))


# ------------------------------------------------------------------
# SpreadForecaster
# ------------------------------------------------------------------

def test_spread_forecast_blends_baseline_and_reverting_deviation():
    sf = SpreadForecaster(3, 0.5, SEC)
    sf.seed_baseline([0.05, 0.01, 0.05])
    sf.on_spread(1, 0.03, SEC)
    assert sf.forecast(1, SEC) > 0.01
    later = sf.forecast(1, SEC + SEC)
    assert later < sf.forecast(1, SEC) and later > 0.01
    assert sf.forecast(1, SEC + 100 * SEC) == pytest.approx(0.01, abs=1e-3)


def test_spread_forecast_knows_the_close_is_wide_before_it_arrives():
    sf = SpreadForecaster(3, 0.5, SEC)
    sf.seed_baseline([0.05, 0.01, 0.06])
    assert sf.forecast(2, SEC) > sf.forecast(1, SEC)


def test_unseeded_day0_forecast_returns_the_live_spread_not_zero():
    sf = SpreadForecaster(3, 0.1, SEC)
    assert math.isnan(sf.forecast(1, SEC))
    sf.on_spread(1, 0.02, SEC)
    assert sf.forecast(1, SEC) == pytest.approx(0.02, abs=1e-12)


def test_baseline_is_learned_across_days_via_roll_day():
    sf = SpreadForecaster(2, 0.5, SEC)
    sf.on_spread(0, 0.04, SEC)
    sf.on_spread(0, 0.06, SEC + 1)
    sf.on_spread(1, 0.01, SEC + 2)
    sf.roll_day()
    assert sf.baseline(0) == pytest.approx(0.05, abs=1e-9)
    assert sf.baseline(1) == pytest.approx(0.01, abs=1e-9)
    sf.on_spread(0, 0.09, 2 * SEC)
    sf.roll_day()
    assert sf.baseline(0) == pytest.approx(0.07, abs=1e-9)
    assert sf.baseline(1) == pytest.approx(0.01, abs=1e-9)


def test_a_bucket_without_a_baseline_injects_no_deviation():
    sf = SpreadForecaster(2, 0.5, SEC)
    sf.on_spread(0, 0.02, SEC)
    sf.roll_day()                       # bucket 1 unseen on day 1
    sf.on_spread(1, 0.03, 2 * SEC)
    assert sf.current_deviation(2 * SEC) == pytest.approx(0, abs=1e-12)
    assert sf.forecast(0, 2 * SEC) == pytest.approx(0.02, abs=1e-12)
    sf.roll_day()
    assert sf.baseline(1) == pytest.approx(0.03, abs=1e-9)


def test_infinite_spread_does_not_poison_a_bucket_baseline():
    sf = SpreadForecaster(2, 0.5, SEC)
    sf.seed_baseline([0.02, 0.02])
    sf.on_spread(0, math.inf, SEC)
    sf.on_spread(0, math.nan, SEC + 1)
    assert math.isfinite(sf.forecast(0, SEC + 2))
    assert sf.forecast(0, SEC + 2) == pytest.approx(0.02, abs=1e-3)


def test_spread_forecaster_validation():
    with pytest.raises(ValueError):
        SpreadForecaster(4, 0.5, 0)


# ------------------------------------------------------------------
# DayTypeProfiles
# ------------------------------------------------------------------

def test_day_types_learn_independent_shapes():
    # 0 = regular, 1 = expiry. Regular days are flat; expiry days trade
    # 3x into the close. One averaged curve would be wrong on both;
    # per-type curves are right on each.
    volume = DayTypeProfiles(2, lambda _: VolumeCurve(2, 0.5))

    regular = volume.profile(0)
    regular.on_volume(0, 100)
    regular.on_volume(1, 100)
    regular.roll_day()

    expiry = volume.profile(1)
    expiry.on_volume(0, 100)
    expiry.on_volume(1, 300)
    expiry.roll_day()

    assert volume.profile(0).profile_volume(1) == pytest.approx(100, abs=1e-9)
    assert volume.profile(1).profile_volume(1) == pytest.approx(300, abs=1e-9)
    assert volume.profile(0).days_learned() == 1
    assert (volume.profile(1).expected_fraction_elapsed(0, 1.0)
           < volume.profile(0).expected_fraction_elapsed(0, 1.0))


def test_day_type_factory_variants_and_validation():
    regular_shape = [100, 100]
    vols = DayTypeProfiles(2, lambda t: VolumeCurve(2, 0.5).seed_profile(regular_shape))
    assert vols.day_types() == 2
    assert vols.profile(1).profile_volume(0) == pytest.approx(100, abs=1e-9)
    with pytest.raises(ValueError):
        DayTypeProfiles(0, lambda t: VolumeCurve())
