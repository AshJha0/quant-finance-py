"""YieldCurve pins, ported from Java RatesTest (curve half).

Bootstrap identity: flat 5% annual par swaps imply DF_n = 1.05^-n
exactly, hence a flat cc zero of ln(1.05).
"""

import math

import pytest

from quantfinlib.rates import YieldCurve


def test_bootstrap_flat_par_curve_recovers_flat_zeros():
    # Flat 5% annual par swaps -> DF_n = 1.05^-n, zero_cc = ln(1.05).
    curve = YieldCurve.bootstrap_annual_par_swaps(
        [1, 2, 3, 5], [0.05, 0.05, 0.05, 0.05])
    for y in range(1, 6):
        assert curve.discount_factor(y) == pytest.approx(1.05 ** -y, abs=1e-10)
        assert curve.zero_rate(y) == pytest.approx(math.log(1.05), abs=1e-10)


def test_upward_sloping_par_curve_implies_higher_forwards():
    curve = YieldCurve.bootstrap_annual_par_swaps([1, 2, 3], [0.02, 0.03, 0.04])
    assert curve.zero_rate(3) > curve.zero_rate(1)
    # Forward 2y->3y must exceed the 3y zero on an upward-sloping curve.
    assert curve.forward_rate(2, 3) > curve.zero_rate(3)
    assert curve.discount_factor(3) < curve.discount_factor(2)


def test_zero_rate_interpolates_and_extrapolates_flat():
    curve = YieldCurve.of_zero_rates([1, 3], [0.02, 0.04])
    assert curve.zero_rate(2) == pytest.approx(0.03, abs=1e-12)
    assert curve.zero_rate(0.5) == pytest.approx(0.02, abs=1e-12)   # flat short end
    assert curve.zero_rate(10) == pytest.approx(0.04, abs=1e-12)    # flat long end
    assert curve.discount_factor(0) == pytest.approx(1.0, abs=1e-12)


def test_curve_gates_refuse_nonsense():
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([1, 2], [0.02])            # misaligned
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([], [])                    # empty
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([0, 1], [0.02, 0.03])      # zero tenor
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([], [])       # empty
    curve = YieldCurve.of_zero_rates([1, 3], [0.02, 0.04])
    with pytest.raises(ValueError):
        curve.forward_rate(3, 3)                            # to <= from


def test_tenors_are_ascending_pillars():
    # Insertion order does not matter: the TreeMap-equivalent sorts keys.
    curve = YieldCurve.of_zero_rates([5, 1, 3], [0.05, 0.01, 0.03])
    assert curve.tenors() == [1.0, 3.0, 5.0]
    assert curve.zero_rate(1) == pytest.approx(0.01, abs=1e-15)
    assert curve.zero_rate(4) == pytest.approx(0.04, abs=1e-15)  # midpoint of 3y/5y


# ----------------------------------------------------------------------
# Review M6: input hygiene for both factories.
# ----------------------------------------------------------------------

def test_bootstrap_rejects_unsorted_fractional_and_nan_pillars():
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([2, 1, 3], [0.03, 0.031, 0.032])   # unsorted
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([1, 2, 2], [0.03, 0.031, 0.032])   # duplicate
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([0.5], [0.03])                     # sub-year
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([1, 2.5], [0.03, 0.031])           # fractional
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([1, 2], [0.03, math.nan])          # NaN par
    with pytest.raises(ValueError):
        YieldCurve.bootstrap_annual_par_swaps([1, math.nan], [0.03, 0.031])      # NaN tenor
    # The sorted version of the unsorted input is the correct curve.
    curve = YieldCurve.bootstrap_annual_par_swaps([1, 2, 3], [0.031, 0.03, 0.032])
    assert curve.zero_rate(1.0) == pytest.approx(math.log(1.031), abs=1e-12)


def test_of_zero_rates_rejects_nan_and_keeps_last_duplicate():
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([1, 2], [0.03, math.nan])
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([1, math.nan], [0.03, 0.03])
    with pytest.raises(ValueError):
        YieldCurve.of_zero_rates([1, math.inf], [0.03, 0.03])
    # Documented TreeMap.put semantics: a duplicate tenor keeps the LAST rate.
    curve = YieldCurve.of_zero_rates([1, 2, 2], [0.03, 0.04, 0.05])
    assert curve.tenors() == [1.0, 2.0]
    assert curve.zero_rate(2.0) == 0.05
    # Unsorted input is fine here: the curve sorts its pillars.
    curve = YieldCurve.of_zero_rates([3, 1, 2], [0.05, 0.03, 0.04])
    assert curve.tenors() == [1.0, 2.0, 3.0]
    assert curve.zero_rate(1.5) == pytest.approx(0.035, abs=1e-12)
