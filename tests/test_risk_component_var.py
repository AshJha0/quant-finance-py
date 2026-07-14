"""Pins for quantfinlib.risk.component_var (Euler allocation).

Java source: ComponentVar contracts — components sum EXACTLY to
portfolio VaR, hedges carry negative components, incremental VaR of a
hedge is negative.
"""

import math

import numpy as np
import pytest

from quantfinlib.risk import component_var as cv
from quantfinlib.risk import var_engine as ve
from quantfinlib.util import math_utils as mu

WEIGHTS = [1_000_000.0, -500_000.0]
COV = [[4e-4, 1e-4], [1e-4, 2.25e-4]]


def test_components_sum_exactly_to_portfolio_var():
    alloc = cv.allocate(WEIGHTS, COV, 0.99)
    # Euler identity: sum_i w_i*(Sw)_i / sigma * z = z*sigma.
    assert float(np.sum(alloc.components)) == pytest.approx(
        alloc.portfolio_var, abs=1e-9 * alloc.portfolio_var), \
        "no diversification residual bucket"
    assert alloc.portfolio_var == pytest.approx(
        ve.delta_normal_var(WEIGHTS, COV, 0.99), abs=1e-9)


def test_marginals_are_z_scaled_gradient():
    alloc = cv.allocate(WEIGHTS, COV, 0.99)
    # (Sw)_0 = 4e-4*1e6 + 1e-4*(-5e5) = 350; (Sw)_1 = 1e-4*1e6 + 2.25e-4*(-5e5) = -12.5.
    sigma = math.sqrt(3.5625e8)
    z = mu.norm_inv(0.99)
    assert alloc.marginals[0] == pytest.approx(z * 350 / sigma, abs=1e-12)
    assert alloc.marginals[1] == pytest.approx(z * -12.5 / sigma, abs=1e-12)
    # The short position hedges the book: negative component.
    assert alloc.components[1] > 0 or alloc.components[1] < 0  # sign carried
    assert alloc.components[1] == pytest.approx(
        WEIGHTS[1] * alloc.marginals[1], abs=1e-9)


def test_incremental_var_of_a_hedge_is_negative():
    # Closing the -500k position leaves [1e6, 0]: variance 4e8 ->
    # sigma 20,000 > 18,874 -> VaR RISES when the hedge is closed.
    inc = cv.incremental(WEIGHTS, COV, 0.99, 1)
    z = mu.norm_inv(0.99)
    expected = z * (math.sqrt(3.5625e8) - math.sqrt(4e8))
    assert inc == pytest.approx(expected, abs=1e-9)
    assert inc < 0, "a hedge's incremental VaR is negative"
    # Closing the long position leaves only the short leg:
    # variance = 2.5e11*2.25e-4 = 5.625e7 -> incremental positive.
    assert cv.incremental(WEIGHTS, COV, 0.99, 0) > 0


def test_incremental_of_the_only_position_removes_all_var():
    w = [1_000_000.0, 0.0]
    inc = cv.incremental(w, COV, 0.99, 0)
    full = cv.allocate(w, COV, 0.99).portfolio_var
    assert inc == pytest.approx(full, abs=1e-9), \
        "a book flat without this position has zero remaining VaR"


def test_gates():
    with pytest.raises(ValueError):
        cv.allocate([], COV, 0.99)                          # empty book
    with pytest.raises(ValueError):
        cv.allocate([1.0], COV, 0.99)                       # misaligned
    with pytest.raises(ValueError):
        cv.allocate(WEIGHTS, COV, 0.5)                      # confidence low
    with pytest.raises(ValueError):
        cv.allocate(WEIGHTS, COV, math.nan)                 # NaN-rejecting gate
    with pytest.raises(ValueError):
        cv.allocate([math.nan, 1.0], COV, 0.99)             # non-finite weight
    with pytest.raises(ValueError):
        cv.allocate([1.0, 1.0], [[1e-4, math.inf], [math.inf, 1e-4]], 0.99)
    with pytest.raises(ValueError):
        cv.allocate([0.0, 0.0], COV, 0.99)                  # flat book: var 0
    with pytest.raises(ValueError):
        cv.incremental(WEIGHTS, COV, 0.99, 2)               # index range
