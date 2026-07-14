"""Pins for quantfinlib.volatility.range_volatility.

Java sources: RangeVolatilityTest (closed-form constant-bar pins, GBM
recovery, Parkinson-vs-close-to-close efficiency, gates) and
RangeVolatilityEdgeTest (degenerate bars, scale invariance). The
closed-form and edge pins transfer exactly; the GBM path uses numpy's
Generator with the Java tolerances. The Java BarSeries-overload test has
no counterpart (no BarSeries in the Python port — arrays are the API).
"""

import math

import numpy as np
import pytest

from quantfinlib.util import math_utils as mu
from quantfinlib.volatility import RangeVolatility

PPY = 252.0


# ---------------------------------------------------------- closed forms

def test_parkinson_matches_its_closed_form_on_constant_range_bars():
    n = 10
    h = np.full(n, 105.0)
    l = np.full(n, 100.0)
    # sigma^2 = ln(105/100)^2 / (4 ln 2) per bar, identical bars -> mean
    # is the single-bar value; annualized by sqrt(* 252).
    ln_hl = math.log(105.0 / 100.0)
    expected = math.sqrt(ln_hl * ln_hl / (4 * math.log(2)) * PPY)
    assert RangeVolatility.parkinson(h, l, PPY) == pytest.approx(expected, abs=1e-15)


def test_garman_klass_and_rogers_satchell_match_their_closed_forms():
    n = 7
    o, hi, lo, c = 100.0, 108.0, 99.0, 104.0
    open_ = np.full(n, o)
    high = np.full(n, hi)
    low = np.full(n, lo)
    close = np.full(n, c)

    hl = math.log(hi / lo)
    co = math.log(c / o)
    gk_var = 0.5 * hl * hl - (2 * math.log(2) - 1) * co * co
    assert RangeVolatility.garman_klass(open_, high, low, close, PPY) == pytest.approx(
        math.sqrt(gk_var * PPY), abs=1e-15)

    rs_var = (math.log(hi / c) * math.log(hi / o)
              + math.log(lo / c) * math.log(lo / o))
    assert RangeVolatility.rogers_satchell(open_, high, low, close, PPY) == pytest.approx(
        math.sqrt(rs_var * PPY), abs=1e-15)


def test_yang_zhang_collapses_to_weighted_rogers_satchell_without_gaps_or_drift():
    # O_i = C_{i-1} (no overnight gap) and C_i = O_i (no open-to-close
    # move): both sample variances are exactly zero, so YZ^2 must be
    # exactly (1 - k) * RS^2 — the k weighting pinned by hand.
    n = 5  # m = 4 estimation periods
    open_ = np.full(n, 100.0)
    high = np.full(n, 110.0)
    low = np.full(n, 95.0)
    close = np.full(n, 100.0)
    rs = (math.log(110.0 / 100) * math.log(110.0 / 100)
          + math.log(95.0 / 100) * math.log(95.0 / 100))
    m = n - 1
    k = 0.34 / (1.34 + (m + 1.0) / (m - 1.0))
    assert RangeVolatility.yang_zhang(open_, high, low, close, PPY) == pytest.approx(
        math.sqrt((1 - k) * rs * PPY), abs=1e-15)


# ------------------------------------------------- GBM sanity + efficiency

def test_gbm_bars_recover_the_true_sigma_and_parkinson_beats_close_to_close():
    # Driftless GBM at sigma = 20%, 390 intraday steps per bar (the
    # observed high/low undershoot the continuous extremes by ~1-2% of
    # vol at this sampling — tolerances leave room for that bias).
    sigma = 0.20
    bars, steps = 500, 390
    dt = 1.0 / PPY
    dts = dt / steps
    rng = np.random.default_rng(7)
    z = rng.standard_normal((bars, steps))
    log_steps = -0.5 * sigma * sigma * dts + sigma * math.sqrt(dts) * z
    prices = 100.0 * np.exp(np.cumsum(log_steps.ravel())).reshape(bars, steps)
    close = prices[:, -1]
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, prices.max(axis=1))
    low = np.minimum(open_, prices.min(axis=1))

    assert RangeVolatility.garman_klass(open_, high, low, close, PPY) == pytest.approx(
        sigma, abs=0.02)
    assert RangeVolatility.rogers_satchell(open_, high, low, close, PPY) == pytest.approx(
        sigma, abs=0.02)
    assert RangeVolatility.yang_zhang(open_, high, low, close, PPY) == pytest.approx(
        sigma, abs=0.02)
    assert RangeVolatility.parkinson(high, low, PPY) == pytest.approx(sigma, abs=0.03)

    # Efficiency: over 25 independent 20-bar blocks the Parkinson
    # estimates scatter visibly less than close-to-close estimates —
    # the ~4.9x variance advantage shows up as a smaller sample std.
    blocks = 25
    length = bars // blocks
    park = np.empty(blocks)
    c2c = np.empty(blocks)
    for b in range(blocks):
        lo_i = b * length
        park[b] = RangeVolatility.parkinson(high[lo_i:lo_i + length],
                                            low[lo_i:lo_i + length], PPY)
        r = np.log(close[lo_i + 1:lo_i + length] / close[lo_i:lo_i + length - 1])
        c2c[b] = math.sqrt(float(np.sum(r * r)) / (length - 1) * PPY)
    assert mu.std_dev(park) < mu.std_dev(c2c), \
        f"Parkinson std {mu.std_dev(park)} must beat close-to-close {mu.std_dev(c2c)}"


# ---------------------------------------------------------------- edges

def test_constant_price_bars_have_exactly_zero_vol_under_every_estimator():
    # O = H = L = C = 100 for every bar: every log ratio is ln(1) = 0,
    # both Yang-Zhang sample variances are 0/0-free zeros, so all four
    # estimators must return 0.0 exactly — no tolerance.
    n = 5
    o = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    c = np.full(n, 100.0)
    assert RangeVolatility.parkinson(h, l, PPY) == 0.0
    assert RangeVolatility.garman_klass(o, h, l, c, PPY) == 0.0
    assert RangeVolatility.rogers_satchell(o, h, l, c, PPY) == 0.0
    assert RangeVolatility.yang_zhang(o, h, l, c, PPY) == 0.0


def test_estimators_are_scale_invariant():
    # Volatility is about RATIOS: quoting the same bars in cents instead
    # of dollars (scale by 4 — a power of two, so the division h/l is
    # bit-identical) cannot change any estimate.
    o = np.array([100.0, 104, 101, 103])
    h = np.array([108.0, 107, 106, 105])
    l = np.array([99.0, 100, 98, 101])
    c = np.array([104.0, 101, 105, 102])
    assert RangeVolatility.parkinson(h, l, PPY) == RangeVolatility.parkinson(
        4 * h, 4 * l, PPY)
    assert RangeVolatility.garman_klass(o, h, l, c, PPY) == \
        RangeVolatility.garman_klass(4 * o, 4 * h, 4 * l, 4 * c, PPY)
    assert RangeVolatility.rogers_satchell(o, h, l, c, PPY) == \
        RangeVolatility.rogers_satchell(4 * o, 4 * h, 4 * l, 4 * c, PPY)
    assert RangeVolatility.yang_zhang(o, h, l, c, PPY) == \
        RangeVolatility.yang_zhang(4 * o, 4 * h, 4 * l, 4 * c, PPY)


# ---------------------------------------------------------------- gates

def test_gates_refuse_malformed_bars():
    ok = np.array([100.0, 100.0])
    with pytest.raises(ValueError):  # misaligned
        RangeVolatility.parkinson([105.0], [105.0, 100.0], PPY)
    with pytest.raises(ValueError):  # H < L
        RangeVolatility.parkinson([99.0, 99.0], ok, PPY)
    with pytest.raises(ValueError):  # L <= 0
        RangeVolatility.parkinson([105.0, 105.0], [0.0, 100.0], PPY)
    with pytest.raises(ValueError):  # NaN
        RangeVolatility.parkinson([math.nan, 105.0], ok, PPY)
    with pytest.raises(ValueError):  # ppy <= 0
        RangeVolatility.parkinson([105.0, 105.0], ok, 0)
    with pytest.raises(ValueError):  # close above high
        RangeVolatility.garman_klass([100.0], [101.0], [99.0], [102.0], PPY)
    with pytest.raises(ValueError):  # open below low
        RangeVolatility.rogers_satchell([98.0], [101.0], [99.0], [100.0], PPY)
    with pytest.raises(ValueError):  # YZ needs >= 3 bars
        RangeVolatility.yang_zhang(ok, [105.0, 105.0], [99.0, 99.0], ok, PPY)
