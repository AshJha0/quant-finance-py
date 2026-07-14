"""Limit pins for the Asian layer, ported from AsianStructuredEdgeTest.java
(the structured-notes vol-direction test lives with the other
StructuredNotes pins): Kemna-Vorst and Turnbull-Wakeman converge as vol
goes to zero (the AM-GM gap is O(vol^2)) and tiny expiry collapses both
to intrinsic.
"""

import pytest

from quantfinlib.pricing import AsianOption, OptionType

CALL = OptionType.CALL
S, R, Q = 100.0, 0.05, 0.01


def test_kemna_vorst_and_turnbull_wakeman_agree_at_low_vol_and_diverge_with_vol():
    # With r = q the fixing forwards are all S, so the drift Jensen
    # term vanishes and the two prices differ ONLY through the AM-GM
    # gap between the average's distributions — which scales with
    # vol^2: sub-penny at 1% vol, visibly larger at 40%. (With r != q
    # the gap has a vol-independent drift floor, ~0.007 here at
    # g = 4%, which is why this pin sets g = 0.)
    n = 12
    t = 1.0
    r, q = 0.03, 0.03
    geo_low = AsianOption.geometric_price(CALL, S, 100, r, q, 0.01, t, n)
    arith_low = AsianOption.arithmetic_price(CALL, S, 100, r, q, 0.01, t, n)
    gap_low = arith_low - geo_low
    assert gap_low >= 0            # AM-GM: arithmetic >= geometric
    assert gap_low < 1e-3          # 1% vol, zero carry gap: KV ~ TW

    geo_high = AsianOption.geometric_price(CALL, S, 100, r, q, 0.40, t, n)
    arith_high = AsianOption.arithmetic_price(CALL, S, 100, r, q, 0.40, t, n)
    gap_high = arith_high - geo_high
    assert gap_high > 10 * gap_low  # vol^2 scaling


def test_tiny_expiry_collapses_to_intrinsic():
    # T = 1e-6: every fixing is (essentially) spot, so the ITM call is
    # worth its intrinsic S - K and the OTM call nothing.
    t = 1e-6
    assert AsianOption.geometric_price(CALL, S, 90, R, Q, 0.25, t, 4) == pytest.approx(
        10.0, abs=1e-3)
    assert AsianOption.arithmetic_price(CALL, S, 90, R, Q, 0.25, t, 4) == pytest.approx(
        10.0, abs=1e-3)
    assert AsianOption.geometric_price(CALL, S, 110, R, Q, 0.25, t, 4) == pytest.approx(
        0.0, abs=1e-6)
    assert AsianOption.arithmetic_price(CALL, S, 110, R, Q, 0.25, t, 4) == pytest.approx(
        0.0, abs=1e-6)
