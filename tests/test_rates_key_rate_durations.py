"""Key-rate duration pins, ported from Java MarketRiskPricingTest.

The slices must reassemble the parallel DV01 (within the documented
curve-interpolation tolerance) and a 5y bond's risk must live at the 5y
node.
"""

import pytest

from quantfinlib.rates import KeyRateDurations, YieldCurve


def test_key_rate_slices_add_back_up_to_the_parallel_dv01():
    curve = YieldCurve.of_zero_rates([1, 2, 5, 10], [0.02, 0.025, 0.03, 0.032])
    krd = KeyRateDurations.key_rate_dv01s(100, 0.04, 2, 5, curve)
    parallel = KeyRateDurations.parallel_dv01(100, 0.04, 2, 5, curve)

    total = sum(krd)
    max_at = max(range(len(krd)), key=lambda i: krd[i])
    assert total == pytest.approx(parallel, abs=0.02 * parallel), \
        "the slices reassemble the parallel move (interpolation tolerance)"
    assert max_at == 2, "a 5y bond's rate risk lives at the 5y node"
    assert parallel > 0, "rates up, bond down — DV01 sign convention"
    assert krd[3] < krd[2], "little risk beyond the bond's maturity"
