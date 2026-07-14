"""Pins for quantfinlib.pricing.incremental_greeks, ported from
IncrementalGreeksTest.java (the JVM allocation-count test has no Python
equivalent and stays in Java).
"""

import pytest

from quantfinlib.pricing import BlackScholes, IncrementalGreeks, OptionType

S, K, R, Q, VOL, T = 100.0, 105.0, 0.04, 0.01, 0.20, 0.5


def test_small_moves_track_the_full_reprice_to_quadratic_accuracy():
    inc = IncrementalGreeks()
    inc.reprice(OptionType.CALL, S, K, R, Q, VOL, T)
    # Anchor state reproduces the full pricer exactly.
    assert inc.estimated_price() == pytest.approx(
        BlackScholes.price(OptionType.CALL, S, K, R, Q, VOL, T), abs=1e-12)
    assert inc.anchor_spot() == S

    # A 0.3% move: Taylor error is O(dS^3 * speed) — far below a cent.
    moved = S * 1.003
    inc.on_tick(moved)
    full_price = BlackScholes.price(OptionType.CALL, moved, K, R, Q, VOL, T)
    full_delta = BlackScholes.delta(OptionType.CALL, moved, K, R, Q, VOL, T)
    assert inc.estimated_price() == pytest.approx(full_price, abs=2e-4)
    assert inc.estimated_delta() == pytest.approx(full_delta, abs=2e-4)
    # Anchor Greeks are exposed for vega/theta risk (constant between anchors).
    assert inc.vega() == pytest.approx(BlackScholes.vega(S, K, R, Q, VOL, T), abs=1e-12)
    assert inc.gamma() > 0


def test_reprice_signal_fires_on_drift_and_clears_on_reanchor():
    inc = IncrementalGreeks()
    inc.reprice(OptionType.PUT, S, K, R, Q, VOL, T)
    inc.on_tick(S * 1.002)
    assert not inc.needs_reprice(S * 0.005)  # 0.2% < 0.5% drift budget
    inc.on_tick(S * 1.02)
    assert inc.needs_reprice(S * 0.005)      # 2% blows the budget
    # Re-anchoring at the new spot clears the signal.
    inc.reprice(OptionType.PUT, S * 1.02, K, R, Q, VOL, T)
    assert not inc.needs_reprice(S * 0.005)
