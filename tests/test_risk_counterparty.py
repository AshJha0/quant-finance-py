"""Pins for quantfinlib.risk.counterparty_exposure_tracker (hand-derived).

The load-bearing formula: PFE = (0.4 + 0.6 * NGR) * grossAddOn, with
NGR = net current exposure / gross positive MTM.
"""

import pytest

from quantfinlib.risk.counterparty_exposure_tracker import (
    CounterpartyExposureTracker,
    CounterpartyTrade,
    add_on_factor,
)


def _tracker():
    return (CounterpartyExposureTracker()
            .add_trade(CounterpartyTrade("A", "fx-fwd", 100.0, 10.0, 0.5))
            .add_trade(CounterpartyTrade("A", "fx-swap", 200.0, -5.0, 3.0))
            .add_trade(CounterpartyTrade("B", "fx-fwd", 50.0, -2.0, 0.5)))


def test_current_exposure_nets_within_the_set():
    t = _tracker()
    # A: 10 - 5 = 5 net; B: -2 floors at 0.
    assert t.current_exposure("A") == pytest.approx(5.0, abs=1e-15)
    assert t.current_exposure("B") == 0.0
    assert t.current_exposure("unknown") == 0.0


def test_pfe_net_to_gross_pin():
    t = _tracker()
    # grossAddOn(A) = 100*0.01 + 200*0.05 = 11; grossPositiveMtm = 10;
    # NGR = 5/10 = 0.5 -> PFE = (0.4 + 0.6*0.5)*11 = 0.7*11 = 7.7.
    assert t.potential_future_exposure("A") == pytest.approx(7.7, abs=1e-12)
    # B has no positive MTM: NGR = 1 (no relief) -> PFE = 1.0*(50*0.01) = 0.5.
    assert t.potential_future_exposure("B") == pytest.approx(0.5, abs=1e-12)


def test_total_and_all_exposures():
    t = _tracker()
    assert t.total_exposure("A") == pytest.approx(12.7, abs=1e-12)
    allx = t.all_exposures()
    assert list(allx) == ["A", "B"], "insertion order preserved"
    assert allx["A"] == pytest.approx(12.7, abs=1e-12)
    assert allx["B"] == pytest.approx(0.5, abs=1e-12)


def test_add_on_factor_tenor_buckets():
    # BIS CEM-style FX buckets: <1y 1%, 1-5y (inclusive) 5%, >5y 7.5%.
    assert add_on_factor(0.99) == 0.01
    assert add_on_factor(1.0) == 0.05
    assert add_on_factor(5.0) == 0.05
    assert add_on_factor(5.01) == 0.075
