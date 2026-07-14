"""Pins for quantfinlib.risk.settlement_risk_analyzer (hand-derived).

The load-bearing detail: at equal timestamps payments apply BEFORE
receipts — the conservative (Herstatt) reading of a simultaneous
pay/receive.
"""

import pytest

from quantfinlib.risk import settlement_risk_analyzer as sra
from quantfinlib.risk.settlement_risk_analyzer import SettlementLeg


def _leg(cp, pay_t, recv_t, amount):
    return SettlementLeg(cp, "USD", amount, pay_t, "EUR", amount, recv_t)


def test_herstatt_window_detection():
    assert _leg("A", 1, 3, 100.0).has_herstatt_window()
    assert not _leg("A", 3, 1, 100.0).has_herstatt_window()
    assert not _leg("A", 3, 3, 100.0).has_herstatt_window()  # simultaneous: no window


def test_herstatt_exposure_sums_at_risk_legs():
    legs = [
        _leg("A", 1, 3, 100.0),
        _leg("A", 2, 4, 50.0),
        _leg("A", 5, 4, 999.0),   # we receive first: no Herstatt window
        _leg("B", 1, 2, 25.0),
    ]
    out = sra.herstatt_exposure(legs)
    assert out["A"] == pytest.approx(150.0, abs=1e-15)
    assert out["B"] == pytest.approx(25.0, abs=1e-15)
    assert list(out) == ["A", "B"], "insertion order preserved"


def test_peak_exposure_payments_first_tie_break():
    # Leg 1 pays at t=1, receives at t=3 (100); leg 2 pays at t=3,
    # receives at t=5 (50). At t=3 the payment (+50) applies BEFORE the
    # receipt (-100): outstanding hits 150. Receipts-first would quietly
    # understate the peak as 100 — the wrong direction for a number
    # named after Herstatt.
    legs = [_leg("A", 1, 3, 100.0), _leg("A", 3, 5, 50.0)]
    assert sra.peak_exposure(legs, "A") == pytest.approx(150.0, abs=1e-15)


def test_peak_exposure_filters_counterparty_and_windowless_legs():
    legs = [
        _leg("A", 1, 4, 100.0),
        _leg("B", 1, 4, 500.0),   # other counterparty
        _leg("A", 5, 2, 999.0),   # receive-first: not at risk
        _leg("A", 2, 3, 50.0),    # overlaps leg 1: peak 150 in [2, 3)
    ]
    assert sra.peak_exposure(legs, "A") == pytest.approx(150.0, abs=1e-15)
    assert sra.peak_exposure(legs, "C") == 0.0
