"""Pins for quantfinlib.pricing.structured_notes, ported from
StructuredNotesTest.java and the vol-direction pins of
AsianStructuredEdgeTest.java: each product must equal its decomposition
into bond + vanilla pieces to machine precision, and every solver must
round-trip.
"""

import math

import pytest

from quantfinlib.pricing import BlackScholes, OptionType, StructuredNotes

S, R, Q, VOL, T = 100.0, 0.03, 0.01, 0.25, 1.0


def test_reverse_convertible_equals_bond_minus_put_replication():
    par, coupon, k = 1000.0, 0.09, 90.0
    note = StructuredNotes.reverse_convertible(par, coupon, S, k, R, Q, VOL, T)
    replication = (par * 1.09 * math.exp(-R * T)
                   - (par / k) * BlackScholes.price(OptionType.PUT, S, k, R, Q, VOL, T))
    assert note == pytest.approx(replication, abs=1e-12)  # the note IS its replication
    # The fat coupon is put premium: the note is worth LESS than the
    # same bond without the embedded short put.
    assert note < par * 1.09 * math.exp(-R * T)
    # Zero vol, strike comfortably below spot: the put is worthless
    # and the note is pure bond + coupon.
    assert StructuredNotes.reverse_convertible(par, coupon, S, 60, R, Q, 0, T) == pytest.approx(
        par * 1.09 * math.exp(-R * T), abs=1e-9)
    # Short put = long the stock.
    assert StructuredNotes.reverse_convertible_delta(par, S, k, R, Q, VOL, T) > 0


def test_capital_protected_note_floors_and_participation_solver_round_trips():
    par, protection = 1000.0, 0.95
    # Zero vol: the ATM call under carry Q > 0 still has forward value;
    # use the floor identity instead at zero participation.
    assert StructuredNotes.capital_protected_note(
        par, protection, 0, S, R, Q, VOL, T) == pytest.approx(
            protection * par * math.exp(-R * T), abs=1e-12)

    # Solver round trip: the participation the budget affords reprices
    # to exactly that budget.
    issue_price = 990.0
    p = StructuredNotes.participation_for(par, protection, issue_price, S, R, Q, VOL, T)
    assert p > 0
    assert StructuredNotes.capital_protected_note(
        par, protection, p, S, R, Q, VOL, T) == pytest.approx(issue_price, abs=1e-9)

    # The zero-rate era lesson: same budget, lower rates -> the bond
    # floor eats more of it -> thinner participation.
    p_low_rate = StructuredNotes.participation_for(par, protection, issue_price,
                                                   S, 0.001, Q, VOL, T)
    assert p_low_rate < p

    # A budget below the protected floor is unbuildable.
    with pytest.raises(ValueError):
        StructuredNotes.participation_for(par, 1.0, 900, S, 0.001, Q, VOL, T)


def test_discount_certificate_is_the_covered_call():
    cap = 110.0
    cert = StructuredNotes.discount_certificate(S, cap, R, Q, VOL, T)
    covered_call = (S * math.exp(-Q * T)
                    - BlackScholes.price(OptionType.CALL, S, cap, R, Q, VOL, T))
    assert cert == pytest.approx(covered_call, abs=1e-12)
    # The discount to (carry-adjusted) spot is the call premium: cert < S e^{-qT}.
    assert cert < S * math.exp(-Q * T)
    # A cap far above spot gives away almost nothing: cert -> S e^{-qT}.
    assert StructuredNotes.discount_certificate(S, 10_000, R, Q, VOL, T) == pytest.approx(
        S * math.exp(-Q * T), abs=1e-6)
    delta = StructuredNotes.discount_certificate_delta(S, cap, R, Q, VOL, T)
    assert 0 < delta < 1  # long stock short call: delta in (0,1)


def test_structured_notes_move_in_vol_with_their_embedded_option_sign():
    # From AsianStructuredEdgeTest.java.
    t = 1.0
    # Reverse convertible = bond MINUS put: more vol, dearer put, cheaper note.
    rc_low = StructuredNotes.reverse_convertible(1000, 0.09, S, 90, R, Q, 0.15, t)
    rc_mid = StructuredNotes.reverse_convertible(1000, 0.09, S, 90, R, Q, 0.25, t)
    rc_high = StructuredNotes.reverse_convertible(1000, 0.09, S, 90, R, Q, 0.35, t)
    assert rc_low > rc_mid > rc_high

    # Capital-protected note = bond PLUS participation * call: more vol,
    # dearer call, dearer note.
    cp_low = StructuredNotes.capital_protected_note(1000, 0.95, 1.0, S, R, Q, 0.15, t)
    cp_high = StructuredNotes.capital_protected_note(1000, 0.95, 1.0, S, R, Q, 0.35, t)
    assert cp_high > cp_low

    # Discount certificate = stock MINUS call: falls with vol.
    dc_low = StructuredNotes.discount_certificate(S, 110, R, Q, 0.15, t)
    dc_high = StructuredNotes.discount_certificate(S, 110, R, Q, 0.35, t)
    assert dc_high < dc_low


def test_structured_note_gates():
    with pytest.raises(ValueError):
        StructuredNotes.reverse_convertible(1000, -0.01, S, 90, R, Q, VOL, T)
    with pytest.raises(ValueError):
        StructuredNotes.capital_protected_note(1000, 1.5, 0.5, S, R, Q, VOL, T)
    with pytest.raises(ValueError):
        StructuredNotes.discount_certificate(S, 0, R, Q, VOL, T)
    with pytest.raises(ValueError):
        StructuredNotes.reverse_convertible(1000, 0.09, S, 90, R, Q, math.nan, T)
