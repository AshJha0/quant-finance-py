"""SwapPricer pins, ported from Java SwapPricerTest plus the swap pins
in AssetClassRoundTest.

PV is annuity * (par - K), so payer/receiver symmetry around par is
EXACT, the annuity on a flat cc curve is a hand-summable geometric
series, and PV falls one-for-one (times the annuity) in the fixed rate.
"""

import math

import pytest

from quantfinlib.rates import RatesOptions, SwapPricer, YieldCurve


def flat5() -> YieldCurve:
    t = [1, 2, 3, 4, 5, 7, 10]
    return YieldCurve.of_zero_rates(t, [0.05] * len(t))


def test_payer_pv_is_exactly_antisymmetric_around_par():
    # payerPv(K) = annuity * (par - K): equal strikes above and below
    # par give PVs of equal size and opposite sign, and the receiver
    # (the documented negation) is the mirror trade.
    c = flat5()
    par = SwapPricer.par_rate(c, 5)
    d = 0.01
    below = SwapPricer.payer_pv(c, 5, par - d)
    above = SwapPricer.payer_pv(c, 5, par + d)
    assert above == pytest.approx(-below, abs=1e-15), "linear in K: exact antisymmetry"
    assert below == pytest.approx(SwapPricer.annuity(c, 5) * d, abs=1e-15)
    assert below > 0, "paying below par is an asset to the payer"
    assert above < 0, "paying above par is a liability to the payer"


def test_annuity_on_a_flat_curve_is_the_hand_geometric_sum():
    # Flat 5% cc: DF(i) = e^{-0.05 i}, so the 5y annual annuity is
    # e^{-0.05} + e^{-0.10} + e^{-0.15} + e^{-0.20} + e^{-0.25}.
    c = flat5()
    expected = sum(math.exp(-0.05 * i) for i in range(1, 6))
    assert SwapPricer.annuity(c, 5) == pytest.approx(expected, abs=1e-12)
    assert SwapPricer.annuity(c, 1) > 0
    # Longer tenor adds strictly positive discount factors.
    assert SwapPricer.annuity(c, 10) > SwapPricer.annuity(c, 5)


def test_payer_pv_falls_strictly_as_the_fixed_rate_rises():
    c = flat5()
    prev = math.inf
    for k in [0.01, 0.03, 0.05, 0.07, 0.09]:
        pv = SwapPricer.payer_pv(c, 5, k)
        assert pv < prev, f"paying a higher fixed rate must cost more, K={k}"
        prev = pv


def test_par_swap_prices_to_zero_and_matches_rates_options():
    c = flat5()
    par = SwapPricer.par_rate(c, 5)
    assert SwapPricer.payer_pv(c, 5, par) == pytest.approx(0.0, abs=1e-12), \
        "a swap struck at par is worth exactly zero"
    # Same object RatesOptions computes as the 0-into-5y forward rate.
    assert par == pytest.approx(RatesOptions.forward_swap_rate(c, 0, 5), abs=1e-12)
    # Flat cc curve: par = the simple annual forward.
    assert par == pytest.approx(math.exp(0.05) - 1, abs=1e-12)
    # Below-par fixed rate favors the payer.
    assert SwapPricer.payer_pv(c, 5, par - 0.01) > 0


def test_dv01_is_annuity_times_the_par_rate_sensitivity():
    # Subtlety worth pinning: a 1bp bump to the CC ZERO curve moves the
    # SIMPLE par rate by e^z bp (d(e^z - 1)/dz), so the payer DV01 on a
    # flat 5% curve is annuity * e^0.05 * 1bp — the naive
    # "annuity * 1bp" is the sensitivity to the PAR rate, a different
    # derivative.
    c = flat5()
    par = SwapPricer.par_rate(c, 5)
    dv01 = SwapPricer.dv01(c, 5, par)
    assert dv01 > 0, "rates up helps the fixed payer"
    expected = SwapPricer.annuity(c, 5) * 1e-4 * math.exp(0.05)
    assert dv01 == pytest.approx(expected, abs=SwapPricer.annuity(c, 5) * 1e-4 * 0.01), \
        "DV01 = annuity * e^z * 1bp on a flat cc curve (within 1%)"


def test_swap_gates_refuse_nonsense():
    c = flat5()
    with pytest.raises(ValueError):
        SwapPricer.annuity(c, 0)
    with pytest.raises(ValueError):
        SwapPricer.par_rate(c, -1)
    with pytest.raises(ValueError):
        SwapPricer.payer_pv(c, 5, float("nan"))
    with pytest.raises(ValueError):
        SwapPricer.payer_pv(c, 0, 0.05)
