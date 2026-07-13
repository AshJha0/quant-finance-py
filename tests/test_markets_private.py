"""Private-market analytics pins, ported from Java AssetClassRoundTest
and AssetClassEdgeTest (private-markets sections).

IRR recovers planted rates and refuses signless flows, the multiples
are exact fractions with TVPI = DPI + RVPI, KS-PME is exactly one when
the fund IS the index, and Geltner desmoothing inverts smoothing to
machine precision.
"""

import numpy as np
import pytest

from quantfinlib.markets import PrivateMarketAnalytics


def test_irr_recovers_planted_rates_and_refuses_signless_flows():
    assert PrivateMarketAnalytics.irr([-100, 110]) == pytest.approx(0.10, abs=1e-9)
    # -100 + 10/1.1 + 110/1.21 = 0 exactly: IRR 10% with an interim flow.
    assert PrivateMarketAnalytics.irr([-100, 10, 110]) == pytest.approx(0.10, abs=1e-9)
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.irr([-100, -10, -5])
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.irr([100])


def test_irr_hits_planted_doubling_and_halving_rates():
    # Invest 100 today, receive 200 in one period:
    # -100 + 200/(1+r) = 0 -> r = 1.0 (a 100% money-weighted return).
    assert PrivateMarketAnalytics.irr([-100, 200]) == pytest.approx(1.0, abs=1e-9)
    # Receive only 50: 1+r = 0.5 -> r = -0.5.
    assert PrivateMarketAnalytics.irr([-100, 50]) == pytest.approx(-0.5, abs=1e-9)
    # Break-even: r = 0 exactly.
    assert PrivateMarketAnalytics.irr([-100, 100]) == pytest.approx(0.0, abs=1e-9)


def test_multiples_are_exact_fractions():
    assert PrivateMarketAnalytics.tvpi(100, 80, 40) == pytest.approx(1.2, abs=1e-12)
    assert PrivateMarketAnalytics.dpi(100, 80, 40) == pytest.approx(0.8, abs=1e-12)
    assert PrivateMarketAnalytics.rvpi(100, 80, 40) == pytest.approx(0.4, abs=1e-12)
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.tvpi(0, 80, 40)


def test_tvpi_is_dpi_plus_rvpi_on_arbitrary_numbers():
    # (D + NAV)/C = D/C + NAV/C: the accounting identity must hold on
    # numbers with no nice factors.
    c, d, nav = 137, 61, 45
    assert PrivateMarketAnalytics.tvpi(c, d, nav) == pytest.approx(
        PrivateMarketAnalytics.dpi(c, d, nav)
        + PrivateMarketAnalytics.rvpi(c, d, nav), abs=1e-15)


def test_ks_pme_is_exactly_one_when_the_fund_is_the_index():
    # Invest 100 at t0, let it ride the index to 121: NAV 121,
    # FV(contribution) = 100 * 121/100 = 121 -> PME = 1 exactly.
    pme = PrivateMarketAnalytics.ks_pme([100, 0, 0], [0, 0, 0], 121,
                                        [100, 110, 121])
    assert pme == pytest.approx(1.0, abs=1e-12)
    # Beating the index: same flows, higher NAV.
    assert PrivateMarketAnalytics.ks_pme([100, 0, 0], [0, 0, 0], 150,
                                         [100, 110, 121]) > 1


def test_ks_pme_with_distributions_matches_the_hand_fraction_and_flags_laggards():
    # Flat index: every growth factor is 1, so
    # PME = (30 + 100) / (50 + 50) = 1.3 exactly.
    assert PrivateMarketAnalytics.ks_pme([50, 50, 0], [0, 0, 30], 100,
                                         [100, 100, 100]) == pytest.approx(
        1.3, abs=1e-15)
    # Index up 10% after the contribution, fund NAV only 80:
    # PME = 80 / (100 * 110/100) = 8/11 < 1 — the fund lagged.
    assert PrivateMarketAnalytics.ks_pme([100, 0], [0, 0], 80,
                                         [100, 110]) == pytest.approx(
        80.0 / 110.0, abs=1e-15)
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.ks_pme([0, 0], [0, 0], 100, [100, 110])


def test_geltner_desmoothing_inverts_smoothing_exactly():
    truth = [0.02, -0.01, 0.03, 0.015, -0.02, 0.01]
    phi = 0.4
    observed = [0.0] * len(truth)
    observed[0] = truth[0]
    for t in range(1, len(truth)):
        observed[t] = (1 - phi) * truth[t] + phi * observed[t - 1]
    recovered = PrivateMarketAnalytics.geltner_desmooth(observed, phi)
    np.testing.assert_allclose(recovered, truth, atol=1e-12,
                               err_msg="smoothing then desmoothing must round-trip")
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.geltner_desmooth(observed, 1.0)


def test_geltner_with_zero_phi_is_the_identity_and_negative_phi_raises():
    # phi = 0: r_true = (r_obs - 0) / 1 = r_obs, element for element.
    obs = [0.02, -0.01, 0.03, 0.005]
    np.testing.assert_array_equal(
        PrivateMarketAnalytics.geltner_desmooth(obs, 0.0), obs)
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.geltner_desmooth(obs, -0.1)
    with pytest.raises(ValueError):
        PrivateMarketAnalytics.geltner_desmooth([0.02], 0.4)
