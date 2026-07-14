"""Pins for BinomialTree and SabrModel, ported from AmericanAndSabrTest.java
plus the SABR level pins from FormulaPinsTest.java.
"""

import pytest

from quantfinlib.pricing import (BinomialTree, BlackScholes, ExerciseStyle,
                                 OptionType, SabrModel)

# ---- CRR binomial tree --------------------------------------------------


def test_european_tree_converges_to_black_scholes():
    bs = BlackScholes.price(OptionType.CALL, 100, 100, 0.05, 0, 0.2, 1)
    tree = BinomialTree.price(OptionType.CALL, ExerciseStyle.EUROPEAN,
                              100, 100, 0.05, 0, 0.2, 1, 500)
    assert tree == pytest.approx(bs, abs=0.05)

    bs_put = BlackScholes.price(OptionType.PUT, 100, 95, 0.03, 0.01, 0.25, 0.5)
    tree_put = BinomialTree.price(OptionType.PUT, ExerciseStyle.EUROPEAN,
                                  100, 95, 0.03, 0.01, 0.25, 0.5, 500)
    assert tree_put == pytest.approx(bs_put, abs=0.05)


def test_american_put_carries_early_exercise_premium():
    # Deep ITM American put on a high-rate asset: early exercise is valuable.
    premium = BinomialTree.early_exercise_premium(
        OptionType.PUT, 80, 100, 0.08, 0, 0.2, 1, 400)
    assert premium > 0.1
    # American never below intrinsic.
    american = BinomialTree.price(OptionType.PUT, ExerciseStyle.AMERICAN,
                                  80, 100, 0.08, 0, 0.2, 1, 400)
    assert american >= 20 - 1e-9


def test_american_call_without_carry_equals_european():
    # No dividends: never optimal to exercise a call early.
    premium = BinomialTree.early_exercise_premium(
        OptionType.CALL, 100, 95, 0.05, 0, 0.25, 1, 400)
    assert premium == pytest.approx(0, abs=1e-9)


def test_tree_delta_matches_black_scholes_for_european():
    bs_delta = BlackScholes.delta(OptionType.CALL, 100, 100, 0.05, 0, 0.2, 1)
    tree_delta = BinomialTree.delta(OptionType.CALL, ExerciseStyle.EUROPEAN,
                                    100, 100, 0.05, 0, 0.2, 1, 500)
    assert tree_delta == pytest.approx(bs_delta, abs=0.01)


def test_tree_gates():
    with pytest.raises(ValueError):
        BinomialTree.price(OptionType.CALL, ExerciseStyle.EUROPEAN,
                           100, 100, 0.05, 0, 0.2, 1, 0)   # steps < 1


# ---- SABR ---------------------------------------------------------------

F, T, BETA = 100.0, 1.0, 1.0
ALPHA, RHO, NU = 0.20, -0.30, 0.60


def test_hagan_atm_formula_consistent_with_smile():
    atm = SabrModel.implied_vol(F, F, T, ALPHA, BETA, RHO, NU)
    near_atm = SabrModel.implied_vol(F, F + 1e-6, T, ALPHA, BETA, RHO, NU)
    assert near_atm == pytest.approx(atm, abs=1e-6)
    # Negative rho: downside strikes carry higher vol.
    assert (SabrModel.implied_vol(F, 80, T, ALPHA, BETA, RHO, NU)
            > SabrModel.implied_vol(F, 120, T, ALPHA, BETA, RHO, NU))


def test_sabr_atm_levels_pinned_for_beta_one_and_beta_half():
    # From FormulaPinsTest.java.
    # beta = 1, ATM Hagan: alpha*(1 + T*(rho*nu*alpha/4 + (2-3rho^2)nu^2/24))
    # = 0.2*(1 + (-0.3*0.6*0.2/4 + (2 - 0.27)*0.36/24)) = 0.2*1.016950.
    assert SabrModel.implied_vol(100, 100, 1.0, 0.20, 1.0, -0.30, 0.60) == pytest.approx(
        0.2033900, abs=1e-7)
    # beta = 0.5: alpha/f^0.5 * (1 + T*((1-b)^2 a^2/(24 f) + rho b nu a/(4 f^0.5)
    # + (2-3rho^2)nu^2/24)) = 0.2*(1 + 4.1667e-4 - 0.0045 + 0.02595).
    assert SabrModel.implied_vol(100, 100, 1.0, 2.0, 0.5, -0.30, 0.60) == pytest.approx(
        0.2043733, abs=1e-7)


def test_calibration_recovers_generated_smile():
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    vols = [SabrModel.implied_vol(F, k, T, ALPHA, BETA, RHO, NU) for k in strikes]
    fit = SabrModel.calibrate(F, T, BETA, strikes, vols)

    assert fit.rmse < 5e-4
    assert fit.alpha == pytest.approx(ALPHA, abs=0.02)
    assert fit.rho == pytest.approx(RHO, abs=0.10)
    assert fit.nu == pytest.approx(NU, abs=0.15)
    # Fitted smile reproduces an unquoted strike.
    interp = SabrModel.implied_vol(F, 85, T, fit.alpha, fit.beta, fit.rho, fit.nu)
    truth = SabrModel.implied_vol(F, 85, T, ALPHA, BETA, RHO, NU)
    assert interp == pytest.approx(truth, abs=0.003)


def test_sabr_gates():
    with pytest.raises(ValueError):
        SabrModel.implied_vol(0, 100, 1, 0.2, 1, 0, 0.5)
    with pytest.raises(ValueError):
        SabrModel.calibrate(F, T, BETA, [90.0, 100.0], [0.2, 0.2])
