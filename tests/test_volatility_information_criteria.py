"""Pins for quantfinlib.volatility.information_criteria.

Java source: InformationCriteriaTest — exact arithmetic, ranking
direction, and the n-dependent penalty. All pins transfer exactly.
"""

import math

import pytest

from quantfinlib.volatility import InformationCriteria


def test_formulas_are_exact():
    assert InformationCriteria.aic(-100, 3) == 2 * 3 - 2 * (-100.0)
    assert InformationCriteria.aic(-100, 3) == 206.0
    assert InformationCriteria.bic(-100, 3, 500) == 3 * math.log(500) - 2 * (-100.0)
    # Zero parameters: the criteria are just -2 ln L.
    assert InformationCriteria.aic(-100, 0) == 200.0
    assert InformationCriteria.bic(-100, 0, 500) == 200.0


def test_better_likelihood_wins_at_equal_complexity():
    # Same k (and n): the model with the higher log-likelihood must score
    # LOWER on both criteria — lower is better.
    assert InformationCriteria.aic(-95, 4) < InformationCriteria.aic(-100, 4)
    assert InformationCriteria.bic(-95, 4, 250) < InformationCriteria.bic(-100, 4, 250)


def test_bic_penalizes_parameters_harder_than_aic_on_large_samples():
    # One extra parameter costs 2 under AIC but ln(n) under BIC: for
    # n >= 8 (ln 8 > 2) BIC can reject an extra parameter AIC accepts.
    gain = 1.2  # log-likelihood improvement from the extra parameter
    aic_simple = InformationCriteria.aic(-100, 1)
    aic_rich = InformationCriteria.aic(-100 + gain, 2)
    bic_simple = InformationCriteria.bic(-100, 1, 1000)
    bic_rich = InformationCriteria.bic(-100 + gain, 2, 1000)
    assert aic_rich < aic_simple, "AIC accepts: 2*1.2 > 2"
    assert bic_rich > bic_simple, "BIC rejects: 2*1.2 < ln(1000)"


def test_gates_refuse_nonsense():
    with pytest.raises(ValueError):
        InformationCriteria.aic(math.nan, 3)
    with pytest.raises(ValueError):
        InformationCriteria.aic(math.inf, 3)
    with pytest.raises(ValueError):
        InformationCriteria.aic(-100, -1)
    with pytest.raises(ValueError):
        InformationCriteria.bic(-100, 3, 0)
    with pytest.raises(ValueError):
        InformationCriteria.bic(-math.inf, 3, 100)
