"""Pins for Fama-MacBeth cross-sectional regression.

Java source: AlphaResearchRoundTest.java (famaMacBethPricesThePlantedFactorAndNotTheDud).
"""

import math

import numpy as np
import pytest

from quantfinlib.alpha.fama_macbeth import FamaMacBeth


def _plant(periods=60, assets=50, seed=17):
    rng = np.random.default_rng(seed)
    x = [[[0.0, 0.0] for _ in range(assets)] for _ in range(periods)]
    r = [[0.0] * assets for _ in range(periods)]
    for t in range(periods):
        for a in range(assets):
            x[t][a][0] = rng.standard_normal()          # the priced factor
            x[t][a][1] = rng.standard_normal()           # the dud
            r[t][a] = 0.01 * x[t][a][0] + 0.005 * rng.standard_normal()
    return x, r


def test_fama_macbeth_prices_the_planted_factor_and_not_the_dud():
    x, r = _plant()
    # A NaN asset (out of the cross-section that period) must be
    # skipped, not poison the regression.
    r[7][3] = math.nan

    fm = FamaMacBeth.fit(x, r)
    assert fm.periods_used == 60
    assert fm.premia[0] == pytest.approx(0.01, abs=0.001)
    assert fm.t_stats[0] > 5
    assert fm.premia[1] == pytest.approx(0.0, abs=0.001)
    assert abs(fm.t_stats[1]) < 2.5
    assert abs(fm.intercept_t_stat) < 2.5

    with pytest.raises(ValueError):
        FamaMacBeth.fit([[[0.0, 0.0]] * 10 for _ in range(5)],
                        [[0.0] * 10 for _ in range(5)])

    # A SINGULAR cross-section (a factor constant across assets that
    # period) is skipped and counted, like a thin one -- one bad
    # period must not abort 59 good ones.
    for a in range(50):
        x[30][a][1] = 1.0                      # collinear with the intercept
    skipped = FamaMacBeth.fit(x, r)
    assert skipped.periods_used == 59
    assert skipped.premia[0] == pytest.approx(0.01, abs=0.001)

    # Infinity is a data error, not a missing name: fail fast.
    r_bad = [row[:] for row in r]
    r_bad[5][5] = math.inf
    with pytest.raises(ValueError):
        FamaMacBeth.fit(x, r_bad)


def test_fama_macbeth_usable_cross_sections_gate():
    # Enough periods supplied, but too many are too thin to price.
    rng = np.random.default_rng(17)
    periods, assets = 20, 50
    thin_x = [[[rng.standard_normal(), rng.standard_normal()]
              for _ in range(assets)] for _ in range(periods)]
    thin_r = [[math.nan if t < 10 else 0.01 * thin_x[t][a][0]
              for a in range(assets)] for t in range(periods)]
    with pytest.raises(ValueError):
        FamaMacBeth.fit(thin_x, thin_r)
