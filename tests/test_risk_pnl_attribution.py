"""Pins for quantfinlib.risk.pnl_attribution (FRTB PLAT), ported from Java
MarketRiskTest (platPassesAFaithfulModel / reviewRegressionsStayFixed).

The 250-day P&L simulation uses numpy's default_rng instead of
java.util.Random; the zone assertions are structural (GREEN vs
not-GREEN), unaffected by the stream change. Module-qualified calls
(pa.test) keep pytest from collecting the ported Java method name
``test`` as a test function.
"""

import math

import numpy as np
import pytest

from quantfinlib.risk import pnl_attribution as pa


def _pnl_series(seed=9, days=250):
    rng = np.random.default_rng(seed)
    factor1 = rng.standard_normal(days)
    factor2 = rng.standard_normal(days)
    hpl = 100 * factor1 + 80 * factor2
    rtpl_good = hpl + 3 * rng.standard_normal(days)
    rtpl_bad = 100 * factor1                    # the model MISSES factor 2
    return hpl, rtpl_good, rtpl_bad


def test_plat_passes_a_faithful_model():
    hpl, rtpl_good, _ = _pnl_series()
    good = pa.test(hpl, rtpl_good)
    assert good.zone is pa.Zone.GREEN, \
        f"a faithful model passes: corr {good.spearman_correlation}, ks {good.ks_statistic}"


def test_plat_flags_a_missing_factor():
    hpl, _, rtpl_bad = _pnl_series()
    bad = pa.test(hpl, rtpl_bad)
    assert bad.zone is not pa.Zone.GREEN, \
        f"a missing risk factor cannot pass PLAT: corr {bad.spearman_correlation}, " \
        f"ks {bad.ks_statistic}"


def test_identical_series_score_perfectly():
    hpl, _, _ = _pnl_series()
    perfect = pa.test(hpl, hpl.copy())
    assert perfect.spearman_correlation == pytest.approx(1.0, abs=1e-9)
    assert perfect.ks_statistic == pytest.approx(0.0, abs=1e-9)


def test_ks_statistic_hand_pin():
    # a=[1,2,3,4], b=[1,2,3,5]: CDFs agree until 4 is consumed only by a:
    # gap |4/4 - 3/4| = 0.25.
    assert pa.ks_statistic([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]) == \
        pytest.approx(0.25, abs=1e-15)
    # Disjoint samples: gap reaches 1 after the smaller sample drains.
    assert pa.ks_statistic([1.0, 2.0], [3.0, 4.0]) == pytest.approx(1.0, abs=1e-15)
    # Ties must not register a transient gap: identical series score 0.
    assert pa.ks_statistic([5.0, 5.0, 7.0], [5.0, 5.0, 7.0]) == 0.0


def test_nan_pnl_day_throws_instead_of_hanging():
    # One NaN P&L day used to HANG ksStatistic in an infinite loop
    # (NaN == NaN is false, so neither pointer advanced) — raises now.
    good = np.arange(20.0)
    bad = np.arange(20.0)
    bad[7] = math.nan
    with pytest.raises(ValueError):
        pa.ks_statistic(good, bad)
    with pytest.raises(ValueError):
        pa.test(bad, good)


def test_gates():
    with pytest.raises(ValueError):
        pa.test(np.arange(19.0), np.arange(19.0))   # < 20 days
    with pytest.raises(ValueError):
        pa.test(np.arange(20.0), np.arange(21.0))   # misaligned
    with pytest.raises(ValueError):
        pa.ks_statistic([], [1.0])                  # empty sample
