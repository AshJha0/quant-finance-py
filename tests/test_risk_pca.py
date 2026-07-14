"""Pins for quantfinlib.risk.pca, ported from Java MarketRiskTest
(pcaRecoversAnalyticEigenstructure / reviewRegressionsStayFixed)."""

import numpy as np
import pytest

from quantfinlib.risk.pca import Pca


def test_recovers_analytic_eigenstructure():
    # [[2,1],[1,2]]: eigenvalues 3 and 1, eigenvectors (1,1)/sqrt2, (1,-1)/sqrt2.
    pca = Pca([[2.0, 1.0], [1.0, 2.0]])
    assert pca.eigenvalue(0) == pytest.approx(3, abs=1e-9)
    assert pca.eigenvalue(1) == pytest.approx(1, abs=1e-9)
    assert pca.explained_variance(1) == pytest.approx(0.75, abs=1e-9), \
        "3 of 4 total variance"
    ratio = pca.loading(0, 0) / pca.loading(0, 1)
    assert ratio == pytest.approx(1, abs=1e-6), \
        "first component loads both factors equally"
    assert pca.size() == 2


def test_rank_one_level_curve():
    # A one-factor 'curve': all tenors driven by one level shock.
    level = np.full((3, 3), 1e-4)          # rank-1: pure level
    curve = Pca(level)
    assert curve.explained_variance(1) == pytest.approx(1.0, abs=1e-9), \
        "one real factor -> one component explains everything"


def test_asymmetric_matrix_rejected():
    with pytest.raises(ValueError):
        Pca([[1.0, 0.5], [0.2, 1.0]])


def test_currency_unit_scale():
    # Currency-unit covariance (~1e8 entries): convergence thresholds are
    # relative to the matrix's scale, so this decomposes exactly like its
    # rate-fraction twin.
    big = Pca([[2e8, 1e8], [1e8, 2e8]])
    assert big.eigenvalue(0) == pytest.approx(3e8, abs=1e-2)
    assert big.eigenvalue(1) == pytest.approx(1e8, abs=1e-2)


def test_extreme_magnitudes_do_not_overflow():
    # Java review regression: norm accumulators must not overflow at 1e155.
    huge = Pca([[2e155, 1e155], [1e155, 2e155]])
    assert huge.eigenvalue(0) == pytest.approx(3e155, abs=1e144)
    assert huge.eigenvalue(1) == pytest.approx(1e155, abs=1e144)


def test_zero_matrix_is_already_diagonal():
    z = Pca(np.zeros((2, 2)))
    assert z.eigenvalue(0) == 0.0
    assert z.explained_variance(1) == 0.0   # no variance to explain


def test_psd_clip_and_gates():
    # A borderline matrix with a tiny negative eigenvalue clips at zero.
    eps = 1e-14
    p = Pca([[1.0, 1.0], [1.0, 1.0 - eps]])
    assert p.eigenvalue(1) >= 0.0, "numerical noise, not an imaginary risk factor"
    with pytest.raises(ValueError):
        Pca(np.zeros((0, 0)))                       # empty
    with pytest.raises(ValueError):
        Pca([[1.0, np.nan], [np.nan, 1.0]])         # non-finite
    with pytest.raises(ValueError):
        Pca([[1.0, 2.0]])                           # non-square
    with pytest.raises(ValueError):
        Pca([[2.0, 1.0], [1.0, 2.0]]).explained_variance(0)  # k range
