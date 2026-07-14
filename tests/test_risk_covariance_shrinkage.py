"""Pins for quantfinlib.risk.covariance_shrinkage (Ledoit-Wolf, /T convention).

Hand-derived pins in the Java house style; the /T (population) sample
covariance convention is load-bearing for every value below.
"""

import numpy as np
import pytest

from quantfinlib.risk import covariance_shrinkage as cs


def test_pure_signal_data_gets_zero_shrinkage():
    # returns [[1,-1],[-1,1]]: means 0, S = [[1,-1],[-1,1]] (/T = /2 of
    # {2, -2}). Every single-observation outer product x_t x_t' equals S
    # exactly -> b2 = 0 -> delta = 0: the sample matrix speaks for itself.
    r = [[1.0, -1.0], [-1.0, 1.0]]
    res = cs.ledoit_wolf(r)
    assert res.intensity == pytest.approx(0.0, abs=1e-15)
    assert res.target == pytest.approx(1.0, abs=1e-15)  # mu = avg variance = 1
    assert np.allclose(res.matrix, [[1.0, -1.0], [-1.0, 1.0]], atol=1e-15)


def test_noise_dominated_data_clamps_to_full_shrinkage():
    # returns [[1,0],[0,1],[1,1]] (T=3, N=2): means [2/3, 2/3];
    # S = [[2/9, -1/9], [-1/9, 2/9]] (population /3); mu = 2/9.
    # d2 = ||S - mu I||_F^2 / n = (2*(1/81))/2 = 1/81 = 3/243.
    # b2 = sum_t ||x_t x_t' - S||_F^2 / (T^2 n) = (7+7+10)/81 / 18 = 4/243
    #    > d2 -> clamped -> delta = 1 -> Sigma* = mu*I exactly.
    r = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    res = cs.ledoit_wolf(r)
    assert res.intensity == pytest.approx(1.0, abs=1e-12)
    assert res.target == pytest.approx(2.0 / 9.0, abs=1e-15)
    assert np.allclose(res.matrix, (2.0 / 9.0) * np.eye(2), atol=1e-15)


def test_sample_covariance_is_population_convention():
    # N=1: S = ss/T (not /(T-1)). returns [1, 3]: mean 2, ss = 2 -> S = 1.
    res = cs.ledoit_wolf([[1.0], [3.0]])
    # N=1 target mu equals the variance, so shrinkage is invisible in the
    # matrix — pinning the matrix pins the /T convention.
    assert res.matrix[0][0] == pytest.approx(1.0, abs=1e-15)
    assert res.target == pytest.approx(1.0, abs=1e-15)


def test_long_history_earns_low_intensity():
    # T >> N: the sample matrix is trusted; delta must be small and the
    # shrunk matrix close to the sample.
    rng = np.random.default_rng(3)
    chol = np.linalg.cholesky([[1.0, 0.3], [0.3, 1.0]])
    r = rng.standard_normal((5_000, 2)) @ chol.T * 0.01
    res = cs.ledoit_wolf(r)
    assert res.intensity < 0.05, f"delta -> 0 for huge T, got {res.intensity}"
    x = r - r.mean(axis=0)
    s = x.T @ x / r.shape[0]
    assert np.allclose(res.matrix, s, rtol=0.06)


def test_shrunk_matrix_lifts_the_small_eigenvalue():
    # Two nearly-identical assets over few observations: the sample matrix
    # is near-singular; any delta > 0 lifts the floor eigenvalue toward mu.
    rng = np.random.default_rng(9)
    base = rng.standard_normal(6) * 0.01
    r = np.column_stack([base, base + 1e-6 * rng.standard_normal(6)])
    res = cs.ledoit_wolf(r)
    x = r - r.mean(axis=0)
    s = x.T @ x / r.shape[0]
    assert res.intensity > 0
    assert np.linalg.eigvalsh(res.matrix)[0] > np.linalg.eigvalsh(s)[0], \
        "a convex combination with mu*I lifts every eigenvalue toward mu"


def test_shrink_convenience_and_gates():
    r = [[1.0, -1.0], [-1.0, 1.0]]
    assert np.allclose(cs.shrink(r), cs.ledoit_wolf(r).matrix)
    with pytest.raises(ValueError):
        cs.ledoit_wolf([[1.0, 2.0]])                 # T < 2
    with pytest.raises(ValueError):
        cs.ledoit_wolf([[1.0, 2.0], [1.0, np.nan]])  # non-finite
    with pytest.raises(ValueError):
        cs.ledoit_wolf([[1.0, 2.0], [1.0]])          # ragged
