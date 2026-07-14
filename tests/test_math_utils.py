"""Pins for quantfinlib.util.math_utils, ported from the Java suite.

Sources on the Java side:
    * MathUtilsPairSortTest.java — the dual-array sort (agreement with a
      reference sort on random data, adversarial shapes, length gate).
    * MarketRiskTest.java — t_cdf closed-form pins (Cauchy df=1,
      algebraic df=2, normal limit) and the cholesky indefinite raise.
Everything else follows the house rule: every value hand-pinned with a
derivation comment, same tolerances as the Java asserts.
"""

import math

import numpy as np
import pytest

from quantfinlib.util import math_utils as mu


# ---------------------------------------------------------------- moments

def test_mean_full_array():
    # (1 + 2 + 3 + 4) / 4 = 2.5
    assert mu.mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5, abs=1e-15)


def test_mean_window():
    # Java overload mean(v, from, to): (2 + 3 + 4) / 3 = 3 over [1, 4).
    assert mu.mean([1.0, 2.0, 3.0, 4.0, 5.0], 1, 4) == pytest.approx(3.0, abs=1e-15)


def test_mean_empty_is_nan():
    # Java 0.0 / 0 == NaN — kept rather than Python's ZeroDivisionError.
    assert math.isnan(mu.mean([]))


def test_variance_sample_denominator():
    # [1,2,3,4]: mean 2.5, squared devs 2.25+0.25+0.25+2.25 = 5, n-1 = 3.
    assert mu.variance([1.0, 2.0, 3.0, 4.0]) == pytest.approx(5.0 / 3.0, abs=1e-15)


def test_variance_degenerate_is_zero():
    assert mu.variance([7.0]) == 0.0
    assert mu.variance([]) == 0.0


def test_std_dev():
    # sqrt of the sample variance above.
    assert mu.std_dev([1.0, 2.0, 3.0, 4.0]) == pytest.approx(math.sqrt(5.0 / 3.0), abs=1e-15)


def test_std_dev_p_window():
    # [9,1,3] over [1,3): mean(1,3) = 2, devs -1,+1, population n=2 -> sqrt(2/2) = 1.
    assert mu.std_dev_p([9.0, 1.0, 3.0], 1, 3) == pytest.approx(1.0, abs=1e-15)
    assert mu.std_dev_p([9.0, 1.0, 3.0], 1, 1) == 0.0  # empty window


def test_std_dev_sample_window():
    # Full window of [1,2,3,4] must agree with std_dev: sqrt(5/3).
    assert mu.std_dev_sample([1.0, 2.0, 3.0, 4.0], 0, 4) == pytest.approx(
        math.sqrt(5.0 / 3.0), abs=1e-15)
    assert mu.std_dev_sample([1.0, 2.0, 3.0, 4.0], 2, 3) == 0.0  # n < 2


def test_skewness_pins():
    # Symmetric [1,2,3]: m3 = (-1)^3 + 0 + 1^3 = 0 -> skew 0.
    assert mu.skewness([1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-15)
    # [0,0,3]: mean 1, devs (-1,-1,2); m2 = 6/3 = 2, m3 = 6/3 = 2;
    # skew = 2 / 2^1.5 = 1/sqrt(2).
    assert mu.skewness([0.0, 0.0, 3.0]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert mu.skewness([5.0, 5.0, 5.0]) == 0.0  # zero-variance guard


def test_kurtosis_pins():
    # [-1,1]: m2 = 1, m4 = 1 -> 1.0 (two-point distribution, minimum kurtosis).
    assert mu.kurtosis([-1.0, 1.0]) == pytest.approx(1.0, abs=1e-15)
    # [1,2,3]: m2 = 2/3, m4 = 2/3 -> (2/3)/(4/9) = 1.5.
    assert mu.kurtosis([1.0, 2.0, 3.0]) == pytest.approx(1.5, abs=1e-12)
    assert mu.kurtosis([5.0, 5.0]) == 0.0  # zero-variance guard


# ---------------------------------------------------------- percentiles

def test_percentile_exact_index():
    # sorted [1,2,3], p=0.5 -> idx = 1.0 exactly -> 2.
    assert mu.percentile([3.0, 1.0, 2.0], 0.5) == pytest.approx(2.0, abs=1e-15)


def test_percentile_interpolates():
    # sorted [1,2,3], p=0.25 -> idx = 0.5 -> 1*(0.5) + 2*(0.5) = 1.5.
    assert mu.percentile([3.0, 1.0, 2.0], 0.25) == pytest.approx(1.5, abs=1e-15)
    # Endpoints: p=0 -> min, p=1 -> max.
    assert mu.percentile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert mu.percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_percentile_does_not_mutate_input():
    v = np.array([3.0, 1.0, 2.0])
    mu.percentile(v, 0.5)
    assert v.tolist() == [3.0, 1.0, 2.0]


def test_percentile_sorted_pin_and_empty():
    # [10,20,30,40], p=0.5 -> idx = 1.5 -> 20*0.5 + 30*0.5 = 25.
    assert mu.percentile_sorted([10.0, 20.0, 30.0, 40.0], 0.5) == pytest.approx(25.0, abs=1e-15)
    assert math.isnan(mu.percentile_sorted([], 0.5))


# ------------------------------------------------------- linear algebra

def test_dot_pin_and_length_gate():
    # 1*4 + 2*5 + 3*6 = 32.
    assert mu.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0, abs=1e-15)
    with pytest.raises(ValueError):
        mu.dot([1.0, 2.0], [1.0, 2.0, 3.0])


def test_mat_vec_pin_and_gate():
    # [[1,2],[3,4]] @ [5,6] = [5+12, 15+24] = [17, 39].
    out = mu.mat_vec([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    assert out.tolist() == pytest.approx([17.0, 39.0], abs=1e-15)
    with pytest.raises(ValueError):
        mu.mat_vec([[1.0, 2.0, 3.0]], [1.0, 2.0])


def test_quadratic_form_pin():
    # w=[1,2], M=diag(2,3): w'Mw = 2*1 + 3*4 = 14.
    assert mu.quadratic_form([1.0, 2.0], [[2.0, 0.0], [0.0, 3.0]]) == pytest.approx(
        14.0, abs=1e-15)
    # Non-symmetric M=[[1,2],[3,4]], w=[1,1]: Mw = [3,7], w'Mw = 10.
    assert mu.quadratic_form([1.0, 1.0], [[1.0, 2.0], [3.0, 4.0]]) == pytest.approx(
        10.0, abs=1e-15)


def test_covariance_pin_and_gates():
    # a=[1,2,3] (mean 2), b=[2,4,6] (mean 4): sum (-1)(-2)+0+(1)(2) = 4; /(n-1)=2 -> 2.
    assert mu.covariance([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(2.0, abs=1e-15)
    with pytest.raises(ValueError):
        mu.covariance([1.0, 2.0], [1.0, 2.0, 3.0])  # length mismatch
    with pytest.raises(ValueError):
        mu.covariance([1.0], [1.0])  # n < 2


def test_correlation_pins():
    # Perfectly linear b = 2a: cov 2, sa 1, sb 2 -> exactly +1.
    assert mu.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0, abs=1e-15)
    # Perfectly anti-linear: cov = -1, sa = sb = 1 -> exactly -1.
    assert mu.correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0, abs=1e-15)
    # Zero-variance guard yields 0 (no information), not a fabricated sign.
    assert mu.correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0


def test_cholesky_pins():
    # [[4,2],[2,5]]: l00 = 2, l10 = 2/2 = 1, l11 = sqrt(5 - 1) = 2.
    l = mu.cholesky([[4.0, 2.0], [2.0, 5.0]])
    assert l[0, 0] == pytest.approx(2.0, abs=1e-15)
    assert l[1, 0] == pytest.approx(1.0, abs=1e-15)
    assert l[1, 1] == pytest.approx(2.0, abs=1e-15)
    assert l[0, 1] == 0.0
    # Identity round-trips to identity.
    assert mu.cholesky(np.eye(3)).tolist() == np.eye(3).tolist()


def test_cholesky_borderline_psd_gets_jitter():
    # Rank-1 [[1,1],[1,1]]: pivot 1 is exactly 0 -> clamped to 1e-12,
    # l11 = sqrt(1e-12) = 1e-6 instead of a hard failure.
    l = mu.cholesky([[1.0, 1.0], [1.0, 1.0]])
    assert l[1, 1] == pytest.approx(1e-6, rel=1e-12)


def test_cholesky_indefinite_fails_loudly():
    # Ported from MarketRiskTest: an indefinite "covariance" (typo'd
    # correlation 1.3) fails loudly instead of silently simulating a
    # clamped dependence structure. Pivot 1 = 1e-4 - 1.69e-4 = -6.9e-5,
    # far below the -1e-8 * maxDiag threshold.
    with pytest.raises(ValueError):
        mu.cholesky([[1e-4, 1.3e-4], [1.3e-4, 1e-4]])


def test_solve_linear_pin_and_gates():
    # 2x + y = 3, x + 3y = 5 -> x = 0.8, y = 1.4 (Cramer: det = 5).
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([3.0, 5.0])
    x = mu.solve_linear(a, b)
    assert x.tolist() == pytest.approx([0.8, 1.4], abs=1e-12)
    # Inputs are not modified.
    assert a.tolist() == [[2.0, 1.0], [1.0, 3.0]]
    assert b.tolist() == [3.0, 5.0]
    with pytest.raises(ValueError):
        mu.solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])  # singular


def test_inverse_pin_and_gate():
    # [[2,1],[1,3]]^-1 = (1/5) [[3,-1],[-1,2]] = [[0.6,-0.2],[-0.2,0.4]].
    inv = mu.inverse([[2.0, 1.0], [1.0, 3.0]])
    assert inv.flatten().tolist() == pytest.approx([0.6, -0.2, -0.2, 0.4], abs=1e-12)
    with pytest.raises(ValueError):
        mu.inverse([[1.0, 2.0], [2.0, 4.0]])  # singular


# ------------------------------------------------------------ pair_sort
# Port of MathUtilsPairSortTest. The Java version sorts in place with a
# median-of-three quicksort; the Python port returns sorted copies, so
# the asserts check the returned pair instead of the mutated inputs.

def test_pair_sort_agrees_with_reference_sort_on_random_data():
    rng = np.random.default_rng(11)
    for trial in range(50):
        n = 1 + int(rng.integers(500))  # 1..500, mirroring nextInt(500)
        keys = rng.integers(50, size=n).astype(float) - 25  # plenty of ties
        values = np.arange(n)
        sorted_keys, perm = mu.pair_sort(keys, values)
        assert sorted_keys.tolist() == sorted(keys.tolist()), f"trial {trial}"
        # The permutation must carry each original key with its index.
        for i in range(n):
            assert sorted_keys[i] == keys[perm[i]], f"trial {trial} slot {i}"


def test_pair_sort_adversarial_shapes_and_small_arrays():
    # Already sorted, reversed, constant, singleton, empty.
    shapes = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [7] * 16,
        [3.0],
        [],
    ]
    for shape in shapes:
        keys = np.asarray(shape, dtype=float)
        values = np.arange(len(shape))
        sorted_keys, _ = mu.pair_sort(keys, values)
        assert sorted_keys.tolist() == sorted(shape)
    with pytest.raises(ValueError):
        mu.pair_sort(np.zeros(2), np.zeros(3, dtype=int))


# --------------------------------------------------- normal distribution

def test_norm_pdf_pin():
    # phi(0) = 1/sqrt(2 pi) = 0.3989422804014327.
    assert mu.norm_pdf(0.0) == pytest.approx(0.3989422804014327, abs=1e-15)


def test_norm_cdf_pins():
    # A&S 26.2.17 claims |error| < 7.5e-8 against the true CDF.
    assert mu.norm_cdf(0.0) == pytest.approx(0.5, abs=1e-7)
    # Phi(1.96) = 0.9750021048517795 (the classic 97.5% point).
    assert mu.norm_cdf(1.96) == pytest.approx(0.9750021048517795, abs=1e-7)
    # Negative side is the exact complement by construction.
    assert mu.norm_cdf(-1.96) == 1 - mu.norm_cdf(1.96)


def test_norm_inv_pins():
    # Central branch at exactly 0.5: q = 0 makes the numerator 0 exactly.
    assert mu.norm_inv(0.5) == 0.0
    # Acklam |error| < 1.15e-9; true quantiles from the standard tables:
    # Phi^-1(0.975) = 1.9599639845400545, Phi^-1(0.01) = -2.3263478740408408,
    # Phi^-1(0.001) = -3.0902323061678132 (exercises the p < 0.02425 tail branch).
    assert mu.norm_inv(0.975) == pytest.approx(1.9599639845400545, abs=1e-8)
    assert mu.norm_inv(0.01) == pytest.approx(-2.3263478740408408, abs=1e-8)
    assert mu.norm_inv(0.001) == pytest.approx(-3.0902323061678132, abs=1e-8)


def test_norm_inv_symmetry():
    # Phi^-1(p) = -Phi^-1(1-p); both sides use the same central branch.
    assert mu.norm_inv(0.3) == pytest.approx(-mu.norm_inv(0.7), abs=1e-9)


def test_norm_inv_round_trips_norm_cdf():
    # Composite accuracy is bounded by the A&S CDF error (7.5e-8 in
    # p-space), well inside 1e-6.
    for p in (0.01, 0.25, 0.5, 0.9, 0.99):
        assert mu.norm_cdf(mu.norm_inv(p)) == pytest.approx(p, abs=1e-6)


def test_norm_inv_domain_gate():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            mu.norm_inv(bad)


# ----------------------------------------------------- gamma / beta / t

def test_log_gamma_pins():
    # Gamma(1) = 1, Gamma(5) = 24, Gamma(0.5) = sqrt(pi); Lanczos claims
    # |relative error| < 2e-10, so abs=1e-9 is comfortable at these scales.
    assert mu.log_gamma(1.0) == pytest.approx(0.0, abs=1e-9)
    assert mu.log_gamma(5.0) == pytest.approx(math.log(24.0), abs=1e-9)
    assert mu.log_gamma(0.5) == pytest.approx(0.5 * math.log(math.pi), abs=1e-9)


def test_log_gamma_gate_rejects_nonpositive_and_nan():
    # The `not (x > 0)` gate is deliberately NaN-rejecting.
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            mu.log_gamma(bad)


def test_regularized_incomplete_beta_pins():
    # I_x(1,1) = x (uniform CDF).
    assert mu.regularized_incomplete_beta(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-13)
    # I_x(2,2) = x^2 (3 - 2x): at x=0.25 -> 0.0625 * 2.5 = 0.15625.
    assert mu.regularized_incomplete_beta(2.0, 2.0, 0.25) == pytest.approx(0.15625, abs=1e-12)
    # At x=0.5 (symmetric a=b) -> exactly one half.
    assert mu.regularized_incomplete_beta(2.0, 2.0, 0.5) == pytest.approx(0.5, abs=1e-12)
    # Endpoints return x itself.
    assert mu.regularized_incomplete_beta(3.0, 4.0, 0.0) == 0.0
    assert mu.regularized_incomplete_beta(3.0, 4.0, 1.0) == 1.0


def test_regularized_incomplete_beta_gates():
    for a, b, x in ((0.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, 1.5),
                    (math.nan, 1.0, 0.5), (1.0, 1.0, math.nan)):
        with pytest.raises(ValueError):
            mu.regularized_incomplete_beta(a, b, x)


def test_t_cdf_closed_form_pins():
    # Ported from MarketRiskTest: the t-CDF against closed forms —
    # df = 1 is Cauchy, df = 2 is algebraic, df -> infinity is the normal.
    assert mu.t_cdf(0.5, 1) == pytest.approx(0.5 + math.atan(0.5) / math.pi, abs=1e-12)
    assert mu.t_cdf(1, 2) == pytest.approx(0.5 * (1 + 1 / math.sqrt(3)), abs=1e-12)
    assert mu.t_cdf(0, 7) == pytest.approx(0.5, abs=1e-12)
    assert mu.t_cdf(-1.5, 1_000_000) == pytest.approx(mu.norm_cdf(-1.5), abs=1e-4)


def test_t_cdf_symmetry_and_gate():
    # t < 0 returns the half-tail directly, t > 0 its complement, with
    # identical |t|: the two sides sum to exactly 1.
    assert mu.t_cdf(-2.0, 5) == pytest.approx(1 - mu.t_cdf(2.0, 5), abs=1e-15)
    for bad in (0.0, -3.0, math.nan):
        with pytest.raises(ValueError):
            mu.t_cdf(1.0, bad)


# ------------------------------------------------------------- helpers

def test_clamp_pins():
    assert mu.clamp(5.0, 0.0, 3.0) == 3.0
    assert mu.clamp(-1.0, 0.0, 3.0) == 0.0
    assert mu.clamp(2.0, 0.0, 3.0) == 2.0


def test_clamp_propagates_nan_like_java_math_min_max():
    # Java's Math.max(lo, Math.min(hi, v)) returns NaN when v is NaN.
    # Python's builtin min/max do NOT propagate NaN the same way --
    # min(hi, nan) silently evaluates to hi because the NaN comparison
    # is always false -- so a naive max(lo, min(hi, v)) would turn a
    # NaN sentinel into hi instead of preserving "unknown".
    assert math.isnan(mu.clamp(math.nan, 0.0, 1.0))
    assert math.isnan(mu.clamp(math.nan, -1.0, 1.0))


def test_nan_array():
    a = mu.nan_array(3)
    assert a.shape == (3,)
    assert np.isnan(a).all()


def test_decay_factor_pins():
    # Non-positive dt decays nothing.
    assert mu.decay_factor(0, 100) == 1.0
    assert mu.decay_factor(-5, 100) == 1.0
    # One half-life halves: exp(-ln2) = 0.5 — the ln2 factor is exactly
    # the constant that goes missing when this is re-spelled per class.
    assert mu.decay_factor(100, 100) == pytest.approx(0.5, abs=1e-15)
    # Two half-lives quarter.
    assert mu.decay_factor(200, 100) == pytest.approx(0.25, abs=1e-15)
