"""Numerical primitives shared across the library.

Port of Java ``com.quantfinlib.util.MathUtils``. Array helpers take
array-likes and return floats / ``np.ndarray``; the scalar special
functions (``norm_inv``, ``log_gamma``, ``regularized_incomplete_beta``,
``t_cdf``) and the Cholesky pivot logic are faithful transcriptions of
the exact Java arithmetic so hand-pinned test values transfer
identically between the two ports.

House rules carried over from Java:
    * Validation gates use the NaN-rejecting ``not (x > 0)`` idiom
      (true for NaN) and raise ``ValueError``.
    * Variance-style statistics are two-pass (mean first, then squared
      deviations) — never the cancellation-prone E[x^2] - E[x]^2 form.
"""

from __future__ import annotations

import math

import numpy as np

_MIN_NORMAL = 2.2250738585072014e-308  # Java Double.MIN_NORMAL
_LN2 = math.log(2.0)


def mean(v, start: int = 0, stop: int | None = None) -> float:
    """Arithmetic mean of ``v`` over the half-open window [start, stop).

    Mirrors the Java overloads ``mean(v)`` / ``mean(v, from, to)``.
    An empty window yields NaN (Java's 0/0 IEEE semantics).
    """
    v = np.asarray(v, dtype=float)
    if stop is None:
        stop = v.shape[0]
    n = stop - start
    if n == 0:
        return math.nan  # Java: 0.0 / 0 == NaN
    return float(np.sum(v[start:stop])) / n


def variance(v) -> float:
    """Sample variance (n - 1 denominator); 0 for fewer than 2 points."""
    v = np.asarray(v, dtype=float)
    if v.shape[0] < 2:
        return 0.0
    d = v - mean(v)  # two-pass: mean first, then squared deviations
    return float(np.sum(d * d)) / (v.shape[0] - 1)


def std_dev(v) -> float:
    """Sample standard deviation."""
    return math.sqrt(variance(v))


def std_dev_p(v, start: int, stop: int) -> float:
    """Population standard deviation over [start, stop); 0 if empty."""
    v = np.asarray(v, dtype=float)
    n = stop - start
    if n < 1:
        return 0.0
    d = v[start:stop] - mean(v, start, stop)
    return math.sqrt(float(np.sum(d * d)) / n)


def std_dev_sample(v, start: int, stop: int) -> float:
    """Sample standard deviation over [start, stop); 0 for n < 2."""
    v = np.asarray(v, dtype=float)
    n = stop - start
    if n < 2:
        return 0.0
    d = v[start:stop] - mean(v, start, stop)
    return math.sqrt(float(np.sum(d * d)) / (n - 1))


def percentile(values, p: float) -> float:
    """Linear-interpolated percentile, ``p`` in [0, 1].

    Copies and sorts the input (the caller's array is not modified).
    """
    return percentile_sorted(np.sort(np.asarray(values, dtype=float)), p)


def percentile_sorted(sorted_values, p: float) -> float:
    """Percentile on an already-sorted array (no copy). NaN if empty."""
    sorted_values = np.asarray(sorted_values, dtype=float)
    n = sorted_values.shape[0]
    if n == 0:
        return math.nan
    idx = p * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def dot(a, b) -> float:
    """Dot product with an explicit length gate.

    The gate localizes the error here rather than truncating silently
    or failing three frames deep in a risk calculation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")
    return float(np.dot(a, b))


def mat_vec(m, v) -> np.ndarray:
    """Matrix-vector product ``m @ v`` with the same length gate as dot."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    if m.shape[1] != v.shape[0]:
        raise ValueError(f"length mismatch: {m.shape[1]} vs {v.shape[0]}")
    return m @ v


def quadratic_form(w, m) -> float:
    """``w' * M * w`` (quadratic form)."""
    return dot(w, mat_vec(m, w))


def covariance(a, b) -> float:
    """Sample covariance of two equally-sized series (n >= 2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")
    if a.shape[0] < 2:
        raise ValueError(f"need at least 2 observations, got {a.shape[0]}")
    da = a - mean(a)
    db = b - mean(b)
    return float(np.sum(da * db)) / (a.shape[0] - 1)


def correlation(a, b) -> float:
    """Pearson correlation; 0 (no information) if either side is constant."""
    sa = std_dev(a)
    sb = std_dev(b)
    if sa == 0 or sb == 0:
        return 0.0
    return covariance(a, b) / (sa * sb)


def cholesky(a) -> np.ndarray:
    """Cholesky decomposition: lower-triangular L with A = L @ L.T.

    Adds tiny diagonal jitter if the matrix is BORDERLINE non-PSD
    (rank-deficient factor models produce pivots a hair below zero),
    but a pivot grossly negative relative to the diagonal scale means
    the input is genuinely indefinite — a typo'd correlation > 1, or
    inconsistent pairwise estimates — and simulating a silently
    clamped, DIFFERENT dependence structure would misstate risk, so
    that fails loudly instead.

    Faithful loop port of the Java arithmetic (including the
    ``-1e-8 * max(maxDiag, MIN_NORMAL)`` indefinite threshold and the
    1e-12 pivot clamp) so pinned factors match across ports.

    Raises:
        ValueError: if a pivot falls below the indefinite threshold.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    max_diag = 0.0
    for i in range(n):
        max_diag = max(max_diag, abs(a[i, i]))
    indefinite = -1e-8 * max(max_diag, _MIN_NORMAL)
    l = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = float(a[i, j])
            for k in range(j):
                s -= l[i, k] * l[j, k]
            if i == j:
                if s < indefinite:
                    raise ValueError(
                        "matrix is not positive semi-definite (pivot "
                        f"{i} = {s}) — check for correlations beyond 1")
                if s <= 0:
                    s = 1e-12
                l[i, i] = math.sqrt(s)
            else:
                l[i, j] = s / l[j, j]
    return l


# Acklam's coefficients, copied digit-for-digit from the Java source.
_NORM_INV_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_NORM_INV_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01)
_NORM_INV_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_NORM_INV_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
               3.754408661907416e+00)


def norm_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, |error| < 1.15e-9).

    Raises:
        ValueError: if p is outside the open interval (0, 1).
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1): {p}")
    a, b, c, d = _NORM_INV_A, _NORM_INV_B, _NORM_INV_C, _NORM_INV_D
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return (-(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def skewness(v) -> float:
    """Population skewness: m3 / m2^1.5; 0 for a constant series."""
    v = np.asarray(v, dtype=float)
    d = v - mean(v)
    m2 = float(np.sum(d * d)) / v.shape[0]
    m3 = float(np.sum(d * d * d)) / v.shape[0]
    return 0.0 if m2 == 0 else m3 / m2 ** 1.5


def kurtosis(v) -> float:
    """Population kurtosis: m4 / m2^2 (3 for a normal, not excess)."""
    v = np.asarray(v, dtype=float)
    d = v - mean(v)
    m2 = float(np.sum(d * d)) / v.shape[0]
    m4 = float(np.sum(d * d * d * d)) / v.shape[0]
    return 0.0 if m2 == 0 else m4 / (m2 * m2)


def pair_sort(keys, values) -> tuple[np.ndarray, np.ndarray]:
    """Sorts ``keys`` ascending while permuting ``values`` identically.

    Python idiom: returns sorted COPIES ``(sorted_keys, permuted_values)``
    instead of mutating the caller's arrays like the Java in-place
    quicksort. NaN keys are not supported (callers filter NaN before
    ranking/selecting), matching the Java contract.

    Raises:
        ValueError: if the two arrays differ in length.
    """
    keys = np.asarray(keys, dtype=float)
    values = np.asarray(values)
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must align")
    order = np.argsort(keys, kind="stable")
    return keys[order], values[order]


def norm_pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17, |error| < 7.5e-8)."""
    if x < 0:
        return 1 - norm_cdf(-x)
    t = 1 / (1 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))))
    return 1 - norm_pdf(x) * poly


def solve_linear(a, b) -> np.ndarray:
    """Solves ``A x = b`` by Gaussian elimination with partial pivoting.

    Inputs are not modified.

    Raises:
        ValueError: if a pivot magnitude falls below 1e-12 (singular).
    """
    m = np.array(a, dtype=float)
    rhs = np.array(b, dtype=float)
    n = rhs.shape[0]
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(m[col:, col])))
        if abs(m[pivot, col]) < 1e-12:
            raise ValueError(f"singular system at column {col}")
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
            rhs[[col, pivot]] = rhs[[pivot, col]]
        for r in range(col + 1, n):
            f = m[r, col] / m[col, col]
            rhs[r] -= f * rhs[col]
            m[r, col:] -= f * m[col, col:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - float(np.dot(m[i, i + 1:], x[i + 1:]))) / m[i, i]
    return x


def inverse(a) -> np.ndarray:
    """Matrix inverse by Gauss-Jordan elimination with partial pivoting.

    Raises:
        ValueError: if a pivot magnitude falls below 1e-12 (singular).
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    aug = np.hstack([a.copy(), np.eye(n)])
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) < 1e-12:
            raise ValueError(f"singular matrix at column {col}")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]
        for r in range(n):
            if r == col:
                continue
            aug[r] -= aug[r, col] * aug[col]
    return aug[:, n:].copy()


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamps ``v`` into [lo, hi]."""
    return max(lo, min(hi, v))


def nan_array(n: int) -> np.ndarray:
    """A length-n array filled with NaN (the 'no value yet' sentinel)."""
    return np.full(n, np.nan)


def decay_factor(dt_nanos: int, half_life_nanos: int) -> float:
    """Exponential decay factor for a half-life over an elapsed interval.

    ``exp(-dt * ln2 / half_life)``; 1.0 for non-positive ``dt``. The
    single home for the half-life -> decay conversion the streaming
    estimators use — the ln2 factor is exactly the constant that goes
    missing when this is re-spelled per class.
    """
    return 1.0 if dt_nanos <= 0 else math.exp(-dt_nanos * _LN2 / half_life_nanos)


def log_gamma(x: float) -> float:
    """Natural log of the gamma function (Lanczos, |relative error| < 2e-10).

    Raises:
        ValueError: if x is not strictly positive (NaN included — the
            ``not (x > 0)`` gate is deliberately NaN-rejecting).
    """
    if not (x > 0):
        raise ValueError(f"x must be > 0: {x}")
    cof = (76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Continued fraction (modified Lentz), switching to the symmetry
    ``I_x(a,b) = 1 - I_{1-x}(b,a)`` where the fraction converges
    fastest. Accurate to ~1e-13 across (0, 1).

    Raises:
        ValueError: unless a > 0, b > 0 and x in [0, 1] (NaN-rejecting).
    """
    if not (a > 0) or not (b > 0) or not (0 <= x <= 1):
        raise ValueError("need a, b > 0 and x in [0, 1]")
    if x == 0 or x == 1:
        return float(x)
    ln_front = (log_gamma(a + b) - log_gamma(a) - log_gamma(b)
                + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln_front) * _beta_continued_fraction(a, b, x) / a
    return 1 - math.exp(ln_front) * _beta_continued_fraction(b, a, 1 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    eps = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1) < eps:
            break
    return h


def t_cdf(t: float, df: float) -> float:
    """Student-t CDF with ``df`` degrees of freedom.

    Exact via the regularized incomplete beta
    (``P(T <= t) = 1 - 0.5 * I_{v/(v+t^2)}(v/2, 1/2)`` for t >= 0),
    no normal approximation: the tails are precisely where a t
    distribution and its moment-matched normal disagree most.

    Raises:
        ValueError: if df is not strictly positive (NaN-rejecting).
    """
    if not (df > 0):
        raise ValueError(f"df must be > 0: {df}")
    if t == 0:
        return 0.5
    half_tail = 0.5 * regularized_incomplete_beta(df / 2, 0.5, df / (df + t * t))
    return 1 - half_tail if t > 0 else half_tail
