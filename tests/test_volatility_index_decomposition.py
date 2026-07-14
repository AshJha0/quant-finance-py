"""Pins for quantfinlib.volatility.{volatility_index,volatility_decomposition}.

Java source: VolatilityIndexAndDecompositionTest. The option chains are
priced with a local Black-Scholes helper built on math_utils.norm_cdf —
the same A&S 26.2.17 approximation Java's BlackScholes uses, so the
"deep-OTM prices can be ~-1e-9" caveat (and the max(0, .) guard)
transfers. The Java cross-check against RiskMetrics.beta has no
counterpart yet (risk package not ported); the OLS identity asserts
cover the same arithmetic.
"""

import math

import numpy as np
import pytest

from quantfinlib.util import math_utils as mu
from quantfinlib.volatility import VolatilityDecomposition, VolatilityIndex


def bs_price(is_call: bool, spot: float, strike: float, vol: float, t: float) -> float:
    """Black-Scholes with r = carry = 0 (all the chain tests use), via
    the library norm_cdf (A&S 26.2.17), matching Java's BlackScholes."""
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if is_call:
        return spot * mu.norm_cdf(d1) - strike * mu.norm_cdf(d2)
    return strike * mu.norm_cdf(-d2) - spot * mu.norm_cdf(-d1)


# ------------------------------------------------------------------
# VIX-style index — model-free means it must RECOVER a flat vol
# ------------------------------------------------------------------

def test_index_recovers_a_flat_vol_and_reads_the_smile():
    # A chain priced at flat 20% vol: the variance-swap replication must
    # hand the 20% back — that is what "model-free" MEANS.
    t = 30.0 / 365
    n = 81
    strikes = 60.0 + np.arange(n)  # 60..140: 8.9 sigma down, 5.9 sigma up
    # max(0, .): the norm_cdf approximation can leave deep-OTM prices at
    # ~-1e-9; real chains quote >= 0 by construction.
    puts = np.array([max(0.0, bs_price(False, 100, k, 0.20, t)) for k in strikes])
    calls = np.array([max(0.0, bs_price(True, 100, k, 0.20, t)) for k in strikes])
    flat = VolatilityIndex.index(strikes, puts, calls, 100, 0, t)
    assert flat == pytest.approx(0.20, abs=2e-3), \
        "the fear gauge reads the market's own number"

    # F STRICTLY BETWEEN strikes: with F on a strike (above) the
    # (F/K0-1)^2 correction term is exactly zero and untestable —
    # dropping it would pass. At F = 100.9 the omission error is
    # ~+2.9e-3, so THIS assertion is what makes the term load-bearing
    # (chain re-priced at the shifted forward via spot = 100.9, r=0).
    puts_f = np.array([max(0.0, bs_price(False, 100.9, k, 0.20, t)) for k in strikes])
    calls_f = np.array([max(0.0, bs_price(True, 100.9, k, 0.20, t)) for k in strikes])
    off_grid = VolatilityIndex.index(strikes, puts_f, calls_f, 100.9, 0, t)
    assert off_grid == pytest.approx(0.20, abs=2e-3), \
        "the K0 != F correction term earns its keep here"

    # A put SKEW (downside priced at 25%) must RAISE the index above the
    # ATM 20% — the wings carry real premium and the replication weights
    # them in. That is why the index > ATM implied vol.
    skewed_puts = np.array([max(0.0, bs_price(False, 100, k, 0.25, t)) for k in strikes])
    skewed = VolatilityIndex.index(strikes, skewed_puts, calls, 100, 0, t)
    assert flat + 0.005 < skewed < 0.25, f"the smile is IN the index: {skewed}"

    # Gates: extrapolation is an opinion, not a measurement.
    with pytest.raises(ValueError):
        VolatilityIndex.index(strikes, puts, calls, 150, 0, t)
    with pytest.raises(ValueError):
        VolatilityIndex.index([90.0, 100.0], np.zeros(2), np.zeros(2), 95, 0, t)
    with pytest.raises(ValueError):
        # an all-zero chain implies no variance: inconsistent
        VolatilityIndex.index(strikes, np.zeros(n), np.zeros(n), 100, 0, t)


def test_index_input_gates():
    strikes = [90.0, 100.0, 110.0]
    q = [1.0, 2.0, 1.0]
    with pytest.raises(ValueError):
        VolatilityIndex.index(strikes, q, q, 100, 0, 0)  # t <= 0
    with pytest.raises(ValueError):
        VolatilityIndex.index(strikes, q, q, 100, math.nan, 0.1)  # NaN rate
    with pytest.raises(ValueError):
        VolatilityIndex.index([100.0, 90.0, 110.0], q, q, 100, 0, 0.1)  # not ascending
    with pytest.raises(ValueError):
        VolatilityIndex.index(strikes, [1.0, -2.0, 1.0], q, 100, 0, 0.1)  # negative mid


# ------------------------------------------------------------------
# Systematic vs idiosyncratic — the split is EXACT, not approximate
# ------------------------------------------------------------------

def test_decomposition_recovers_planted_beta_and_splits_exactly():
    rng = np.random.default_rng(11)
    n = 5_000
    market = 0.02 * rng.standard_normal(n)
    asset = 1.5 * market + 0.01 * rng.standard_normal(n)
    d = VolatilityDecomposition.decompose(asset, market)
    assert d.beta == pytest.approx(1.5, abs=0.05), "the planted beta"
    # One beta in this library: it must equal cov/var directly.
    assert d.beta == pytest.approx(
        mu.covariance(asset, market) / mu.variance(market), abs=1e-12)
    assert d.idiosyncratic_variance == pytest.approx(1e-4, abs=1.5e-5), \
        "the residual is the planted 1% noise"
    assert d.systematic_variance + d.idiosyncratic_variance == pytest.approx(
        d.total_variance, abs=1e-18), \
        "the OLS split is an identity, not an approximation"
    assert d.r_squared > 0.85, f"mostly a market story: {d.r_squared}"
    assert d.systematic_vol(252) == pytest.approx(
        math.sqrt(d.systematic_variance * 252), abs=1e-15)

    # A clone of the market IS the market: beta 1, idio 0, R^2 1.
    clone = VolatilityDecomposition.decompose(market.copy(), market)
    assert clone.beta == pytest.approx(1.0, abs=1e-12)
    assert clone.idiosyncratic_variance == pytest.approx(0.0, abs=1e-15), \
        "no company story in a market clone"
    assert clone.r_squared == pytest.approx(1.0, abs=1e-12)

    # Pure noise vs the market: nothing systematic to find.
    noise = 0.015 * rng.standard_normal(n)
    idio = VolatilityDecomposition.decompose(noise, market)
    assert idio.r_squared < 0.02, f"a biotech, not a utility: {idio.r_squared}"

    with pytest.raises(ValueError):
        VolatilityDecomposition.decompose(np.zeros(30), market)  # misaligned
    with pytest.raises(ValueError):
        VolatilityDecomposition.decompose(noise, np.zeros(n))  # flat benchmark
    bad = asset.copy()
    bad[7] = math.nan
    with pytest.raises(ValueError):
        VolatilityDecomposition.decompose(bad, market)
