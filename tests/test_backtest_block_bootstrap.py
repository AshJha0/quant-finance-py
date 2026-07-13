"""Pins for quantfinlib.backtest.validation.block_bootstrap.

Java source: AlphaResearchRoundTest.java (block bootstrap section). The
Java java.util.Random stream is not reproduced — the pinned property set
is: sorted output of the requested size, the distribution centering on
the sample Sharpe, determinism per seed, the blocked-vs-iid spread
ordering on autocorrelated data (THE point of blocks), and the gates.
"""

import numpy as np
import pytest

from quantfinlib.backtest import _risk
from quantfinlib.backtest.validation import BlockBootstrap
from quantfinlib.util import math_utils as mu


def _iid_returns(n: int = 300, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 5e-4 + 0.01 * rng.standard_normal(n)


def test_sharpe_samples_sorted_centered_and_deterministic():
    iid = _iid_returns()
    sample_sharpe = _risk.sharpe_ratio(iid, 0, 252)
    samples = BlockBootstrap.sharpe_samples(iid, 10, 500, 252, 7)
    assert samples.shape[0] == 500
    # Sorted ascending, ready for percentile_sorted.
    assert np.all(samples[:-1] <= samples[1:])
    # The distribution centers on the sample estimate.
    assert mu.percentile_sorted(samples, 0.5) == pytest.approx(
        sample_sharpe, abs=0.35)
    # Deterministic per seed: replayable.
    assert np.array_equal(samples,
                          BlockBootstrap.sharpe_samples(iid, 10, 500, 252, 7))


def test_blocks_preserve_the_dependence_the_iid_resample_destroys():
    # AR(1) with phi = 0.9: the honest Sharpe uncertainty is wider than
    # an iid resample admits — blocked spread must exceed iid spread.
    rng = np.random.default_rng(17)
    n = 300
    ar = np.empty(n)
    prev = 0.0
    for i in range(n):
        prev = 0.9 * prev + 0.01 * rng.standard_normal()
        ar[i] = prev
    std_blocked = mu.std_dev(BlockBootstrap.sharpe_samples(ar, 25, 400, 252, 3))
    std_iid = mu.std_dev(BlockBootstrap.sharpe_samples(ar, 1, 400, 252, 3))
    assert std_blocked > 1.2 * std_iid


def test_resample_is_a_multiset_over_the_original_values():
    series = np.arange(50, dtype=float)
    rng = np.random.default_rng(5)
    path = BlockBootstrap.resample(series, 5, rng)
    assert path.shape[0] == 50
    # Every resampled value is an original observation (circular wrap).
    assert np.all(np.isin(path, series))


def test_block_bootstrap_gates():
    iid = _iid_returns()
    with pytest.raises(ValueError):
        BlockBootstrap.sharpe_samples(iid, 0, 500, 252, 7)     # L < 1
    with pytest.raises(ValueError):
        BlockBootstrap.sharpe_samples(iid, 10, 50, 252, 7)     # < 100 resamples
    with pytest.raises(ValueError):
        BlockBootstrap.sharpe_samples(iid, 10, 500, 0, 7)      # bad periods
    with pytest.raises(ValueError):
        BlockBootstrap.sharpe_samples(iid[:30], 10, 500, 252, 7)  # < 50 obs
    bad = iid.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        BlockBootstrap.sharpe_samples(bad, 10, 500, 252, 7)    # non-finite
