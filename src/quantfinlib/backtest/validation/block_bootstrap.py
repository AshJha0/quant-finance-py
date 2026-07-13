"""Stationary block bootstrap (port of Java ``validation.BlockBootstrap``).

Politis-Romano stationary bootstrap — the confidence interval your
backtest's Sharpe ratio deserves and almost never gets. A single
historical path yields ONE Sharpe estimate; resampling the path in
blocks yields its sampling DISTRIBUTION, and the honest question becomes
"is the 5th percentile still positive?" rather than "is 1.2 a good
number?".

Why BLOCKS: returns are autocorrelated (vol clusters, trends persist),
and resampling single observations (an iid bootstrap) destroys that
structure and UNDERSTATES the uncertainty — the classic way to be
falsely confident. Blocks of geometric mean length L preserve local
dependence; the stationary variant restarts blocks with probability 1/L
and wraps circularly, so every resampled path has the full original
length. Rule of thumb: L ~ n^(1/3) (about 10 for a 1,000-day history).

Deterministic per seed (replayable) via ``np.random.default_rng`` (port
note: the Java reference draws from ``java.util.Random``; the RNG stream
is not reproduced across ports — the pinned properties are the sorted
output, determinism-per-seed, the distribution centering on the sample
estimate and the blocked-vs-iid spread ordering). Honest about what it
is NOT: the bootstrap resamples the history you had — it cannot
manufacture regimes the sample never contained. Pair with
``SharpeValidation`` (multiple-testing haircut); this class quantifies
the sampling error that remains even for an honest, single-trial
backtest. Research lane.
"""

from __future__ import annotations

import numpy as np

from quantfinlib.backtest import _risk


class BlockBootstrap:
    """Static stationary-bootstrap resampling; see the module docstring."""

    @staticmethod
    def sharpe_samples(returns, mean_block_length: int, resamples: int,
                       periods_per_year: int, seed: int) -> np.ndarray:
        """Bootstrap distribution of ANNUALIZED Sharpe, sorted ascending.

        Read percentiles with ``math_utils.percentile_sorted``.

        Args:
            returns: Per-period strategy returns, >= 50 finite values.
            mean_block_length: Geometric mean block length L, >= 1
                (1 = iid bootstrap — only for demonstrating why you
                should not use it).
            resamples: Bootstrap paths, >= 100.
            periods_per_year: Annualization (252 for daily).
            seed: Deterministic seed.

        Raises:
            ValueError: on a degenerate configuration or non-finite input.
        """
        if resamples < 100:
            raise ValueError("need >= 100 resamples for a distribution")
        if periods_per_year < 1:
            raise ValueError("periodsPerYear must be >= 1")
        r = _required_series(returns, mean_block_length)
        rng = np.random.default_rng(seed)
        samples = np.empty(resamples)
        for s in range(resamples):
            path = _resample(r, mean_block_length, rng)
            samples[s] = _risk.sharpe_ratio(path, 0, periods_per_year)
        samples.sort()
        return samples

    @staticmethod
    def resample(series, mean_block_length: int,
                 rng: np.random.Generator) -> np.ndarray:
        """One stationary-bootstrap path (circular, geometric blocks)."""
        r = _required_series(series, mean_block_length)
        return _resample(r, mean_block_length, rng)


def _resample(series: np.ndarray, mean_block_length: int,
              rng: np.random.Generator) -> np.ndarray:
    """Builds one path: blocks restart with probability 1/L, wrap circularly.

    Vectorized form of the Java per-step walk: position 0 always starts a
    block; each later position restarts with probability 1/L, otherwise
    continues the previous index + 1 (mod n).
    """
    n = series.shape[0]
    restart_p = 1.0 / mean_block_length
    restarts = np.empty(n, dtype=bool)
    restarts[0] = True
    restarts[1:] = rng.random(n - 1) < restart_p
    group = np.cumsum(restarts) - 1              # block id per position
    starts = rng.integers(0, n, size=int(group[-1]) + 1)
    first_pos = np.flatnonzero(restarts)          # position where each block begins
    offset = np.arange(n) - first_pos[group]
    return series[(starts[group] + offset) % n]


def _required_series(series, mean_block_length: int) -> np.ndarray:
    r = np.asarray(series, dtype=float)
    if r.shape[0] < 50:
        raise ValueError("need >= 50 observations")
    if mean_block_length < 1:
        raise ValueError("meanBlockLength must be >= 1")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    return r
