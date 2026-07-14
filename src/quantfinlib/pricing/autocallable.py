"""Autocallable note pricer (port of Java
``com.quantfinlib.pricing.Autocallable``).

A note that pays a fat coupon and redeems early ("autocalls") the first
observation date the underlier closes at or above the autocall barrier.
If it survives to maturity, the principal is protected UNLESS the
underlier has fallen through the knock-in barrier, in which case the
holder takes the equity loss — the note is, economically, a bond plus
sold down-and-in put, funded by the coupons.

**Model honesty.** Monte Carlo under Black-Scholes GBM with flat
volatility and rates — the standard first pricer, NOT a desk-grade one.
European knock-in (observed at maturity only), observation-date
monitoring, no issuer credit spread. Antithetic variates halve the
variance; a fixed seed makes every price reproducible.

Port note: the Java implementation draws from ``java.util.Random``; this
port draws the same path structure from ``numpy.random.default_rng``
(vectorized over paths). Prices are deterministic per seed within this
port but not bit-identical to the Java port — the Java suite pins only
zero-vol exact values, inequalities and same-seed reproducibility, all
of which transfer.
"""

from __future__ import annotations

import math

import numpy as np


class Autocallable:
    """Immutable autocallable terms; ``price`` runs the Monte Carlo."""

    def __init__(self, notional: float, observation_years, autocall_barrier: float,
                 coupon_barrier: float, knock_in_barrier: float, coupon_per_period: float,
                 memory_coupons: bool) -> None:
        """
        Args:
            notional: redemption amount, e.g. 1_000_000.
            observation_years: strictly increasing observation times in
                years; the last is maturity.
            autocall_barrier: autocall trigger as a fraction of initial spot.
            coupon_barrier: coupon trigger as a fraction of initial spot,
                <= autocall_barrier.
            knock_in_barrier: protection barrier as a fraction of initial spot.
            coupon_per_period: coupon per observation period as a fraction
                of notional.
            memory_coupons: missed coupons are caught up at the next paying
                observation (Phoenix memory).
        """
        observation_years = [float(t) for t in observation_years]
        # not (x > 0) rejects NaN as well as non-positives: a NaN term must
        # fail HERE, not surface as a NaN price later. The knock-in must
        # sit at or below the autocall (protection above the early-redemption
        # trigger is unreachable); its relation to the COUPON barrier is
        # deliberately free — structures place it on either side.
        if (not (notional > 0) or notional == math.inf
                or len(observation_years) == 0
                or not (autocall_barrier > 0) or autocall_barrier == math.inf
                or not (coupon_barrier > 0) or not (knock_in_barrier > 0)
                or coupon_barrier > autocall_barrier or knock_in_barrier > autocall_barrier
                or not (coupon_per_period >= 0) or coupon_per_period == math.inf):
            raise ValueError("invalid autocallable terms")
        prev = 0.0
        for t in observation_years:
            if not (t > prev):
                raise ValueError("observation times must be strictly increasing and positive")
            prev = t
        self._notional = notional
        self._observation_years = tuple(observation_years)
        self._autocall_barrier = autocall_barrier
        self._coupon_barrier = coupon_barrier
        self._knock_in_barrier = knock_in_barrier
        self._coupon_per_period = coupon_per_period
        self._memory_coupons = memory_coupons

    def price(self, spot: float, initial: float, vol: float, rate: float,
              div_yield: float, paths: int, seed: int) -> float:
        """Monte Carlo present value under GBM.

        Args:
            spot: current underlier level (S0 for a new issue).
            initial: the strike-setting initial level S0 (equal to ``spot``
                at issue; differs for a seasoned note).
            vol: Black-Scholes volatility, per sqrt(year) — pick it from
                the downside smile region, not ATM (see module doc).
            rate: continuously-compounded discount rate.
            div_yield: continuous dividend yield.
            paths: Monte Carlo paths (antithetic: 2 per draw), e.g. 100_000.
            seed: RNG seed — fixed seed = reproducible price.
        """
        if (not (spot > 0) or not (initial > 0) or paths < 1
                or spot == math.inf or initial == math.inf
                or not (vol >= 0) or vol == math.inf
                or not (math.isfinite(rate) and math.isfinite(div_yield))):
            raise ValueError("invalid market inputs")
        n = len(self._observation_years)
        drift = np.empty(n)
        vol_step = np.empty(n)
        discount = np.empty(n)
        prev_t = 0.0
        for i, t in enumerate(self._observation_years):
            dt = t - prev_t
            drift[i] = (rate - div_yield - 0.5 * vol * vol) * dt
            vol_step[i] = vol * math.sqrt(dt)
            discount[i] = math.exp(-rate * t)
            prev_t = t

        rng = np.random.default_rng(seed)
        gaussians = rng.standard_normal((paths, n))
        total = 0.0
        for sign in (1.0, -1.0):
            total += self._path_values(spot, initial, drift, vol_step, discount,
                                       gaussians, sign)
        return total / (2.0 * paths)

    def _path_values(self, spot: float, initial: float, drift: np.ndarray,
                     vol_step: np.ndarray, discount: np.ndarray,
                     gaussians: np.ndarray, sign: float) -> float:
        """Sum of discounted payoffs across all paths; ``sign`` flips the
        draws (antithetic). Vectorized transcription of the Java per-path
        observation walk."""
        n = len(self._observation_years)
        paths = gaussians.shape[0]
        coupon = self._coupon_per_period * self._notional
        # Level at each observation: cumulative log-return from spot.
        log_inc = drift[np.newaxis, :] + sign * vol_step[np.newaxis, :] * gaussians
        levels = spot * np.exp(np.cumsum(log_inc, axis=1)) / initial

        alive = np.ones(paths, dtype=bool)
        value = np.zeros(paths)
        missed = np.zeros(paths)
        for i in range(n):
            lvl = levels[:, i]
            df = discount[i]
            autocall = alive & (lvl >= self._autocall_barrier)
            # Early redemption: notional + this coupon + any memory.
            value[autocall] += df * (self._notional + coupon)
            if self._memory_coupons:
                value[autocall] += df * missed[autocall]
            alive = alive & ~autocall
            pays = alive & (lvl >= self._coupon_barrier)
            value[pays] += df * coupon
            if self._memory_coupons:
                value[pays] += df * missed[pays]
            missed[pays] = 0.0
            misses = alive & ~pays
            missed[misses] += coupon
            if i == n - 1:
                # Maturity without autocall: protected unless knocked in.
                protected = alive & (lvl >= self._knock_in_barrier)
                value[protected] += df * self._notional
                knocked = alive & ~protected
                value[knocked] += df * self._notional * lvl[knocked]
        return float(np.sum(value))

    def notional(self) -> float:
        return self._notional

    def observations(self) -> int:
        return len(self._observation_years)
