"""Jump-robust streaming volatility (port of Java
``microstructure.JumpRobustVolatility``). A squared-return estimator
cannot tell a news gap from diffusion: one headline print enters as
r^2 and reads as a volatility regime shift for the estimator's whole
memory. Bipower variation (Barndorff-Nielsen & Shephard, 2004) fixes
this with a beautifully simple trick: use the product of CONSECUTIVE
absolute returns, ``(pi/2)*|r_t|*|r_(t-1)|``, instead of r^2. Diffusion
moves both factors together, so the product estimates the same
sigma^2; a single jump inflates only ONE factor of two neighboring
products instead of one whole squared term -- its weight in the
estimate collapses.

Both estimators run side by side on time-decayed rates per second:
:meth:`vol_per_sqrt_second` is the jump-robust (bipower) volatility --
the one to feed
:class:`~quantfinlib.microstructure.volatility_curve.VolatilityCurve`
and any model that should read regimes, not headlines -- while
:meth:`raw_vol_per_sqrt_second` is the squared-return volatility, and
:meth:`jump_fraction` is the share of raw variance the robust
estimator attributes to jumps (``1 - BV/RV``, clamped to [0,1]).

Gap discipline: a non-finite return or non-positive dt drops the
sample AND resets the consecutive-return pairing -- multiplying across
a feed gap would pair returns that were never neighbors. The first
return after a gap therefore updates only the raw estimator; the
bipower leg resumes one sample later. Irregular sampling is handled
exactly: the two-return product is normalized by
``sqrt(dt_t * dt_(t-1))`` (each ``|r|`` scales with sqrt of ITS OWN
interval), so event-time feeds -- where activity accelerates precisely
when volatility bursts -- do not bias the estimator.
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils

_HALF_PI = math.pi / 2


class JumpRobustVolatility:
    """Bipower-variation streaming volatility with jump-fraction
    diagnostics; see the module docstring."""

    __slots__ = ("_half_life_nanos", "_raw_rate_per_sec",
                 "_bipower_rate_per_sec", "_prev_abs_return", "_prev_dt_sec",
                 "_has_prev", "_bipower_seeded", "_samples")

    def __init__(self, half_life_nanos: int = 10_000_000_000) -> None:
        """``half_life_nanos``: decay half-life, e.g. 10s."""
        if half_life_nanos <= 0:
            raise ValueError("halfLifeNanos must be positive")
        self._half_life_nanos = half_life_nanos
        self._raw_rate_per_sec = 0.0        # decayed E[r^2/dt]
        self._bipower_rate_per_sec = 0.0    # decayed E[(pi/2)|r||r-1|/sqrt(dt*dt-1)]
        self._prev_abs_return = 0.0
        self._prev_dt_sec = 0.0
        self._has_prev = False
        self._bipower_seeded = False
        self._samples = 0

    def on_return(self, ret: float, dt_nanos: int) -> None:
        """One return observation: the relative mid change over the
        elapsed ``dt_nanos``. Non-finite returns or non-positive gaps
        drop the sample and break the pairing (see module doc)."""
        if not math.isfinite(ret) or dt_nanos <= 0:
            self._has_prev = False          # gap: never pair across it
            return
        dt_sec = dt_nanos * 1e-9
        a = 1 - math_utils.decay_factor(dt_nanos, self._half_life_nanos)

        raw_obs = ret * ret / dt_sec
        self._raw_rate_per_sec = (
            raw_obs if self._samples == 0
            else self._raw_rate_per_sec + a * (raw_obs - self._raw_rate_per_sec))

        abs_ret = abs(ret)
        if self._has_prev:
            # Each |r| carries sqrt of its OWN interval, so the
            # product is normalized by the geometric mean of the two
            # -- exact under irregular sampling, where
            # dt-of-the-moment normalization would read a cadence
            # change as a volatility change.
            bp_obs = (_HALF_PI * abs_ret * self._prev_abs_return
                     / math.sqrt(dt_sec * self._prev_dt_sec))
            self._bipower_rate_per_sec = (
                self._bipower_rate_per_sec + a * (bp_obs - self._bipower_rate_per_sec)
                if self._bipower_seeded else bp_obs)
            self._bipower_seeded = True
        self._prev_abs_return = abs_ret
        self._prev_dt_sec = dt_sec
        self._has_prev = True
        self._samples += 1

    def vol_per_sqrt_second(self) -> float:
        """The jump-robust volatility, as return per sqrt(second) --
        the diffusion component, with jumps down-weighted. 0 until two
        consecutive valid returns exist."""
        return math.sqrt(max(self._bipower_rate_per_sec, 0.0))

    def raw_vol_per_sqrt_second(self) -> float:
        """The plain squared-return volatility (jumps and all), per
        sqrt(second)."""
        return math.sqrt(max(self._raw_rate_per_sec, 0.0))

    def jump_fraction(self) -> float:
        """The share of raw variance attributed to jumps:
        ``clamp(1 - bipower/raw, 0, 1)``. Near 0 in pure diffusion,
        spikes after a discontinuous move, and decays back as the jump
        washes out of the raw estimator's memory. 0 while either
        estimator is unlearned."""
        if self._raw_rate_per_sec <= 0 or not self._bipower_seeded:
            return 0.0
        return math_utils.clamp(
            1 - self._bipower_rate_per_sec / self._raw_rate_per_sec, 0, 1)

    def samples(self) -> int:
        return self._samples
