"""Self-exciting (Hawkes) event intensity (port of Java
``microstructure.HawkesIntensity``) -- the model behind the trader's
observation that "activity breeds activity": one trade raises the
probability of the next, so order flow arrives in bursts, not as a
steady Poisson drizzle. The exponential Hawkes form keeps the whole
process in two numbers::

    lambda(t) = mu + S(t),   S(t) = sum over past events of alpha*exp(-beta*(t-t_i))

where ``mu`` is the baseline arrival rate, ``alpha`` the excitation
each event adds, and ``beta`` the decay. ``S`` updates in O(1) per
event -- decay what is there, add alpha -- which is what makes Hawkes
streaming-friendly while richer clustering models are not.

**Stability is enforced, not assumed:** the branching ratio
``alpha/beta`` is the expected number of children each event spawns;
at >= 1 the process is explosive (each event begets more than one, and
the intensity diverges), so the constructor rejects it. Read
:meth:`burst_score` as the regime signal: 0 = baseline flow, 1 = the
self-excited component equals the baseline (activity running 2x), and
it decays back with the configured half-life when the burst ends.

Feed every arrival of the event type you care about via
:meth:`on_event`; timestamps must be non-decreasing (a backwards
timestamp is ignored -- feed-merge jitter must not inject negative
decay).
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils


class HawkesIntensity:
    """Streaming self-exciting event intensity with a
    constructor-enforced stability check; see the module docstring."""

    __slots__ = ("_baseline_rate_per_sec", "_excitation",
                 "_decay_half_life_nanos", "_excited", "_last_event_nanos",
                 "_has_event", "_events")

    def __init__(self, baseline_rate_per_sec: float = 2.0,
                 excitation: float = 0.1,
                 decay_half_life_nanos: int = 2_000_000_000) -> None:
        """
        Args:
            baseline_rate_per_sec: mu -- arrival rate with no
                excitation, e.g. 2.0.
            excitation: alpha -- intensity each event adds (per
                second), e.g. 0.1.
            decay_half_life_nanos: how fast excitation fades, e.g. 2s.
                Stability requires branching ratio ``alpha/beta < 1``
                where ``beta = ln2/halfLife`` (per second): at a 2s
                half-life, beta ~= 0.35/s, so alpha must stay below
                ~0.35 -- rejected otherwise.
        """
        if (baseline_rate_per_sec <= 0 or excitation < 0
                or decay_half_life_nanos <= 0):
            raise ValueError(
                "need baselineRate > 0, excitation >= 0, halfLife > 0")
        beta = math.log(2) / (decay_half_life_nanos * 1e-9)
        if excitation / beta >= 1:
            raise ValueError(
                f"explosive: branching ratio {excitation / beta} >= 1 "
                "(each event spawns >= 1 child)")
        self._baseline_rate_per_sec = baseline_rate_per_sec
        self._excitation = excitation
        self._decay_half_life_nanos = decay_half_life_nanos
        self._excited = 0.0
        self._last_event_nanos = 0
        self._has_event = False
        self._events = 0

    def on_event(self, timestamp_nanos: int) -> None:
        """One event arrival. Timestamps must be non-decreasing; an
        out-of-order timestamp is dropped (negative decay would GROW
        past excitation instead of fading it)."""
        if self._has_event:
            dt = timestamp_nanos - self._last_event_nanos
            if dt < 0:
                return
            self._excited = (self._excited
                             * math_utils.decay_factor(dt, self._decay_half_life_nanos)
                             + self._excitation)
        else:
            self._excited = self._excitation
            self._has_event = True
        self._last_event_nanos = timestamp_nanos
        self._events += 1

    def intensity(self, now_nanos: int) -> float:
        """The current intensity lambda(now) in events per second."""
        return self._baseline_rate_per_sec + self._excited_at(now_nanos)

    def burst_score(self, now_nanos: int) -> float:
        """The dimensionless burst regime: self-excited intensity over
        the baseline, clamped to [0, 1] at "activity running 2x
        baseline". 0 in steady flow; decays back with the configured
        half-life."""
        return math_utils.clamp(
            self._excited_at(now_nanos) / self._baseline_rate_per_sec, 0, 1)

    def _excited_at(self, now_nanos: int) -> float:
        if not self._has_event:
            return 0.0
        dt = now_nanos - self._last_event_nanos
        if dt <= 0:
            return self._excited
        return self._excited * math_utils.decay_factor(dt, self._decay_half_life_nanos)

    def events(self) -> int:
        return self._events
