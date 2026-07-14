"""Port of Java ``com.quantfinlib.volatility.VolatilityIndex``.

A VIX-style MARKET volatility index — the "fear gauge": the market's own
30-day volatility expectation, read model-free out of an option chain.
No pricing model is assumed; the index is the variance-swap replication
(Carr-Madan / CBOE methodology, single expiry):

    sigma^2 = (2/T) * sum_i (dK_i / K_i^2) * e^{rT} * Q(K_i)
              - (1/T) * (F/K0 - 1)^2

where Q(K) is the OUT-OF-THE-MONEY option mid at strike K (puts below
the forward, calls above, the put/call average at the pivot K0 — the
highest strike at or below F), dK the half-distance between neighboring
strikes, and the last term corrects for K0 != F.

Why OTM options? Each strike's option contributes exactly the 1/K^2
slice needed to build a constant-dollar-gamma payoff — a position whose
P&L IS realized variance. The market prices that portfolio, so the
portfolio's price reveals the market's variance expectation, whatever
model anyone used. That is why a vol SMILE raises the index above ATM
implied vol: the wings carry real premium and the replication weights
them in.

Honesty notes: single-expiry (the CBOE interpolates two expiries to
exactly 30 days — supply the chain nearest your target tenor, or
compute two indices and interpolate variance in time); truncation bias:
strikes should span several sigma*sqrt(T) or the index reads LOW (the
tails you cannot see are variance you do not count). Styled after the
methodology, not certified.
"""

from __future__ import annotations

import math

import numpy as np


class VolatilityIndex:
    """Static model-free volatility index from a single-expiry chain."""

    @staticmethod
    def index(strikes, put_mids, call_mids, forward: float,
              rate: float, t_years: float) -> float:
        """The index (annualized volatility, e.g. 0.20 = "a VIX of 20")
        from one expiry's chain.

        Args:
            strikes: ascending strikes, >= 3, all > 0.
            put_mids: put mid prices per strike, >= 0, finite.
            call_mids: call mid prices per strike, >= 0, finite.
            forward: the forward F for this expiry, strictly inside
                (strikes[0], strikes[-1]) — an index built on
                extrapolation would be an opinion, not a measurement.
            rate: continuously-compounded rate to expiry.
            t_years: time to expiry, > 0.

        Raises:
            ValueError: on any malformed input or a chain implying
                non-positive variance.
        """
        strikes = np.asarray(strikes, dtype=float)
        put_mids = np.asarray(put_mids, dtype=float)
        call_mids = np.asarray(call_mids, dtype=float)
        n = strikes.shape[0]
        if n < 3 or put_mids.shape[0] != n or call_mids.shape[0] != n:
            raise ValueError("need >= 3 aligned strikes/puts/calls")
        if not (t_years > 0) or t_years == math.inf:
            raise ValueError("tYears must be positive and finite")
        if not math.isfinite(rate):
            raise ValueError("rate must be finite")
        # Ascending, positive, finite strikes (NaN fails the comparison).
        prev = np.concatenate([[0.0], strikes[:-1]])
        if bool(np.any(~(strikes > prev) | (strikes == np.inf))):
            raise ValueError("strikes must be ascending, positive and finite")
        if bool(np.any(~(put_mids >= 0) | (put_mids == np.inf)
                       | ~(call_mids >= 0) | (call_mids == np.inf))):
            raise ValueError("option mids must be >= 0 and finite")
        if not (forward > strikes[0] and forward < strikes[n - 1]):
            raise ValueError(
                f"forward {forward} must sit strictly inside the strike range "
                "— the index cannot be measured from extrapolation")

        # K0: highest strike at or below the forward.
        pivot = 0
        for i in range(n):
            if strikes[i] <= forward:
                pivot = i
        # e^{+rT}: FORWARD-values the option mids (the CBOE construction)
        # — this is a growth factor, not a discount factor; "fixing" it
        # to e^{-rT} would introduce a 2rT relative error.
        growth = math.exp(rate * t_years)
        dk = np.empty(n)
        dk[0] = strikes[1] - strikes[0]
        dk[-1] = strikes[-1] - strikes[-2]
        if n > 2:
            dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2
        idx = np.arange(n)
        q = np.where(idx < pivot, put_mids,
                     np.where(idx > pivot, call_mids,
                              (put_mids + call_mids) / 2))  # the K0 straddle average
        s = float(np.sum(dk / (strikes * strikes) * growth * q))
        k0 = float(strikes[pivot])
        variance = (2 / t_years) * s - (1 / t_years) * (forward / k0 - 1) ** 2
        if not (variance > 0):
            raise ValueError(f"chain implies non-positive variance ({variance}) "
                             "— the quotes are inconsistent")
        return math.sqrt(variance)
