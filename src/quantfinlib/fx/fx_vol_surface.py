"""FX delta-quoted volatility surface (port of Java
``com.quantfinlib.fx.FxVolSurface``).

Builds an absolute strike/vol smile from the market's delta quotes —
ATM (delta-neutral straddle), 25-delta risk reversal and butterfly,
optionally 10-delta wings:

    vol(25d call) = atm + bf25 + rr25 / 2
    vol(25d put)  = atm + bf25 - rr25 / 2

then solves each pillar's STRIKE from its delta and its own vol.
Forward delta (``N(d1)``) by default; premium-adjusted forward delta
(``(K/F) N(d2)``) with ``premium_adjusted``, taking the market's OTM
(higher-strike) branch for calls.

Interpolation: linear in vol against log-moneyness within an expiry
(flat wings); linear in total variance across expiries (flat outside);
forwards log-linear in time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.util import math_utils as mu


@dataclass(frozen=True, slots=True)
class SmilePillar:
    """One expiry's solved smile: absolute strikes and vols, low to
    high strike (port of the Java record)."""

    expiry_years: float
    forward: float
    strikes: list[float]
    vols: list[float]


class FxVolSurfaceBuilder:
    """Accumulates per-expiry delta quotes; strikes solve in ``build``."""

    def __init__(self):
        self._quotes: list[tuple] = []
        self._premium_adjusted = False

    def add(self, expiry_years: float, forward: float, atm_vol: float,
            rr25: float, bf25: float, rr10: float = math.nan,
            bf10: float = math.nan) -> "FxVolSurfaceBuilder":
        """25-delta quote (rr/bf in absolute vol, 0.01 = 1 vol point);
        pass ``rr10``/``bf10`` for the full five-pillar smile."""
        if expiry_years <= 0 or forward <= 0 or atm_vol <= 0:
            raise ValueError("expiry, forward and atm vol must be > 0")
        self._quotes.append((expiry_years, forward, atm_vol, rr25, bf25,
                             rr10, bf10))
        return self

    def premium_adjusted(self, value: bool) -> "FxVolSurfaceBuilder":
        """Switches strike solving to premium-adjusted forward delta."""
        self._premium_adjusted = value
        return self

    def build(self) -> "FxVolSurface":
        if not self._quotes:
            raise RuntimeError("at least one expiry quote required")
        quotes = sorted(self._quotes, key=lambda q: q[0])
        pa = self._premium_adjusted
        ts, fwds, lm, vv = [], [], [], []
        for t, fwd, atm, rr25, bf25, rr10, bf10 in quotes:
            ten_delta = not math.isnan(rr10)
            # Pillar vols from the broker quote (see module doc).
            v25c = atm + bf25 + rr25 / 2
            v25p = atm + bf25 - rr25 / 2
            # Each pillar's strike is solved with its own vol — the
            # market's definition of the quoted smile points.
            k_atm = FxVolSurface.dns_strike(fwd, atm, t, pa)
            k25c = FxVolSurface.strike_for_delta(fwd, v25c, t, 0.25, True, pa)
            k25p = FxVolSurface.strike_for_delta(fwd, v25p, t, -0.25, False, pa)
            if ten_delta:
                v10c = atm + bf10 + rr10 / 2
                v10p = atm + bf10 - rr10 / 2
                k10c = FxVolSurface.strike_for_delta(fwd, v10c, t, 0.10, True, pa)
                k10p = FxVolSurface.strike_for_delta(fwd, v10p, t, -0.10, False, pa)
                ks = [k10p, k25p, k_atm, k25c, k10c]
                vs = [v10p, v25p, atm, v25c, v10c]
            else:
                ks = [k25p, k_atm, k25c]
                vs = [v25p, atm, v25c]
            # Store as log-moneyness so time interp is forward-relative.
            x = []
            for j, k in enumerate(ks):
                if j > 0 and k <= ks[j - 1]:
                    raise RuntimeError(
                        f"solved strikes not increasing at expiry {t} — "
                        "check rr/bf signs and magnitudes")
                x.append(math.log(k / fwd))
            ts.append(t)
            fwds.append(fwd)
            lm.append(x)
            vv.append(vs)
        return FxVolSurface(ts, fwds, lm, vv, pa)


class FxVolSurface:
    """Construct via :meth:`builder`."""

    def __init__(self, expiries: list[float], forwards: list[float],
                 log_moneyness: list[list[float]], vols: list[list[float]],
                 premium_adjusted: bool):
        self._expiries = expiries
        self._forwards = forwards
        self._log_moneyness = log_moneyness
        self._vols = vols
        self._premium_adjusted = premium_adjusted

    @staticmethod
    def builder() -> FxVolSurfaceBuilder:
        return FxVolSurfaceBuilder()

    # ------------------------------------------------------------------
    # Delta <-> strike (static, reusable by hedgers and exotic pricers)
    # ------------------------------------------------------------------

    @staticmethod
    def dns_strike(forward: float, vol: float, t_years: float,
                   premium_adjusted: bool) -> float:
        """Delta-neutral-straddle (ATM) strike: ``F e^{+s^2 t/2}`` for
        forward delta, ``F e^{-s^2 t/2}`` premium-adjusted."""
        half = 0.5 * vol * vol * t_years
        return forward * math.exp(-half if premium_adjusted else half)

    @staticmethod
    def strike_for_delta(forward: float, vol: float, t_years: float,
                         delta: float, is_call: bool,
                         premium_adjusted: bool) -> float:
        """Strike for a target forward delta (call delta in (0,1), put
        delta in (-1,0)). Unadjusted deltas invert in closed form;
        premium-adjusted deltas are solved by bisection on the OTM
        branch."""
        if (delta <= 0 or delta >= 1) if is_call else (delta >= 0 or delta <= -1):
            side = "call" if is_call else "put"
            raise ValueError(f"delta out of range for {side}: {delta}")
        sv = vol * math.sqrt(t_years)
        if not premium_adjusted:
            # Call: d = N(d1) -> d1 = Ninv(d). Put: d = -N(-d1).
            d1 = mu.norm_inv(delta) if is_call else -mu.norm_inv(-delta)
            return forward * math.exp(-d1 * sv + 0.5 * sv * sv)
        # Premium-adjusted: call dpa = (K/F) N(d2), put dpa = -(K/F) N(-d2).
        lo = forward * math.exp(-8 * sv)
        hi = forward * math.exp(8 * sv)
        if is_call:
            # (K/F)N(d2) rises then falls in K; the market takes the OTM
            # (falling, higher-strike) branch. Coarse-scan for the peak,
            # then bisect to its right.
            peak = lo
            peak_val = -1.0
            for i in range(201):
                k = lo * (hi / lo) ** (i / 200.0)
                v = _pa_call_delta(forward, k, sv)
                if v > peak_val:
                    peak_val = v
                    peak = k
            if delta > peak_val:
                raise ValueError(
                    f"premium-adjusted call delta {delta} unattainable "
                    f"(max {peak_val}) at vol {vol}, t {t_years}")
            return _bisect(lambda k: _pa_call_delta(forward, k, sv) - delta,
                           peak, hi)
        # Put delta is monotone decreasing in K over the whole bracket.
        return _bisect(
            lambda k: -(k / forward) * mu.norm_cdf(-_d2(forward, k, sv)) - delta,
            lo, hi)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def vol(self, expiry_years: float, strike: float) -> float:
        """Interpolated vol for an absolute strike at an expiry. Flat
        wings beyond quoted pillars; linear in total variance across
        expiries, flat outside the quoted range."""
        if strike <= 0:
            raise ValueError(f"strike must be > 0: {strike}")
        ts = self._expiries
        n = len(ts)
        # Boundary expiries: flat in the edge smile, at that pillar's forward.
        if expiry_years <= ts[0]:
            return self._smile_vol(0, math.log(strike / self._forwards[0]))
        if expiry_years >= ts[n - 1]:
            return self._smile_vol(n - 1,
                                   math.log(strike / self._forwards[n - 1]))
        lo = self._bracket(expiry_years)
        hi = lo + 1
        w = (expiry_years - ts[lo]) / (ts[hi] - ts[lo])
        forward = math.exp(math.log(self._forwards[lo])
                           + w * (math.log(self._forwards[hi])
                                  - math.log(self._forwards[lo])))
        x = math.log(strike / forward)
        # Total-variance interpolation keeps calendar arbitrage at bay
        # when the smiles are themselves arbitrage-free.
        v_lo = self._smile_vol(lo, x)
        v_hi = self._smile_vol(hi, x)
        w_lo = v_lo * v_lo * ts[lo]
        w_hi = v_hi * v_hi * ts[hi]
        total_var = w_lo + (w_hi - w_lo) * w
        return math.sqrt(total_var / expiry_years)

    def atm_vol(self, expiry_years: float) -> float:
        """ATM (delta-neutral straddle) vol at an expiry."""
        nearest = self._nearest_expiry(expiry_years)
        seed = self._vols[nearest][len(self._vols[nearest]) // 2]
        return self.vol(expiry_years,
                        FxVolSurface.dns_strike(self.forward_at(expiry_years),
                                                seed, expiry_years,
                                                self._premium_adjusted))

    def forward_at(self, expiry_years: float) -> float:
        """Log-linear interpolated forward, flat outside pillars."""
        ts = self._expiries
        n = len(ts)
        if expiry_years <= ts[0]:
            return self._forwards[0]
        if expiry_years >= ts[n - 1]:
            return self._forwards[n - 1]
        lo = self._bracket(expiry_years)
        w = (expiry_years - ts[lo]) / (ts[lo + 1] - ts[lo])
        return math.exp(math.log(self._forwards[lo])
                        + w * (math.log(self._forwards[lo + 1])
                               - math.log(self._forwards[lo])))

    def pillar(self, i: int) -> SmilePillar:
        """The solved pillar smile at index ``i``."""
        ks = [self._forwards[i] * math.exp(x) for x in self._log_moneyness[i]]
        return SmilePillar(self._expiries[i], self._forwards[i], ks,
                           list(self._vols[i]))

    def pillar_count(self) -> int:
        return len(self._expiries)

    def is_premium_adjusted(self) -> bool:
        return self._premium_adjusted

    def _bracket(self, t: float) -> int:
        """Largest pillar index with expiry <= t (interior t only)."""
        lo, hi = 0, len(self._expiries) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._expiries[mid] <= t:
                lo = mid
            else:
                hi = mid
        return lo

    def _smile_vol(self, i: int, x: float) -> float:
        """Linear interp in log-moneyness within expiry ``i``, flat wings."""
        xs = self._log_moneyness[i]
        vs = self._vols[i]
        if x <= xs[0]:
            return vs[0]
        last = len(xs) - 1
        if x >= xs[last]:
            return vs[last]
        lo = 0
        while xs[lo + 1] < x:
            lo += 1
        w = (x - xs[lo]) / (xs[lo + 1] - xs[lo])
        return vs[lo] + w * (vs[lo + 1] - vs[lo])

    def _nearest_expiry(self, t: float) -> int:
        best = 0
        for i in range(1, len(self._expiries)):
            if abs(self._expiries[i] - t) < abs(self._expiries[best] - t):
                best = i
        return best


def _pa_call_delta(forward: float, strike: float, sv: float) -> float:
    return (strike / forward) * mu.norm_cdf(_d2(forward, strike, sv))


def _d2(forward: float, strike: float, sv: float) -> float:
    return math.log(forward / strike) / sv - 0.5 * sv


def _bisect(f, lo: float, hi: float) -> float:
    """Bisection for a function decreasing over [lo, hi] with a sign
    change."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
