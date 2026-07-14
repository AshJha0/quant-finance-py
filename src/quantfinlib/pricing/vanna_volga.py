"""Vanna-volga smile-consistent pricing (port of Java
``com.quantfinlib.pricing.VannaVolga``).

The FX desk's standard smile adjustment built from exactly three market
pillars (25-delta put, ATM, 25-delta call). Hedge the flat-vol
Black-Scholes price's vega, vanna and volga with a portfolio of the
three pillars; the market cost of that hedge is the smile adjustment.
The classic log-strike weight form used here makes the construction
exact AT the pillars — pricing a pillar strike returns the pillar's own
market vol — and interpolates smoothly between and beyond them::

    price(K) = BS(K; vol_atm) + sum_i w_i(K) * [BS(K_i; vol_i) - BS(K_i; vol_atm)]
    w_1(K) = vega(K)/vega(K_1) * ln(K2/K) ln(K3/K) / (ln(K2/K1) ln(K3/K1))

(cyclic for w2, w3). ``implied_vol`` inverts the adjusted price back
through Black-Scholes, giving a full smile from three quotes.
Conventions match ``BlackScholes``.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class VannaVolga:
    """Three-pillar smile pricer; immutable after construction."""

    def __init__(self, strikes, vols, rate: float, carry: float,
                 time_years: float) -> None:
        """
        Args:
            strikes: three ascending pillar strikes (25d put, ATM, 25d call).
            vols: their market vols; vols[1] is the ATM anchor.
        """
        strikes = [float(k) for k in strikes]
        vols = [float(v) for v in vols]
        if len(strikes) != 3 or len(vols) != 3:
            raise ValueError("exactly three pillars required")
        if not (strikes[0] < strikes[1] < strikes[2]):
            raise ValueError("strikes must be strictly ascending")
        for v in vols:
            if v <= 0:
                raise ValueError("vols must be > 0")
        if time_years <= 0:
            raise ValueError("timeYears must be > 0")
        self._strikes = strikes
        self._vols = vols
        self._rate = rate
        self._carry = carry
        self._time_years = time_years

    @staticmethod
    def of_pillars(strikes, vols, rate: float, carry: float,
                   time_years: float) -> "VannaVolga":
        """Builds directly from a solved FX-surface pillar set (25-delta triple)."""
        if len(strikes) == 5:
            # Five-pillar smile (10-delta wings present): vanna-volga uses
            # the classic 25-delta triple — indices 1, 2, 3.
            return VannaVolga([strikes[1], strikes[2], strikes[3]],
                              [vols[1], vols[2], vols[3]], rate, carry, time_years)
        return VannaVolga(strikes, vols, rate, carry, time_years)

    def price(self, option_type: OptionType, spot: float, strike: float) -> float:
        """Smile-consistent price of a vanilla at any strike."""
        if spot <= 0 or strike <= 0:
            raise ValueError("spot and strike must be > 0")
        atm = self._vols[1]
        r, q, t = self._rate, self._carry, self._time_years
        base = BlackScholes.price(option_type, spot, strike, r, q, atm, t)
        vega_k = BlackScholes.vega(spot, strike, r, q, atm, t)
        adjustment = 0.0
        for i in range(3):
            vega_i = BlackScholes.vega(spot, self._strikes[i], r, q, atm, t)
            # Pillar hedge cost: market vol vs flat ATM vol at the pillar.
            # OptionType is irrelevant to the vol difference (put-call parity
            # makes call and put vega identical); use calls throughout.
            market_i = BlackScholes.price(OptionType.CALL, spot, self._strikes[i], r, q,
                                          self._vols[i], t)
            flat_i = BlackScholes.price(OptionType.CALL, spot, self._strikes[i], r, q,
                                        atm, t)
            adjustment += self._weight(i, strike, vega_k, vega_i) * (market_i - flat_i)
        return base + adjustment

    def implied_vol(self, spot: float, strike: float) -> float:
        """Smile-consistent implied vol: the vanna-volga price inverted
        through Black-Scholes. Pillar strikes recover their market vols
        exactly."""
        # Calls are numerically safest OTM-forward and equivalent by parity.
        p = self.price(OptionType.CALL, spot, strike)
        return BlackScholes.implied_vol(OptionType.CALL, p, spot, strike,
                                        self._rate, self._carry, self._time_years)

    def _weight(self, i: int, k: float, vega_k: float, vega_i: float) -> float:
        """The classic log-strike interpolation weights (exact at the pillars)."""
        x = math.log(k)
        x1 = math.log(self._strikes[0])
        x2 = math.log(self._strikes[1])
        x3 = math.log(self._strikes[2])
        if i == 0:
            num = (x2 - x) * (x3 - x)
            den = (x2 - x1) * (x3 - x1)
        elif i == 1:
            num = (x - x1) * (x3 - x)
            den = (x2 - x1) * (x3 - x2)
        else:
            num = (x - x1) * (x - x2)
            den = (x3 - x1) * (x3 - x2)
        return (vega_k / vega_i) * (num / den)
