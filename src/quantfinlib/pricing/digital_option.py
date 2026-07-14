"""European digital (binary) options under Black-Scholes (port of Java
``com.quantfinlib.pricing.DigitalOption``).

Parameter conventions match ``BlackScholes``: ``rate`` is the domestic
(quote-currency) rate, ``carry`` the continuous yield on the underlying
— the foreign rate for FX (Garman-Kohlhagen), the dividend yield for
equities.

* **Cash-or-nothing**: pays a fixed amount if the option finishes in the
  money — ``payout * e^{-rt} * N(+-d2)``. The market-standard
  "European digital".
* **Asset-or-nothing**: pays the underlying itself —
  ``S * e^{-qt} * N(+-d1)``. A vanilla decomposes exactly into
  asset-or-nothing minus strike x cash-or-nothing, which the tests
  assert.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import OptionType
from quantfinlib.util import math_utils as mu


class DigitalOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def cash_or_nothing(option_type: OptionType, spot: float, strike: float,
                        rate: float, carry: float, vol: float,
                        time_years: float, payout: float) -> float:
        """Fixed payout if spot finishes beyond the strike (call: above, put: below)."""
        _validate(spot, strike, vol, time_years)
        if time_years == 0:
            itm = spot > strike if option_type is OptionType.CALL else spot < strike
            return payout if itm else 0.0
        d2 = _d2(spot, strike, rate, carry, vol, time_years)
        df = math.exp(-rate * time_years)
        return payout * df * mu.norm_cdf(option_type.sign() * d2)

    @staticmethod
    def asset_or_nothing(option_type: OptionType, spot: float, strike: float,
                         rate: float, carry: float, vol: float,
                         time_years: float) -> float:
        """Pays one unit of the underlying if the option finishes in the money."""
        _validate(spot, strike, vol, time_years)
        if time_years == 0:
            itm = spot > strike if option_type is OptionType.CALL else spot < strike
            return spot if itm else 0.0
        d1 = _d2(spot, strike, rate, carry, vol, time_years) + vol * math.sqrt(time_years)
        return spot * math.exp(-carry * time_years) * mu.norm_cdf(option_type.sign() * d1)


def _d2(spot: float, strike: float, rate: float, carry: float,
        vol: float, t: float) -> float:
    return ((math.log(spot / strike) + (rate - carry - 0.5 * vol * vol) * t)
            / (vol * math.sqrt(t)))


def _validate(spot: float, strike: float, vol: float, t: float) -> None:
    if spot <= 0 or strike <= 0 or vol <= 0 or t < 0:
        raise ValueError("spot, strike, vol must be > 0 and timeYears >= 0")
