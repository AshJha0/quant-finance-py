"""Rates volatility products (port of Java ``com.quantfinlib.rates.RatesOptions``).

Priced off the curve — the bridge between ``YieldCurve`` (where
forwards and discount factors live) and Black-76 (the market-standard
lognormal quoter for anything written on a forward rate).

* Swaption — an option on a forward-starting swap. The underlying is
  the forward swap rate ``F = (DF(start) - DF(end)) / annuity``, the
  natural numeraire is the ANNUITY (the PV of the fixed leg's
  1%-per-year), and the price is simply
  ``annuity * Black76(F, K, vol, expiry)`` — discounting lives entirely
  in the annuity, which is why the Black-76 leg is called with rate 0.
  A PAYER swaption (right to pay fixed) is a CALL on the swap rate: it
  pays when rates rise.
* Cap / floor — a strip of independent options (caplets), one per
  accrual period, each a Black-76 call (put) on that period's SIMPLE
  forward rate ``f_i = (DF(t_{i-1})/DF(t_i) - 1)/tau``, fixing at the
  period START and paying at its end. The first period's rate is
  already fixed today, so its caplet is pure intrinsic — included, as
  the market convention includes it unless the trade says otherwise.

Two identities keep this honest (both pinned by tests):
payer - receiver = annuity*(F - K) (swaption put-call parity), and
cap - floor = the PV of the matching vanilla swap. Annual fixed legs
and accrual periods (tau = 1), matching
``YieldCurve.bootstrap_annual_par_swaps``; one flat lognormal vol per
product (no smile).

Python port note: the Java class calls ``pricing.Black76``, which is
outside this port's scope, so the discounted Black-76 formula it needs
(price only, including the T <= 0 / vol <= 0 discounted-intrinsic
branch) is transcribed here as a module-private helper.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve
from quantfinlib.util import math_utils


def _black76_price(is_call: bool, forward: float, strike: float,
                   rate: float, vol: float, time_years: float) -> float:
    """Discounted Black-76 price of a call/put on a forward.

    Faithful transcription of Java ``pricing.Black76.price`` (only the
    price is needed here): at ``time_years <= 0`` or ``vol <= 0`` the
    option is worth its discounted intrinsic value.
    """
    if time_years <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return math.exp(-rate * max(time_years, 0.0)) * intrinsic
    df = math.exp(-rate * time_years)
    sqrt_t = math.sqrt(time_years)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if is_call:
        return df * (forward * math_utils.norm_cdf(d1)
                     - strike * math_utils.norm_cdf(d2))
    return df * (strike * math_utils.norm_cdf(-d2)
                 - forward * math_utils.norm_cdf(-d1))


class RatesOptions:
    """Static Black-76 swaption and cap/floor pricers off the curve."""

    @staticmethod
    def annuity(curve: YieldCurve, start_years: int, tenor_years: int) -> float:
        """PV of 1 per year paid annually over (start_years, start_years+tenor_years]."""
        _validate(start_years, tenor_years)
        a = 0.0
        for i in range(1, tenor_years + 1):
            a += curve.discount_factor(start_years + i)
        return a

    @staticmethod
    def forward_swap_rate(curve: YieldCurve, start_years: int,
                          tenor_years: int) -> float:
        """Forward par swap rate for a swap starting at ``start_years``."""
        _validate(start_years, tenor_years)
        df_start = curve.discount_factor(start_years)
        df_end = curve.discount_factor(start_years + tenor_years)
        return (df_start - df_end) / RatesOptions.annuity(curve, start_years,
                                                          tenor_years)

    @staticmethod
    def swaption(curve: YieldCurve, start_years: int, tenor_years: int,
                 strike: float, vol: float, payer: bool) -> float:
        """Black-76 swaption price per unit notional.

        Args:
            payer: True = payer swaption (call on the swap rate).
            vol: flat lognormal vol of the forward swap rate.
        """
        _require_positive_vol_and_strike(strike, vol)
        fsr = RatesOptions.forward_swap_rate(curve, start_years, tenor_years)
        a = RatesOptions.annuity(curve, start_years, tenor_years)
        return a * _black76_price(payer, fsr, strike, 0.0, vol, start_years)

    @staticmethod
    def cap(curve: YieldCurve, maturity_years: int, strike: float,
            vol: float) -> float:
        """Cap: strip of annual Black-76 caplets to ``maturity_years``."""
        return _caplet_strip(curve, maturity_years, strike, vol, is_call=True)

    @staticmethod
    def floor(curve: YieldCurve, maturity_years: int, strike: float,
              vol: float) -> float:
        """Floor: the matching strip of floorlets."""
        return _caplet_strip(curve, maturity_years, strike, vol, is_call=False)


def _caplet_strip(curve: YieldCurve, maturity_years: int, strike: float,
                  vol: float, is_call: bool) -> float:
    if maturity_years < 1:
        raise ValueError(f"maturityYears must be >= 1, got {maturity_years}")
    _require_positive_vol_and_strike(strike, vol)
    pv = 0.0
    for i in range(1, maturity_years + 1):
        df_pay = curve.discount_factor(i)
        simple_forward = curve.discount_factor(i - 1) / df_pay - 1
        # Fixes at i-1 (the first period is already fixed: T=0 -> Black76
        # returns discounted intrinsic, i.e. pure intrinsic here since its
        # own rate argument is 0), pays at i.
        pv += df_pay * _black76_price(is_call, simple_forward, strike, 0.0,
                                      vol, i - 1)
    return pv


def _validate(start_years: int, tenor_years: int) -> None:
    if start_years < 0 or tenor_years < 1:
        raise ValueError(
            f"need startYears >= 0 and tenorYears >= 1: {start_years}/{tenor_years}")


def _require_positive_vol_and_strike(strike: float, vol: float) -> None:
    if not (strike > 0) or strike == math.inf:
        raise ValueError(f"strike must be positive and finite, got {strike}")
    if not (vol > 0) or vol == math.inf:
        raise ValueError(f"vol must be positive and finite, got {vol}")
