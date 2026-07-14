"""Structured notes priced by decomposition (port of Java
``com.quantfinlib.pricing.StructuredNotes``).

Every structured product is a bond plus options in a costume; the
costume is what the issuer charges for. Pricing by replication makes the
margin visible, and the tests ARE the decompositions (each note must
equal its replicating portfolio to machine precision).

* **Reverse convertible** — par bond + fat coupon, but the investor has
  SOLD a put struck at K:
  ``value = (par + coupon) DF(T) - (par/K) put(K)``. The "9% coupon"
  is put premium in disguise. (The knock-in variant needs a down-and-in
  put — stated, not approximated.)
* **Capital-protected note** — ``protection * par`` floor plus
  ``participation`` of the upside; ``participation_for`` inverts the
  pricing for the participation a budget affords.
* **Discount certificate** — the covered call
  ``value = S e^{-qT} - call(cap)``.

Values are per note of face ``par`` (certificate: per unit of
underlying). Research lane, deterministic.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class StructuredNotes:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def reverse_convertible(par: float, coupon_rate: float,
                            spot: float, strike: float, rate: float,
                            carry: float, vol: float, time_years: float) -> float:
        """Fair value of a vanilla reverse convertible of face ``par``.

        Args:
            coupon_rate: total coupon rate for the LIFE of the note
                (0.09 = 9% paid at maturity with the redemption).
            strike: the conversion strike K (shares delivered are worth
                ``S_T/K * par`` when ``S_T < K``).
        """
        _require_common(par, spot, rate, carry, vol, time_years)
        if not (strike > 0) or strike == math.inf:
            raise ValueError("strike must be positive and finite")
        if not (coupon_rate >= 0) or coupon_rate == math.inf:
            raise ValueError("couponRate must be >= 0 and finite")
        df = math.exp(-rate * time_years)
        put = BlackScholes.price(OptionType.PUT, spot, strike, rate, carry, vol, time_years)
        return par * (1 + coupon_rate) * df - (par / strike) * put

    @staticmethod
    def reverse_convertible_delta(par: float, spot: float, strike: float,
                                  rate: float, carry: float, vol: float,
                                  time_years: float) -> float:
        """Delta of the reverse convertible: short put makes the holder LONG the stock."""
        _require_common(par, spot, rate, carry, vol, time_years)
        return -(par / strike) * BlackScholes.delta(OptionType.PUT, spot, strike, rate,
                                                    carry, vol, time_years)

    @staticmethod
    def capital_protected_note(par: float, protection: float, participation: float,
                               spot: float, rate: float, carry: float,
                               vol: float, time_years: float) -> float:
        """Fair value of a capital-protected note: ``protection`` of par
        floored, plus ``participation`` of the underlying's upside from
        ``spot``."""
        _require_common(par, spot, rate, carry, vol, time_years)
        if not (protection > 0) or protection > 1:
            raise ValueError(f"protection must be in (0, 1], got {protection}")
        if not (participation >= 0) or participation == math.inf:
            raise ValueError("participation must be >= 0 and finite")
        df = math.exp(-rate * time_years)
        call = BlackScholes.price(OptionType.CALL, spot, spot, rate, carry, vol, time_years)
        return protection * par * df + participation * (par / spot) * call

    @staticmethod
    def participation_for(par: float, protection: float, issue_price: float,
                          spot: float, rate: float, carry: float,
                          vol: float, time_years: float) -> float:
        """The participation rate a given ISSUE PRICE affords:
        ``(issue_price - protection * par * DF) / ((par/S0) * call)``.

        This is the issuer's product-design equation solved for its one
        free variable — and the reason zero-rate eras produce notes with
        embarrassing participation: the bond floor eats the whole budget.
        """
        _require_common(par, spot, rate, carry, vol, time_years)
        if not (issue_price > 0) or issue_price == math.inf:
            raise ValueError("issuePrice must be positive and finite")
        if not (protection > 0) or protection > 1:
            raise ValueError(f"protection must be in (0, 1], got {protection}")
        df = math.exp(-rate * time_years)
        floor = protection * par * df
        if issue_price <= floor:
            raise ValueError(f"issue price {issue_price} does not even cover the "
                             f"protected floor {floor}: no participation is affordable")
        call = BlackScholes.price(OptionType.CALL, spot, spot, rate, carry, vol, time_years)
        if not (call > 0):
            raise ValueError(
                "ATM call is worthless under these parameters: participation undefined")
        return (issue_price - floor) / ((par / spot) * call)

    @staticmethod
    def discount_certificate(spot: float, cap: float, rate: float,
                             carry: float, vol: float, time_years: float) -> float:
        """Fair value per unit of underlying of a discount certificate capped
        at ``cap``: the covered call ``S e^{-qT} - call(cap)``."""
        _require_common(1, spot, rate, carry, vol, time_years)
        if not (cap > 0) or cap == math.inf:
            raise ValueError("cap must be positive and finite")
        return (spot * math.exp(-carry * time_years)
                - BlackScholes.price(OptionType.CALL, spot, cap, rate, carry, vol, time_years))

    @staticmethod
    def discount_certificate_delta(spot: float, cap: float, rate: float,
                                   carry: float, vol: float, time_years: float) -> float:
        """Delta of the discount certificate: long stock, short call."""
        _require_common(1, spot, rate, carry, vol, time_years)
        return (math.exp(-carry * time_years)
                - BlackScholes.delta(OptionType.CALL, spot, cap, rate, carry, vol, time_years))


def _require_common(par: float, spot: float, rate: float, carry: float,
                    vol: float, time_years: float) -> None:
    if not (par > 0) or par == math.inf:
        raise ValueError("par must be positive and finite")
    if not (spot > 0) or spot == math.inf:
        raise ValueError("spot must be positive and finite")
    if not (math.isfinite(rate) and math.isfinite(carry)):
        raise ValueError("rate and carry must be finite")
    if not (vol >= 0) or vol == math.inf:
        raise ValueError("vol must be >= 0 and finite")
    if not (time_years > 0) or time_years == math.inf:
        raise ValueError("timeYears must be positive and finite")
