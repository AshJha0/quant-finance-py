"""Non-deliverable forward (port of Java ``com.quantfinlib.fx.Ndf``).

The FX forward for restricted currencies: cash-settles in the
deliverable (base, usually USD) currency against an official fixing
published before settlement. Settlement to the base buyer:

    amount = base_notional * (fixing - contract_rate) / fixing

— the division by the fixing converts the quote-currency difference
back into deliverable currency.
"""

from __future__ import annotations

import datetime as dt

from quantfinlib.fx.currency_pair import CurrencyPair
from quantfinlib.fx.swap_points_curve import SwapPointsCurve
from quantfinlib.rates.yield_curve import YieldCurve

# Fixing lags (local business days between fixing and settlement) for
# the common NDF currencies; PTAX fixes one day before settlement, the
# Asian fixings two. Unlisted currencies default to 2.
_FIXING_LAG = {"INR": 2, "KRW": 2, "TWD": 2, "IDR": 2, "MYR": 2,
               "PHP": 1, "CNY": 2, "VND": 2, "BRL": 1, "CLP": 1}


class Ndf:

    def __init__(self, pair: CurrencyPair, base_notional: float,
                 contract_rate: float, fixing_date: dt.date,
                 settlement_date: dt.date):
        if contract_rate <= 0 or base_notional == 0:
            raise ValueError("contract_rate must be > 0 and notional non-zero")
        if fixing_date >= settlement_date:
            raise ValueError("fixing must precede settlement")
        self._pair = pair
        self._base_notional = base_notional
        self._contract_rate = contract_rate
        self._fixing_date = fixing_date
        self._settlement_date = settlement_date

    @staticmethod
    def of_tenor(pair: CurrencyPair, trade_date: dt.date, tenor: str,
                 contract_rate: float, base_notional: float) -> "Ndf":
        """Books an NDF at a market tenor: settlement from the pair's
        tenor arithmetic, fixing walked back by the restricted (quote)
        currency's lag counted in LOCAL (quote-calendar) business days
        — an RBI/KFTC/PTAX fixing publishes on its local business days
        regardless of USD holidays."""
        settlement = pair.tenor_date(trade_date, tenor)
        fixing = pair.quote_calendar().subtract_business_days(
            settlement, _FIXING_LAG.get(pair.quote(), 2))
        return Ndf(pair, base_notional, contract_rate, fixing, settlement)

    @staticmethod
    def of(pair: CurrencyPair, base_notional: float, contract_rate: float,
           fixing_date: dt.date, settlement_date: dt.date) -> "Ndf":
        """Explicit dates (broken dates, historical bookings)."""
        return Ndf(pair, base_notional, contract_rate, fixing_date,
                   settlement_date)

    @staticmethod
    def fixing_lag_days(currency: str) -> int:
        """The fixing lag booked for a restricted currency code."""
        return _FIXING_LAG.get(currency, 2)

    # ------------------------------------------------------------------
    # Settlement and valuation
    # ------------------------------------------------------------------

    def settlement_amount(self, fixing_rate: float) -> float:
        """Cash settlement in base (deliverable) currency once the
        official fixing publishes. Positive pays the base buyer."""
        if fixing_rate <= 0:
            raise ValueError(f"fixing_rate must be > 0: {fixing_rate}")
        return (self._base_notional
                * (fixing_rate - self._contract_rate) / fixing_rate)

    def mark_to_market(self, current: SwapPointsCurve,
                       base_discount: YieldCurve | None = None) -> float:
        """Mark-to-market in base currency: the settlement formula at
        the curve's forward to the FIXING date. Inside the fixing
        window (fixing at or before the curve's spot) the mark degrades
        to the spot outright instead of throwing mid-lifecycle. With
        ``base_discount``, discounted from the settlement date on a
        base-currency zero curve, ACT/365 from the curve's spot."""
        spot = current.spot_date()
        forward_estimate = (current.outright(self._fixing_date)
                            if self._fixing_date > spot
                            else current.outright(spot))
        mtm = self.settlement_amount(forward_estimate)
        if base_discount is None:
            return mtm
        t = (self._settlement_date - spot).days / 365.0
        return mtm * (base_discount.discount_factor(t) if t > 0 else 1.0)

    def pair(self) -> CurrencyPair:
        return self._pair

    def base_notional(self) -> float:
        return self._base_notional

    def contract_rate(self) -> float:
        return self._contract_rate

    def fixing_date(self) -> dt.date:
        return self._fixing_date

    def settlement_date(self) -> dt.date:
        return self._settlement_date

    def __repr__(self) -> str:
        return (f"{self._pair.symbol()} NDF {self._base_notional} "
                f"{self._pair.base()} @{self._contract_rate} "
                f"fix {self._fixing_date} settle {self._settlement_date}")
