"""CDS pricing (port of Java ``com.quantfinlib.credit.CdsPricer``).

Prices off a ``CreditCurve``: the two legs, the par spread, and the
upfront that post-2009 standardized contracts actually exchange.

A credit default swap is insurance with a running premium: the
protection BUYER pays ``spread`` per year (quarterly, accruing to the
default date) and receives ``1 - R`` of notional if the name defaults
before maturity. The pricing identities, per unit notional:

    riskyAnnuity  = sum dt * DF(t_i) * Q(t_i)  +  accrual-on-default term
    premiumLegPv  = spread * riskyAnnuity
    protectionPv  = (1 - R) * sum DF(t_i) * (Q(t_{i-1}) - Q(t_i))
    parSpread     = protectionPv / riskyAnnuity     (zero-upfront spread)
    upfront(S_c)  = protectionPv - S_c * riskyAnnuity

Positive upfront means the protection buyer pays points up front (the
contract's fixed coupon ``S_c`` is too small for the risk — the
standard 100bp coupon on a 300bp name). The risky annuity is also the
desk's "risky DV01": the PnL per 1bp of spread move, which is why it
gets its own accessor rather than living inside the leg. Same quarterly
discretization as the bootstrap, stated there.
"""

from __future__ import annotations

import math

from quantfinlib.credit.credit_curve import CreditCurve
from quantfinlib.rates.yield_curve import YieldCurve


class CdsPricer:
    """Static CDS leg/par-spread/upfront pricers on the quarterly grid."""

    @staticmethod
    def risky_annuity(credit: CreditCurve, discount: YieldCurve,
                      maturity_years: float) -> float:
        """PV of 1-per-year premium stream per unit spread (the risky annuity
        / risky DV01 base).
        """
        _validate(maturity_years)
        dt = CreditCurve.grid_step()
        annuity = 0.0
        prev_q = 1.0
        t = dt
        while t <= maturity_years + 1e-12:
            q = credit.survival_probability(t)
            df = discount.discount_factor(t)
            # Full period if it survives; half a period of accrual if it
            # defaults inside the period (the standard convention).
            annuity += dt * df * q + 0.5 * dt * df * (prev_q - q)
            prev_q = q
            t += dt
        return annuity

    @staticmethod
    def premium_leg_pv(credit: CreditCurve, discount: YieldCurve,
                       spread: float, maturity_years: float) -> float:
        """PV of the premium leg at the given running spread."""
        if not (spread > 0) or spread == math.inf:
            raise ValueError(f"spread must be positive and finite, got {spread}")
        return spread * CdsPricer.risky_annuity(credit, discount, maturity_years)

    @staticmethod
    def protection_leg_pv(credit: CreditCurve, discount: YieldCurve,
                          maturity_years: float) -> float:
        """PV of the protection leg: (1-R) paid at default."""
        _validate(maturity_years)
        dt = CreditCurve.grid_step()
        pv = 0.0
        prev_q = 1.0
        t = dt
        while t <= maturity_years + 1e-12:
            q = credit.survival_probability(t)
            pv += discount.discount_factor(t) * (prev_q - q)
            prev_q = q
            t += dt
        return (1 - credit.recovery()) * pv

    @staticmethod
    def par_spread(credit: CreditCurve, discount: YieldCurve,
                   maturity_years: float) -> float:
        """The zero-upfront (par) spread for this maturity."""
        return (CdsPricer.protection_leg_pv(credit, discount, maturity_years)
                / CdsPricer.risky_annuity(credit, discount, maturity_years))

    @staticmethod
    def upfront(credit: CreditCurve, discount: YieldCurve,
                contract_spread: float, maturity_years: float) -> float:
        """Upfront points (per unit notional) the protection BUYER pays on a
        contract with fixed running coupon ``contract_spread``.
        """
        if not (contract_spread > 0) or contract_spread == math.inf:
            raise ValueError(
                f"contractSpread must be positive and finite, got {contract_spread}")
        return (CdsPricer.protection_leg_pv(credit, discount, maturity_years)
                - contract_spread * CdsPricer.risky_annuity(credit, discount,
                                                            maturity_years))


def _validate(maturity_years: float) -> None:
    if not (maturity_years > 0) or maturity_years == math.inf:
        raise ValueError(
            f"maturityYears must be positive and finite, got {maturity_years}")
    # The legs are summed on the quarterly grid starting at grid_step();
    # a maturity below one grid step has NO coupon dates, so the annuity
    # and protection leg are both empty -- par_spread would be 0/0 = NaN.
    # Raise instead of leaking NaN (house rule).
    if maturity_years < CreditCurve.grid_step():
        raise ValueError(f"maturityYears must be >= the pricing grid step "
                         f"{CreditCurve.grid_step()}, got {maturity_years}")
