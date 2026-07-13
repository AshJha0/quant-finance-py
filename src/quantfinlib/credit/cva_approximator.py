"""Unilateral CVA (port of Java ``com.quantfinlib.credit.CvaApproximator``).

The price of the counterparty in every derivative you hold: the
expected loss from their default before your trades' cash flows finish
arriving. The standard discrete approximation the desks carry:

    CVA = LGD * sum_i EE(t_i) * [ Q(t_{i-1}) - Q(t_i) ] * DF(t_i)

expected exposure at each bucket, times the probability of defaulting
IN that bucket (read off the ``CreditCurve``'s survival function),
times the discount factor, times loss-given-default. The three
ingredients are deliberately separate objects: EXPOSURE comes from your
pricing/simulation stack (this class does not know your portfolio),
CREDIT comes from the CDS-bootstrapped curve, DISCOUNT from the
``YieldCurve``.

Approximations, stated: exposure is evaluated at the bucket END and
assumed constant across the bucket (O(dt) bias, shrink the grid to
shrink it); default and exposure are INDEPENDENT — no wrong-way risk,
which UNDERSTATES CVA when exposure grows exactly when the counterparty
weakens (the FX-forward-with-an-EM-sovereign classic); unilateral —
your own default (DVA) is not netted; LGD is a constant you pass,
usually ``1 - recovery`` on the same convention as the curve's
bootstrap, but kept separate because the curve's recovery is a quoting
convention while CVA's LGD is a modeling choice.
"""

from __future__ import annotations

import math

from quantfinlib.credit.credit_curve import CreditCurve
from quantfinlib.rates.yield_curve import YieldCurve


class CvaApproximator:
    """Static discrete unilateral CVA."""

    @staticmethod
    def cva(expected_exposure, bucket_end_years, counterparty: CreditCurve,
            discount: YieldCurve, lgd: float) -> float:
        """Discrete unilateral CVA over the given exposure profile.

        Args:
            expected_exposure: EE(t_i) per bucket, >= 0 (same currency units
                as the answer).
            bucket_end_years: bucket end times t_i in years, strictly
                ascending, all > 0; t_0 = 0 is implicit.
            counterparty: the counterparty's bootstrapped credit curve.
            discount: risk-free discounting curve.
            lgd: loss given default in (0, 1].

        Returns:
            The CVA charge (positive; subtract it from the risk-free PV).
        """
        n = len(expected_exposure)
        if n == 0 or len(bucket_end_years) != n:
            raise ValueError("need aligned, non-empty exposure/time arrays, "
                             f"got {n}/{len(bucket_end_years)}")
        if not (lgd > 0) or not (lgd <= 1):
            raise ValueError(f"lgd must be in (0, 1], got {lgd}")
        prev_t = 0.0
        for i in range(n):
            if not (bucket_end_years[i] > prev_t) or bucket_end_years[i] == math.inf:
                raise ValueError("bucket times must be strictly ascending, "
                                 f"positive and finite: t[{i}]={bucket_end_years[i]}")
            prev_t = bucket_end_years[i]
            if not (expected_exposure[i] >= 0) or expected_exposure[i] == math.inf:
                raise ValueError("expected exposure must be >= 0 and finite: "
                                 f"EE[{i}]={expected_exposure[i]}")
        total = 0.0
        prev_q = 1.0    # Q(0)
        for i in range(n):
            q = counterparty.survival_probability(bucket_end_years[i])
            total += (expected_exposure[i] * (prev_q - q)
                      * discount.discount_factor(bucket_end_years[i]))
            prev_q = q
        return lgd * total
