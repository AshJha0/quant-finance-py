"""Private-market analytics (port of Java
``com.quantfinlib.markets.PrivateMarketAnalytics``).

The toolkit for the asset class where the usual machinery fails on
purpose: no daily prices, cash flows the manager (not the investor)
times, and NAVs that are appraisals rather than trades.

* IRR — the money-weighted return: the rate that zeroes the NPV of the
  fund's cash flows plus terminal NAV. It rewards the manager's TIMING
  (which a time-weighted return deliberately ignores), which is why PE
  quotes IRR and mutual funds may not. Solved by bisection with an
  explicit sign-change/bracket check — cash flows that never change
  sign have no IRR, and this raises rather than inventing one.
* Multiples — TVPI = (distributions + NAV)/contributions,
  DPI = distributions/contributions (the "cash back" ratio),
  RVPI = NAV/contributions (the part still an appraisal). DPI is the
  honest one: you cannot spend RVPI.
* Kaplan-Schoar PME — the public-market equivalent: grow every
  contribution and distribution forward at the INDEX's return and take
  ``(FV(distributions) + NAV) / FV(contributions)``. PME > 1 means the
  fund beat just buying the index with the same cash flows on the same
  dates — the only fair benchmark for irregular cash flows, and the
  reason "our IRR beat the S&P's return" is not evidence.
* Geltner desmoothing — appraisal NAVs are AR(1)-smoothed versions of
  true returns (``r_obs_t = (1-phi) r_true_t + phi r_obs_{t-1}``),
  which UNDERSTATES volatility and correlation to public markets
  ("volatility laundering"). Inverting,
  ``r_true_t = (r_obs_t - phi r_obs_{t-1}) / (1 - phi)``, recovers a
  series whose risk numbers can sit honestly next to public-market
  ones. The inversion is exact: smoothing then desmoothing round-trips
  to machine precision (pinned).

Cash-flow sign convention throughout: contributions (money in)
NEGATIVE, distributions (money out) POSITIVE — the investor's
perspective, matching every spreadsheet's XIRR. Period-indexed flows
(annual/quarterly — caller's choice, IRR is per period).
"""

from __future__ import annotations

import math

import numpy as np


class PrivateMarketAnalytics:
    """Static IRR, fund multiples, KS-PME and Geltner desmoothing."""

    @staticmethod
    def irr(cashflows) -> float:
        """Money-weighted return per period: solves
        ``sum cf_t / (1+irr)^t = 0``. The final period's cash flow should
        include terminal NAV as a distribution.

        Args:
            cashflows: period-indexed, index 0 = today; must contain at
                least one negative and one positive flow.
        """
        if len(cashflows) < 2:
            raise ValueError(f"need >= 2 cash flows, got {len(cashflows)}")
        has_neg = False
        has_pos = False
        for cf in cashflows:
            if not math.isfinite(cf):
                raise ValueError("non-finite cash flow")
            has_neg = has_neg or cf < 0
            has_pos = has_pos or cf > 0
        if not has_neg or not has_pos:
            raise ValueError("cash flows never change sign: no IRR exists")
        lo, hi = -0.9999, 100.0
        npv_lo = _npv(cashflows, lo)
        npv_hi = _npv(cashflows, hi)
        if npv_lo * npv_hi > 0:
            raise ValueError("no IRR in (-99.99%, 10000%)")
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            v = _npv(cashflows, mid)
            if v * npv_lo > 0:
                lo = mid
                npv_lo = v
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def tvpi(contributions: float, distributions: float, nav: float) -> float:
        """TVPI: total value (distributions + NAV) to paid-in."""
        _validate_multiples(contributions, distributions, nav)
        return (distributions + nav) / contributions

    @staticmethod
    def dpi(contributions: float, distributions: float, nav: float) -> float:
        """DPI: realized distributions to paid-in — the cash-back multiple."""
        _validate_multiples(contributions, distributions, nav)
        return distributions / contributions

    @staticmethod
    def rvpi(contributions: float, distributions: float, nav: float) -> float:
        """RVPI: remaining (appraised) value to paid-in."""
        _validate_multiples(contributions, distributions, nav)
        return nav / contributions

    @staticmethod
    def ks_pme(contributions, distributions, terminal_nav: float,
               index_levels) -> float:
        """Kaplan-Schoar PME. Arrays are period-aligned with ``index_levels``
        (same length); contributions/distributions are the POSITIVE amounts
        flowing in each period.

        Returns:
            PME; > 1 = fund beat the index on its own cash-flow dates.
        """
        n = len(index_levels)
        if n < 2 or len(contributions) != n or len(distributions) != n:
            raise ValueError("need aligned series of length >= 2")
        last = index_levels[n - 1]
        fv_contrib = 0.0
        fv_distrib = 0.0
        for t in range(n):
            if not (index_levels[t] > 0) or index_levels[t] == math.inf:
                raise ValueError(f"index level must be positive: {index_levels[t]}")
            if (not (contributions[t] >= 0) or not (distributions[t] >= 0)
                    or contributions[t] == math.inf or distributions[t] == math.inf):
                raise ValueError("flows must be >= 0 and finite (amounts, not signed)")
            growth = last / index_levels[t]
            fv_contrib += contributions[t] * growth
            fv_distrib += distributions[t] * growth
        if not (terminal_nav >= 0) or terminal_nav == math.inf:
            raise ValueError("terminalNav must be >= 0 and finite")
        if fv_contrib == 0:
            raise ValueError("no contributions: PME undefined")
        return (fv_distrib + terminal_nav) / fv_contrib

    @staticmethod
    def geltner_desmooth(observed_returns, phi: float) -> np.ndarray:
        """Geltner desmoothing: inverts AR(1) appraisal smoothing with
        parameter ``phi`` in [0, 1). Element 0 is kept as observed (no lag
        exists for it).
        """
        if len(observed_returns) < 2:
            raise ValueError("need >= 2 returns")
        if not (phi >= 0) or not (phi < 1):
            raise ValueError(f"phi must be in [0, 1), got {phi}")
        r = np.empty(len(observed_returns))
        r[0] = _require_finite(observed_returns[0])
        for t in range(1, len(r)):
            now = _require_finite(observed_returns[t])
            r[t] = (now - phi * observed_returns[t - 1]) / (1 - phi)
        return r


def _npv(cashflows, rate: float) -> float:
    v = 0.0
    for t in range(len(cashflows)):
        v += cashflows[t] / (1 + rate) ** t
    return v


def _validate_multiples(contributions: float, distributions: float,
                        nav: float) -> None:
    if not (contributions > 0) or contributions == math.inf:
        raise ValueError("contributions must be positive and finite")
    if (not (distributions >= 0) or distributions == math.inf
            or not (nav >= 0) or nav == math.inf):
        raise ValueError("distributions and nav must be >= 0 and finite")


def _require_finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("non-finite return")
    return v
