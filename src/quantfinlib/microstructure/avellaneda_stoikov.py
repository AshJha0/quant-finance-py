"""Avellaneda-Stoikov (2008) optimal market-making quotes (port of Java
``trading.AvellanedaStoikov``; ported here into
:mod:`quantfinlib.microstructure` alongside the other quoting/impact
models per this port's package layout). Two closed-form answers from
one utility-maximization problem:

1. **Where is MY mid?** The reservation price shades the market mid
   against your inventory: ``r = mid - q*gamma*sigma^2*tau``. Long
   inventory (q > 0) pushes both quotes down -- you want to sell, so
   you make selling attractive and buying not. The shade grows with
   risk aversion (gamma), variance (sigma^2) and remaining horizon
   (tau): a big position in a wild market you must carry for hours is
   worth shading hard;
2. **How wide?** The optimal total spread
   ``delta = gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/kappa)``
   balances the volatility cost of holding inventory against the
   fill-rate cost of quoting wide, where kappa is the order-arrival
   decay (how fast fill intensity drops as you quote away from the
   touch -- bigger kappa = thicker flow near the mid = quote tighter).
   As gamma -> 0 the spread collapses to the pure liquidity floor
   ``2/kappa``.

**Units contract:**

* ``price_variance_per_second`` -- variance of the PRICE per second,
  not the return: e.g. ``(mid * vol_per_sqrt_second) ** 2``. NaN or
  negative reads as 0 (the inventory term disables; the liquidity
  floor keeps quoting sane);
* ``horizon_seconds`` -- time to the moment inventory must be flat
  (the close, the fixing). Continuous 24/5 FX has no terminal time:
  use a fixed risk horizon (how long you are willing to sit on a
  position) -- the standard practitioner reading of tau;
* ``inventory`` -- signed position in UNITS OF THE INSTRUMENT the mid
  prices (shares of the stock, base-currency units of the pair). This
  is dimensional, not stylistic: the shade ``q*gamma*sigma^2*tau`` is
  in price units only when q counts the same thing sigma^2 is quoted
  per -- pass round lots instead of shares and the shade silently
  shrinks 100x while the spread term (which never sees q) is
  unchanged, gutting the skew that is the model's point. Calibrate
  gamma with the inventory unit fixed first.

Pure static-shape math on primitives. Like every model here, gamma and
kappa deserve calibration against your own fill data before the output
is trusted with size.
"""

from __future__ import annotations

import math


class AvellanedaStoikov:
    """Reservation price and optimal half-spread quoter; see the
    module docstring."""

    __slots__ = ("_gamma", "_kappa", "_liquidity_half_spread")

    def __init__(self, gamma: float, kappa: float) -> None:
        """
        Args:
            gamma: risk aversion, e.g. 0.1 (bigger = shade and widen
                more).
            kappa: fill-intensity decay per unit distance from the
                touch, e.g. 1.5 (bigger = flow concentrates at the
                mid).
        """
        if gamma <= 0 or kappa <= 0:
            raise ValueError("need gamma > 0 and kappa > 0")
        self._gamma = gamma
        self._kappa = kappa
        # log1p keeps the liquidity floor exact as gamma -> 0
        # (log(1+x) rounds to 0 below x ~ 1e-16; log1p does not).
        self._liquidity_half_spread = math.log1p(gamma / kappa) / gamma

    def reservation_price(self, mid: float, inventory: float,
                          price_variance_per_second: float,
                          horizon_seconds: float) -> float:
        """The inventory-shaded fair value:
        ``mid - inventory*gamma*sigma^2*tau``. Flat inventory or a
        dead/garbage variance returns the mid itself."""
        return mid - inventory * self._gamma * self._variance_term(
            price_variance_per_second, horizon_seconds)

    def optimal_half_spread(self, price_variance_per_second: float,
                            horizon_seconds: float) -> float:
        """Half of the optimal total spread:
        ``(gamma*sigma^2*tau)/2 + (1/gamma)*ln(1 + gamma/kappa)``.
        Never below the liquidity floor -- even a becalmed market pays
        for immediacy."""
        return (0.5 * self._gamma * self._variance_term(
                    price_variance_per_second, horizon_seconds)
                + self._liquidity_half_spread)

    def bid_quote(self, mid: float, inventory: float,
                 price_variance_per_second: float,
                 horizon_seconds: float) -> float:
        """The bid to quote: reservation price minus the optimal
        half-spread."""
        return (self.reservation_price(mid, inventory, price_variance_per_second,
                                       horizon_seconds)
                - self.optimal_half_spread(price_variance_per_second, horizon_seconds))

    def ask_quote(self, mid: float, inventory: float,
                 price_variance_per_second: float,
                 horizon_seconds: float) -> float:
        """The ask to quote: reservation price plus the optimal
        half-spread."""
        return (self.reservation_price(mid, inventory, price_variance_per_second,
                                       horizon_seconds)
                + self.optimal_half_spread(price_variance_per_second, horizon_seconds))

    def gamma(self) -> float:
        return self._gamma

    def kappa(self) -> float:
        return self._kappa

    @staticmethod
    def _variance_term(price_variance_per_second: float,
                       horizon_seconds: float) -> float:
        """``sigma^2*tau`` with the gap discipline: non-finite or
        negative inputs read as 0."""
        if (not (price_variance_per_second > 0) or not (horizon_seconds > 0)
                or price_variance_per_second == math.inf
                or horizon_seconds == math.inf):
            return 0.0                      # !(x > 0) also catches NaN
        return price_variance_per_second * horizon_seconds
