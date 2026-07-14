"""Triangular arbitrage checks over three related FX pairs (port of Java
``com.quantfinlib.pricing.TriangularArbitrage``).

Uses dealable bid/ask quotes (not mids), so a positive result is
executable edge before fees.

Conventions: ``ab`` is the price of one unit of A in B, ``bc`` of one B
in C, ``ac`` of one A in C — e.g. A=EUR, B=USD, C=JPY: ab=EURUSD,
bc=USDJPY, ac=EURJPY.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Quote:
    """A dealable two-way quote; construction rejects crossed markets."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: bid {self.bid} > ask {self.ask}")

    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class TriangularArbitrage:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def arbitrage_bps(ab: Quote, bc: Quote, ac: Quote) -> float:
        """Best executable round-trip edge in basis points (positive = arbitrage):

        * Path 1 — buy A synthetically via B (``ab.ask * bc.ask``) and sell
          it directly at ``ac.bid``.
        * Path 2 — buy A directly at ``ac.ask`` and sell it via B at
          ``ab.bid * bc.bid``.
        """
        synthetic_ask = ab.ask * bc.ask     # cost in C to build one A via B
        synthetic_bid = ab.bid * bc.bid     # proceeds in C unwinding one A via B
        path1 = (ac.bid - synthetic_ask) / synthetic_ask
        path2 = (synthetic_bid - ac.ask) / ac.ask
        return max(path1, path2) * 1e4

    @staticmethod
    def exists(ab: Quote, bc: Quote, ac: Quote, threshold_bps: float) -> bool:
        """True when the executable edge exceeds ``threshold_bps`` (e.g. costs)."""
        return TriangularArbitrage.arbitrage_bps(ab, bc, ac) > threshold_bps

    @staticmethod
    def implied_cross_mid(ab: Quote, bc: Quote) -> float:
        """The no-arbitrage cross rate implied by the two leg mids."""
        return ab.mid() * bc.mid()
