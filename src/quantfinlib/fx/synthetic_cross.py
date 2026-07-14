"""Direct-versus-synthetic cross execution arithmetic (port of Java
``com.quantfinlib.fx.SyntheticCross``; ``CrossOp`` ports the Java
``CrossRateEngine.Op`` composition — the streaming engine itself is
bus-coupled and not ported).

An FX cross (EURJPY) can be dealt directly or replicated through its
liquid legs, and the cheaper route changes with every quote. Spread
composition is the whole point: a synthetic buy pays the ask on both
legs (MULTIPLY: askA x askB) or the ask of one and the bid of the
other (DIVIDE: askA / bidB) — two half-spreads against one on the
direct route.
"""

from __future__ import annotations

import math
from enum import Enum


class CrossOp(Enum):
    """How two leg prices compose into the cross."""

    MULTIPLY = "multiply"
    """A/B x B/C = A/C (shared middle currency)."""

    DIVIDE = "divide"
    """A/C / B/C = A/B (shared quote currency)."""


def _priced(p: float) -> bool:
    """A dealable price: finite and strictly positive."""
    return p > 0 and p < math.inf


class SyntheticCross:
    """Static route-comparison namespace, mirroring the Java final class."""

    @staticmethod
    def synthetic_ask(op: CrossOp, bid_a: float, ask_a: float,
                      bid_b: float, ask_b: float) -> float:
        """All-in synthetic ASK (cost to BUY the cross via the legs).
        MULTIPLY: buy both legs -> askA x askB. DIVIDE: buy leg A, sell
        leg B -> askA / bidB."""
        return ask_a * ask_b if op is CrossOp.MULTIPLY else ask_a / bid_b

    @staticmethod
    def synthetic_bid(op: CrossOp, bid_a: float, ask_a: float,
                      bid_b: float, ask_b: float) -> float:
        """All-in synthetic BID (proceeds of SELLING via the legs)."""
        return bid_a * bid_b if op is CrossOp.MULTIPLY else bid_a / ask_b

    @staticmethod
    def buy_savings(direct_ask: float, op: CrossOp, bid_a: float,
                    ask_a: float, bid_b: float, ask_b: float) -> float:
        """Savings per unit of buying synthetically instead of directly
        (positive = the legs are cheaper). NaN when either route is
        unpriced — NaN, zero (an empty tier default) or negative — so
        an unquoted book can never masquerade as an attractive route,
        including via a divide-by-zero infinity."""
        if (not _priced(direct_ask) or not _priced(ask_a)
                or not _priced(ask_b if op is CrossOp.MULTIPLY else bid_b)):
            return math.nan
        return direct_ask - SyntheticCross.synthetic_ask(op, bid_a, ask_a,
                                                         bid_b, ask_b)

    @staticmethod
    def sell_savings(direct_bid: float, op: CrossOp, bid_a: float,
                     ask_a: float, bid_b: float, ask_b: float) -> float:
        """Mirror: extra proceeds per unit of selling via the legs
        (positive = legs win)."""
        if (not _priced(direct_bid) or not _priced(bid_a)
                or not _priced(bid_b if op is CrossOp.MULTIPLY else ask_b)):
            return math.nan
        return SyntheticCross.synthetic_bid(op, bid_a, ask_a,
                                            bid_b, ask_b) - direct_bid

    @staticmethod
    def buy_synthetic_wins(direct_ask: float, op: CrossOp, bid_a: float,
                           ask_a: float, bid_b: float, ask_b: float) -> bool:
        """True when buying through the legs beats the direct ask
        (NaN-safe: False)."""
        return SyntheticCross.buy_savings(direct_ask, op, bid_a, ask_a,
                                          bid_b, ask_b) > 0

    @staticmethod
    def sell_synthetic_wins(direct_bid: float, op: CrossOp, bid_a: float,
                            ask_a: float, bid_b: float, ask_b: float) -> bool:
        """True when selling through the legs beats the direct bid
        (NaN-safe: False)."""
        return SyntheticCross.sell_savings(direct_bid, op, bid_a, ask_a,
                                           bid_b, ask_b) > 0
