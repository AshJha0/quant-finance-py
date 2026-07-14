"""SBE-style binary market/order codecs (port of the flyweight subset
of Java ``com.quantfinlib.sbe``).

Fixed-offset, little-endian flyweights over a caller-owned
``bytearray``: :class:`TradeFlyweight` (trade in), :class:`OrderFlyweight`
(order out), :class:`QuoteFlyweight` (two-sided quote out). Each
message type carries an ``int32`` discriminator at offset 0
(``MESSAGE_TYPE``), so a receiver dispatches on
``TradeFlyweight.type_at(buffer, offset)`` before wrapping the right
flyweight.

The live network adapters (``BinaryMarketDataClient``,
``BinaryOrderPublisher``, ``BinaryOrderReceiver`` -- UDP/TCP wiring
around these codecs) are **not ported**: they have no meaning without
a running network process, out of scope for a library port.
"""

from quantfinlib.sbe.order_flyweight import OrderFlyweight
from quantfinlib.sbe.quote_flyweight import QuoteFlyweight
from quantfinlib.sbe.trade_flyweight import TradeFlyweight

__all__ = [
    "OrderFlyweight",
    "QuoteFlyweight",
    "TradeFlyweight",
]
