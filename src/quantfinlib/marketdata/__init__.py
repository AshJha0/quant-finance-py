"""Market-data feed handling (port of the codec/book/consolidation
subset of Java ``com.quantfinlib.marketdata``).

:mod:`~quantfinlib.marketdata.itch_codec` is the ITCH 5.0-style binary
message codec; :class:`L3BookBuilder` reconstructs a participant-side
full-depth book from that feed, including exact queue-position
tracking; :class:`Nbbo` consolidates per-venue top-of-book quotes into
the national best bid/offer.

The live/streaming bus lane (``HftMarketDataBus``, ``MarketDataProcessor``,
``RingBuffer``/``TickRingBuffer``, ``SymbolRegistry``, ``HistoricalDataStore``,
``MarketDataListener``/``MarketDataEvent``, ``TickListener``) is out of
scope for this port -- it wires the codecs above onto a live network/
disruptor pipeline, which has no meaning outside a running process.
"""

from quantfinlib.marketdata import itch_codec
from quantfinlib.marketdata.l3_book_builder import L3BookBuilder
from quantfinlib.marketdata.nbbo import NO_ASK, NO_BID, Nbbo

__all__ = [
    "L3BookBuilder",
    "NO_ASK",
    "NO_BID",
    "Nbbo",
    "itch_codec",
]
