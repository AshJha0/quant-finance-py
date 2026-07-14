"""Participant-side full-depth (L3) book builder (port of Java
``marketdata.L3BookBuilder``): reconstructs a venue's book from an
ITCH-style event stream (add / execute / cancel / delete / replace) and
answers the questions an execution engine actually asks -- best
bid/ask, depth, and *exactly how many shares are queued ahead of my
order*.

This is the consumer of a venue's L3 feed, not the venue. The Java
original is a dense fixed-capacity tick ladder with pooled intrusive
nodes and an open-addressing ref map (zero allocation per event, a JVM
hot-path concern); this port keeps the exact behavior -- FIFO queue
priority per level, O(1) shares-ahead maintenance, the same
unknown/out-of-band/duplicate-ref counters, idempotent :meth:`track`
-- over plain dicts/lists keyed by tick and by order ref, the Python
idiom for this kind of sparse book.

Queue position
--------------
Call :meth:`track` with your own order's reference (learned from the
order-entry gateway's ack) once its add has appeared on the feed. The
initial shares-ahead is computed by one walk of the level's FIFO; from
then on it is maintained in O(1) per event using two facts of
price-time priority: executions always consume the queue head (so any
execution at your level that isn't you happened ahead of you), and a
cancel is ahead of you iff it entered the queue before you (insertion
sequence numbers). When your order fills, is deleted, or is replaced,
tracking ends automatically.

Order references must be positive (0 would be a natural empty
sentinel, matching real ITCH feeds); non-positive refs, non-positive
share counts, and out-of-band prices are rejected via the out-of-band
counter rather than raising, since these are feed anomalies a
participant must survive, not program bugs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from quantfinlib.marketdata import itch_codec
from quantfinlib.microstructure.execution import Side

_BUY = 0
_SELL = 1
_NONE = -1
_MAX_TRACKED = 64
_INT_MIN = -(1 << 31)
_INT_MAX = (1 << 31) - 1


class _Node:
    __slots__ = ("ref", "side", "tick", "qty", "seq")

    def __init__(self, ref: int, side: int, tick: int, qty: int, seq: int) -> None:
        self.ref = ref
        self.side = side
        self.tick = tick
        self.qty = qty
        self.seq = seq


class L3BookBuilder:
    """Full-depth book for one symbol, driven by an ITCH-style feed."""

    def __init__(self, stock_locate: int, min_price_tick: int,
                max_price_tick: int, max_orders: int) -> None:
        """
        Args:
            stock_locate: the feed's locate code for the symbol this
                book tracks; messages for other locates are ignored by
                :meth:`on_message`.
            min_price_tick: lowest representable price in 0.0001 ticks
                (inclusive).
            max_price_tick: highest representable price in 0.0001
                ticks (inclusive).
            max_orders: resting-order capacity, fixed forever.
        """
        if max_price_tick < min_price_tick or max_orders <= 0:
            raise ValueError(
                "need maxPriceTick >= minPriceTick, maxOrders > 0")
        self._stock_locate = stock_locate
        self._min_tick = min_price_tick
        self._ladder = max_price_tick - min_price_tick + 1
        self._max_orders = max_orders

        self._levels: Dict[int, Dict[int, List[_Node]]] = {_BUY: {}, _SELL: {}}
        self._qty: Dict[int, Dict[int, int]] = {_BUY: {}, _SELL: {}}
        self._occupied: Dict[int, set] = {_BUY: set(), _SELL: set()}
        self._best_bid_idx = _NONE
        self._best_ask_idx = _NONE

        self._orders: Dict[int, _Node] = {}
        self._next_seq = 1

        self._tracked: List[dict] = []

        self._add_count = 0
        self._execute_count = 0
        self._cancel_count = 0
        self._delete_count = 0
        self._replace_count = 0
        self._trade_count = 0
        self._unknown_ref_count = 0
        self._out_of_band_count = 0
        self._duplicate_ref_count = 0
        self._resting_orders = 0
        self._last_trade_tick = _INT_MIN

        self._view = itch_codec.ItchView()

    # ------------------------------------------------------------------
    # Feed entry points
    # ------------------------------------------------------------------

    def on_message(self, buf: bytes, offset: int) -> int:
        """Applies one wire message starting at ``offset``. Returns
        the wire length consumed, or 0 when the message is for another
        stock locate or outside the supported subset."""
        v = self._view.wrap(buf, offset)
        msg_type = v.type()
        ln = itch_codec.length(msg_type)
        if ln < 0 or v.stock_locate() != self._stock_locate:
            return 0
        if msg_type in (itch_codec.ADD, itch_codec.ADD_MPID):
            side = Side.BUY if v.side() == itch_codec.BUY else Side.SELL
            self.on_add(v.order_ref(), side, v.shares(), v.price_tick())
        elif msg_type == itch_codec.EXECUTED:
            self.on_execute(v.order_ref(), v.delta_shares())
        elif msg_type == itch_codec.CANCEL:
            self.on_cancel(v.order_ref(), v.delta_shares())
        elif msg_type == itch_codec.DELETE:
            self.on_delete(v.order_ref())
        elif msg_type == itch_codec.REPLACE:
            self.on_replace(v.orig_ref(), v.new_ref(), v.shares(), v.price_tick())
        elif msg_type == itch_codec.TRADE:
            self.on_trade(v.price_tick())
        else:
            return 0
        return ln

    def on_add(self, ref: int, side: Side, shares: int, price_tick: int) -> bool:
        """Add order: appends to its level's FIFO. False when the
        order was dropped (off-band price, exhausted pool, or a
        duplicate ref -- a feed anomaly that would otherwise corrupt
        the ref map)."""
        s = _BUY if side == Side.BUY else _SELL
        if not self._insert(ref, s, shares, price_tick):
            return False
        self._add_count += 1
        return True

    def _insert(self, ref: int, side: int, shares: int, price_tick: int) -> bool:
        """Places an order without event counting -- shared by add and
        replace. Failures are counted here (out-of-band / duplicate)
        because they mean the same thing on both paths."""
        idx = price_tick - self._min_tick
        if (idx < 0 or idx >= self._ladder or shares <= 0 or ref <= 0
                or len(self._orders) >= self._max_orders):
            self._out_of_band_count += 1
            return False
        if ref in self._orders:
            # Re-delivered add (gap-recovery replay / simulator bug): a
            # blind insert would leave a phantom second node the
            # venue's future delete can never remove.
            self._duplicate_ref_count += 1
            return False
        node = _Node(ref, side, idx, shares, self._next_seq)
        self._next_seq += 1
        self._orders[ref] = node
        self._levels[side].setdefault(idx, []).append(node)
        self._qty[side][idx] = self._qty[side].get(idx, 0) + shares
        self._occupied[side].add(idx)
        if side == _BUY:
            if self._best_bid_idx == _NONE or idx > self._best_bid_idx:
                self._best_bid_idx = idx
        else:
            if self._best_ask_idx == _NONE or idx < self._best_ask_idx:
                self._best_ask_idx = idx
        self._resting_orders += 1
        return True

    def on_execute(self, ref: int, shares: int) -> None:
        """Execution against a resting order (always the queue head
        under price-time priority -- which is what makes O(1) queue
        tracking sound)."""
        if self._reduce(ref, shares, False):
            self._execute_count += 1
        else:
            self._unknown_ref_count += 1

    def on_cancel(self, ref: int, shares: int) -> None:
        """Partial cancel: reduces a resting order in place (keeps its
        priority)."""
        if self._reduce(ref, shares, True):
            self._cancel_count += 1
        else:
            self._unknown_ref_count += 1

    def on_delete(self, ref: int) -> None:
        """Full removal of a resting order."""
        node = self._orders.get(ref)
        if node is None:
            self._unknown_ref_count += 1
            return
        self._delete_count += 1
        self._remove_whole(node)

    def on_replace(self, orig_ref: int, new_ref: int, shares: int, price_tick: int) -> None:
        """Cancel/replace: the original order is removed and the new
        reference joins the back of the (possibly different) level's
        queue -- priority is lost, exactly as on a real venue. A
        replace re-pricing to an off-band level drops the order
        entirely, consistent with off-band adds."""
        node = self._orders.get(orig_ref)
        if node is None:
            self._unknown_ref_count += 1
            return
        self._replace_count += 1
        side = node.side
        self._remove_whole(node)
        self._insert(new_ref, side, shares, price_tick)

    def _reduce(self, ref: int, shares: int, by_seq: bool) -> bool:
        """Shared reduction for executions and cancels: clamps to the
        resting quantity, credits tracked orders, and removes the node
        when it empties."""
        node = self._orders.get(ref)
        if node is None:
            return False
        cut = min(shares, node.qty)
        self._credit_ahead(node, cut, by_seq)
        node.qty -= cut
        self._qty[node.side][node.tick] -= cut
        if node.qty == 0:
            self._remove_node(node)
        return True

    def _remove_whole(self, node: _Node) -> None:
        """Full removal shared by delete and replace: credit, level
        total, unlink."""
        self._credit_ahead(node, node.qty, True)
        self._qty[node.side][node.tick] -= node.qty
        self._remove_node(node)

    def on_trade(self, price_tick: int) -> None:
        """Off-book/non-displayed trade print: records it, book
        unchanged."""
        self._trade_count += 1
        self._last_trade_tick = price_tick

    # ------------------------------------------------------------------
    # Own-order queue tracking
    # ------------------------------------------------------------------

    def track(self, ref: int) -> bool:
        """Starts queue tracking for a resting order (yours, learned
        from your gateway ack). The initial shares-ahead is one FIFO
        walk; maintenance is O(1) per event afterwards. Returns False
        when the ref is unknown or the tracking table is full."""
        for t in self._tracked:
            if t["ref"] == ref:
                return True    # idempotent: a retried ack must not
                                # create a leaking duplicate row
        node = self._orders.get(ref)
        if node is None or len(self._tracked) == _MAX_TRACKED:
            return False
        level = self._levels[node.side].get(node.tick, [])
        ahead = 0
        for n in level:
            if n is node:
                break
            ahead += n.qty
        self._tracked.append({
            "ref": ref, "seq": node.seq, "ahead": ahead,
            "tick": node.tick, "side": node.side,
        })
        return True

    def untrack(self, ref: int) -> None:
        """Stops tracking a ref (no-op when not tracked)."""
        for i, t in enumerate(self._tracked):
            if t["ref"] == ref:
                del self._tracked[i]
                return

    def shares_ahead(self, ref: int) -> int:
        """Shares queued ahead of a tracked order right now; -1 when
        the ref is not tracked (never was, or it filled / was deleted
        / was replaced)."""
        for t in self._tracked:
            if t["ref"] == ref:
                return t["ahead"]
        return -1

    def _credit_ahead(self, node: _Node, qty: int, by_seq: bool) -> None:
        """Reduces shares-ahead for every tracked order that the
        removed/reduced quantity was actually ahead of. Executions
        consume the head, so they are ahead of every other order at
        the level; cancels are ahead only of orders they precede in
        insertion order (``by_seq``)."""
        for t in self._tracked:
            if (t["tick"] == node.tick and t["side"] == node.side
                    and t["ref"] != node.ref
                    and (not by_seq or node.seq < t["seq"])):
                t["ahead"] = max(0, t["ahead"] - qty)

    # ------------------------------------------------------------------
    # Node/level plumbing
    # ------------------------------------------------------------------

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a zero/removed node, clears occupancy, recycles the
        ref map."""
        self._levels[node.side][node.tick].remove(node)
        self.untrack(node.ref)
        del self._orders[node.ref]
        self._resting_orders -= 1

        if self._qty[node.side].get(node.tick, 0) == 0:
            self._occupied[node.side].discard(node.tick)
            self._qty[node.side].pop(node.tick, None)
            self._levels[node.side].pop(node.tick, None)
            if node.side == _BUY and node.tick == self._best_bid_idx:
                occ = self._occupied[_BUY]
                self._best_bid_idx = max(occ) if occ else _NONE
            elif node.side == _SELL and node.tick == self._best_ask_idx:
                occ = self._occupied[_SELL]
                self._best_ask_idx = min(occ) if occ else _NONE

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best_bid_tick(self) -> int:
        """Best bid in absolute 0.0001 ticks; ``INT_MIN`` when none."""
        return self._best_bid_idx + self._min_tick if self._best_bid_idx != _NONE else _INT_MIN

    def best_ask_tick(self) -> int:
        """Best ask in absolute 0.0001 ticks; ``INT_MAX`` when none."""
        return self._best_ask_idx + self._min_tick if self._best_ask_idx != _NONE else _INT_MAX

    def best_bid_size(self) -> int:
        return self._qty[_BUY].get(self._best_bid_idx, 0) if self._best_bid_idx != _NONE else 0

    def best_ask_size(self) -> int:
        return self._qty[_SELL].get(self._best_ask_idx, 0) if self._best_ask_idx != _NONE else 0

    def qty_at_tick(self, side: Side, price_tick: int) -> int:
        """Resting quantity at an absolute tick (0 when off-band or empty)."""
        idx = price_tick - self._min_tick
        if idx < 0 or idx >= self._ladder:
            return 0
        s = _BUY if side == Side.BUY else _SELL
        return self._qty[s].get(idx, 0)

    def open_quantity(self, ref: int) -> int:
        """Open shares of any resting order by ref; 0 when gone/unknown."""
        node = self._orders.get(ref)
        return node.qty if node is not None else 0

    def snapshot(self, side: Side, max_levels: int) -> List[Tuple[int, int]]:
        """Depth snapshot, best-first: a list of (tick, qty) pairs, at
        most ``max_levels`` long."""
        result: List[Tuple[int, int]] = []
        if side == Side.BUY:
            idx = self._best_bid_idx
            occ = self._occupied[_BUY]
            qty = self._qty[_BUY]
            while idx != _NONE and len(result) < max_levels:
                result.append((idx + self._min_tick, qty[idx]))
                idx = self._next_at_or_below(occ, idx - 1)
        else:
            idx = self._best_ask_idx
            occ = self._occupied[_SELL]
            qty = self._qty[_SELL]
            while idx != _NONE and len(result) < max_levels:
                result.append((idx + self._min_tick, qty[idx]))
                idx = self._next_at_or_above(occ, idx + 1)
        return result

    @staticmethod
    def _next_at_or_below(occupied: set, frm: int) -> int:
        candidates = [i for i in occupied if i <= frm]
        return max(candidates) if candidates else _NONE

    @staticmethod
    def _next_at_or_above(occupied: set, frm: int) -> int:
        candidates = [i for i in occupied if i >= frm]
        return min(candidates) if candidates else _NONE

    def last_trade_tick(self) -> int:
        return self._last_trade_tick

    def resting_orders(self) -> int:
        return self._resting_orders

    def add_count(self) -> int:
        return self._add_count

    def execute_count(self) -> int:
        return self._execute_count

    def cancel_count(self) -> int:
        return self._cancel_count

    def delete_count(self) -> int:
        return self._delete_count

    def replace_count(self) -> int:
        return self._replace_count

    def trade_count(self) -> int:
        return self._trade_count

    def unknown_ref_count(self) -> int:
        """Events referencing unknown orders (feed gap symptom --
        resubscribe/snapshot)."""
        return self._unknown_ref_count

    def out_of_band_count(self) -> int:
        """Orders dropped for off-band prices or an exhausted pool --
        adds AND replace re-adds."""
        return self._out_of_band_count

    def duplicate_ref_count(self) -> int:
        """Adds re-delivering a live ref, rejected to protect the book
        (replay symptom)."""
        return self._duplicate_ref_count
