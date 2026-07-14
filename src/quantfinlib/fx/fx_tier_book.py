"""Tiered multi-LP FX book (port of Java ``com.quantfinlib.fx.FxTierBook``).

Equities venues publish one anonymous order book; FX liquidity
providers each stream a private ladder of size tiers -- 1M at
1.08500/02, 5M at 1.08498/04, 10M wider still -- and the practical
questions are per-clip: what does 20M actually cost if swept
(``sweep_buy_cost``/``sweep_sell_proceeds``), and which single LP fills
the whole clip at one price (``best_full_amount_ask``/
``best_full_amount_bid`` -- the full-amount convention that avoids
signaling the market with a spray of children).

``AggregatedBook`` remains the top-of-book composite; this class holds
the tiers underneath it. Tiers are expected best-first per LP (tier 0 =
tightest); sizes are per-tier clip capacity; prices are the all-in rate
for a clip up to that size. Dealable = strictly positive: 0.0 (an
uninitialized/empty tier) and NaN both read as "no quote".
"""

from __future__ import annotations

import math


class FxTierBook:

    def __init__(self, lp_count: int, max_tiers: int):
        if lp_count < 1 or max_tiers < 1:
            raise ValueError("need lp_count >= 1, max_tiers >= 1")
        self._lp_count = lp_count
        self._max_tiers = max_tiers
        n = lp_count * max_tiers
        self._bid_px = [0.0] * n
        self._bid_sz = [0.0] * n
        self._ask_px = [0.0] * n
        self._ask_sz = [0.0] * n
        self._bid_tiers = [0] * lp_count
        self._ask_tiers = [0] * lp_count
        self._last_full_amount_price = math.nan
        self._update_count = 0

    # ------------------------------------------------------------------
    # Feed side
    # ------------------------------------------------------------------

    def tier(self, lp: int, bid: bool, tier: int, price: float, size: float) -> None:
        """Replaces one tier of one LP's ladder (``tier < max_tiers``).
        Call :meth:`tier_count` after the last tier of an update so
        partially written ladders are never visible to later queries."""
        if lp < 0 or lp >= self._lp_count or tier < 0 or tier >= self._max_tiers:
            raise ValueError(
                f"lp/tier out of range: lp={lp} tier={tier} "
                f"(lp_count={self._lp_count}, max_tiers={self._max_tiers})")
        i = lp * self._max_tiers + tier
        if bid:
            self._bid_px[i] = price
            self._bid_sz[i] = size
        else:
            self._ask_px[i] = price
            self._ask_sz[i] = size

    def tier_count(self, lp: int, bid: bool, count: int) -> None:
        """Declares how many tiers of ``lp``'s side are now active (0
        pulls the side)."""
        if count < 0 or count > self._max_tiers:
            raise ValueError(f"tier count out of range: {count}")
        if bid:
            self._bid_tiers[lp] = count
        else:
            self._ask_tiers[lp] = count
        self._update_count += 1

    def clear(self, lp: int) -> None:
        """Pulls an LP entirely (disconnect / last-look withdrawal)."""
        self._bid_tiers[lp] = 0
        self._ask_tiers[lp] = 0
        self._update_count += 1

    # ------------------------------------------------------------------
    # Composite queries
    # ------------------------------------------------------------------

    def best_bid(self) -> float:
        """Best (highest) bid across LPs, taken at each LP's frontier
        tier. NaN when nobody bids."""
        best = math.nan
        for lp in range(self._lp_count):
            t = self._frontier(lp, self._bid_px, self._bid_sz, self._bid_tiers[lp], 0)
            if t < self._bid_tiers[lp]:
                p = self._bid_px[lp * self._max_tiers + t]
                if math.isnan(best) or p > best:
                    best = p
        return best

    def best_ask(self) -> float:
        """Best (lowest) ask across LPs at each LP's frontier tier; NaN
        when nobody offers."""
        best = math.nan
        for lp in range(self._lp_count):
            t = self._frontier(lp, self._ask_px, self._ask_sz, self._ask_tiers[lp], 0)
            if t < self._ask_tiers[lp]:
                p = self._ask_px[lp * self._max_tiers + t]
                if math.isnan(best) or p < best:
                    best = p
        return best

    def sweep_buy_cost(self, size: float) -> float:
        """All-in cost of BUYING ``size`` by sweeping ask tiers across
        LPs, cheapest tier first. NaN when the book cannot fill the
        size."""
        return self._sweep(size, True, None)

    def sweep_sell_proceeds(self, size: float) -> float:
        """Mirror: proceeds of SELLING ``size`` into the bid tiers; NaN
        if unfillable."""
        return self._sweep(size, False, None)

    def sweep_plan(self, buy: bool, size: float, out_lp_qty: list[float]) -> float:
        """Sweep with a plan: ``out_lp_qty[lp]`` receives the quantity
        taken from each LP (length >= lp_count, fully overwritten)."""
        return self._sweep(size, buy, out_lp_qty)

    def _sweep(self, size: float, buy: bool, out_lp_qty: list[float] | None) -> float:
        if out_lp_qty is not None:
            for lp in range(self._lp_count):
                out_lp_qty[lp] = 0.0
        if size <= 0:
            return math.nan
        px = self._ask_px if buy else self._bid_px
        sz = self._ask_sz if buy else self._bid_sz
        tiers = self._ask_tiers if buy else self._bid_tiers
        sweep_tier = [0] * self._lp_count
        sweep_rem = [0.0] * self._lp_count
        for lp in range(self._lp_count):
            sweep_tier[lp] = self._frontier(lp, px, sz, tiers[lp], 0)
            sweep_rem[lp] = (sz[lp * self._max_tiers + sweep_tier[lp]]
                             if sweep_tier[lp] < tiers[lp] else 0.0)
        remaining = size
        notional = 0.0
        while remaining > 0:
            best = -1
            best_px = 0.0
            for lp in range(self._lp_count):
                if sweep_tier[lp] >= tiers[lp]:
                    continue
                p = px[lp * self._max_tiers + sweep_tier[lp]]
                if best == -1 or (p < best_px if buy else p > best_px):
                    best = lp
                    best_px = p
            if best == -1:
                return math.nan
            take = min(remaining, sweep_rem[best])
            sweep_rem[best] -= take
            notional += take * best_px
            remaining -= take
            if out_lp_qty is not None:
                out_lp_qty[best] += take
            if sweep_rem[best] <= 0:
                sweep_tier[best] = self._frontier(best, px, sz, tiers[best],
                                                  sweep_tier[best] + 1)
                if sweep_tier[best] < tiers[best]:
                    sweep_rem[best] = sz[best * self._max_tiers + sweep_tier[best]]
        return notional

    def _frontier(self, lp: int, px: list[float], sz: list[float], n: int,
                 from_: int) -> int:
        """First tier at or after ``from_`` with a dealable price and
        positive size; ``n`` = exhausted."""
        base = lp * self._max_tiers
        for t in range(from_, n):
            if px[base + t] > 0 and sz[base + t] > 0:
                return t
        return n

    def best_full_amount_ask(self, size: float) -> float:
        """Best single-LP full-amount ASK for ``size``: the lowest tier
        price whose clip capacity covers the whole size at one LP. NaN
        when no LP quotes the size."""
        return self._full_amount(size, True)

    def best_full_amount_bid(self, size: float) -> float:
        return self._full_amount(size, False)

    def best_full_amount_ask_lp(self, size: float) -> int:
        """LP index behind :meth:`best_full_amount_ask`; -1 when none."""
        return self._full_amount_lp(size, True)

    def best_full_amount_bid_lp(self, size: float) -> int:
        return self._full_amount_lp(size, False)

    def _full_amount(self, size: float, buy: bool) -> float:
        self._full_amount_lp(size, buy)
        return self._last_full_amount_price

    def _full_amount_lp(self, size: float, buy: bool) -> int:
        best_lp = -1
        best_px = 0.0
        for lp in range(self._lp_count):
            p = self.full_amount_price(lp, buy, size)
            if math.isnan(p):
                continue
            if best_lp == -1 or (p < best_px if buy else p > best_px):
                best_lp = lp
                best_px = p
        self._last_full_amount_price = math.nan if best_lp == -1 else best_px
        return best_lp

    def full_amount_price(self, lp: int, buy: bool, size: float) -> float:
        """One LP's full-amount price for a clip: the tightest tier
        whose clip capacity covers ``size``, NaN when the LP doesn't
        quote it."""
        if size <= 0:
            return math.nan
        base = lp * self._max_tiers
        n = self._ask_tiers[lp] if buy else self._bid_tiers[lp]
        px = self._ask_px if buy else self._bid_px
        sz = self._ask_sz if buy else self._bid_sz
        for t in range(n):
            if sz[base + t] >= size and px[base + t] > 0:
                return px[base + t]
        return math.nan

    def lp_count(self) -> int:
        return self._lp_count

    def max_tiers(self) -> int:
        return self._max_tiers

    def tier_count_of(self, lp: int, bid: bool) -> int:
        return self._bid_tiers[lp] if bid else self._ask_tiers[lp]

    def price(self, lp: int, bid: bool, tier: int) -> float:
        i = lp * self._max_tiers + tier
        return self._bid_px[i] if bid else self._ask_px[i]

    def size(self, lp: int, bid: bool, tier: int) -> float:
        i = lp * self._max_tiers + tier
        return self._bid_sz[i] if bid else self._ask_sz[i]

    def update_count(self) -> int:
        return self._update_count
