"""Maker-side symmetric last-look price check (port of Java
``trading.LastLookGate``).

The mechanism FX liquidity providers apply to incoming deal requests,
implemented the way the FX Global Code (Principle 17) says it must be:
SYMMETRIC. At the end of the hold window the quoted price is compared
to the current fair price, and the request is rejected when the market
has moved beyond the tolerance in EITHER direction -- protecting the
maker from being picked off, without free-optioning the taker
(accepting only the moves that favor the maker is the asymmetric
practice the Code prohibits).

This class is the decision arithmetic plus its disclosure statistics
(accept/reject counts split by who the reject protected -- the numbers
an LP publishes and a taker's LP scorecard measures from the other
side). The hold window itself belongs to the caller's timer/session
machinery.
"""

from __future__ import annotations

import math


class LastLookGate:
    """Symmetric last-look accept/reject decision; see the module
    docstring."""

    def __init__(self, tolerance: float) -> None:
        """
        Args:
            tolerance: maximum |current fair - quoted| move, in price
                units (e.g. 0.0001 = 1 pip on EURUSD), beyond which the
                request is rejected -- in both directions.
        """
        if tolerance <= 0 or math.isnan(tolerance):
            raise ValueError("tolerance must be positive")
        self._tolerance = tolerance
        self._accepts = 0
        self._rejects = 0
        self._maker_protective_rejects = 0
        self._taker_protective_rejects = 0

    def accept(self, maker_sells: bool, quoted_price: float, current_fair: float) -> bool:
        """The decision at the end of the hold: accept iff the fair
        price is still within tolerance of the quote. Symmetric -- the
        direction of the move never changes the outcome, only the
        statistics.

        Args:
            maker_sells: true when the taker is buying (maker sells at
                the quote).
            quoted_price: the price the maker showed.
            current_fair: the maker's current fair value (e.g.
                composite mid).
        """
        move = current_fair - quoted_price
        if abs(move) <= self._tolerance:
            self._accepts += 1
            return True
        self._rejects += 1
        # Classification only (the decision is already made): a maker
        # who sells is hurt by a rising fair; falling fair means the
        # reject "protected" a taker who would have overpaid.
        hurts_maker = move > 0 if maker_sells else move < 0
        if hurts_maker:
            self._maker_protective_rejects += 1
        else:
            self._taker_protective_rejects += 1
        return False

    def accepts(self) -> int:
        return self._accepts

    def rejects(self) -> int:
        return self._rejects

    def maker_protective_rejects(self) -> int:
        """Rejects where the move was against the maker (the classic
        pick-off)."""
        return self._maker_protective_rejects

    def taker_protective_rejects(self) -> int:
        """Rejects where the move favored the maker -- a symmetric gate
        produces these in roughly equal measure; their absence in an
        LP's disclosures is the signature of asymmetric last look."""
        return self._taker_protective_rejects

    def reject_rate(self) -> float:
        """Reject fraction of all decisions (NaN before any decision)."""
        total = self._accepts + self._rejects
        return math.nan if total == 0 else self._rejects / total
