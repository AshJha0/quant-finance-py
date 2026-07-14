"""Professional Stock Screener (port of Java ``screener.StockScreener``).

Applies technical and fundamental filters to a universe, optionally
ranks matches, and exports results to CSV.

Survivorship caution: the screener sees exactly the universe it is
given. A universe built from TODAY's constituents has already dropped
every delisted or acquired name -- the bias enters before any filter
runs. For historical screens, construct the snapshot list point-in-time
via :meth:`members_as_of` with a
:class:`~quantfinlib.data.point_in_time_universe.PointInTimeUniverse`
that includes dead tickers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

from quantfinlib.data.point_in_time_universe import PointInTimeUniverse
from quantfinlib.screener.ranking_engine import RankingEngine, ScoredStock
from quantfinlib.screener.screen_filter import ScreenFilter
from quantfinlib.screener.stock_snapshot import StockSnapshot


class StockScreener:
    def __init__(self, universe: Sequence[StockSnapshot]) -> None:
        self._universe: tuple = tuple(universe)

    @staticmethod
    def members_as_of(
        snapshots: Sequence[StockSnapshot],
        universe: PointInTimeUniverse,
        as_of_timestamp: int,
    ) -> List[StockSnapshot]:
        """Filters snapshots to the point-in-time members at
        ``as_of_timestamp`` -- the survivorship-safe way to build a
        historical screening universe (assuming the snapshot list itself
        includes the dead tickers)."""
        return [s for s in snapshots if universe.is_member(s.symbol, as_of_timestamp)]

    def screen(self, *filters: ScreenFilter) -> List[StockSnapshot]:
        """Returns stocks matching every supplied filter."""
        out = []
        for s in self._universe:
            if all(f.matches(s) for f in filters):
                out.append(s)
        return out

    def screen_and_rank(self, ranking: RankingEngine, *filters: ScreenFilter) -> List[ScoredStock]:
        """Screens then ranks the survivors best-first."""
        return ranking.rank(self.screen(*filters))

    @staticmethod
    def export_csv(path: str, results: Sequence[ScoredStock]) -> None:
        """Exports ranked results (symbol, score, last close, fundamentals) to CSV."""
        p = Path(path)
        if p.parent and str(p.parent) not in ("", "."):
            os.makedirs(p.parent, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as w:
            w.write("symbol,score,lastClose,marketCap,peRatio,pbRatio,eps,roe,dividendYield,debtToEquity\n")
            for r in results:
                f = r.stock.fundamentals
                w.write(
                    "%s,%.4f,%.4f,%.0f,%.2f,%.2f,%.2f,%.4f,%.4f,%.2f\n"
                    % (
                        r.stock.symbol,
                        r.score,
                        r.stock.last_close(),
                        f.market_cap,
                        f.pe_ratio,
                        f.pb_ratio,
                        f.eps,
                        f.roe,
                        f.dividend_yield,
                        f.debt_to_equity,
                    )
                )
