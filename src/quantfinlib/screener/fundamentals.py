"""Fundamental data record (port of Java ``screener.Fundamentals``)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Fundamental data for one instrument.

    Use ``float('nan')`` for unknown fields; fundamental filters never
    match on NaN values (NaN comparisons are always false, in Python as
    in Java).
    """

    market_cap: float
    pe_ratio: float
    pb_ratio: float
    eps: float
    roe: float
    dividend_yield: float
    debt_to_equity: float

    @staticmethod
    def unknown() -> "Fundamentals":
        nan = math.nan
        return Fundamentals(nan, nan, nan, nan, nan, nan, nan)
