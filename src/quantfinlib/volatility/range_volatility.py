"""Port of Java ``com.quantfinlib.volatility.RangeVolatility``.

RANGE-BASED volatility estimators — the free lunch hiding inside every
OHLC bar: the high-low range carries far more information about the
day's variance than the close alone, so a range estimator reaches a
given precision with several times fewer bars than close-to-close.
Four classics, in increasing order of what they use:

* Parkinson (1980) — range only:
  ``sigma^2 = mean( ln(H/L)^2 ) / (4 ln 2)``. About 4.9x more efficient
  than close-to-close under driftless GBM; biased UP by drift (it books
  trend as range) and DOWN by discrete sampling.
* Garman-Klass (1980) — range plus open/close:
  ``sigma^2 = mean( 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 )``.
  Roughly 7.4x efficient; still assumes zero drift.
* Rogers-Satchell (1991) — drift-INDEPENDENT:
  ``sigma^2 = mean( ln(H/C) ln(H/O) + ln(L/C) ln(L/O) )``.
  The one to reach for on trending series.
* Yang-Zhang (2000) — adds the OVERNIGHT gap the others ignore:
  ``sigma^2 = var_o + k var_c + (1-k) rs`` with ``var_o`` the sample
  variance of open-over-prior-close log returns, ``var_c`` the sample
  variance of close-over-open log returns, ``rs`` the Rogers-Satchell
  term, and ``k = 0.34 / (1.34 + (m+1)/(m-1))`` over m periods — the
  weighting that minimizes the estimator's variance. Drift independent
  AND jump-aware; the practical default for daily bars on markets that
  close.

All methods return ANNUALIZED volatility (not variance):
``sqrt(per_period_variance * periods_per_year)``, with the
annualization factor supplied by the caller (252 for daily bars, 52 for
weekly, ...) — this class does not choose your calendar. Estimates are
computed over the full arrays passed in; slice for rolling windows.
(The Java BarSeries overloads have no Python counterpart — pass the
O/H/L/C arrays directly.)
"""

from __future__ import annotations

import math

import numpy as np

_FOUR_LN_2 = 4 * math.log(2)
_GK_CLOSE_WEIGHT = 2 * math.log(2) - 1


class RangeVolatility:
    """Static Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang."""

    @staticmethod
    def parkinson(high, low, periods_per_year: float) -> float:
        """Parkinson estimator from highs/lows, annualized."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        _check_aligned(high.shape[0], low.shape[0], 1)
        _check_periods(periods_per_year)
        _check_bars(high, high, low, high)  # H/L only
        hl = np.log(high / low)
        return math.sqrt(float(np.sum(hl * hl)) / high.shape[0]
                         / _FOUR_LN_2 * periods_per_year)

    @staticmethod
    def garman_klass(open_, high, low, close, periods_per_year: float) -> float:
        """Garman-Klass estimator, annualized."""
        open_, high, low, close = _check_ohlc(open_, high, low, close, 1)
        _check_periods(periods_per_year)
        hl = np.log(high / low)
        co = np.log(close / open_)
        s = float(np.sum(0.5 * hl * hl - _GK_CLOSE_WEIGHT * co * co))
        # The close term can push a single bar negative; the average of a
        # valid sample cannot reasonably be, but floor for safety.
        return math.sqrt(max(0.0, s / open_.shape[0]) * periods_per_year)

    @staticmethod
    def rogers_satchell(open_, high, low, close, periods_per_year: float) -> float:
        """Rogers-Satchell (drift-independent) estimator, annualized."""
        open_, high, low, close = _check_ohlc(open_, high, low, close, 1)
        _check_periods(periods_per_year)
        s = float(np.sum(_rs_term(open_, high, low, close)))
        return math.sqrt(s / open_.shape[0] * periods_per_year)

    @staticmethod
    def yang_zhang(open_, high, low, close, periods_per_year: float) -> float:
        """Yang-Zhang estimator, annualized.

        Uses bars ``1..n-1`` as the estimation periods (bar 0 only
        supplies the prior close for the first overnight return), so it
        needs at least 3 bars for the sample variances to exist.
        """
        open_, high, low, close = _check_ohlc(open_, high, low, close, 3)
        _check_periods(periods_per_year)
        m = open_.shape[0] - 1  # estimation periods
        # Overnight (close-to-open) and open-to-close log returns.
        o_ret = np.log(open_[1:] / close[:-1])
        c_ret = np.log(close[1:] / open_[1:])
        mean_o = float(np.sum(o_ret)) / m
        mean_c = float(np.sum(c_ret)) / m
        do = o_ret - mean_o
        dc = c_ret - mean_c
        var_o = float(np.sum(do * do)) / (m - 1)
        var_c = float(np.sum(dc * dc)) / (m - 1)
        rs = float(np.sum(_rs_term(open_[1:], high[1:], low[1:], close[1:]))) / m
        k = 0.34 / (1.34 + (m + 1.0) / (m - 1.0))
        return math.sqrt((var_o + k * var_c + (1 - k) * rs) * periods_per_year)


# ----------------------------------------------------------------------


def _rs_term(o, h, l, c):
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def _check_ohlc(open_, high, low, close, min_bars: int):
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    _check_aligned(open_.shape[0], high.shape[0], min_bars)
    _check_aligned(open_.shape[0], low.shape[0], min_bars)
    _check_aligned(open_.shape[0], close.shape[0], min_bars)
    _check_bars(open_, high, low, close)
    return open_, high, low, close


def _check_aligned(a: int, b: int, min_bars: int) -> None:
    if a != b:
        raise ValueError(f"O/H/L/C arrays must be aligned: {a} vs {b}")
    if a < min_bars:
        raise ValueError(f"need >= {min_bars} bars, got {a}")


def _check_bars(o, h, l, c) -> None:
    # NaN-rejecting: any NaN fails one of these comparisons.
    bad = (~(l > 0) | ~(h >= l) | (h == np.inf)
           | ~(o >= l) | ~(o <= h) | ~(c >= l) | ~(c <= h))
    if bool(np.any(bad)):
        i = int(np.argmax(bad))
        raise ValueError(
            f"bar {i} violates H >= O,C >= L > 0: O={o[i]} H={h[i]} "
            f"L={l[i]} C={c[i]}")


def _check_periods(periods_per_year: float) -> None:
    if not (periods_per_year > 0) or periods_per_year == np.inf:
        raise ValueError(
            f"periodsPerYear must be positive and finite, got {periods_per_year}")
