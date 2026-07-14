"""The standard alpha factor library (port of Java ``alpha.Factors``) —
nine signal generators covering the classic technical, factor-investing
and defensive families. Each returns an :class:`AlphaFactor` producing
raw cross-sectional scores where **higher = more attractive long**.

Sign conventions are chosen so every factor is usable long/short as-is:

* **Trend** (MA crossover, MACD, momentum) score positively when the
  trend is up;
* **Mean reversion** (RSI, Bollinger, mean reversion) score positively
  when the price is *depressed* — the contrarian orientation, since
  these signals bet on the snap-back;
* **Defensive/fundamental** (value, quality, low volatility) score
  positively for cheap, profitable, calm names — the Fama-French/AQR
  orientation of each anomaly.

All computations read only bars ``<= index`` (the no-look-ahead
contract) and cost O(window) per symbol per call — stateless by design
so factors are trivially safe to evaluate at arbitrary dates. EMAs are
truncated at 4x their period, where the dropped tail weight is
``(1 - 2/(p+1))^{4p} < 0.04%`` — far below signal noise.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext, Fundamentals
from quantfinlib.alpha.alpha_factor import AlphaFactor
from quantfinlib.data.bar_series import BarSeries


class _Named(AlphaFactor):
    """Wraps per-symbol window math into the cross-sectional contract."""

    def __init__(self, name: str, min_bars: int,
                 per_symbol: Callable[[BarSeries, int], float]) -> None:
        self._name = name
        self._min_bars = min_bars
        self._per_symbol = per_symbol

    def scores(self, ctx: AlphaContext, index: int) -> np.ndarray:
        out = np.empty(ctx.symbol_count())
        for i in range(ctx.symbol_count()):
            if index < self._min_bars:
                # NaN below the warm-up: downstream steps skip it.
                out[i] = math.nan
            elif not ctx.is_active(i, index):
                # Point-in-time universe gate: dead/non-member names
                # never enter the cross-section (see AlphaContext).
                out[i] = math.nan
            else:
                out[i] = self._per_symbol(ctx.series(i), index)
        return out

    def name(self) -> str:
        return self._name


class _Fundamental(AlphaFactor):
    """Wraps a fundamentals read; NaN for symbols without a snapshot."""

    def __init__(self, name: str,
                 fn: Callable[[Fundamentals], float]) -> None:
        self._name = name
        self._fn = fn

    def scores(self, ctx: AlphaContext, index: int) -> np.ndarray:
        out = np.empty(ctx.symbol_count())
        for i in range(ctx.symbol_count()):
            fu = ctx.fundamentals(i)
            out[i] = (math.nan if fu is None or not ctx.is_active(i, index)
                      else self._fn(fu))
        return out

    def name(self) -> str:
        return self._name


def _sma(s: BarSeries, index: int, period: int) -> float:
    closes = s.closes()
    return float(np.mean(closes[index - period + 1:index + 1]))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class Factors:
    """Static factory methods; see the module docstring."""

    # ------------------------------------------------------------------
    # Trend family
    # ------------------------------------------------------------------

    @staticmethod
    def moving_average_crossover(fast: int, slow: int) -> AlphaFactor:
        """Moving-average crossover: ``(SMA_fast - SMA_slow) / SMA_slow``.
        Positive when the fast average rides above the slow — the %
        spread makes scores comparable across price levels."""
        _require(fast > 0 and slow > fast, "need slow > fast > 0")

        def score(s: BarSeries, i: int) -> float:
            f = _sma(s, i, fast)
            sl = _sma(s, i, slow)
            return (f - sl) / sl

        return _Named(f"MA_CROSS({fast},{slow})", slow, score)

    @staticmethod
    def macd(fast: int, slow: int, signal: int) -> AlphaFactor:
        """MACD histogram normalized by price:
        ``(macd_line - signal_line) / close``. Positive while bullish
        momentum is accelerating; the price normalization keeps a $10
        and a $1000 stock on one scale."""
        _require(fast > 0 and slow > fast and signal > 0,
                 "need slow > fast > 0, signal > 0")
        # Warm-up: one slow window plus the signal window. The truncated
        # EMA self-seeds from whatever history exists, so longer waits
        # buy only the sub-0.04% truncation tail.
        warmup = slow + signal

        def score(s: BarSeries, i: int) -> float:
            # Single forward pass carrying both price EMAs and the
            # signal EMA together; the start clamp at bar 0 means the
            # seeds are real prices, never fabricated zeros.
            start = max(0, i - 4 * slow + 1)
            k_fast = 2.0 / (fast + 1)
            k_slow = 2.0 / (slow + 1)
            k_signal = 2.0 / (signal + 1)
            ema_fast = s.close(start)
            ema_slow = s.close(start)
            signal_line = math.nan
            macd_line = 0.0
            for j in range(start, i + 1):
                if j > start:
                    close = s.close(j)
                    ema_fast += k_fast * (close - ema_fast)
                    ema_slow += k_slow * (close - ema_slow)
                macd_line = ema_fast - ema_slow
                # The signal EMA seeds from the first MACD value.
                signal_line = (macd_line if math.isnan(signal_line)
                               else signal_line
                               + k_signal * (macd_line - signal_line))
            return (macd_line - signal_line) / s.close(i)

        return _Named(f"MACD({fast},{slow},{signal})", warmup, score)

    @staticmethod
    def momentum(lookback: int, skip: int) -> AlphaFactor:
        """Cross-sectional momentum:
        ``close[i-skip] / close[i-lookback] - 1``. The academic 12-1
        form when called as ``momentum(252, 21)`` — skipping the last
        month sidesteps short-term reversal (Jegadeesh-Titman 1993)."""
        _require(lookback > 0 and 0 <= skip < lookback,
                 "need lookback > skip >= 0")
        return _Named(
            f"MOMENTUM({lookback}-{skip})", lookback,
            lambda s, i: s.close(i - skip) / s.close(i - lookback) - 1)

    # ------------------------------------------------------------------
    # Mean-reversion family (contrarian sign: depressed prices score high)
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(period: int) -> AlphaFactor:
        """Contrarian RSI: ``(50 - RSI) / 50`` in [-1, +1]. Oversold
        names (RSI 30 -> +0.4) score positively.

        Definition note: this is *Cutler's* RSI (arithmetic average of
        gains/losses over the window), chosen because it is stateless
        and exactly recomputable at any bar. It is NOT the Wilder-
        smoothed RSI from the indicators package — after a trend the
        two can disagree near the 30/70 thresholds. The factor name
        says so to keep reports unambiguous."""
        _require(period > 0, "need period > 0")

        def score(s: BarSeries, i: int) -> float:
            gain = 0.0
            loss = 0.0
            for j in range(i - period + 1, i + 1):
                change = s.close(j) - s.close(j - 1)
                if change > 0:
                    gain += change
                else:
                    loss -= change
            if gain + loss == 0:
                return 0.0  # flat window: neither overbought nor oversold
            rsi_val = 100 * gain / (gain + loss)
            return (50 - rsi_val) / 50

        return _Named(f"RSI_CUTLER_REV({period})", period + 1, score)

    @staticmethod
    def bollinger(period: int, std_devs: float) -> AlphaFactor:
        """Bollinger mean reversion: ``-(close - SMA) / (k*sigma)`` —
        the negative band position, +1 at the lower band, -1 at the
        upper."""
        _require(period > 1 and std_devs > 0,
                 "need period > 1 and stdDevs > 0")

        def score(s: BarSeries, i: int) -> float:
            window = s.closes()[i - period + 1:i + 1]
            mean = float(np.mean(window))
            sd = math.sqrt(float(np.mean((window - mean) ** 2)))
            if sd == 0:
                return 0.0  # flat window: no band to revert within
            return -(s.close(i) - mean) / (std_devs * sd)

        return _Named(f"BOLL_REV({period},{std_devs})", period, score)

    @staticmethod
    def mean_reversion(lookback: int) -> AlphaFactor:
        """Plain mean reversion: ``-(close / SMA - 1)`` — how far the
        price sits below its own average, as a fraction."""
        _require(lookback > 0, "need lookback > 0")
        return _Named(
            f"MEAN_REV({lookback})", lookback,
            lambda s, i: -(s.close(i) / _sma(s, i, lookback) - 1))

    # ------------------------------------------------------------------
    # Fundamental / defensive family
    # ------------------------------------------------------------------

    @staticmethod
    def value() -> AlphaFactor:
        """Value composite: the average of earnings yield (``1/PE``) and
        book yield (``1/PB``) — yields, not ratios, so "cheap" is high
        and negative-earnings names contribute a negative yield rather
        than a meaningless negative PE rank. NaN without fundamentals."""

        def score(f: Fundamentals) -> float:
            total = 0.0
            n = 0
            if not math.isnan(f.pe_ratio) and f.pe_ratio != 0:
                total += 1 / f.pe_ratio
                n += 1
            if not math.isnan(f.pb_ratio) and f.pb_ratio != 0:
                total += 1 / f.pb_ratio
                n += 1
            return math.nan if n == 0 else total / n

        return _Fundamental("VALUE", score)

    @staticmethod
    def quality() -> AlphaFactor:
        """Quality composite: profitability minus leverage —
        ``ROE - 0.1 * debt/equity``. The 0.1 haircut puts one turn of
        leverage on the same scale as 10 points of ROE, the usual
        quality-minus-junk shape (profitable AND conservatively
        financed)."""

        def score(f: Fundamentals) -> float:
            leverage = 0.0 if math.isnan(f.debt_to_equity) else f.debt_to_equity
            return (math.nan if math.isnan(f.roe)
                    else f.roe - 0.1 * leverage)

        return _Fundamental("QUALITY", score)

    @staticmethod
    def low_volatility(lookback: int) -> AlphaFactor:
        """Low-volatility anomaly: ``-sigma(returns)`` over the lookback
        — calm names score high. Ranking only needs the negative sign,
        not annualization."""
        _require(lookback > 1, "need lookback > 1")

        def score(s: BarSeries, i: int) -> float:
            closes = s.closes()
            r = (closes[i - lookback + 1:i + 1]
                 / closes[i - lookback:i] - 1)
            mean = float(np.mean(r))
            return -math.sqrt(float(np.mean((r - mean) ** 2)))

        return _Named(f"LOW_VOL({lookback})", lookback + 1, score)
