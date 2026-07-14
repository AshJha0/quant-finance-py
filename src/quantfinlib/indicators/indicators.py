"""Port of Java ``com.quantfinlib.indicators.Indicators``.

Technical Indicator Engine: the standard technical analysis toolkit.
All indicators operate on arrays and return NumPy arrays (or small
records of arrays) aligned with the input: index i of the output
corresponds to bar i, with NaN for warm-up bars.

Included: RSI, SMA, EMA, WMA, VWAP, MACD, ATR, ADX, CCI, ROC, Momentum,
OBV, CMF, SuperTrend, Ichimoku Cloud, Stochastic RSI, Williams %R,
Parabolic SAR, Bollinger Bands, Keltner Channel, Donchian Channel.

The Java methods that take a ``BarSeries`` take the raw O/H/L/C/volume
arrays here (there is no BarSeries container in the Python port).
Stateless window sweeps are vectorized with NumPy; every smoothing
recurrence (EMA seeding, Wilder RSI/ATR/ADX, SuperTrend, Parabolic SAR)
and the MACD signal construction (single pass over the valid MACD line
only — no NaN pre-history biasing the seed) are transcribed exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantfinlib.util import math_utils as mu


@dataclass(frozen=True)
class Macd:
    line: np.ndarray
    signal: np.ndarray
    histogram: np.ndarray


@dataclass(frozen=True)
class Bollinger:
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray


@dataclass(frozen=True)
class Adx:
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray


@dataclass(frozen=True)
class SuperTrend:
    """direction: +1 = uptrend (value is support), -1 = downtrend
    (value is resistance)."""
    value: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class Ichimoku:
    tenkan: np.ndarray
    kijun: np.ndarray
    senkou_a: np.ndarray
    senkou_b: np.ndarray
    chikou: np.ndarray


@dataclass(frozen=True)
class StochRsi:
    k: np.ndarray
    d: np.ndarray


@dataclass(frozen=True)
class Keltner:
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray


@dataclass(frozen=True)
class Donchian:
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray


class Indicators:
    """Static batch indicator engine (Java parity)."""

    # Result records re-exported on the class, mirroring Java's nesting.
    Macd = Macd
    Bollinger = Bollinger
    Adx = Adx
    SuperTrend = SuperTrend
    Ichimoku = Ichimoku
    StochRsi = StochRsi
    Keltner = Keltner
    Donchian = Donchian

    # ------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------

    @staticmethod
    def sma(v, period: int) -> np.ndarray:
        """Simple moving average."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] < period:
            return out
        out[period - 1:] = sliding_window_view(v, period).sum(axis=1) / period
        return out

    @staticmethod
    def ema(v, period: int) -> np.ndarray:
        """Exponential moving average, seeded with the SMA of the first
        ``period`` values."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] < period:
            return out
        prev = float(np.sum(v[:period])) / period
        out[period - 1] = prev
        k = 2.0 / (period + 1)
        for i in range(period, v.shape[0]):
            prev += (v[i] - prev) * k
            out[i] = prev
        return out

    @staticmethod
    def wma(v, period: int) -> np.ndarray:
        """Linearly weighted moving average."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] < period:
            return out
        denom = period * (period + 1) / 2.0
        weights = np.arange(1, period + 1, dtype=float)  # oldest 1 .. newest p
        out[period - 1:] = sliding_window_view(v, period) @ weights / denom
        return out

    # ------------------------------------------------------------------
    # Momentum / oscillators
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(v, period: int) -> np.ndarray:
        """Relative Strength Index (Wilder smoothing)."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] <= period:
            return out
        gain = 0.0
        loss = 0.0
        for i in range(1, period + 1):
            d = v[i] - v[i - 1]
            if d > 0:
                gain += d
            else:
                loss -= d
        avg_gain = gain / period
        avg_loss = loss / period
        out[period] = _to_rsi(avg_gain, avg_loss)
        for i in range(period + 1, v.shape[0]):
            d = v[i] - v[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
            out[i] = _to_rsi(avg_gain, avg_loss)
        return out

    @staticmethod
    def roc(v, period: int) -> np.ndarray:
        """Rate of change, percent: ``(v[i] / v[i-period] - 1) * 100``."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        out[period:] = (v[period:] / v[:-period] - 1) * 100
        return out

    @staticmethod
    def momentum(v, period: int) -> np.ndarray:
        """Momentum: ``v[i] - v[i-period]``."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        out[period:] = v[period:] - v[:-period]
        return out

    @staticmethod
    def macd(close, fast_period: int, slow_period: int, signal_period: int) -> Macd:
        """MACD: EMA(fast) - EMA(slow), with an EMA signal line and
        histogram. The signal EMA runs single-pass over the valid MACD
        line only, so the NaN warm-up never biases its seed."""
        close = np.asarray(close, dtype=float)
        fast = Indicators.ema(close, fast_period)
        slow = Indicators.ema(close, slow_period)
        n = close.shape[0]
        line = fast - slow  # NaN while either EMA is warming up
        signal = mu.nan_array(n)
        hist = mu.nan_array(n)
        start = slow_period - 1
        if start < n:
            valid = line[start:].copy()
            sig_valid = Indicators.ema(valid, signal_period)
            signal[start:] = sig_valid
            hist[start:] = line[start:] - sig_valid
        return Macd(line, signal, hist)

    @staticmethod
    def stochastic_rsi(close, rsi_period: int, stoch_period: int,
                       k_smooth: int, d_smooth: int) -> StochRsi:
        """Stochastic RSI: stochastic oscillator applied to RSI, with %K
        and %D smoothing."""
        close = np.asarray(close, dtype=float)
        r = Indicators.rsi(close, rsi_period)
        n = close.shape[0]
        raw = mu.nan_array(n)
        for i in range(rsi_period + stoch_period - 1, n):
            hi = -math.inf
            lo = math.inf
            for j in range(i - stoch_period + 1, i + 1):
                if not math.isnan(r[j]):
                    hi = max(hi, r[j])
                    lo = min(lo, r[j])
            raw[i] = (r[i] - lo) / (hi - lo) * 100 if hi > lo else 50.0
        k = _smooth_ignoring_nan(raw, k_smooth)
        d = _smooth_ignoring_nan(k, d_smooth)
        return StochRsi(k, d)

    @staticmethod
    def williams_r(high, low, close, period: int) -> np.ndarray:
        """Williams %R: ``-100 * (highestHigh - close) / (highestHigh -
        lowestLow)``."""
        _check_period(period)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        out = mu.nan_array(n)
        if n < period:
            return out
        hh = sliding_window_view(high, period).max(axis=1)
        ll = sliding_window_view(low, period).min(axis=1)
        c = close[period - 1:]
        out[period - 1:] = np.where(hh > ll, -100 * (hh - c) / (hh - ll), -50.0)
        return out

    @staticmethod
    def cci(high, low, close, period: int) -> np.ndarray:
        """Commodity Channel Index over typical price."""
        _check_period(period)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        tp = (high + low + close) / 3.0
        out = mu.nan_array(n)
        if n < period:
            return out
        windows = sliding_window_view(tp, period)
        m = windows.sum(axis=1) / period
        dev = np.abs(windows - m[:, None]).sum(axis=1) / period
        centered = tp[period - 1:] - m
        out[period - 1:] = np.where(dev == 0, 0.0,
                                    centered / np.where(dev == 0, 1.0, 0.015 * dev))
        return out

    # ------------------------------------------------------------------
    # Volatility / range
    # ------------------------------------------------------------------

    @staticmethod
    def true_range(high, low, close) -> np.ndarray:
        """True range series."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        if n > 1:
            hl = high[1:] - low[1:]
            hc = np.abs(high[1:] - close[:-1])
            lc = np.abs(low[1:] - close[:-1])
            tr[1:] = np.maximum(hl, np.maximum(hc, lc))
        return tr

    @staticmethod
    def atr(high, low, close, period: int) -> np.ndarray:
        """Average True Range (Wilder smoothing)."""
        _check_period(period)
        tr = Indicators.true_range(high, low, close)
        n = tr.shape[0]
        out = mu.nan_array(n)
        if n < period:
            return out
        prev = float(np.sum(tr[:period])) / period
        out[period - 1] = prev
        for i in range(period, n):
            prev = (prev * (period - 1) + tr[i]) / period
            out[i] = prev
        return out

    @staticmethod
    def adx(high, low, close, period: int) -> Adx:
        """Average Directional Index with +DI / -DI (Wilder)."""
        _check_period(period)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        adx_out = mu.nan_array(n)
        plus_di = mu.nan_array(n)
        minus_di = mu.nan_array(n)
        if n <= 2 * period:
            return Adx(adx_out, plus_di, minus_di)
        tr = Indicators.true_range(high, low, close)
        sm_tr = 0.0
        sm_plus = 0.0
        sm_minus = 0.0
        for i in range(1, period + 1):
            sm_tr += tr[i]
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            sm_plus += up if (up > dn and up > 0) else 0.0
            sm_minus += dn if (dn > up and dn > 0) else 0.0
        dx = mu.nan_array(n)
        for i in range(period, n):
            if i > period:
                up = high[i] - high[i - 1]
                dn = low[i - 1] - low[i]
                sm_tr = sm_tr - sm_tr / period + tr[i]
                sm_plus = sm_plus - sm_plus / period + (up if (up > dn and up > 0) else 0.0)
                sm_minus = sm_minus - sm_minus / period + (dn if (dn > up and dn > 0) else 0.0)
            pdi = 0.0 if sm_tr == 0 else 100 * sm_plus / sm_tr
            mdi = 0.0 if sm_tr == 0 else 100 * sm_minus / sm_tr
            plus_di[i] = pdi
            minus_di[i] = mdi
            s = pdi + mdi
            dx[i] = 0.0 if s == 0 else 100 * abs(pdi - mdi) / s
        # ADX = Wilder average of DX
        acc = 0.0
        for i in range(period, 2 * period):
            acc += dx[i]
        prev = acc / period
        adx_out[2 * period - 1] = prev
        for i in range(2 * period, n):
            prev = (prev * (period - 1) + dx[i]) / period
            adx_out[i] = prev
        return Adx(adx_out, plus_di, minus_di)

    @staticmethod
    def bollinger(close, period: int, k: float) -> Bollinger:
        """Bollinger Bands: SMA middle band with k POPULATION standard
        deviations (not sample — the Java engine pins this)."""
        close = np.asarray(close, dtype=float)
        mid = Indicators.sma(close, period)
        n = close.shape[0]
        up = mu.nan_array(n)
        lo = mu.nan_array(n)
        if n >= period:
            windows = sliding_window_view(close, period)
            mean = windows.sum(axis=1) / period
            sd = np.sqrt(((windows - mean[:, None]) ** 2).sum(axis=1) / period)
            up[period - 1:] = mid[period - 1:] + k * sd
            lo[period - 1:] = mid[period - 1:] - k * sd
        return Bollinger(up, mid, lo)

    @staticmethod
    def keltner(high, low, close, ema_period: int, atr_period: int,
                multiplier: float) -> Keltner:
        """Keltner Channel: EMA middle band with ATR-based envelope."""
        close = np.asarray(close, dtype=float)
        mid = Indicators.ema(close, ema_period)
        a = Indicators.atr(high, low, close, atr_period)
        ok = ~np.isnan(mid) & ~np.isnan(a)
        n = close.shape[0]
        up = mu.nan_array(n)
        lo = mu.nan_array(n)
        up[ok] = mid[ok] + multiplier * a[ok]
        lo[ok] = mid[ok] - multiplier * a[ok]
        return Keltner(up, mid, lo)

    @staticmethod
    def donchian(high, low, period: int) -> Donchian:
        """Donchian Channel: highest high / lowest low over the period."""
        _check_period(period)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        up = Indicators.highest(high, period)
        lo = Indicators.lowest(low, period)
        mid = mu.nan_array(high.shape[0])
        ok = ~np.isnan(up)
        mid[ok] = (up[ok] + lo[ok]) / 2
        return Donchian(up, mid, lo)

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    @staticmethod
    def obv(close, volume) -> np.ndarray:
        """On-Balance Volume."""
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = close.shape[0]
        out = np.zeros(n)
        if n > 1:
            d = close[1:] - close[:-1]
            step = np.where(d > 0, volume[1:], np.where(d < 0, -volume[1:], 0.0))
            out[1:] = np.cumsum(step)
        return out

    @staticmethod
    def vwap(high, low, close, volume) -> np.ndarray:
        """Cumulative Volume-Weighted Average Price (anchored at the
        series start)."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        tp = (high + low + close) / 3.0
        pv = np.cumsum(tp * volume)
        vol = np.cumsum(volume)
        return np.where(vol == 0, tp, pv / np.where(vol == 0, 1.0, vol))

    @staticmethod
    def cmf(high, low, close, volume, period: int) -> np.ndarray:
        """Chaikin Money Flow."""
        _check_period(period)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = high.shape[0]
        rng = high - low
        mult = np.where(rng == 0, 0.0,
                        ((close - low) - (high - close)) / np.where(rng == 0, 1.0, rng))
        mfv = mult * volume
        out = mu.nan_array(n)
        sum_mfv = 0.0
        sum_vol = 0.0
        for i in range(n):
            sum_mfv += mfv[i]
            sum_vol += volume[i]
            if i >= period:
                sum_mfv -= mfv[i - period]
                sum_vol -= volume[i - period]
            if i >= period - 1:
                out[i] = 0.0 if sum_vol == 0 else sum_mfv / sum_vol
        return out

    # ------------------------------------------------------------------
    # Trend systems
    # ------------------------------------------------------------------

    @staticmethod
    def super_trend(high, low, close, period: int, multiplier: float) -> SuperTrend:
        """SuperTrend with ATR bands."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        a = Indicators.atr(high, low, close, period)
        value = mu.nan_array(n)
        direction = np.zeros(n, dtype=int)
        f_up = math.nan
        f_lo = math.nan
        trend = 1
        for i in range(period - 1, n):
            mid = (high[i] + low[i]) / 2
            b_up = mid + multiplier * a[i]
            b_lo = mid - multiplier * a[i]
            if math.isnan(f_up):
                f_up = b_up
                f_lo = b_lo
            else:
                f_up = b_up if (b_up < f_up or close[i - 1] > f_up) else f_up
                f_lo = b_lo if (b_lo > f_lo or close[i - 1] < f_lo) else f_lo
            if trend == 1 and close[i] < f_lo:
                trend = -1
            elif trend == -1 and close[i] > f_up:
                trend = 1
            direction[i] = trend
            value[i] = f_lo if trend == 1 else f_up
        return SuperTrend(value, direction)

    @staticmethod
    def ichimoku(high, low, close, tenkan_period: int, kijun_period: int,
                 senkou_b_period: int) -> Ichimoku:
        """Ichimoku Cloud with standard forward/backward displacement:
        senkou spans are plotted ``kijun_period`` bars ahead, chikou
        ``kijun_period`` bars behind."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        tenkan = _mid_channel(high, low, tenkan_period)
        kijun = _mid_channel(high, low, kijun_period)
        senkou_a = mu.nan_array(n)
        senkou_b = mu.nan_array(n)
        chikou = mu.nan_array(n)
        span_b_base = _mid_channel(high, low, senkou_b_period)
        if kijun_period < n:
            src = slice(0, n - kijun_period)
            dst = slice(kijun_period, n)
            # NaN in either input propagates, matching the Java guard.
            senkou_a[dst] = (tenkan[src] + kijun[src]) / 2
            senkou_b[dst] = span_b_base[src]
            chikou[src] = close[dst]
        return Ichimoku(tenkan, kijun, senkou_a, senkou_b, chikou)

    @staticmethod
    def parabolic_sar(high, low, close, af_start: float, af_step: float,
                      af_max: float) -> np.ndarray:
        """Parabolic SAR (standard Wilder acceleration schedule)."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = high.shape[0]
        out = mu.nan_array(n)
        if n < 2:
            return out
        up = close[1] >= close[0]
        sar = min(low[0], low[1]) if up else max(high[0], high[1])
        ep = max(high[0], high[1]) if up else min(low[0], low[1])
        af = af_start
        out[1] = sar
        for i in range(2, n):
            sar += af * (ep - sar)
            if up:
                sar = min(sar, min(low[i - 1], low[i - 2]))
                if low[i] < sar:            # reversal to downtrend
                    up = False
                    sar = ep
                    ep = low[i]
                    af = af_start
                elif high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
            else:
                sar = max(sar, max(high[i - 1], high[i - 2]))
                if high[i] > sar:           # reversal to uptrend
                    up = True
                    sar = ep
                    ep = high[i]
                    af = af_start
                elif low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
            out[i] = sar
        return out

    # ------------------------------------------------------------------
    # Rolling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def highest(v, period: int) -> np.ndarray:
        """Rolling maximum over the trailing window."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] >= period:
            out[period - 1:] = sliding_window_view(v, period).max(axis=1)
        return out

    @staticmethod
    def lowest(v, period: int) -> np.ndarray:
        """Rolling minimum over the trailing window."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] >= period:
            out[period - 1:] = sliding_window_view(v, period).min(axis=1)
        return out

    @staticmethod
    def rolling_std(v, period: int) -> np.ndarray:
        """Rolling POPULATION standard deviation."""
        _check_period(period)
        v = np.asarray(v, dtype=float)
        out = mu.nan_array(v.shape[0])
        if v.shape[0] >= period:
            windows = sliding_window_view(v, period)
            mean = windows.sum(axis=1) / period
            out[period - 1:] = np.sqrt(
                ((windows - mean[:, None]) ** 2).sum(axis=1) / period)
        return out


# ----------------------------------------------------------------------


def _to_rsi(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def _mid_channel(high, low, period: int) -> np.ndarray:
    hh = Indicators.highest(high, period)
    ll = Indicators.lowest(low, period)
    out = mu.nan_array(high.shape[0])
    ok = ~np.isnan(hh)
    out[ok] = (hh[ok] + ll[ok]) / 2
    return out


def _smooth_ignoring_nan(v: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return v.copy()
    out = mu.nan_array(v.shape[0])
    for i in range(v.shape[0]):
        s = 0.0
        cnt = 0
        for j in range(max(0, i - period + 1), i + 1):
            if not math.isnan(v[j]):
                s += v[j]
                cnt += 1
        if cnt == period:
            out[i] = s / period
    return out


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1: {period}")
