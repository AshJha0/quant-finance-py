"""Pins for quantfinlib.indicators.indicators.

Java sources: IndicatorsTest (the behavioral suite over GBM bars) and
FormulaPinsTest.rsiWilderSmoothingAndAtrGapTermsPinned (the exact
hand-computed Wilder/true-range pins, transferred verbatim). GBM bars
come from a numpy port of the Java TestData.gbmSeries shape (numpy
Generator instead of SplittableRandom; all asserts are behavioral).
"""

import math

import numpy as np
import pytest

from quantfinlib.indicators import Indicators


def gbm_bars(days, start_price, annual_drift, annual_vol, seed):
    """Numpy port of Java TestData.gbmSeries: (open, high, low, close, volume)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    price = start_price
    o = np.empty(days)
    h = np.empty(days)
    l = np.empty(days)
    c = np.empty(days)
    v = np.empty(days)
    for d in range(days):
        z = rng.standard_normal()
        nxt = price * math.exp((annual_drift - 0.5 * annual_vol * annual_vol) * dt
                               + annual_vol * math.sqrt(dt) * z)
        o[d] = price
        h[d] = max(price, nxt) * (1 + rng.random() * 0.004)
        l[d] = min(price, nxt) * (1 - rng.random() * 0.004)
        c[d] = nxt
        v[d] = 1_000_000 * (0.5 + rng.random())
        price = nxt
    return o, h, l, c, v


# ---------------------------------------------------------------- averages

def test_sma_known_values():
    out = Indicators.sma([1.0, 2, 3, 4, 5], 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(2.0, abs=1e-12)
    assert out[3] == pytest.approx(3.0, abs=1e-12)
    assert out[4] == pytest.approx(4.0, abs=1e-12)


def test_ema_seeds_with_sma_and_converges():
    flat = np.full(50, 10.0)
    out = Indicators.ema(flat, 10)
    assert out[9] == pytest.approx(10.0, abs=1e-12)
    assert out[49] == pytest.approx(10.0, abs=1e-12)


def test_wma_weights_recent_more():
    out = Indicators.wma([1.0, 2, 3], 3)
    # (1*1 + 2*2 + 3*3) / 6 = 14/6
    assert out[2] == pytest.approx(14.0 / 6, abs=1e-12)


# ------------------------------------------------------------- oscillators

def test_rsi_extremes_and_range():
    up = 100.0 + np.arange(30)
    rsi = Indicators.rsi(up, 14)
    assert rsi[29] == pytest.approx(100.0, abs=1e-9)

    _, _, _, close, _ = gbm_bars(300, 100, 0.05, 0.2, 1)
    r = Indicators.rsi(close, 14)
    assert np.all((r[14:] >= 0) & (r[14:] <= 100))


def test_rsi_wilder_smoothing_pinned():
    # Ported from FormulaPinsTest. Diffs {+1, -0.5, +1}: avgGain 2/3,
    # avgLoss 1/6 -> RS 4 -> RSI 80. Next diff +0.5 under WILDER
    # smoothing: avgGain (2/3*2 + 0.5)/3, avgLoss (1/6*2)/3 -> RS 5.5 ->
    # RSI 100 - 100/6.5 = 84.6154 (a plain rolling mean gives 75 here).
    rsi = Indicators.rsi([100.0, 101, 100.5, 101.5, 102, 101, 102.5], 3)
    assert rsi[3] == pytest.approx(80.0, abs=1e-9)
    assert rsi[4] == pytest.approx(100 - 100 / 6.5, abs=1e-9), \
        "Wilder, not a rolling mean"


def test_macd_histogram_is_line_minus_signal():
    _, _, _, close, _ = gbm_bars(200, 100, 0.1, 0.2, 2)
    m = Indicators.macd(close, 12, 26, 9)
    last = close.shape[0] - 1
    assert not math.isnan(m.signal[last])
    assert m.histogram[last] == pytest.approx(m.line[last] - m.signal[last], abs=1e-12)


def test_stochastic_rsi_within_range():
    _, _, _, close, _ = gbm_bars(300, 100, 0.05, 0.25, 9)
    sr = Indicators.stochastic_rsi(close, 14, 14, 3, 3)
    last = close.shape[0] - 1
    assert not math.isnan(sr.k[last])
    assert 0 <= sr.k[last] <= 100
    assert 0 <= sr.d[last] <= 100


def test_roc_and_momentum():
    v = [100.0, 110, 121]
    assert Indicators.roc(v, 2)[2] == pytest.approx(21.0, abs=1e-9)
    assert Indicators.momentum(v, 2)[2] == pytest.approx(21.0, abs=1e-9)


# ------------------------------------------------------- volatility / range

def test_true_range_and_atr_gap_terms_pinned():
    # Ported from FormulaPinsTest. Bars (9,10,8,9), (9,9.5,8.5,9),
    # (7,7.5,6.5,7): the third bar GAPS down, so its true range is
    # |low - prevClose| = 2.5, not high-low = 1.
    high = [10.0, 9.5, 7.5]
    low = [8.0, 8.5, 6.5]
    close = [9.0, 9.0, 7.0]
    tr = Indicators.true_range(high, low, close)
    assert tr[0] == pytest.approx(2.0, abs=1e-12)
    assert tr[1] == pytest.approx(1.0, abs=1e-12)
    assert tr[2] == pytest.approx(2.5, abs=1e-12), "the gap term |low - prevClose|"
    atr = Indicators.atr(high, low, close, 2)
    assert atr[1] == pytest.approx(1.5, abs=1e-12), "seed: mean of the first two TRs"
    assert atr[2] == pytest.approx(2.0, abs=1e-12), \
        "Wilder: (1.5*1 + 2.5)/2 — hl-only gives 1.25"


def test_atr_positive_after_warmup():
    _, h, l, c, _ = gbm_bars(100, 100, 0.05, 0.25, 4)
    atr = Indicators.atr(h, l, c, 14)
    assert math.isnan(atr[12])
    assert np.all(atr[13:] > 0)


def test_adx_within_range():
    _, h, l, c, _ = gbm_bars(300, 100, 0.1, 0.25, 5)
    adx = Indicators.adx(h, l, c, 14)
    last = c.shape[0] - 1
    assert not math.isnan(adx.adx[last])
    assert 0 <= adx.adx[last] <= 100
    assert adx.plus_di[last] >= 0 and adx.minus_di[last] >= 0


def test_bollinger_bands_bracket_middle_with_population_stdev():
    _, _, _, close, _ = gbm_bars(100, 100, 0, 0.3, 3)
    b = Indicators.bollinger(close, 20, 2)
    assert np.all(b.upper[19:] >= b.middle[19:])
    assert np.all(b.lower[19:] <= b.middle[19:])
    # Population stdev pin: window [1,2,3,4], mean 2.5, pop var 1.25.
    bb = Indicators.bollinger([1.0, 2, 3, 4], 4, 2)
    sd = math.sqrt(1.25)
    assert bb.upper[3] == pytest.approx(2.5 + 2 * sd, abs=1e-12)
    assert bb.lower[3] == pytest.approx(2.5 - 2 * sd, abs=1e-12), \
        "k POPULATION standard deviations (sample would give sqrt(5/3))"


def test_keltner_and_donchian_channels_ordered():
    _, h, l, c, _ = gbm_bars(120, 100, 0.05, 0.2, 11)
    last = c.shape[0] - 1
    k = Indicators.keltner(h, l, c, 20, 10, 2)
    assert k.upper[last] > k.middle[last] > k.lower[last]
    d = Indicators.donchian(h, l, 20)
    assert d.upper[last] >= d.middle[last] >= d.lower[last]


# ---------------------------------------------------------------- volume

def test_vwap_obv_cmf_cci_williams_r_produce_values():
    _, h, l, c, v = gbm_bars(150, 100, 0.08, 0.2, 6)
    last = c.shape[0] - 1
    assert Indicators.vwap(h, l, c, v)[last] > 0
    assert not math.isnan(Indicators.obv(c, v)[last])
    cmf = Indicators.cmf(h, l, c, v, 20)[last]
    assert -1 <= cmf <= 1
    assert not math.isnan(Indicators.cci(h, l, c, 20)[last])
    wr = Indicators.williams_r(h, l, c, 14)[last]
    assert -100 <= wr <= 0


def test_obv_and_vwap_hand_pins():
    # OBV: closes {10, 11, 11, 9}, volumes {5, 7, 3, 2}:
    # +7 (up), +0 (flat), -2 (down) -> {0, 7, 7, 5}.
    obv = Indicators.obv([10.0, 11, 11, 9], [5.0, 7, 3, 2])
    assert obv.tolist() == pytest.approx([0.0, 7.0, 7.0, 5.0], abs=1e-12)
    # VWAP with H=L=C: cum(p*v)/cum(v); (10*100 + 20*100)/200 = 15.
    vwap = Indicators.vwap([10.0, 20], [10.0, 20], [10.0, 20], [100.0, 100])
    assert vwap[1] == pytest.approx(15.0, abs=1e-12)


# ------------------------------------------------------------ trend systems

def test_super_trend_direction_follows_strong_trend():
    # Strongly rising market: direction should end bullish.
    _, h, l, c, _ = gbm_bars(400, 100, 0.5, 0.08, 7)
    st = Indicators.super_trend(h, l, c, 10, 3)
    assert st.direction[-1] == 1
    assert not math.isnan(st.value[-1])


def test_ichimoku_cloud_spans_present():
    _, h, l, c, _ = gbm_bars(300, 100, 0.1, 0.2, 8)
    ich = Indicators.ichimoku(h, l, c, 9, 26, 52)
    last = c.shape[0] - 1
    assert not math.isnan(ich.tenkan[last])
    assert not math.isnan(ich.kijun[last])
    assert not math.isnan(ich.senkou_a[last])
    assert not math.isnan(ich.senkou_b[last])
    # chikou is close displaced backwards
    assert ich.chikou[last - 26] == pytest.approx(c[last], abs=1e-12)


def test_parabolic_sar_stays_on_correct_side_in_trend():
    _, h, l, c, _ = gbm_bars(300, 100, 0.6, 0.06, 10)
    sar = Indicators.parabolic_sar(h, l, c, 0.02, 0.02, 0.2)
    assert not math.isnan(sar[-1])
    # In a strong uptrend the last SAR should sit below the close.
    assert sar[-1] < c[-1]


# ---------------------------------------------------------------- gates

def test_period_gate():
    with pytest.raises(ValueError):
        Indicators.sma([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        Indicators.rsi([1.0, 2.0], -3)
