"""Pins for alpha decay, OLS attribution and curve/rolling metrics.

Java source: AlphaBacktesterReportTest.java (AlphaReport half).
"""

import math

import numpy as np
import pytest

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_report import AlphaReport
from quantfinlib.alpha.factors import Factors
from quantfinlib.data.bar_series import BarSeries

BARS = 260
DRIFTS = (0.004, 0.002, 0.001, -0.001, -0.002, -0.004)


def _panel() -> AlphaContext:
    data = {}
    for s, drift in enumerate(DRIFTS):
        b = BarSeries.builder(f"S{s}")
        close = 100.0
        for i in range(BARS):
            open_ = close
            close = 100 * (1 + drift) ** (i + 1)
            b.add(i, open_, max(open_, close), min(open_, close), close, 500_000)
        data[f"S{s}"] = b.build()
    return AlphaContext.of(data)


def _zigzag_panel() -> AlphaContext:
    """Synchronized zigzags with distinct amplitudes and shuffled tiny
    drifts."""
    amps = (0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)
    drift_ranks = (4, 8, 2, 6, 1, 5, 3, 7)
    drifts = [(r - 4.5) * 1e-4 for r in drift_ranks]
    data = {}
    for s, amp in enumerate(amps):
        b = BarSeries.builder(f"S{s}")
        for i in range(120):
            close = (100 * (1 + drifts[s]) ** i
                    * (1 + amp * (1 if i % 2 == 0 else -1)))
            b.add(i, close, close * 1.0001, close * 0.9999, close, 1_000)
        data[f"S{s}"] = b.build()
    return AlphaContext.of(data)


def test_decay_is_flat_for_persistent_signals_and_finite_for_fading_ones():
    # Persistent: drift panel momentum never decays -> infinite half-life.
    persistent = AlphaReport.decay_profile(
        _panel(), Factors.momentum(20, 0), 30, [1, 2, 5, 10])
    for ic in persistent.mean_ics:
        assert ic == pytest.approx(1.0, abs=1e-9)
    assert math.isinf(persistent.half_life_bars)
    assert "half-life" in persistent.format()

    # Fading: zigzag panel -- mean reversion predicts the next bar
    # exactly (the zigzag flips), but two bars out the zigzag cancels
    # and only the small random drift remains -> IC collapses.
    fading = AlphaReport.decay_profile(
        _zigzag_panel(), Factors.mean_reversion(2), 10, [1, 2])
    assert fading.mean_ics[0] > 0.9
    assert abs(fading.mean_ics[1]) < 0.5
    assert 1 < fading.half_life_bars < 2
    with pytest.raises(ValueError):
        AlphaReport.decay_profile(_panel(), Factors.momentum(20, 0), 30, [5, 2])


def test_attribution_recovers_known_betas_exactly():
    # Synthetic returns with known loadings and zero noise: OLS must
    # recover alpha and betas to machine precision with R^2 = 1.
    n = 100
    rng = np.random.default_rng(3)
    f1 = (rng.random(n) - 0.5) * 0.02
    f2 = (rng.random(n) - 0.5) * 0.02
    y = 0.0002 + 1.5 * f1 - 0.5 * f2
    a = AlphaReport.attribute(y, [f1, f2], ["MOM", "VAL"])
    assert a.alpha_per_bar == pytest.approx(0.0002, abs=1e-12)
    assert a.betas[0] == pytest.approx(1.5, abs=1e-9)
    assert a.betas[1] == pytest.approx(-0.5, abs=1e-9)
    assert a.r_squared == pytest.approx(1.0, abs=1e-9)
    assert "MOM" in a.format()
    with pytest.raises(ValueError):
        AlphaReport.attribute(y, [f1], ["A", "B"])


def test_attribution_rejects_nan_inputs_and_sharpes_agree():
    # NaN = missing everywhere in this package: attribution fails with
    # the offending stream and index instead of returning all-NaN betas.
    y = np.array([0.01, 0.02, -0.01, 0.005, 0.0, 0.01])
    f = np.array([0.01, math.nan, -0.01, 0.005, 0.0, 0.01])
    with pytest.raises(ValueError, match="MOM"):
        AlphaReport.attribute(y, [f], ["MOM"])

    # One Sharpe definition per report: the full-sample rolling window
    # must reproduce summarize()'s headline Sharpe (sample stdDev).
    equity = np.array([1.0, 1.02, 1.01, 1.05, 1.04, 1.08, 1.07, 1.12])
    returns = AlphaReport.returns_of(equity)
    rolling = AlphaReport.rolling_sharpe(returns, len(returns), 252)
    assert AlphaReport.summarize(equity, 252).sharpe_ratio == \
        pytest.approx(rolling[-1], abs=1e-9)


def test_curves_and_rolling_metrics():
    equity = np.array([1.0, 1.1, 1.05, 1.2, 1.1, 1.3])
    dd = AlphaReport.drawdown_curve(equity)
    assert dd[0] == 0.0
    assert dd[2] == pytest.approx(1.05 / 1.1 - 1, abs=1e-12)  # below the 1.1 peak
    assert dd[5] == pytest.approx(0.0, abs=1e-12)             # new high: no drawdown
    returns = AlphaReport.returns_of(equity)
    assert len(returns) == 5
    assert returns[0] == pytest.approx(0.1, abs=1e-12)

    rolling = AlphaReport.rolling_sharpe(returns, 3, 252)
    assert math.isnan(rolling[0]) and math.isnan(rolling[1])
    assert not math.isnan(rolling[2])
    # The shared ratio engine backs the summary.
    assert AlphaReport.summarize(equity, 252).sharpe_ratio > 0
    with pytest.raises(ValueError):
        AlphaReport.rolling_sharpe(returns, 1, 252)
    with pytest.raises(ValueError):
        AlphaReport.returns_of(np.array([1.0]))
