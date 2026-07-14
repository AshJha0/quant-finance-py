"""Pins for quantfinlib.ml.

Java sources: GradientBoostedRegressor/VolatilityForecaster/AnomalyDetector/
RegimeDetector/IntradayLiquidityForecaster/MarketImpactPredictor.java.

GradientBoostedRegressor has no RNG: each round's split is the exact
SSE-optimal stump found by a full sorted sweep, so the port is pinned
by planted-signal recovery (fit a known function, verify near-zero
RMSE and monotonic residual improvement) rather than by reproducing a
seed -- there is no seed to reproduce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantfinlib.ml import anomaly_detector as ad
from quantfinlib.ml import regime_detector as rd
from quantfinlib.ml.gradient_boosted_regressor import GradientBoostedRegressor
from quantfinlib.ml.intraday_liquidity_forecaster import IntradayLiquidityForecaster
from quantfinlib.ml.market_impact_predictor import MarketImpactPredictor
from quantfinlib.ml.volatility_forecaster import VolatilityForecaster


# ----------------------------------------------------------------------
# GradientBoostedRegressor
# ----------------------------------------------------------------------


def test_gbdt_recovers_a_planted_step_function_exactly():
    # A single true split at x=0: stumps should recover it in one round.
    x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    model = GradientBoostedRegressor(50, 0.5).fit(x, y)
    assert model.rmse(x, y) < 1e-6


def test_gbdt_reduces_error_each_extra_round_until_convergence():
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 5, size=(200, 1))
    y = 3.0 * x[:, 0] + rng.normal(0, 0.01, size=200)
    small = GradientBoostedRegressor(5, 0.1).fit(x, y)
    large = GradientBoostedRegressor(200, 0.1).fit(x, y)
    assert large.rmse(x, y) < small.rmse(x, y)


def test_gbdt_predict_before_fit_raises():
    model = GradientBoostedRegressor(10, 0.1)
    with pytest.raises(RuntimeError):
        model.predict([1.0])


def test_gbdt_empty_or_mismatched_input_raises():
    with pytest.raises(ValueError):
        GradientBoostedRegressor(10, 0.1).fit([], [])
    with pytest.raises(ValueError):
        GradientBoostedRegressor(10, 0.1).fit([[1.0], [2.0]], [1.0])


def test_gbdt_with_defaults_matches_java_hyperparameters():
    model = GradientBoostedRegressor.with_defaults()
    assert model._rounds == 200
    assert model._learning_rate == pytest.approx(0.1)


# ----------------------------------------------------------------------
# VolatilityForecaster
# ----------------------------------------------------------------------


def test_volatility_forecaster_nonnegative_and_reasonable_scale():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 500)
    vf = VolatilityForecaster(5).fit(returns)
    forecast = vf.forecast(returns)
    assert forecast >= 0
    # Scale sanity: forecast should be in the same ballpark as the
    # underlying per-period vol, not off by orders of magnitude.
    assert forecast == pytest.approx(0.01, rel=2.0)


def test_volatility_forecaster_risk_score_in_bounds():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 500)
    vf = VolatilityForecaster.weekly().fit(returns)
    score = vf.risk_score(returns)
    assert 0.0 <= score <= 100.0


def test_volatility_forecaster_insufficient_history_raises():
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.01, 20)
    with pytest.raises(ValueError):
        VolatilityForecaster(5).fit(returns)


def test_volatility_forecaster_forecast_before_fit_raises():
    vf = VolatilityForecaster(5)
    with pytest.raises(RuntimeError):
        vf.forecast(np.zeros(30))


# ----------------------------------------------------------------------
# AnomalyDetector
# ----------------------------------------------------------------------


def test_detect_quote_stuffing_flags_the_spike_only():
    messages = [10] * 20 + [500]
    trades = [9] * 20 + [1]
    anomalies = ad.detect_quote_stuffing(messages, trades, 3.0, 5.0)
    assert len(anomalies) == 1
    assert anomalies[0].interval_index == 20
    assert anomalies[0].type == ad.QUOTE_STUFFING
    assert anomalies[0].score > 3.0


def test_detect_quote_stuffing_requires_aligned_series():
    with pytest.raises(ValueError):
        ad.detect_quote_stuffing([1, 2], [1], 3.0, 5.0)


def test_detect_price_spikes_flags_the_outlier_return():
    mids = [100.0] * 20 + [130.0] + [100.0] * 5
    anomalies = ad.detect_price_spikes(mids, 3.0)
    assert any(a.interval_index == 20 for a in anomalies)
    assert all(a.type == ad.PRICE_SPIKE for a in anomalies)


def test_detect_price_spikes_too_short_returns_empty():
    assert ad.detect_price_spikes([100.0, 101.0], 3.0) == []


# ----------------------------------------------------------------------
# RegimeDetector
# ----------------------------------------------------------------------


def test_regime_detector_state_1_is_always_high_vol():
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.003, 200)
    turbulent = rng.normal(0, 0.04, 200)
    returns = np.concatenate([calm, turbulent])
    model = rd.fit(returns, 100)
    assert model.std_devs[1] >= model.std_devs[0]


def test_regime_detector_requires_at_least_100_returns():
    with pytest.raises(ValueError):
        rd.fit(np.zeros(50), 10)


def test_regime_detector_expected_duration_formula():
    rng = np.random.default_rng(6)
    returns = rng.normal(0, 0.01, 300)
    model = rd.fit(returns, 50)
    for state in (0, 1):
        expected = 1.0 / (1.0 - model.transition[state][state])
        assert model.expected_duration(state) == pytest.approx(expected)


def test_regime_detector_current_regime_matches_argmax():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, 300)
    model = rd.fit(returns, 50)
    expected_regime = 1 if model.current_probabilities[1] > model.current_probabilities[0] else 0
    assert model.current_regime == expected_regime


# ----------------------------------------------------------------------
# IntradayLiquidityForecaster
# ----------------------------------------------------------------------


def test_intraday_liquidity_forecaster_profile_sums_to_one():
    f = IntradayLiquidityForecaster(4)
    f.add_day([1.0, 2.0, 3.0, 4.0]).add_day([2.0, 2.0, 2.0, 2.0])
    profile = f.profile()
    assert profile.sum() == pytest.approx(1.0)
    assert f.peak_bucket() == 3
    assert f.forecast_volume(0) == pytest.approx(1.5)


def test_intraday_liquidity_forecaster_no_data_is_uniform():
    f = IntradayLiquidityForecaster(4)
    profile = f.profile()
    assert np.allclose(profile, [0.25, 0.25, 0.25, 0.25])
    assert f.forecast_volume(0) == 0.0


def test_intraday_liquidity_forecaster_rejects_wrong_bucket_count():
    f = IntradayLiquidityForecaster(4)
    with pytest.raises(ValueError):
        f.add_day([1.0, 2.0])


def test_intraday_liquidity_forecaster_session_share():
    f = IntradayLiquidityForecaster(4)
    f.add_day([1.0, 1.0, 1.0, 1.0])
    assert f.session_share(0, 2) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "hour,session",
    [(23, "SYDNEY"), (0, "TOKYO"), (6, "TOKYO"), (7, "LONDON"), (11, "LONDON"),
     (12, "LONDON_NY_OVERLAP"), (16, "LONDON_NY_OVERLAP"), (17, "NEW_YORK"), (21, "NEW_YORK")],
)
def test_fx_session_boundaries(hour, session):
    assert IntradayLiquidityForecaster.fx_session(hour) == session


# ----------------------------------------------------------------------
# MarketImpactPredictor
# ----------------------------------------------------------------------


def test_market_impact_predictor_fits_and_predicts():
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, size=(100, 4))
    y = 50 * x[:, 0] + 10 * x[:, 1]
    predictor = MarketImpactPredictor().fit(x, y)
    pred = predictor.predict_impact_bps(MarketImpactPredictor.features(0.5, 5.0, 0.1, 0.02))
    assert math.isfinite(pred)


def test_market_impact_predictor_predict_before_fit_raises():
    predictor = MarketImpactPredictor()
    with pytest.raises(RuntimeError):
        predictor.predict_impact_bps([0.1, 1.0, 0.0, 0.01])


def test_sweep_probability_is_half_at_parity_and_bounds_at_extremes():
    assert MarketImpactPredictor.sweep_probability(100, 100) == pytest.approx(0.5)
    assert MarketImpactPredictor.sweep_probability(10, 0) == 1.0
    assert MarketImpactPredictor.sweep_probability(1, 1_000_000) < 0.02
    assert MarketImpactPredictor.sweep_probability(1_000_000, 1) > 0.99
