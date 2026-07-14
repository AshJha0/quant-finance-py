"""Machine-learning research utilities (port of Java ``com.quantfinlib.ml``).

Pure Python/NumPy, no ML framework dependencies: a deterministic
gradient-boosted stump regressor, volatility and market-impact
forecasters built on it, a two-state Gaussian HMM regime detector, an
intraday liquidity profiler, and robust-statistics surveillance
anomaly detection.
"""

from quantfinlib.ml import anomaly_detector, regime_detector
from quantfinlib.ml.anomaly_detector import Anomaly
from quantfinlib.ml.gradient_boosted_regressor import GradientBoostedRegressor
from quantfinlib.ml.intraday_liquidity_forecaster import IntradayLiquidityForecaster
from quantfinlib.ml.market_impact_predictor import MarketImpactPredictor
from quantfinlib.ml.regime_detector import RegimeModel
from quantfinlib.ml.volatility_forecaster import VolatilityForecaster

__all__ = [
    "GradientBoostedRegressor",
    "VolatilityForecaster",
    "Anomaly",
    "anomaly_detector",
    "RegimeModel",
    "regime_detector",
    "IntradayLiquidityForecaster",
    "MarketImpactPredictor",
]
