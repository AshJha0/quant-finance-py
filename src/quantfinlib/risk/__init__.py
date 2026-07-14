"""Risk domain (port of Java com.quantfinlib.risk) — the pure-math core.

Static Java utility classes map to modules of functions (the
``math_utils`` precedent); stateful classes (``Pca``,
``GaussianCopula``, ``CounterpartyExposureTracker``) map to Python
classes; Java records map to frozen dataclasses.

``pnl_attribution.test`` and ``var_backtest.test`` keep their Java
names — access them module-qualified (``var_backtest.test(...)``) so
pytest never mistakes an import for a test function.

Phase 2 (trading-infra, deliberately not ported here):
GlobalRiskAggregator, HftRiskGate, PreTradeLimitChecker — plus the
registry/aggregation wrappers CorrelationMatrix, Portfolio,
PortfolioRiskAnalyzer, RiskMetric, RiskMetricRegistry.
"""

from quantfinlib.risk import (  # noqa: F401 — submodule namespace exports
    concentration_risk,
    component_var,
    counterparty_exposure_tracker,
    covariance_shrinkage,
    dependence,
    extreme_value_theory,
    frtb_es,
    gaussian_copula,
    pca,
    pnl_attribution,
    risk_metrics,
    settlement_risk_analyzer,
    stress_tester,
    var_backtest,
    var_engine,
)
from quantfinlib.risk.component_var import Allocation
from quantfinlib.risk.counterparty_exposure_tracker import (
    CounterpartyExposureTracker,
    CounterpartyTrade,
)
from quantfinlib.risk.covariance_shrinkage import Result as ShrinkageResult
from quantfinlib.risk.extreme_value_theory import GpdFit
from quantfinlib.risk.frtb_es import TrafficLight
from quantfinlib.risk.gaussian_copula import GaussianCopula
from quantfinlib.risk.pca import Pca
from quantfinlib.risk.pnl_attribution import Result as PlatResult, Zone
from quantfinlib.risk.settlement_risk_analyzer import SettlementLeg
from quantfinlib.risk.stress_tester import ReverseStress
from quantfinlib.risk.var_backtest import VarBacktestResult
from quantfinlib.risk.var_engine import VarResult

__all__ = [
    "Allocation",
    "CounterpartyExposureTracker",
    "CounterpartyTrade",
    "GaussianCopula",
    "GpdFit",
    "Pca",
    "PlatResult",
    "ReverseStress",
    "SettlementLeg",
    "ShrinkageResult",
    "TrafficLight",
    "VarBacktestResult",
    "VarResult",
    "Zone",
    "concentration_risk",
    "component_var",
    "counterparty_exposure_tracker",
    "covariance_shrinkage",
    "dependence",
    "extreme_value_theory",
    "frtb_es",
    "gaussian_copula",
    "pca",
    "pnl_attribution",
    "risk_metrics",
    "settlement_risk_analyzer",
    "stress_tester",
    "var_backtest",
    "var_engine",
]
