"""Port of Java ``com.quantfinlib.volatility.InformationCriteria``.

AIC / BIC — the two numbers that keep model shopping honest. Every
extra parameter raises the maximized log-likelihood by construction;
these criteria charge admission for it:

    AIC = 2k - 2 ln L          (Akaike: prediction-oriented)
    BIC = k ln n - 2 ln L      (Schwarz: consistency-oriented)

LOWER is better for both. BIC's penalty grows with the sample size, so
on large samples it picks smaller models than AIC — BIC will recover
the true model as n grows (consistency) while AIC minimizes
out-of-sample prediction error even when no candidate is "true". The
rule of thumb: AIC for forecasting, BIC for identifying structure.

Both criteria only rank models fitted to the SAME data with likelihoods
on the same scale — comparing an AIC computed on returns against one
computed on squared returns is meaningless, and this class cannot
detect that for you. Made for the volatility-model zoo here (Garch11 vs
GjrGarch11 vs Egarch11: does the leverage parameter pay its way?), but
the arithmetic is model-agnostic.
"""

from __future__ import annotations

import math


class InformationCriteria:
    """Static AIC / BIC with the Java gates (ValueError on nonsense)."""

    @staticmethod
    def aic(log_likelihood: float, parameters: int) -> float:
        """Akaike information criterion ``2k - 2 ln L``.

        Args:
            log_likelihood: maximized log-likelihood ln L (finite).
            parameters: number of fitted parameters k, >= 0.
        """
        _check_log_lik(log_likelihood)
        _check_params(parameters)
        return 2.0 * parameters - 2.0 * log_likelihood

    @staticmethod
    def bic(log_likelihood: float, parameters: int, observations: int) -> float:
        """Bayesian (Schwarz) information criterion ``k ln n - 2 ln L``.

        Args:
            log_likelihood: maximized log-likelihood ln L (finite).
            parameters: number of fitted parameters k, >= 0.
            observations: sample size n the likelihood was computed over, >= 1.
        """
        _check_log_lik(log_likelihood)
        _check_params(parameters)
        if observations < 1:
            raise ValueError(f"observations must be >= 1, got {observations}")
        return parameters * math.log(observations) - 2.0 * log_likelihood


def _check_log_lik(log_likelihood: float) -> None:
    if not math.isfinite(log_likelihood):
        raise ValueError(f"logLikelihood must be finite, got {log_likelihood}")


def _check_params(parameters: int) -> None:
    if parameters < 0:
        raise ValueError(f"parameters must be >= 0, got {parameters}")
