"""Two-state Gaussian Markov-switching model (port of Java ``ml.RegimeDetector``).

A hidden Markov model fitted by Baum-Welch EM with forward-backward
scaling: detects calm/turbulent regimes in a return series. State 1 is
always the HIGH-VOLATILITY regime. Feeds naturally into vol targeting
(de-lever when ``smoothed_high_vol_probability`` rises) and liquidity
forecasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import mean as _mean
from quantfinlib.util import percentile


@dataclass(frozen=True, slots=True)
class RegimeModel:
    means: np.ndarray  # per-state mean return
    std_devs: np.ndarray  # per-state volatility (state 1 = high vol)
    transition: np.ndarray  # transition[i][j] = P(next=j | now=i)
    log_likelihood: float
    smoothed_high_vol_probability: np.ndarray  # P(high vol at t | all data)
    current_probabilities: np.ndarray  # filtered P(state | data up to T)
    current_regime: int  # argmax of current_probabilities

    def expected_duration(self, state: int) -> float:
        """Expected persistence of a regime in periods: 1 / (1 - p_stay)."""
        return 1.0 / (1.0 - self.transition[state][state])


def fit(returns, max_iterations: int) -> RegimeModel:
    returns = np.asarray(returns, dtype=float)
    n = returns.shape[0]
    if n < 100:
        raise ValueError(f"need at least 100 returns, got {n}")

    # Initialize by splitting on absolute-return median.
    abs_r = np.abs(returns)
    median = percentile(abs_r, 0.5)
    overall_mean = _mean(returns)
    low_var = high_var = 0.0
    low_count = high_count = 0
    for t in range(n):
        d = returns[t] - overall_mean
        if abs_r[t] <= median:
            low_var += d * d
            low_count += 1
        else:
            high_var += d * d
            high_count += 1
    mu = [overall_mean, overall_mean]
    variance = [
        max(low_var / max(1, low_count), 1e-12),
        max(high_var / max(1, high_count), 1e-12),
    ]
    a = [[0.95, 0.05], [0.05, 0.95]]
    pi = [0.5, 0.5]

    alpha = np.zeros((n, 2))
    beta = np.zeros((n, 2))
    gamma = np.zeros((n, 2))
    scale = np.zeros(n)
    log_likelihood = -math.inf

    # One extra pass beyond max_iterations: the loop always ENDS on an
    # E-step, so the returned probabilities and log-likelihood are
    # computed under the RETURNED parameters. Ending on the M-step
    # (the old shape) handed back parameters one update ahead of the
    # probabilities -- benign at convergence, visibly inconsistent for
    # small iteration budgets.
    for iteration in range(max_iterations + 1):
        # Forward pass with scaling.
        ll = 0.0
        for t in range(n):
            s = 0.0
            for j in range(2):
                if t == 0:
                    prior = pi[j]
                else:
                    prior = alpha[t - 1][0] * a[0][j] + alpha[t - 1][1] * a[1][j]
                alpha[t][j] = prior * _density(returns[t], mu[j], variance[j])
                s += alpha[t][j]
            scale[t] = s if s > 0 else 1e-300
            alpha[t][0] /= scale[t]
            alpha[t][1] /= scale[t]
            ll += math.log(scale[t])

        # Backward pass (same scaling).
        beta[n - 1][0] = 1.0
        beta[n - 1][1] = 1.0
        for t in range(n - 2, -1, -1):
            for i in range(2):
                s = 0.0
                for j in range(2):
                    s += a[i][j] * _density(returns[t + 1], mu[j], variance[j]) * beta[t + 1][j]
                beta[t][i] = s / scale[t + 1]

        # Smoothed state probabilities and transition statistics.
        xi_sum = [[0.0, 0.0], [0.0, 0.0]]
        for t in range(n):
            norm = alpha[t][0] * beta[t][0] + alpha[t][1] * beta[t][1]
            gamma[t][0] = alpha[t][0] * beta[t][0] / norm
            gamma[t][1] = 1 - gamma[t][0]
            if t < n - 1:
                denom = 0.0
                xi = [[0.0, 0.0], [0.0, 0.0]]
                for i in range(2):
                    for j in range(2):
                        xi[i][j] = (
                            alpha[t][i]
                            * a[i][j]
                            * _density(returns[t + 1], mu[j], variance[j])
                            * beta[t + 1][j]
                        )
                        denom += xi[i][j]
                for i in range(2):
                    for j in range(2):
                        xi_sum[i][j] += xi[i][j] / denom

        # Exit on the E-step: converged, or out of M-step budget.
        last_pass = iteration == max_iterations or abs(ll - log_likelihood) < 1e-9
        log_likelihood = ll
        if last_pass:
            break

        # M-step.
        pi[0] = gamma[0][0]
        pi[1] = gamma[0][1]
        for i in range(2):
            row_sum = xi_sum[i][0] + xi_sum[i][1]
            if row_sum > 0:
                a[i][0] = xi_sum[i][0] / row_sum
                a[i][1] = xi_sum[i][1] / row_sum
        for j in range(2):
            weight = 0.0
            weighted_sum = 0.0
            for t in range(n):
                weight += gamma[t][j]
                weighted_sum += gamma[t][j] * returns[t]
            mu[j] = weighted_sum / weight
            var_sum = 0.0
            for t in range(n):
                d = returns[t] - mu[j]
                var_sum += gamma[t][j] * d * d
            variance[j] = max(var_sum / weight, 1e-14)

    # Canonical ordering: state 1 = high volatility.
    if variance[0] > variance[1]:
        mu[0], mu[1] = mu[1], mu[0]
        variance[0], variance[1] = variance[1], variance[0]
        a[0][0], a[1][1] = a[1][1], a[0][0]
        a[0][1], a[1][0] = a[1][0], a[0][1]
        for t in range(n):
            gamma[t][0], gamma[t][1] = gamma[t][1], gamma[t][0]
            alpha[t][0], alpha[t][1] = alpha[t][1], alpha[t][0]

    high_vol_prob = np.array([gamma[t][1] for t in range(n)])
    current = np.array([alpha[n - 1][0], alpha[n - 1][1]])
    return RegimeModel(
        means=np.array(mu),
        std_devs=np.array([math.sqrt(variance[0]), math.sqrt(variance[1])]),
        transition=np.array(a),
        log_likelihood=log_likelihood,
        smoothed_high_vol_probability=high_vol_prob,
        current_probabilities=current,
        current_regime=1 if current[1] > current[0] else 0,
    )


def _density(x: float, mean_: float, variance_: float) -> float:
    d = x - mean_
    return math.exp(-0.5 * d * d / variance_) / math.sqrt(2 * math.pi * variance_)
