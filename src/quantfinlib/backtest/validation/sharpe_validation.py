"""Sharpe significance tests (port of Java ``validation.SharpeValidation``).

Bailey & Lopez de Prado:

* **Probabilistic Sharpe Ratio** — the probability the true Sharpe
  exceeds a benchmark, adjusting for track length and non-normal returns
  (skew, kurtosis).
* **Deflated Sharpe Ratio** — PSR against the Sharpe you'd expect from
  the *best of N* random trials: the multiple-testing haircut for a
  strategy picked from a parameter grid.
* **Minimum track record length** — PSR's closed-form inverse: how many
  periods of THIS performance before the record means something.

All Sharpe inputs are per-period (not annualized) with observation count
``n_obs``.
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils as mu

_EULER_GAMMA = 0.5772156649015329


class SharpeValidation:
    """Static Sharpe significance tests; see the module docstring."""

    @staticmethod
    def probabilistic_sharpe(observed_sharpe: float, benchmark_sharpe: float,
                             n_obs: int, skewness: float,
                             kurtosis: float) -> float:
        """Probability the true Sharpe exceeds ``benchmark_sharpe``, in [0, 1].

        Raises:
            ValueError: with fewer than 2 observations.
        """
        if n_obs < 2:
            raise ValueError("need at least 2 observations")
        variance = (1 - skewness * observed_sharpe
                    + (kurtosis - 1) / 4.0 * observed_sharpe * observed_sharpe)
        if variance <= 0:
            return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
        z = ((observed_sharpe - benchmark_sharpe)
             * math.sqrt(n_obs - 1) / math.sqrt(variance))
        return mu.norm_cdf(z)

    @staticmethod
    def expected_max_sharpe(trials: int,
                            variance_of_trial_sharpes: float) -> float:
        """Expected maximum Sharpe among ``trials`` independent zero-skill
        strategies whose Sharpe estimates have the given cross-trial variance.

        Raises:
            ValueError: with fewer than 2 trials.
        """
        if trials < 2:
            raise ValueError("need at least 2 trials")
        sd = math.sqrt(max(0.0, variance_of_trial_sharpes))
        return sd * ((1 - _EULER_GAMMA) * mu.norm_inv(1 - 1.0 / trials)
                     + _EULER_GAMMA * mu.norm_inv(1 - 1.0 / (trials * math.e)))

    @staticmethod
    def deflated_sharpe(observed_sharpe: float, trial_sharpes,
                        n_obs: int, skewness: float, kurtosis: float) -> float:
        """Deflated Sharpe: PSR of the winner against the expected-max
        benchmark implied by all the parameter combinations that were tried.

        Values near 1 mean the edge survives the multiple-testing haircut;
        below ~0.95 the "discovery" is likely selection bias.

        Args:
            observed_sharpe: Per-period Sharpe of the winner.
            trial_sharpes: Per-period Sharpe of every trial in the search
                (including the winner).
            n_obs: Observation count of the winner's track.
            skewness: Return skewness of the winner's track.
            kurtosis: Return kurtosis (3 for a normal, not excess).
        """
        trial_sharpes = list(trial_sharpes)
        benchmark = SharpeValidation.expected_max_sharpe(
            len(trial_sharpes), mu.variance(trial_sharpes))
        return SharpeValidation.probabilistic_sharpe(
            observed_sharpe, benchmark, n_obs, skewness, kurtosis)

    @staticmethod
    def min_track_record_length(observed_sharpe: float, benchmark_sharpe: float,
                                skewness: float, kurtosis: float,
                                confidence: float) -> float:
        """Minimum track record length (Bailey & Lopez de Prado).

        How many periods of THIS performance are needed before
        :meth:`probabilistic_sharpe` would clear ``confidence`` that the
        true Sharpe exceeds the benchmark — the allocator's question
        ("how long until this manager's record means something?") in
        closed form::

            n* = 1 + (1 - g3*SR + (g4-1)/4*SR^2) * (z_conf / (SR - SR*))^2

        Returns ``inf`` when the observed Sharpe does not exceed the
        benchmark — no track record length proves an edge the record does
        not show. Sharpe inputs are PER-PERIOD (not annualized), matching
        :meth:`probabilistic_sharpe`.

        Raises:
            ValueError: unless ``confidence`` is in (0, 1) and all inputs
                are finite (NaN-rejecting gates).
        """
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")
        if not (math.isfinite(observed_sharpe) and math.isfinite(benchmark_sharpe)
                and math.isfinite(skewness) and math.isfinite(kurtosis)):
            raise ValueError("inputs must be finite")
        if observed_sharpe <= benchmark_sharpe:
            return math.inf
        variance = (1 - skewness * observed_sharpe
                    + (kurtosis - 1) / 4.0 * observed_sharpe * observed_sharpe)
        if variance <= 0:
            return 2.0                  # PSR is already 1: any record suffices
        z = mu.norm_inv(confidence)
        edge = observed_sharpe - benchmark_sharpe
        return 1 + variance * z * z / (edge * edge)
