"""Kyle's lambda (port of Java ``microstructure.KylesLambda``).

Market impact LEARNED from the tape instead of assumed from a formula.
Kyle (1985): price change is linear in signed order flow,
``dp = lambda * q + noise``, and lambda (price per unit of signed
volume) is the market's depth read off its own behavior. Where
:class:`~quantfinlib.microstructure.market_impact_model.MarketImpactModel`
parameterizes impact (square-root law with calibrated constants), this
class runs the streaming through-origin regression
``lambda = E[q * dp] / E[q^2]`` on time-decayed moments — the
regression-through-origin form is standard here because both the signed
flow and the mid change are zero-mean at these horizons.

Feed one :meth:`KylesLambda.on_sample` per aggregation window (a trade,
a bar, a decision interval): the mid change over the window and the
signed volume that traded in it (buy-aggressor positive).
:meth:`KylesLambda.impact_bps` prices a contemplated child order; a
noisy negative lambda estimate is clamped to zero impact there — "the
market pays you to trade" is an estimation artifact.
:meth:`KylesLambda.lambda_` stays raw for diagnostics.

Gap discipline: non-finite inputs are skipped whole; the moments seed
from the first valid sample. (The Java checkpoint persistence is not
ported — no ``persist`` lane in the Python port.)
"""

from __future__ import annotations

import math


class KylesLambda:
    """Streaming depth estimator; see the module docstring."""

    __slots__ = ("_alpha", "_flow_squared", "_flow_times_move", "_samples")

    def __init__(self, alpha: float = 0.02) -> None:
        """``alpha``: EWMA weight per sample, e.g. 0.02 (~50-sample
        memory)."""
        if alpha <= 0 or alpha > 1:
            raise ValueError("need alpha in (0,1]")
        self._alpha = alpha
        self._flow_squared = 0.0      # decayed E[q^2]
        self._flow_times_move = 0.0   # decayed E[q * dp]
        self._samples = 0

    def on_sample(self, mid_change: float, signed_volume: float) -> None:
        """One aggregation window: the mid change over the window and the
        signed volume traded in it (+ = buyer-initiated). Non-finite or
        volume-free windows are skipped — no flow, no impact information.
        """
        if (not math.isfinite(mid_change) or not math.isfinite(signed_volume)
                or signed_volume == 0):
            return
        q2 = signed_volume * signed_volume
        qp = signed_volume * mid_change
        if self._samples == 0:
            self._flow_squared = q2
            self._flow_times_move = qp
        else:
            self._flow_squared += self._alpha * (q2 - self._flow_squared)
            self._flow_times_move += self._alpha * (qp - self._flow_times_move)
        self._samples += 1

    def lambda_(self) -> float:
        """The learned lambda: price change per unit of signed volume.
        Raw — can be negative while the estimate is noise. 0 until any
        flow is observed."""
        if self._flow_squared > 0:
            return self._flow_times_move / self._flow_squared
        return 0.0

    def impact_bps(self, quantity: float, mid: float) -> float:
        """The estimated impact of trading ``quantity`` now, in basis
        points of ``mid``. The SIGN of the quantity is ignored (impact is
        a cost in both directions — a signed sell size must not read as
        free), a negative lambda estimate is clamped to 0 (noise, never a
        subsidy), and non-finite inputs are neutral."""
        size = abs(quantity)
        if not (mid > 0) or mid == math.inf or not (size > 0) or size == math.inf:
            return 0.0                 # not (x > 0) also catches NaN
        return max(0.0, self.lambda_() * size / mid * 1e4)

    def samples(self) -> int:
        return self._samples
