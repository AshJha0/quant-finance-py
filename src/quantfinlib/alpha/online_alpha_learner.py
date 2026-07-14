"""Online alpha-weight learning (port of Java
``microstructure.OnlineAlphaLearner``): an online ridge regression
(SGD with L2 shrinkage) from four dimensionless signal ingredients
(queue imbalance, trade imbalance, normalized OFI, momentum-Z) to the
next-interval return.

The honesty mechanism: prequential out-of-sample IC
====================================================
The trap with any self-updating alpha is grading its own homework.
This learner can't: :meth:`train` records the prediction made with the
CURRENT weights *before* the realized return updates them
(predict-then-train, "prequential" evaluation), and maintains a
time-decayed correlation between those genuinely out-of-sample
predictions and the outcomes -- :meth:`out_of_sample_ic`. **Gate any
use of the learned alpha on that number**: persistently positive
(intraday, ~0.02-0.10 is real) means the weights found signal; an IC
hovering at zero means they found noise, and
:meth:`normalized_prediction` should be treated as such. This
diagnostic is a live tripwire, not a validation -- before trading a
weighting seriously, run it through
:mod:`quantfinlib.alpha.alpha_validation`'s walk-forward and
permutation machinery like any other signal.

One instance per symbol (or one pooled across a homogeneous group --
pooling trades specificity for sample count; the caller chooses).
Cross-asset: ingredients are dimensionless and the target is a return.

Deviations from the Java source:

* The Java class also exposes ``predictFrom``/``trainFrom`` overloads
  that pull the four ingredients directly from a live
  ``microstructure.SignalEngine`` (a bus/feed-coupled class this port
  does not carry). :meth:`train_snapshot` preserves the same
  "features snapshotted one interval earlier" alignment discipline
  generically, over a caller-supplied feature tuple instead.
* No ``persist.Checkpoint`` lane in this port, so ``write_state``/
  ``read_state`` are not carried over (see
  :mod:`quantfinlib.microstructure.kyles_lambda`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from quantfinlib.util import math_utils

_FEATURES = 4


class OnlineAlphaLearner:
    """Predict/train/IC-gate alpha weight learner; see the module
    docstring."""

    __slots__ = ("_learning_rate", "_ridge_lambda", "_ic_alpha", "_w",
                 "_mean_pred", "_mean_ret", "_var_pred", "_var_ret",
                 "_covar", "_abs_pred_ewma", "_samples", "_snapshot",
                 "_has_snapshot")

    def __init__(self, learning_rate: float = 0.01,
                 ridge_lambda: float = 1e-4,
                 ic_alpha: float = 0.01) -> None:
        """
        Args:
            learning_rate: SGD step, e.g. 0.01 -- larger adapts
                faster, overshoots noisier targets.
            ridge_lambda: L2 shrinkage toward 0 per step, e.g. 1e-4 --
                keeps weights from chasing one lucky streak.
            ic_alpha: EWMA weight of the IC statistics, e.g. 0.01
                (~a few-hundred-sample memory).
        """
        if learning_rate <= 0 or ridge_lambda < 0 or ic_alpha <= 0 or ic_alpha > 1:
            raise ValueError(
                "need learningRate > 0, ridgeLambda >= 0, icAlpha in (0,1]")
        self._learning_rate = learning_rate
        self._ridge_lambda = ridge_lambda
        self._ic_alpha = ic_alpha
        self._w = np.zeros(_FEATURES)
        self._mean_pred = 0.0
        self._mean_ret = 0.0
        self._var_pred = 0.0
        self._var_ret = 0.0
        self._covar = 0.0
        self._abs_pred_ewma = 0.0
        self._samples = 0
        self._snapshot: Optional[Tuple[float, float, float, float]] = None
        self._has_snapshot = False

    # ------------------------------------------------------------------
    # Predict / train
    # ------------------------------------------------------------------

    def predict(self, queue_imbalance: float, trade_imbalance: float,
               normalized_ofi: float, momentum_z: float) -> float:
        """The learned prediction of the next-interval return from the
        four ingredients (each expected in ~[-1, 1]). Raw units =
        whatever return you train against."""
        w = self._w
        return (w[0] * queue_imbalance + w[1] * trade_imbalance
                + w[2] * normalized_ofi + w[3] * momentum_z)

    def train(self, queue_imbalance: float, trade_imbalance: float,
             normalized_ofi: float, momentum_z: float,
             realized_return: float) -> None:
        """One learning step: the prediction made with the current
        weights is scored against ``realized_return`` (this is what
        makes :meth:`out_of_sample_ic` honest), THEN the weights
        update by ridge-SGD. Non-finite inputs are skipped entirely --
        a NaN must neither poison the weights nor sneak into the IC.

        **Alignment is the caller's contract here:** the four features
        must have been observed BEFORE the interval ``realized_return``
        covers. Passing the current features with the return that
        just ended fits a nowcast -- the features already contain the
        move -- and the IC will read high on pure leakage.
        :meth:`train_snapshot` handles this alignment automatically;
        use it unless you keep your own snapshots.
        """
        if (not math.isfinite(queue_imbalance)
                or not math.isfinite(trade_imbalance)
                or not math.isfinite(normalized_ofi)
                or not math.isfinite(momentum_z)
                or not math.isfinite(realized_return)):
            return
        # 1. Score BEFORE learning: genuinely out-of-sample.
        pred = self.predict(queue_imbalance, trade_imbalance,
                            normalized_ofi, momentum_z)
        self._mean_pred += self._ic_alpha * (pred - self._mean_pred)
        self._mean_ret += self._ic_alpha * (realized_return - self._mean_ret)
        dp = pred - self._mean_pred
        dr = realized_return - self._mean_ret
        self._var_pred += self._ic_alpha * (dp * dp - self._var_pred)
        self._var_ret += self._ic_alpha * (dr * dr - self._var_ret)
        self._covar += self._ic_alpha * (dp * dr - self._covar)
        # The scale seeds from the first nonzero |prediction| --
        # ramping from 0 would leave it ~an order of magnitude small
        # early and let normalized_prediction rail-pin at +/-1 on a
        # thin track record.
        ap = abs(pred)
        self._abs_pred_ewma = (ap if self._abs_pred_ewma == 0
                               else self._abs_pred_ewma
                               + self._ic_alpha * (ap - self._abs_pred_ewma))
        self._samples += 1

        # 2. Ridge-SGD step: w += lr*(error*x - lambda*w).
        err = realized_return - pred
        w = self._w
        w[0] += self._learning_rate * (err * queue_imbalance - self._ridge_lambda * w[0])
        w[1] += self._learning_rate * (err * trade_imbalance - self._ridge_lambda * w[1])
        w[2] += self._learning_rate * (err * normalized_ofi - self._ridge_lambda * w[2])
        w[3] += self._learning_rate * (err * momentum_z - self._ridge_lambda * w[3])

    def train_snapshot(self, queue_imbalance: float, trade_imbalance: float,
                       normalized_ofi: float, momentum_z: float,
                       realized_return: float) -> None:
        """The aligned learning step: trains on the ingredients
        snapshotted at the PREVIOUS call (which predate the interval
        ``realized_return`` covers), then snapshots the current
        ingredients for the next call. The first call only snapshots
        -- there is nothing aligned to train on yet. Call once per
        interval, with the return realized since the previous call;
        feeding it the current features directly would let the
        momentum echo of the return grade itself (see :meth:`train`).
        """
        if self._has_snapshot:
            qi, ti, ofi, mz = self._snapshot
            self.train(qi, ti, ofi, mz, realized_return)
        self._snapshot = (queue_imbalance, trade_imbalance,
                          normalized_ofi, momentum_z)
        self._has_snapshot = True

    # ------------------------------------------------------------------
    # Diagnostics and the normalized output
    # ------------------------------------------------------------------

    def out_of_sample_ic(self) -> float:
        """The prequential (out-of-sample) information coefficient:
        time-decayed correlation between the predictions made BEFORE
        each outcome and the outcomes themselves. The gate for using
        the learned alpha; 0 before enough variance exists to
        measure."""
        denom = math.sqrt(self._var_pred * self._var_ret)
        return self._covar / denom if denom > 0 else 0.0

    def normalized_prediction(self, queue_imbalance: float,
                              trade_imbalance: float,
                              normalized_ofi: float,
                              momentum_z: float) -> float:
        """The prediction scaled by its own typical magnitude and
        clamped to [-1, 1]. Returns 0 while the out-of-sample IC is
        not positive OR the track record is shorter than one IC memory
        (~1/ic_alpha samples): a learner that hasn't demonstrated live
        predictive power over a meaningful window emits no signal -- a
        lucky first hour is not evidence."""
        if (self._samples * self._ic_alpha < 1 or self.out_of_sample_ic() <= 0
                or self._abs_pred_ewma <= 0):
            return 0.0
        pred = self.predict(queue_imbalance, trade_imbalance,
                            normalized_ofi, momentum_z)
        return math_utils.clamp(pred / (2 * self._abs_pred_ewma), -1, 1)

    def weight(self, i: int) -> float:
        """The learned weight for feature ``i`` (0=queueImb,
        1=tradeImb, 2=OFI, 3=momZ)."""
        return float(self._w[i])

    def samples(self) -> int:
        return self._samples
