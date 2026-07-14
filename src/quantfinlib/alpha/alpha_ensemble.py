"""IC-weighted alpha ensemble (port of Java
``microstructure.AlphaEnsemble``) -- the layer above the individual
signals. A desk rarely trades one alpha; it blends several, and the
blending question is the same honesty problem
:class:`~quantfinlib.alpha.online_alpha_learner.OnlineAlphaLearner`
solves for its weights: **how much should each component be trusted,
on evidence it could not have memorized?**

The ensemble runs one prequential IC per component: each interval,
:meth:`on_observation` scores the component values snapshotted at the
PREVIOUS call against the return just realized (values now would
contain the move -- the same nowcast trap ``OnlineAlphaLearner``
closes), then snapshots the current values for the next round.
:meth:`combined` weights each component by ``max(0, IC)`` and the
weights ARE the sizing -- deliberately NOT renormalized to sum to 1,
because renormalizing would let a lone component with IC 0.01 emit at
full strength: a barely-trusted blend must be a barely-sized signal. A
component that has not demonstrated live predictive power gets zero
weight, and while the track record spans less than one IC memory the
ensemble emits 0 outright. Output is clamped to [-1, 1].

Same caveat as every learned signal here: the live IC is a tripwire,
not a validation -- run any blend you intend to trade through
:mod:`quantfinlib.alpha.alpha_validation`'s walk-forward machinery.
Components must be dimensionless (~[-1, 1]); non-finite inputs are
handled per component (see :meth:`on_observation`).

Note: the Java class persists its IC evidence via
``persist.Checkpoint``; there is no ``persist`` lane in this port, so
``write_state``/``read_state`` are not carried over (see
:mod:`quantfinlib.microstructure.kyles_lambda` for the same
documented omission).
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils


class AlphaEnsemble:
    """IC-weighted blend of dimensionless component signals; see the
    module docstring."""

    __slots__ = ("_components", "_ic_alpha", "_mean_sig", "_mean_ret",
                 "_var_sig", "_var_ret", "_covar", "_snapshot",
                 "_has_snapshot", "_samples")

    def __init__(self, components: int, ic_alpha: float = 0.01) -> None:
        """``ic_alpha``: EWMA weight of the IC statistics, e.g. 0.01
        (~a few-hundred-observation memory)."""
        if components < 1 or ic_alpha <= 0 or ic_alpha > 1:
            raise ValueError("need components >= 1, icAlpha in (0,1]")
        self._components = components
        self._ic_alpha = ic_alpha
        self._mean_sig = np.zeros(components)
        self._mean_ret = np.zeros(components)
        self._var_sig = np.zeros(components)
        self._var_ret = np.zeros(components)
        self._covar = np.zeros(components)
        self._snapshot = np.zeros(components)
        self._has_snapshot = False
        self._samples = 0

    def on_observation(self, values, realized_return: float) -> None:
        """One interval: the current component values and the return
        realized since the previous call. Scores the PREVIOUS snapshot
        against the return (honest alignment), then snapshots
        ``values``. The first call only snapshots. Non-finite handling
        is per-component: a NaN component skips ITS scoring (each
        component's moments stay conditioned on exactly the returns
        its covariance saw) while finite siblings still score -- but
        an observation where NOTHING scored (NaN return, or every
        snapshot value non-finite) does not count toward the track
        record: the gate must never open on evidence that scored
        nothing."""
        v = np.asarray(values, dtype=float)
        self._require_length(v)
        if self._has_snapshot and math.isfinite(realized_return):
            scored = False
            for c in range(self._components):
                s = self._snapshot[c]
                if not math.isfinite(s):
                    continue
                self._mean_sig[c] += self._ic_alpha * (s - self._mean_sig[c])
                self._mean_ret[c] += self._ic_alpha * (realized_return - self._mean_ret[c])
                ds = s - self._mean_sig[c]
                dr = realized_return - self._mean_ret[c]
                self._var_sig[c] += self._ic_alpha * (ds * ds - self._var_sig[c])
                self._var_ret[c] += self._ic_alpha * (dr * dr - self._var_ret[c])
                self._covar[c] += self._ic_alpha * (ds * dr - self._covar[c])
                scored = True
            if scored:
                self._samples += 1
        self._snapshot[:self._components] = v[:self._components]
        self._has_snapshot = True

    def combined(self, values) -> float:
        """The blended alpha in [-1, 1]:
        ``clamp(sum(max(0, IC_c) * value_c))``. The IC weights are the
        SIZE of the signal, not just its mix (see the class doc for
        why they are not renormalized). 0 while the track record spans
        less than one IC memory or no component has a positive IC --
        an unproven blend is silent, exactly like the learner it sits
        above."""
        v = np.asarray(values, dtype=float)
        self._require_length(v)
        if self._samples * self._ic_alpha < 1:
            return 0.0
        weighted = 0.0
        for c in range(self._components):
            ic = self.component_ic(c)
            if ic > 0 and math.isfinite(v[c]):
                weighted += ic * v[c]
        return math_utils.clamp(weighted, -1, 1)

    def component_ic(self, c: int) -> float:
        """The prequential (out-of-sample) IC of one component -- the
        trust diagnostic per signal. 0 before enough variance
        exists."""
        denom = math.sqrt(self._var_sig[c] * self._var_ret[c])
        return self._covar[c] / denom if denom > 0 else 0.0

    def components(self) -> int:
        return self._components

    def samples(self) -> int:
        return self._samples

    def _require_length(self, a: np.ndarray) -> None:
        if a.shape[0] < self._components:
            raise ValueError(
                f"array has {a.shape[0]} entries, ensemble has "
                f"{self._components}")
