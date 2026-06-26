"""
Representation probe.

Uses the generating model's own hidden-state activations via a trained linear
probe. Only applicable with HuggingFaceAdapter (expose_hidden_states=True).
Overhead: one matrix multiply per check (<2ms). Falls back to a secondary
checker when hidden states are unavailable.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from checker.base import CheckResult, SafetyChecker

if TYPE_CHECKING:
    import numpy.typing as npt


class RepresentationProbeChecker(SafetyChecker):
    """
    Linear probe on intermediate layer activations of the generating model.

    Load at inference time with `RepresentationProbeChecker.from_file(path)`.

    At check time, pass the mean-pooled hidden states from the generating model
    via the `hidden_states` keyword argument. When hidden_states is None (e.g.,
    API-only inference), falls back to `fallback_checker`.
    """

    def __init__(
        self,
        layer_index: int = 16,
        hidden_dim: int = 4096,
        threshold: float = 0.5,
        fallback_checker: SafetyChecker | None = None,
    ):
        self.layer_index = layer_index
        self.threshold = threshold
        self.fallback_checker = fallback_checker
        self.w: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.b: float = 0.0
        self._trained = False

    #  Weight I/O

    def save(self, path: str | Path) -> None:
        np.savez(str(path), w=self.w, b=np.array([self.b]))

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "RepresentationProbeChecker":
        data = np.load(str(path))
        probe = cls(hidden_dim=data["w"].shape[0], **kwargs)
        probe.w = data["w"]
        probe.b = float(data["b"][0])
        probe._trained = True
        return probe


    def check(
        self,
        prompt: str,
        output: str,
        hidden_states: "npt.NDArray[np.float32] | None" = None,
    ) -> CheckResult:
        start = time.perf_counter()

        if hidden_states is None or not self._trained:
            if self.fallback_checker is not None:
                result = self.fallback_checker.check(prompt, output)
                result.category = result.category or "probe_fallback"
                return result
            return CheckResult(
                passed=False,
                confidence=0.5,
                category="probe_unavailable",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        unsafe_prob = float(self._sigmoid(float(np.dot(self.w, hidden_states)) + self.b))
        passed = unsafe_prob < self.threshold
        return CheckResult(
            passed=passed,
            confidence=max(unsafe_prob, 1.0 - unsafe_prob),
            category=None if passed else "probe_flagged",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


#Train offline

def train_probe(
    safe_activations: "npt.NDArray[np.float32]",
    unsafe_activations: "npt.NDArray[np.float32]",
    C: float = 1.0,
    max_iter: int = 1000,
) -> RepresentationProbeChecker:
    """
    Train a logistic regression probe from pre-extracted hidden states.

    safe_activations   : (N_safe,   hidden_dim) float32
    unsafe_activations : (N_unsafe, hidden_dim) float32
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([safe_activations, unsafe_activations])
    y = np.array([0] * len(safe_activations) + [1] * len(unsafe_activations))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_scaled, y)

    hidden_dim = X.shape[1]
    probe = RepresentationProbeChecker(hidden_dim=hidden_dim)
    w_eff = (clf.coef_[0] / scaler.scale_).astype(np.float32)
    b_eff = float(clf.intercept_[0]) - float(np.dot(w_eff, scaler.mean_))
    probe.w = w_eff
    probe.b = b_eff
    probe._trained = True
    return probe