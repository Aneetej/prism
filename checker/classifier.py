"""Small transformer classifier (~125M params), single forward pass."""
from __future__ import annotations

import time

from checker.base import CheckResult, SafetyChecker

_UNSAFE_LABEL = "UNSAFE"
_SAFE_LABEL = "SAFE"


class ClassifierChecker(SafetyChecker):

    def __init__(
        self,
        #Model selected given alignment to problem. Not 100% alignment, but not bad. Looks at toxicity like hate speech, doesn't cover all harms.
        model_name: str = "unitary/toxic-bert",
        device: str = "cpu",
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.max_length = max_length
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=self.max_length,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ClassifierChecker requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from exc

    def check(self, prompt: str, output: str) -> CheckResult:
        """Concatenate prompt + output and run a single classifier forward pass."""
        start = time.perf_counter()
        self._load()

        combined = f"[PROMPT] {prompt} [RESPONSE] {output}"
        result = self._pipeline(combined)[0]
        label: str = result["label"]
        score: float = result["score"]

        unsafe_prob = 1.0 - score if label == _SAFE_LABEL else score
        passed = unsafe_prob < self.threshold

        return CheckResult(
            passed=passed,
            confidence=max(unsafe_prob, 1.0 - unsafe_prob),
            category=None if passed else "classifier_flagged",
            latency_ms=(time.perf_counter() - start) * 1000,
            unsafe_prob=unsafe_prob,
        )