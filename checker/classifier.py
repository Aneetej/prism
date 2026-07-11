"""Small transformer classifier (~125M params), single forward pass."""
from __future__ import annotations

import time

from checker.base import CheckResult, SafetyChecker

# Labels treated as "safe" across common moderation models
_SAFE_LABELS = {"safe", "ok", "non_toxic", "benign", "clean", "ham"}


class ClassifierChecker(SafetyChecker):

    def __init__(
        self,
        model_name: str = "KoalaAI/Text-Moderation",
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
                top_k=None,   # return all labels so we can detect the right one
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
        raw = self._pipeline(combined)

        # Normalise: pipeline returns [[{label, score},...]] with top_k=None
        # or [{label, score}] for single-label models — flatten to a list of dicts.
        entries: list[dict] = raw[0] if isinstance(raw[0], list) else raw

        unsafe_prob = self._unsafe_prob(entries)
        passed = unsafe_prob < self.threshold

        return CheckResult(
            passed=passed,
            confidence=max(unsafe_prob, 1.0 - unsafe_prob),
            category=None if passed else "classifier_flagged",
            latency_ms=(time.perf_counter() - start) * 1000,
            unsafe_prob=unsafe_prob,
        )

    @staticmethod
    def _unsafe_prob(entries: list[dict]) -> float:
        """Derive a single unsafe probability from any label schema.

        Handles:
          - Binary SAFE/UNSAFE  (e.g. custom fine-tunes)
          - toxic/non_toxic     (e.g. unitary/toxic-bert)
          - OK + harm categories (e.g. KoalaAI/Text-Moderation)
          - Any model where safe labels are in _SAFE_LABELS
        """
        # 1. Explicit SAFE label
        for e in entries:
            if e["label"].lower() == "safe":
                return 1.0 - e["score"]

        # 2. Explicit UNSAFE label
        for e in entries:
            if e["label"].lower() == "unsafe":
                return e["score"]

        # 3. toxic / non_toxic (unitary/toxic-bert style)
        for e in entries:
            lbl = e["label"].lower()
            if "toxic" in lbl and "non" not in lbl:
                return e["score"]

        # 4. OK label (KoalaAI style: "OK" = safe, everything else = unsafe signal)
        for e in entries:
            if e["label"].lower() == "ok":
                return 1.0 - e["score"]

        # 5. Fallback: any recognised safe label
        for e in entries:
            if e["label"].lower() in _SAFE_LABELS:
                return 1.0 - e["score"]

        # 6. Last resort: highest score among all entries treated as unsafe probability
        return max(e["score"] for e in entries)
