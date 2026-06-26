"""
Stage 1: Pre-Check — fast synchronous prompt screening before inference.

Three modes:
  hybrid      — regex first (sub-ms, catches instruction-seeking patterns), then
                unitary/toxic-bert if regex passes (~20-40ms). Best coverage. Default.
  regex       — taxonomy patterns only. Fast but brittle; misses novel phrasing.
  classifier  — toxic-bert only. Catches hate/threats but misses harmful instructions.

The pre-check is a first-pass gate. Anything that slips through is caught by Stage 2.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_TAXONOMY_PATH = Path(__file__).parent / "taxonomy" / "llama_guard_patterns.json"
_CLASSIFIER_MODEL = "unitary/toxic-bert"


@dataclass
class PreCheckResult:
    passed: bool
    category: str | None = None
    matched_pattern: str | None = None
    latency_ms: float = 0.0


class PreCheck:

    def __init__(
        self,
        taxonomy: str | Path = "llama_guard",
        mode: Literal["hybrid", "classifier", "regex"] = "hybrid",
        normalize_leet: bool = True,
        normalize_unicode: bool = True,
        classifier_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.mode = mode
        self.normalize_leet = normalize_leet
        self.normalize_unicode = normalize_unicode
        self._classifier_threshold = classifier_threshold
        self._device = device

        path = _TAXONOMY_PATH if taxonomy == "llama_guard" else Path(taxonomy)
        self._taxonomy = json.loads(path.read_text())
        self._leet_map: dict[str, str] = self._taxonomy.get("normalization", {}).get("leet_speak", {})
        self._rules = [
            (name, [kw.lower() for kw in spec.get("keywords", [])],
             [re.compile(p, re.IGNORECASE | re.DOTALL) for p in spec.get("patterns", [])])
            for name, spec in self._taxonomy["categories"].items()
        ]

        # Classifier loaded lazily on first use
        self._classifier = None

    def check(self, prompt: str) -> PreCheckResult:
        """Screen the prompt and return immediately if any rule fires."""
        start = time.perf_counter()

        if self.mode == "classifier":
            return self._check_classifier(prompt, start)
        if self.mode == "hybrid":
            result = self._check_regex(prompt, start)
            if not result.passed:
                return result
            return self._check_classifier(prompt, start)
        return self._check_regex(prompt, start)

    def _check_classifier(self, prompt: str, start: float) -> PreCheckResult:
        """Run toxic-bert; block if the toxicity score meets the threshold."""
        self._load_classifier()

        # Scan first and last 512-char windows so payloads hidden deep in long prompts aren't missed
        chunks = [prompt[:512]]
        if len(prompt) > 512:
            chunks.append(prompt[-512:])

        toxic_score = 0.0
        try:
            for chunk in chunks:
                for item in self._classifier(chunk):
                    label = (item.get("label", "") if isinstance(item, dict) else item.label).lower()
                    score = item.get("score", 0.0) if isinstance(item, dict) else item.score
                    if "toxic" in label and "non" not in label:
                        toxic_score = max(toxic_score, score)
                        break
        except Exception:
            # If classifier fails, fall back to regex rather than letting everything through
            return self._check_regex(prompt, start)

        elapsed = (time.perf_counter() - start) * 1000
        if toxic_score >= self._classifier_threshold:
            return PreCheckResult(
                passed=False,
                category="classifier_toxic",
                matched_pattern=f"toxic_score:{toxic_score:.3f}",
                latency_ms=elapsed,
            )
        return PreCheckResult(passed=True, latency_ms=elapsed)

    def _load_classifier(self) -> None:
        if self._classifier is not None:
            return
        from transformers import pipeline as hf_pipeline
        self._classifier = hf_pipeline(
            "text-classification",
            model=_CLASSIFIER_MODEL,
            device=self._device if self._device != "mps" else -1,  # HF pipeline uses -1 for CPU; MPS via torch
            truncation=True,
            max_length=512,
        )

    def _check_regex(self, prompt: str, start: float) -> PreCheckResult:
        normalised = self._normalise(prompt)
        # Leet normalization converts digits (0→o, 1→i, 3→e …), which breaks numeric patterns.
        # Keep a leet-free version (just lowercased + whitespace-collapsed) for those cases.
        lowered = " ".join(prompt.lower().split())

        for category, keywords, patterns in self._rules:
            for kw in keywords:
                # Word-boundary match prevents "ied" firing inside "died", etc.
                if re.search(r'\b' + re.escape(kw) + r'\b', normalised):
                    return PreCheckResult(
                        passed=False,
                        category=category,
                        matched_pattern=f"keyword:{kw}",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
            for pattern in patterns:
                # Try leet-normalized first; fall back to lowered for patterns using \d
                m = pattern.search(normalised) or pattern.search(lowered)
                if m:
                    return PreCheckResult(
                        passed=False,
                        category=category,
                        matched_pattern=m.group(0),
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

        return PreCheckResult(passed=True, latency_ms=(time.perf_counter() - start) * 1000)

    def _normalise(self, text: str) -> str:
        """Lowercase, collapse whitespace, NFC-normalize, and substitute l33tspeak."""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
            text = text.encode("ascii", errors="ignore").decode("ascii")
        text = " ".join(text.split())
        if self.normalize_leet:
            text = "".join(self._leet_map.get(ch, ch) for ch in text)
        return text.lower()
