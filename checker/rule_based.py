#Rule-based: regex/keyword matching on the generated output using the lexicons provided and examples from testing
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from checker.base import CheckResult, SafetyChecker

_TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy" / "llama_guard_patterns.json"

#To fix: What if I want the historical background of these chemicals? Not directly harmful -- should go through.
_KEYWORD_BLOCKLIST = {
    "sarin", "ricin", "anthrax", "novichok", "tabun", "csam",
}


class RuleBasedChecker(SafetyChecker):

    def __init__(
        self,
        taxonomy_path: str | Path = _TAXONOMY_PATH,
        confidence_on_match: float = 0.97,
    ):
        self.confidence_on_match = confidence_on_match
        taxonomy = json.loads(Path(taxonomy_path).read_text())
        self._rules = self._compile(taxonomy["categories"])

    def check(self, prompt: str, output: str) -> CheckResult:
        """Scan output (and regex patterns over full context) for blocked content.

        The keyword blocklist runs only on the generated output so that a question like
        "What happened in the Tokyo sarin attack?" doesn't trigger a block when the
        model's response is a benign historical summary. Regex patterns run over the
        full prompt+output so cross-turn injection patterns are caught.
        """
        start = time.perf_counter()
        output_lower = output.lower()
        full_text = prompt + " " + output

        for kw in _KEYWORD_BLOCKLIST:
            if kw in output_lower:
                return CheckResult(
                    passed=False,
                    confidence=self.confidence_on_match,
                    category="blocklist_keyword",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

        for category, keywords, patterns in self._rules:
            for kw in keywords:
                if kw in output_lower:
                    return CheckResult(
                        passed=False,
                        confidence=self.confidence_on_match,
                        category=category,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
            for pattern in patterns:
                if pattern.search(full_text):
                    return CheckResult(
                        passed=False,
                        confidence=self.confidence_on_match,
                        category=category,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

        return CheckResult(
            passed=True,
            confidence=0.7,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _compile(categories: dict) -> list[tuple[str, list[str], list[re.Pattern]]]:
        rules = []
        for name, spec in categories.items():
            keywords = [kw.lower() for kw in spec.get("keywords", [])]
            patterns = [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in spec.get("patterns", [])
            ]
            rules.append((name, keywords, patterns))
        return rules