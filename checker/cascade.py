"""CascadeChecker — fast BERT skip gate followed by Llama Guard authoritative judge."""
from __future__ import annotations

from checker.base import CheckResult, SafetyChecker


class CascadeChecker(SafetyChecker):
    """
    Two-tier Stage 2: fast classifier as a skip gate, slow judge for unclear cases.

    If the fast classifier's unsafe_prob falls below skip_below,  output is treated
    as clearly safe and the slow judge is skipped entirely.
    When unsafe_prob >= skip_below, the slow judge makes the final blocking decision.

    The fast checker never blocks — only the slow judge can reject output.

    Stage 1 pre-check is for obviously harmful cases via. the lexicon/regex filter. 
    """

    def __init__(self, fast: SafetyChecker, slow: SafetyChecker, skip_below: float = 0.4):
        self.fast = fast
        self.slow = slow
        self.skip_below = skip_below

    def check(self, prompt: str, output: str) -> CheckResult:
        fast_result = self.fast.check(prompt, output)
        if fast_result.unsafe_prob < self.skip_below:
            # Clearly safe — return BERT result directly, skip Llama Guard
            return fast_result
        # Borderline or suspicious — delegate to the authoritative judge
        return self.slow.check(prompt, output)
