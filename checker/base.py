from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CheckResult:
    passed: bool
    confidence: float       #0.0–1.0
    category: str | None = None
    latency_ms: float = 0.0
    unsafe_prob: float = 0.0  #raw unsafe probability; used by CascadeChecker as a skip gate


class SafetyChecker(ABC):
    """
    Stage 2 safety checker interface.

    Receives the original prompt and the generated output (or a buffer window
    in sliding-window mode) so the checker has full context for its decision.
    """

    @abstractmethod
    def check(self, prompt: str, output: str) -> CheckResult:
        ...