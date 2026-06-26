from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    system_prompt: str | None = None
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    tokens_used: int
    model_id: str
    latency_ms: float = 0.0


class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult: ...

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...