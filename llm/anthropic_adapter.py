"""Anthropic adapter stub — interface only, not implemented this phase."""
from __future__ import annotations

from collections.abc import Iterator

from llm.base import GenerationConfig, GenerationResult, LLMAdapter


class AnthropicAdapter(LLMAdapter):
    """
    Stub for the Anthropic API adapter.

    To implement: pip install anthropic, then fill in generate() and stream()
    using the anthropic.Anthropic client with the Messages API. The rest of the
    pipeline requires no changes.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        self._model_id = model_id
        self._api_key_env = api_key_env

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        raise NotImplementedError(
            "AnthropicAdapter is a stub. Implement using the anthropic SDK."
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        raise NotImplementedError(
            "AnthropicAdapter is a stub. Implement using the anthropic SDK."
        )