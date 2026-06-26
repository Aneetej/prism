"""OpenAI adapter stub — interface only, not implemented this phase."""
from __future__ import annotations

from collections.abc import Iterator

from llm.base import GenerationConfig, GenerationResult, LLMAdapter


class OpenAIAdapter(LLMAdapter):
    """
    Stub for the OpenAI API adapter.

    To implement: pip install openai, then fill in generate() and stream()
    using the openai.OpenAI client. The rest of the pipeline requires no changes.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self._model_id = model_id
        self._api_key_env = api_key_env

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement using the openai SDK."
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        raise NotImplementedError(
            "OpenAIAdapter is a stub. Implement using the openai SDK."
        )