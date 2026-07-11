"""OpenAI adapter — wraps GPT-4o (or any OpenAI chat model) behind the LLMAdapter interface."""
from __future__ import annotations

import os
import time
from collections.abc import Iterator

from llm.base import GenerationConfig, GenerationResult, LLMAdapter


class OpenAIAdapter(LLMAdapter):
    """
    LLMAdapter backed by the OpenAI Chat Completions API.

    Requires the `openai` package (pip install openai>=1.0.0) and an API key in
    the environment variable named by `api_key_env` (default: OPENAI_API_KEY).

    Works with any OpenAI chat model (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.).
    The rest of the PRISM pipeline requires no changes — just swap the adapter.

    Example:
        pipeline = PrismPipeline(
            llm=OpenAIAdapter(model_id="gpt-4o-mini"),
            checker=CascadeChecker(...),
            pre_check=PreCheck(),
        )
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self):
        """Lazily initialise the OpenAI client on first call."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIAdapter requires the openai package. "
                "Install with: pip install openai>=1.0.0"
            ) from exc

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"OpenAI API key not found. Set the {self._api_key_env} environment variable."
            )
        self._client = OpenAI(api_key=api_key)

    def _build_messages(self, prompt: str, config: GenerationConfig) -> list[dict]:
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Single blocking call; returns the complete assistant response."""
        self._load()

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=self._build_messages(prompt, config),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""
        tokens_used = response.usage.completion_tokens if response.usage else len(text.split())

        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream response chunks via the OpenAI streaming API."""
        self._load()

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=self._build_messages(prompt, config),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
