"""Ollama adapter — local inference via the Ollama REST API."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from llm.base import GenerationConfig, GenerationResult, LLMAdapter


class OllamaAdapter(LLMAdapter):
    """
    Calls a locally running Ollama server at base_url (default: http://localhost:11434).

    Requires Ollama to be installed and running:
      ollama serve
      ollama pull <model_id>

    Uses the /api/chat endpoint so system prompts are passed as a proper system message.
    No Python dependencies beyond the stdlib — all HTTP calls use urllib.
    """

    def __init__(
        self,
        model_id: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
    ):
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> None:
        """Verify Ollama is reachable and the requested model is available."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self._base_url}. "
                "Run 'ollama serve' to start it."
            ) from exc

        available = [m.get("name", "") for m in data.get("models", [])]
        # Ollama model names may include a tag (e.g. "llama3.2:1b"); match flexibly
        if not any(self._model_id in name or name.startswith(self._model_id) for name in available):
            raise RuntimeError(
                f"Model '{self._model_id}' is not available in Ollama. "
                f"Run 'ollama pull {self._model_id}' to download it. "
                f"Available: {available or ['(none)']}"
            )

    def _build_messages(self, prompt: str, config: GenerationConfig) -> list[dict]:
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Send a blocking /api/chat request and return the complete response."""
        payload = json.dumps({
            "model": self._model_id,
            "messages": self._build_messages(prompt, config),
            "stream": False,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        text = data.get("message", {}).get("content", "")
        tokens_used = data.get("eval_count", 0)

        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream tokens from /api/chat with stream=true (NDJSON response)."""
        payload = json.dumps({
            "model": self._model_id,
            "messages": self._build_messages(prompt, config),
            "stream": True,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Ollama stream failed: {exc}") from exc
