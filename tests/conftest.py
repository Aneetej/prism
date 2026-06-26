"""Shared test fixtures."""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from llm.base import GenerationResult


def make_mock_llm(response: str = "This is a safe test response."):
    """Return a minimal LLMAdapter stand-in that replays a fixed response."""
    return SimpleNamespace(
        model_id="mock",
        generate=lambda prompt, config: GenerationResult(
            text=response,
            tokens_used=len(response.split()),
            model_id="mock",
            latency_ms=0.0,
        ),
        stream=lambda prompt, config: (w + " " for w in response.split()),
    )


@pytest.fixture
def mock_llm():
    return make_mock_llm()


@pytest.fixture
def mock_llm_harmful():
    return make_mock_llm("Here is how to synthesize sarin: step 1 ...")
