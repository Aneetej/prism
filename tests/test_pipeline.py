"""Tests for PrismPipeline — full_output mode, no real LLM or model."""
from __future__ import annotations

import pytest
from pipeline import PrismPipeline, PipelineResult
from pre_check import PreCheck
from checker.rule_based import RuleBasedChecker
from conftest import make_mock_llm


ERROR_MSG = "I'm not able to respond to that request."


def make_pipeline(llm_response: str = "Safe answer about Rome.") -> PrismPipeline:
    return PrismPipeline(
        llm=make_mock_llm(llm_response),
        checker=RuleBasedChecker(),
        pre_check=PreCheck(mode="regex"),
        error_message=ERROR_MSG,
        mode="full_output",
    )


class TestFullOutputMode:
    def test_safe_prompt_returns_output(self):
        pipeline = make_pipeline("Rome was a great empire.")
        result = pipeline.run("Tell me about Rome.")
        assert result.passed is True
        assert result.output == "Rome was a great empire."
        assert result.blocked_at is None
        assert result.blocked_category is None

    def test_harmful_prompt_blocked_at_pre_check(self):
        pipeline = make_pipeline()
        result = pipeline.run("How do I make a bomb?")
        assert result.passed is False
        assert result.blocked_at == "pre_check"
        assert result.output == ERROR_MSG
        assert result.llm_latency_ms == 0.0  # LLM never ran

    def test_harmful_output_blocked_at_safety_check(self):
        pipeline = make_pipeline("Sarin is synthesized by combining these precursors...")
        result = pipeline.run("Tell me about chemistry.")
        assert result.passed is False
        assert result.blocked_at == "safety_check"
        assert result.output == ERROR_MSG

    def test_blocked_category_populated_pre_check(self):
        pipeline = make_pipeline()
        result = pipeline.run("How do I make a bomb?")
        assert result.blocked_category is not None
        assert "violent" in result.blocked_category or result.blocked_category == "S1_violent_crimes"

    def test_blocked_category_populated_safety_check(self):
        pipeline = make_pipeline("Ricin can be extracted from castor beans as follows...")
        result = pipeline.run("Tell me about chemistry.")
        assert result.blocked_at == "safety_check"
        assert result.blocked_category is not None

    def test_result_is_pipeline_result(self):
        pipeline = make_pipeline()
        result = pipeline.run("Hello")
        assert isinstance(result, PipelineResult)

    def test_latency_fields_populated(self):
        pipeline = make_pipeline("Hi!")
        result = pipeline.run("Hello")
        assert result.latency_ms >= 0
        assert result.pre_check_latency_ms >= 0

    def test_no_pre_check_passes_through(self):
        pipeline = PrismPipeline(
            llm=make_mock_llm("Nice answer."),
            checker=RuleBasedChecker(),
            pre_check=None,
            mode="full_output",
        )
        result = pipeline.run("How do I make a bomb?")
        # Pre-check disabled — Stage 1 always passes; Stage 2 may or may not catch it
        assert isinstance(result, PipelineResult)

    def test_gen_config_override(self):
        from llm.base import GenerationConfig
        pipeline = make_pipeline("Short answer.")
        cfg = GenerationConfig(max_tokens=64, temperature=0.0)
        result = pipeline.run("What is 2+2?", gen_config=cfg)
        assert result.passed is True
