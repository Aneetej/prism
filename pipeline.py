"""
PRISM — Pre/post-inference Runtime Inference Safety Monitor.

Composes pre-check, LLM adapter, and safety checker into a single pipeline.

Two modes:
  full_output     — generate the complete response, run Stage 2 once, deliver or block.
  sliding_window  — generate token-by-token through StreamManager; Stage 2 runs on
                    each buffer window. Verified chunks stream to the user progressively.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from buffer import BufferConfig, TokenBuffer
from checker.base import CheckResult, SafetyChecker
from checker.rule_based import RuleBasedChecker
from llm.base import GenerationConfig, LLMAdapter
from pre_check import PreCheck, PreCheckResult
from stream_manager import StreamConfig, StreamManager


@dataclass
class PipelineResult:
    output: str
    passed: bool
    blocked_at: Literal["pre_check", "safety_check"] | None
    latency_ms: float
    checker_latency_ms: float
    pre_check_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    mode: str = "full_output"
    blocked_category: str | None = None
    borderline: bool = False        # True when checker confidence fell in the 0.4–0.6 range
    checker_confidence: float = 0.0 # raw confidence score from Stage 2 (0.0 when pre-check blocked)


class PrismPipeline:

    def __init__(
        self,
        llm: LLMAdapter,
        checker: SafetyChecker | None = None,
        pre_check: PreCheck | None = None,
        buffer_config: BufferConfig | None = None,
        stream_config: StreamConfig | None = None,
        gen_config: GenerationConfig | None = None,
        error_message: str = "I'm not able to respond to that request.",
        mode: Literal["full_output", "sliding_window"] = "full_output",
    ):
        self.llm = llm
        self.checker = checker or RuleBasedChecker()
        self.pre_check = pre_check
        self.buffer_config = buffer_config or BufferConfig()
        self.stream_config = stream_config or StreamConfig()
        self.gen_config = gen_config or GenerationConfig()
        self.error_message = error_message
        self.mode = mode
        # Token buffer is re-created per stream() call via reset() so state doesn't leak
        self._buf = TokenBuffer(self.buffer_config)

    def run(self, prompt: str, gen_config: GenerationConfig | None = None) -> PipelineResult:
        """Pre-check → LLM → safety check. Returns the verified output or the error message."""
        cfg = gen_config or self.gen_config
        wall_start = time.perf_counter()

        pre_result, pre_ms = self._run_pre_check(prompt)
        if not pre_result.passed:
            return PipelineResult(
                output=self.error_message,
                passed=False,
                blocked_at="pre_check",
                latency_ms=(time.perf_counter() - wall_start) * 1000,
                checker_latency_ms=0.0,
                pre_check_latency_ms=pre_ms,
                mode=self.mode,
                blocked_category=pre_result.category,
            )

        llm_start = time.perf_counter()
        gen_result = self.llm.generate(prompt, cfg)
        llm_ms = (time.perf_counter() - llm_start) * 1000

        check_start = time.perf_counter()
        check_result = self._run_checker(prompt, gen_result.text)
        checker_ms = (time.perf_counter() - check_start) * 1000

        borderline = 0.4 <= check_result.confidence <= 0.6

        if not check_result.passed:
            return PipelineResult(
                output=self.error_message,
                passed=False,
                blocked_at="safety_check",
                latency_ms=(time.perf_counter() - wall_start) * 1000,
                checker_latency_ms=checker_ms,
                pre_check_latency_ms=pre_ms,
                llm_latency_ms=llm_ms,
                mode=self.mode,
                blocked_category=check_result.category,
                borderline=borderline,
                checker_confidence=check_result.confidence,
            )

        return PipelineResult(
            output=gen_result.text,
            passed=True,
            blocked_at=None,
            latency_ms=(time.perf_counter() - wall_start) * 1000,
            checker_latency_ms=checker_ms,
            pre_check_latency_ms=pre_ms,
            llm_latency_ms=llm_ms,
            mode=self.mode,
            borderline=borderline,
            checker_confidence=check_result.confidence,
        )

    def stream(self, prompt: str) -> Iterator[str]:
        """Sliding-window mode: yields verified chunks, injects the error/truncation on FAIL.

        Each chunk yielded by the LLM adapter is treated as one token unit. This matches
        how HuggingFace's TextIteratorStreamer works (it yields sub-word pieces), so
        buffer_size=30 means ~30 sub-word tokens rather than 30 characters.

        On FAIL the StreamManager fallback strategy is honoured:
          CANNED   — yield the configured error message (default)
          TRUNCATE — yield a graceful close sentence so the stream ends cleanly
          REGEN    — yield the canned refusal (regen requires a caller-supplied function
                     and is not wired in here; callers should use the API for that path)
        """
        manager = StreamManager(self.stream_config)
        self._buf.reset()

        # Stage 1: pre-check — run synchronously before any tokens stream
        pre_result, _ = self._run_pre_check(prompt)
        if not pre_result.passed:
            yield self.stream_config.canned_refusal if self.error_message == "I'm not able to respond to that request." else self.error_message
            return

        # chunk_texts[i] holds the text for the i-th chunk (token unit)
        chunk_texts: list[str] = []
        verified_text: list[str] = []
        halted = False

        for chunk in self.llm.stream(prompt, self.gen_config):
            if halted:
                break

            chunk_texts.append(chunk)
            idx = len(chunk_texts) - 1
            ready = self._buf.push(idx)

            if ready:
                window_indices = self._buf.window()
                window_text = "".join(chunk_texts[i] for i in window_indices)
                context = prompt + "".join(verified_text)
                result = self._run_checker(context, window_text)

                if result.passed:
                    released_indices = self._buf.release()
                    released_text = "".join(chunk_texts[i] for i in released_indices)
                    verified_text.append(released_text)
                    yield released_text
                else:
                    halted = True
                    self._buf.drain()
                    manager.fail()
                    yield from manager.stream()
                    return

        # Drain remaining buffer at end of generation
        if not halted:
            remaining_indices = self._buf.drain()
            if remaining_indices:
                window_text = "".join(chunk_texts[i] for i in remaining_indices)
                result = self._run_checker(prompt + "".join(verified_text), window_text)
                if result.passed:
                    yield window_text
                else:
                    manager.fail()
                    yield from manager.stream()

    def _run_pre_check(self, prompt: str) -> tuple[PreCheckResult, float]:
        """Run Stage 1 and return (result, latency_ms). Passes through if pre_check is disabled."""
        if self.pre_check is None:
            from pre_check import PreCheckResult as _PCR
            return _PCR(passed=True), 0.0
        result = self.pre_check.check(prompt)
        return result, result.latency_ms

    def _run_checker(self, prompt: str, output: str) -> CheckResult:
        from checker.probe import RepresentationProbeChecker
        from llm.huggingface_adapter import HuggingFaceAdapter

        # Probe needs the activations the LLM produced during generation; pass them through if available
        if (
            isinstance(self.checker, RepresentationProbeChecker)
            and isinstance(self.llm, HuggingFaceAdapter)
            and self.llm.last_hidden_states is not None
        ):
            return self.checker.check(prompt, output, hidden_states=self.llm.last_hidden_states)
        return self.checker.check(prompt, output)


class ConfigurationError(ValueError):
    """Raised when an incompatible pipeline configuration is detected at startup."""


def validate_config(llm: LLMAdapter, checker) -> None:
    """Fail fast if the checker/adapter combination is incompatible.

    Currently enforces:
    - RepresentationProbeChecker requires HuggingFaceAdapter with expose_hidden_states=True
    """
    from checker.probe import RepresentationProbeChecker
    from llm.huggingface_adapter import HuggingFaceAdapter

    if isinstance(checker, RepresentationProbeChecker):
        if not isinstance(llm, HuggingFaceAdapter):
            raise ConfigurationError(
                "RepresentationProbeChecker requires HuggingFaceAdapter "
                f"(got {type(llm).__name__}). The probe reads hidden states from the "
                "generating model, so only the HuggingFace adapter is supported."
            )
        if not llm._expose_hidden_states:
            raise ConfigurationError(
                "RepresentationProbeChecker requires HuggingFaceAdapter to be "
                "initialized with expose_hidden_states=True."
            )


def from_config(config_path: str = "config.yaml") -> PrismPipeline:
    """Construct a PrismPipeline from config.yaml."""
    import os
    import yaml
    from llm.huggingface_adapter import HuggingFaceAdapter
    from llm.openai_adapter import OpenAIAdapter
    from llm.anthropic_adapter import AnthropicAdapter
    from llm.ollama_adapter import OllamaAdapter
    from checker.rule_based import RuleBasedChecker
    from checker.classifier import ClassifierChecker
    from checker.probe import RepresentationProbeChecker
    from checker.llama_guard import LlamaGuardChecker
    from checker.cascade import CascadeChecker
    from pre_check import PreCheck

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg["llm"]
    chk_cfg = cfg.get("safety_checker", {})
    pre_cfg = cfg.get("pre_check", {})
    exp_cfg = cfg.get("experiment", {})
    device = chk_cfg.get("device", "cpu")
    model_path = chk_cfg.get("model_path", "")
    threshold = chk_cfg.get("confidence_threshold", 0.8)

    def _stub_error(name: str):
        raise NotImplementedError(
            f"{name} adapter is not implemented in this release. "
            f"See llm/{name}_adapter.py to implement it."
        )

    _adapters = {
        "huggingface": lambda: HuggingFaceAdapter(
            model_id=llm_cfg["model_id"],
            hf_token=os.environ.get(llm_cfg.get("hf_token_env", "HF_TOKEN")),
            device=llm_cfg.get("device", "cpu"),
        ),
        "openai":    lambda: OpenAIAdapter(
            model_id=llm_cfg["model_id"],
            api_key_env=llm_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
        ),
        "anthropic": lambda: _stub_error("anthropic"),
        "ollama":    lambda: OllamaAdapter(
            model_id=llm_cfg["model_id"],
            base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
        ),
    }
    hf_token = os.environ.get(llm_cfg.get("hf_token_env", "HF_TOKEN"))
    llm_device = llm_cfg.get("device", "cpu")
    _checkers = {
        "rule_based":  lambda: RuleBasedChecker(),
        "classifier":  lambda: ClassifierChecker(model_name=model_path or "KoalaAI/Text-Moderation", device=device, threshold=threshold),
        "probe":       lambda: RepresentationProbeChecker.from_file(model_path) if model_path and os.path.exists(model_path) else RepresentationProbeChecker(),
        "llama_guard": lambda: LlamaGuardChecker(device=llm_device, hf_token=hf_token),
        "cascade":     lambda: CascadeChecker(
            fast=ClassifierChecker(model_name="KoalaAI/Text-Moderation", device="cpu", threshold=0.5),
            slow=LlamaGuardChecker(device=llm_device, hf_token=hf_token),
            skip_below=chk_cfg.get("cascade_skip_below", 0.4),
        ),
    }

    provider = llm_cfg["provider"]
    if provider not in _adapters:
        raise ValueError(f"Unknown provider: {provider}")
    chk_type = chk_cfg.get("type", "rule_based")
    if chk_type not in _checkers:
        raise ValueError(f"Unknown checker type: {chk_type}")

    pre_check = PreCheck(
        taxonomy=pre_cfg.get("taxonomy", "llama_guard"),
        mode=pre_cfg.get("mode", "classifier"),
        normalize_leet=pre_cfg.get("normalize_leet", True),
        normalize_unicode=pre_cfg.get("normalize_unicode", True),
        classifier_threshold=pre_cfg.get("classifier_threshold", 0.5),
        device=device,
    ) if pre_cfg.get("enabled", True) else None

    mode = exp_cfg.get("mode", "full_output")
    if mode == "both":
        mode = "full_output"

    llm_instance = _adapters[provider]()
    checker_instance = _checkers[chk_type]()
    validate_config(llm_instance, checker_instance)

    return PrismPipeline(
        llm=llm_instance,
        checker=checker_instance,
        pre_check=pre_check,
        buffer_config=BufferConfig(buffer_size=exp_cfg.get("buffer_size", 30), overlap=exp_cfg.get("overlap", 5)),
        gen_config=GenerationConfig(max_tokens=llm_cfg.get("max_tokens", 512), temperature=llm_cfg.get("temperature", 0.7)),
        error_message=cfg.get("error_message", "I'm not able to respond to that request."),
        mode=mode,
    )