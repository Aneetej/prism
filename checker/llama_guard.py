"""Llama Guard 3 1B, Meta's purpose-built prompt+output safety classifier."""
from __future__ import annotations

import time
from typing import Optional

from checker.base import CheckResult, SafetyChecker


class LlamaGuardChecker(SafetyChecker):
    """
    Output format from the model is to give "safe" or "unsafe\\nS{n}" where n is the taxonomy category.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Guard-3-1B",
        device: str = "cpu",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 20,
    ):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Load Llama Guard weights; uses MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise."""
        if self._model is not None:
            return

        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = self.hf_token or os.environ.get("HF_TOKEN")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)

        if self.device == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map="cpu",
                torch_dtype=torch.float16,
            )
            if self.device == "mps":
                self._model = self._model.to("mps")

        self._model.eval()

        import logging
        logging.getLogger(__name__).info(
            "LlamaGuardChecker loaded on device: %s", next(self._model.parameters()).device
        )

    def check(self, prompt: str, output: str) -> CheckResult:
        """Run Llama Guard on the prompt+output pair; parse its safe/unsafe verdict."""
        start = time.perf_counter()
        self._load()

        import torch

        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        first_line = raw.splitlines()[0].strip().lower() if raw else ""
        passed = first_line == "safe"

        category = None
        if not passed:
            lines = raw.strip().splitlines()
            #Second line is the violated category code, e.g. "S1" or "S4,S6"
            category = lines[1].strip() if len(lines) > 1 else "llama_guard_flagged"

        return CheckResult(
            passed=passed,
            confidence=0.9,
            category=category,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
