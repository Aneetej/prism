"""HuggingFace adapter — the only concrete LLM adapter in this phase."""
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from llm.base import GenerationConfig, GenerationResult, LLMAdapter

if TYPE_CHECKING:
    pass


class HuggingFaceAdapter(LLMAdapter):

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        hf_token: str | None = None,
        device: str = "cpu",
        load_in_4bit: bool = False,
        expose_hidden_states: bool = False,
    ):
        self._model_id = model_id
        self._hf_token = hf_token
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._expose_hidden_states = expose_hidden_states
        self._model = None
        self._tokenizer = None
        self.last_hidden_states = None  # populated when expose_hidden_states=True

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> None:
        """Load the tokenizer and model on first call; no-op on subsequent calls."""
        if self._model is not None:
            return

        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = self._hf_token or os.environ.get("HF_TOKEN")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, token=token
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quant_cfg = None
        if self._load_in_4bit and self._device != "cpu":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
            )

        import torch

        if self._device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
        else:
            resolved = self._device

        if resolved == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                token=token,
                quantization_config=quant_cfg,
                device_map="auto",
                output_hidden_states=self._expose_hidden_states,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                token=token,
                device_map="cpu",
                output_hidden_states=self._expose_hidden_states,
            )
            if resolved == "mps":
                self._model = self._model.to("mps")

        self._model.eval()
        import logging
        logging.getLogger(__name__).info("LLM loaded on device: %s", next(self._model.parameters()).device)

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate a complete response in one shot and return it with token count and latency."""
        self._load()
        import torch

        full_prompt = f"<|system|>\n{config.system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n" if config.system_prompt else prompt
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature if config.temperature > 0 else None,
                top_p=config.top_p if config.temperature > 0 else None,
                do_sample=config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                output_hidden_states=self._expose_hidden_states,
                return_dict_in_generate=self._expose_hidden_states,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if self._expose_hidden_states:
            # Store mean-pooled hidden states from the middle layer for the probe
            import numpy as np
            hidden = outputs.hidden_states  # tuple of (num_layers, batch, seq, hidden)
            mid = len(hidden) // 2
            self.last_hidden_states = hidden[mid][0].mean(dim=0).cpu().numpy()
            sequences = outputs.sequences
        else:
            sequences = outputs

        input_len = inputs["input_ids"].shape[1]
        new_tokens = sequences[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            tokens_used=len(new_tokens),
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream tokens via TextIteratorStreamer running in a background thread."""
        self._load()
        import threading
        import torch
        from transformers import TextIteratorStreamer

        full_prompt = f"<|system|>\n{config.system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n" if config.system_prompt else prompt
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(
            target=lambda: self._model.generate(**gen_kwargs),
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()
