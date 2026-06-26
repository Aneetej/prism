"""StreamManager — token buffer and streaming gate for sliding_window mode."""
from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class FallbackStrategy(str, Enum):
    CANNED = "canned"       # static error message
    TRUNCATE = "truncate"   # deliver verified portion + graceful close
    REGEN = "regen"         # caller-supplied regeneration function


@dataclass
class StreamConfig:
    fallback_strategy: FallbackStrategy = FallbackStrategy.CANNED
    canned_refusal: str = "I'm not able to respond to that request."
    graceful_close: str = " [Response ended.]"


_SENTINEL = object()


class StreamManager:
    """
    Thread-safe queue between the generation/checking thread and the user
    output thread. Verified tokens are enqueued here and consumed by stream().

    Used only in sliding_window mode. In full_output mode the pipeline blocks
    until generation completes and runs the checker once.
    """

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self._verified: queue.Queue = queue.Queue()
        self._halted = threading.Event()
        self._done = threading.Event()

    def enqueue_text(self, text: str) -> None:
        for ch in text:
            self._verified.put(ch)

    def fail(
        self,
        regen_fn: Callable[[], str] | None = None,
    ) -> None:
        self._halted.set()
        strategy = self.config.fallback_strategy

        if strategy == FallbackStrategy.CANNED:
            self.enqueue_text(self.config.canned_refusal)
        elif strategy == FallbackStrategy.TRUNCATE:
            self.enqueue_text(self.config.graceful_close)
        elif strategy == FallbackStrategy.REGEN:
            text = regen_fn() if regen_fn else self.config.canned_refusal
            self.enqueue_text(text)

        self.signal_done()

    def signal_done(self) -> None:
        self._done.set()
        self._verified.put(_SENTINEL)

    @property
    def is_halted(self) -> bool:
        return self._halted.is_set()

    def stream(self) -> Generator[str, None, None]:
        while True:
            item = self._verified.get()
            if item is _SENTINEL:
                return
            yield item

    def reset(self) -> None:
        self._halted.clear()
        self._done.clear()
        while not self._verified.empty():
            self._verified.get_nowait()