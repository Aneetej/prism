"""TokenBuffer — rolling FIFO window with overlap for context continuity."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BufferConfig:
    buffer_size: int = 30       # N: tokens before a check is triggered
    overlap: int = 5            # K: tokens retained from previous window


class TokenBuffer:
    """
    Accumulates generated tokens and exposes windows for safety checking.

    At each generation step a new token is appended. When the buffer reaches
    `buffer_size`, a check window is produced containing all N tokens. After a
    passed check, the oldest N-K tokens are released (returned) and removed from
    the buffer. The trailing K tokens are retained as context for the next window.

    This overlap ensures harmful content split across window boundaries is always
    seen in at least one check.
    """

    def __init__(self, config: BufferConfig | None = None):
        self.config = config or BufferConfig()
        self._buf: deque[int] = deque()
        self._total_added: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, token_id: int) -> bool:
        """Append a token. Returns True when the buffer is ready for a check."""
        self._buf.append(token_id)
        self._total_added += 1
        return len(self._buf) >= self.config.buffer_size

    def window(self) -> list[int]:
        """Current buffer contents as a list (check input)."""
        return list(self._buf)

    def release(self) -> list[int]:
        """
        Release the oldest buffer_size-overlap tokens after a passed check.
        Returns the released token IDs (to be forwarded to the verified queue).
        """
        release_count = self.config.buffer_size - self.config.overlap
        released: list[int] = []
        for _ in range(min(release_count, len(self._buf))):
            released.append(self._buf.popleft())
        return released

    def drain(self) -> list[int]:
        """Drain and return all remaining tokens (used on FAIL or end-of-generation)."""
        tokens = list(self._buf)
        self._buf.clear()
        return tokens

    def reset(self) -> None:
        self._buf.clear()
        self._total_added = 0