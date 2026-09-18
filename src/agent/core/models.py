"""
models.py — Base agent shared models.

SlotAttempt is the core per-slot runtime tracker used by ALL agents.
  - attempt_count increments on every failure
  - Confirmed slots emit CallAgentField events via _build()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent.core.constants import MAX_SLOT_ATTEMPTS


@dataclass
class SlotAttempt:
    """
    Per-slot retry tracker.

    attempt_count increments on every failure.
    Confirmed slots are serialized via slots_dict() and persisted in LangGraph state.
    """

    slot_name: str
    max_attempts: int = MAX_SLOT_ATTEMPTS
    attempt_count: int = 0
    confirmed: bool = False
    last_value: Optional[Any] = None

    def record_attempt(
        self, value: Any, success: bool, is_asr: bool = False, reason: Optional[str] = None
    ) -> None:
        self.last_value = value
        if success:
            self.confirmed = True
        else:
            self.attempt_count += 1

    def is_exhausted(self) -> bool:
        return self.attempt_count >= self.max_attempts

    def remaining(self) -> int:
        return max(0, self.max_attempts - self.attempt_count)

    def reset(self) -> None:
        self.attempt_count = 0
        self.confirmed = False
        self.last_value = None
