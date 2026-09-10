"""
conversation_context.py — Accumulated conversation state for humanized responses.

ConversationContext
    Lightweight session-level accumulator. Survives across turns and agent
    switches because it is serialized into LangGraph state as a plain dict.
    Tracks: confirmed slots, caller name.
    Drives name personalization in the response builder.

No external calls. Pure Python, zero latency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from agent.state import State


# ---------------------------------------------------------------------------
# Conversation Context
# ---------------------------------------------------------------------------


@dataclass
class ConversationContext:
    """
    Accumulated conversational state that persists across ALL turns
    within a session. Serialized as a plain dict in LangGraph state.

    This is what the response builder uses to make responses feel like
    they belong to the same continuous conversation, not isolated exchanges.
    """

    # Turn tracking
    session_turn_count: int = 0  # total turns since session start
    agent_turn_count: int = 0  # turns since the current agent started

    # Slot state awareness
    confirmed_slots: List[str] = field(default_factory=list)  # names of confirmed slots
    total_slots_in_pipeline: int = 0  # how many slots to collect total (for "almost there" cues)

    # Personalization
    caller_first_name: str = ""  # set once first_name slot confirmed

    # Current agent tracking
    active_agent_name: str = ""

    # LLM-generated recovery message for the current turn (cleared after use)
    llm_recovery_message: str = ""

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------

    @property
    def slots_remaining(self) -> int:
        """Slots not yet confirmed (when total is known)."""
        if not self.total_slots_in_pipeline:
            return 99
        return max(0, self.total_slots_in_pipeline - len(self.confirmed_slots))

    @property
    def is_final_slot(self) -> bool:
        """True if only one slot remains in the collection pipeline."""
        return self.slots_remaining == 1

    @property
    def should_use_name(self) -> bool:
        """
        Whether to use the caller's name in this response.
        Follows natural cadence — not every turn, but at key moments.
        """
        if not self.caller_first_name:
            return False
        # Always use name on the turn immediately after first_name confirmed
        if "first_name" in self.confirmed_slots and len(self.confirmed_slots) == 1:
            return True
        # Use name for the final slot ("Almost done, Emily — just your date of birth")
        if self.is_final_slot:
            return True
        # Otherwise use name probabilistically to feel natural, not robotic
        return random.random() < 0.4

    # -------------------------------------------------------------------------
    # Mutation methods
    # -------------------------------------------------------------------------

    def record_slot_success(self, slot_name: str) -> None:
        """Call when a slot is confirmed successfully."""
        if slot_name not in self.confirmed_slots:
            self.confirmed_slots.append(slot_name)

    def update_caller_name(self, name: str) -> None:
        """Set first name once confirmed."""
        # if name and not self.caller_first_name:
        if name:
            self.caller_first_name = name.strip().title()

    def increment_turn(self, agent_name: str = "") -> None:
        """Call at the start of each response generation."""
        self.session_turn_count += 1
        if agent_name and agent_name == self.active_agent_name:
            self.agent_turn_count += 1
        elif agent_name:
            self.active_agent_name = agent_name
            self.agent_turn_count = 1

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "session_turn_count": self.session_turn_count,
            "agent_turn_count": self.agent_turn_count,
            "confirmed_slots": self.confirmed_slots,
            "total_slots_in_pipeline": self.total_slots_in_pipeline,
            "caller_first_name": self.caller_first_name,
            "active_agent_name": self.active_agent_name,
            "llm_recovery_message": self.llm_recovery_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        if not data:
            return cls()
        return cls(
            session_turn_count=data.get("session_turn_count", 0),
            agent_turn_count=data.get("agent_turn_count", 0),
            confirmed_slots=data.get("confirmed_slots", []),
            total_slots_in_pipeline=data.get("total_slots_in_pipeline", 0),
            caller_first_name=data.get("caller_first_name", ""),
            active_agent_name=data.get("active_agent_name", ""),
            llm_recovery_message=data.get("llm_recovery_message", ""),
        )

    @classmethod
    def from_state(cls, state: "State") -> "ConversationContext":
        """Load from LangGraph state dict. Returns a fresh context if not set."""
        raw = state.get("conversation_context")
        if raw and isinstance(raw, dict):
            ctx = cls.from_dict(raw)
        else:
            ctx = cls()
        # Sync caller name from state if already confirmed but context is new
        if not ctx.caller_first_name and state.get("first_name"):
            ctx.caller_first_name = (state.get("first_name") or "").strip().title()
        return ctx
