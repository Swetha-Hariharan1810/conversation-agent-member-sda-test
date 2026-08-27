"""
agent.py — BaseAgent: abstract conversational agent.

Composes three mixins into a single base class all agents inherit from:
  ConversationGuardsMixin  (guards.py)    — abuse, ASR, transfer detection
  SlotManagerMixin         (slot_manager.py) — slot state + _collect_slot
  SignalsMixin             (signals.py)   — ask_member, signal_complete/escalate

To build a new agent:
  1. Inherit from BaseAgent
  2. Set AGENT_NAME = "my_agent_name"
  3. Implement async run(self, state) -> dict
  4. Use self._collect_slot(), self.signal_complete(), etc. — they're all inherited

No agent-specific logic lives here. This class is infrastructure only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Set

from agent.core.guards import ConversationGuardsMixin
from agent.core.models import SlotAttempt
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin
from agent.logger import get_logger
from agent.state import State


class BaseAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin, ABC):
    """Abstract base for all conversational agents."""

    AGENT_NAME: str = "base_agent"
    SUPPORTED_TOPICS: Set[str] = set()

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._slots: Dict[str, SlotAttempt] = {}
        self._newly_confirmed: Set[str] = set()
        self._pending_ambiguous_resets: Set[str] = set()

    @classmethod
    def from_state(cls, state: State) -> "BaseAgent":
        """Create an instance with slot state restored from LangGraph state."""
        instance = cls()
        instance._slots = {k: cls._restore_slot(k, v) for k, v in (state.get("slot_attempts") or {}).items()}
        instance._pending_ambiguous_resets = set()
        return instance

    async def execute(self, state: State) -> dict:
        return await self.run(state)

    def consume_cross_agent_request(self, state: State, kinds: tuple, targets: tuple) -> dict:
        """The in-flight cross-agent request this agent should serve now, or {}.

        Re-entry contract helper (Phase 6): owning agents call this at the top
        of run() to detect a routed redo/replay/update aimed at them. A match
        requires the request kind AND target to be in the given sets, and the
        requester to be a DIFFERENT agent — an agent never consumes its own
        outbound request (e.g. delivery_management's routed ZIP update, whose
        pending marker must survive until provider_search finishes).

        The caller owns clearing: either signal COMPLETE with the request
        still set (the orchestrator return hop consumes it) or clear
        pending_cross_agent_request explicitly when handing back directly.
        """
        from agent.state import normalize_cross_agent_request

        request = normalize_cross_agent_request(state)
        if not request or request.get("return_to_agent") == self.AGENT_NAME:
            return {}
        if request.get("kind") in kinds and request.get("target") in targets:
            return request
        return {}

    @abstractmethod
    async def run(self, state: State) -> dict: ...
