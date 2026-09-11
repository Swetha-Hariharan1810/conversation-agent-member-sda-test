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
        # A question the caller asked alongside this turn's answer, recorded by
        # the guard layer and cleared by whoever answers it. See execute().
        self._side_question: Dict[str, str] = {}

    @classmethod
    def from_state(cls, state: State) -> "BaseAgent":
        """Create an instance with slot state restored from LangGraph state."""
        instance = cls()
        instance._slots = {k: cls._restore_slot(k, v) for k, v in (state.get("slot_attempts") or {}).items()}
        instance._pending_ambiguous_resets = set()
        return instance

    async def execute(self, state: State) -> dict:
        """Run this agent's turn, then make sure a side question was answered.

        A caller answers and asks in the same breath — "No. But I lost my ID
        card, can you help me with the new one?" — and both halves are real.
        _collect_slot has always handled both for the slots it collects. Every
        hand-written handler had to remember to, and a handler that forgets
        fails silently: it reads extracted[slot], branches on the value and
        returns, so the question is indistinguishable from never having been
        asked. Nothing downstream catches it either — the guard layer needs
        guard_confidence >= 0.7 to act, such a turn carries 0.0, and the
        repeated-ignored-request escalation only hangs off the OFFTOPIC_AGENT
        branch, so a caller can ask three times and be ignored three times.

        Remembering is not a thing a handler should have to do, so it is not
        asked of them. Every agent node in the graph calls execute(); the guard
        layer records the question; any handler that answers it consumes it on
        the way through _generate_slot_retry_response; and whatever is left
        here is put in front of what the turn says. A new slot cannot silently
        drop a question, because no handler has to do anything to keep it.
        """
        result = await self.run(state)
        return await self._answer_unanswered_side_question(state, result)

    async def _answer_unanswered_side_question(self, state: State, result: dict) -> dict:
        """Put an unanswered side question's answer in front of this turn."""
        pending = self.consume_side_question()
        query = (pending.get("query") or "").strip()
        if not query or not isinstance(result, dict):
            return result

        # An escalating turn speaks through escalation_pre_message and carries
        # no "messages" of its own by design (signals.signal_escalate). Attaching
        # an answer here would invent one, and a caller being transferred does
        # not need an aside first — the representative they are going to is the
        # answer. Guard-driven escalations never reach this (the guard discards
        # the question); this covers the ones raised inside run().
        # model_dump() keeps the enum, so str() would give "AgentStatus.ESCALATE"
        # — the same getattr(.value) idiom the request/disposition reads use.
        raw_status = (result.get("last_agent_signal") or {}).get("status")
        status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        if status in ("escalate", "blocked"):
            self.logger.info(
                "execute: side question dropped — this turn escalates",
                extra={"agent": self.AGENT_NAME, "query": query},
            )
            return result

        answer = await self.answer_side_question(
            state,
            list(state.get("messages") or []),
            followup_query=query,
            slot_name=str(state.get("awaiting_slot") or ""),
            extracted_value=pending.get("value", ""),
        )
        if not answer:
            return result
        self.logger.info(
            "execute: answered a side question the turn left unanswered",
            extra={"agent": self.AGENT_NAME, "query": query},
        )
        return self.prefix_side_answer(result, answer)

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
