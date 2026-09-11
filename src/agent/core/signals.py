"""
signals.py — SignalsMixin: all agent-to-LangGraph communication.

Every method here returns a dict that LangGraph reads to:
  - resume at the right node (next_node)
  - pause for human input (is_interrupt)
  - emit metadata events (slot confirmations, transfers)

Rules:
  ask_member()      → is_interrupt=True,  next_node=AGENT_NAME
  signal_complete() → is_interrupt=False, next_node="orchestrator"
  signal_escalate() → is_interrupt=False, next_node=AgentNode.ESCALATION.value

All return dicts include slot_attempts so LangGraph persists slot state.
"""

from __future__ import annotations

import random
from typing import List, Optional

from agent.core.signal import AgentSignal, AgentStatus
from agent.orchestration.orchestration import AgentNode
from agent.state import State


class SignalsMixin:
    """Mixin that adds agent→LangGraph signal methods to BaseAgent."""

    def ask_member(self, state: State, message: str) -> dict:
        """Interrupt graph and wait for member input.

        Drains any answer waiting in ``pending_side_answer`` into this message.
        The agent that answered a side question may have had nothing of its own
        to say — it handed off — and speaking the answer there would put it in
        the transcript as its own AI turn, ahead of the next agent's opener.
        The caller should hear one turn, so the answer rides along until
        something is actually said to them and goes out in front of it.
        """
        message = self.join_side_answer(str(state.get("pending_side_answer") or ""), message)
        result = {
            "messages": {"role": "assistant", "content": message},
            "pending_side_answer": "",
            "next_node": self.AGENT_NAME,
            "is_interrupt": True,
            "active_agent": self.AGENT_NAME,
            "slot_attempts": self.slots_dict(),
            "metadata_events": [],
            "app_run_id": state.get("app_run_id", ""),
        }
        if self._pending_ambiguous_resets:
            existing = result.get("ambiguous_counts") or {}
            for s in self._pending_ambiguous_resets:
                existing[s] = 0
            result["ambiguous_counts"] = existing
            self._pending_ambiguous_resets = set()
        # Phase 2 fix: persist confirmed slot values to LangGraph state mid-pipeline.
        # A slot record restored from an EARLIER turn (or from another agent —
        # slot_attempts is shared state) may disagree with the live value: the
        # flow can deliberately change a slot after it was confirmed, e.g. a
        # fax→email switch rewrites delivery_method while the delivery_method
        # slot record still holds "fax". Re-persisting that stale record would
        # silently resurrect the abandoned value on the next interrupt, so only
        # a slot confirmed on THIS turn may override a differing live value.
        for slot_name, slot in self._slots.items():
            if not (slot.confirmed and slot.last_value is not None and slot.last_value != ""):
                continue
            if slot_name in result:
                continue
            live = str(state.get(slot_name) or "").strip()
            if slot_name not in self._newly_confirmed and live and live != str(slot.last_value).strip():
                continue
            result[slot_name] = slot.last_value
        return result

    def signal_complete(
        self,
        state: State,
        message: str,
        resolved_intents: Optional[list] = None,
        context_updates: Optional[dict] = None,
        proactive_offer_available: bool = False,
        new_intent_detected: Optional[str] = None,
        closure_requested: bool = False,
        reasoning: Optional[str] = None,
        is_interrupt: bool = False,
    ) -> dict:
        sig = AgentSignal(
            status=AgentStatus.COMPLETE,
            resolved_intents=resolved_intents or [],
            new_intent_detected=new_intent_detected,
            closure_requested=closure_requested,
            context_updates=context_updates or {},
            proactive_offer_available=proactive_offer_available,
            reasoning=reasoning or f"{self.AGENT_NAME}: complete",
        )
        return self._build(state, message, sig, is_interrupt=is_interrupt)

    def signal_escalate(self, state: State, message: str, reason: str, *, initiator: str = "Agent") -> dict:
        """Escalate to a human agent. Emits AgentCallTransfer metadata event."""
        sig = AgentSignal(
            status=AgentStatus.ESCALATE,
            escalation_reason=reason,
            reasoning=f"{self.AGENT_NAME}: escalate — {reason}",
        )
        # result = self._build(state, message or "", sig)
        result = self._build(state, "", sig)
        result["next_node"] = AgentNode.ESCALATION.value
        result["escalation_pre_message"] = message.strip() if message else ""
        # result["metadata_events"] = result.get("metadata_events", []) + [
        #     {
        #         "eventType": "AgentCallEvent",
        #         "data": {
        #             "eventName": "AgentCallTransfer",
        #             "transferInitiator": initiator,
        #             "detail": reason,
        #         },
        #     }
        # ]

        result["metadata_events"] = []
        return result

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _build(self, state: State, message: str, sig: AgentSignal, is_interrupt: bool = False) -> dict:
        """Build the LangGraph state-update dict. Always includes slot_attempts and metadata_events."""
        result = {
            "last_agent_signal": sig.to_state_dict(),
            "next_node": "orchestrator",
            "is_interrupt": is_interrupt,
            "active_agent": self.AGENT_NAME,
            "slot_attempts": self.slots_dict(),
            # "metadata_events": self._build_slot_events(),
            "metadata_events": [],
            "app_run_id": state.get("app_run_id", ""),
            "awaiting_slot": "",
        }
        if isinstance(message, str) and message.strip():
            # Same rule as ask_member: whatever is actually said to the member
            # takes the waiting answer with it. Only reached by the completions
            # that speak — closure's goodbye — since an escalation passes "".
            result["messages"] = {
                "role": "assistant",
                "content": self.join_side_answer(str(state.get("pending_side_answer") or ""), message),
            }
            result["pending_side_answer"] = ""
        for k, v in (sig.context_updates or {}).items():
            result[k] = v
        if self._pending_ambiguous_resets:
            existing = result.get("ambiguous_counts") or {}
            for s in self._pending_ambiguous_resets:
                existing[s] = 0
            result["ambiguous_counts"] = existing
            self._pending_ambiguous_resets = set()
        self._newly_confirmed = set()
        return result

    def _build_slot_events(self) -> List[dict]:
        events = []
        for slot_name in self._newly_confirmed:
            slot = self._slots.get(slot_name)
            value = str(slot.last_value) if slot and slot.last_value is not None else ""
            if value:
                events.append({"eventType": "CallAgentField", "data": {"field": slot_name, "value": value}})
                self.logger.info("Slot event emitted", extra={"field": slot_name, "value": value[:20]})
        return events

    def _emergency(self, state: State, reason: str) -> dict:
        """Last-resort fallback for unhandled exceptions. Escalates with a reference number."""
        ref = state.get("ref_no") or f"REF{random.randint(100000000, 999999999)}"
        sig = AgentSignal(
            status=AgentStatus.ESCALATE, escalation_reason=f"Unhandled error in {self.AGENT_NAME}: {reason}"
        )
        return {
            # "messages": {"role": "assistant", "content": MSG_EMERGENCY.format(ref=ref)},
            "last_agent_signal": sig.to_state_dict(),
            "next_node": "orchestrator",
            "is_interrupt": False,
            "ref_no": ref,
            "active_agent": self.AGENT_NAME,
            "slot_attempts": {},
            "metadata_events": [],
            "app_run_id": state.get("app_run_id", ""),
            "awaiting_slot": "",
        }
