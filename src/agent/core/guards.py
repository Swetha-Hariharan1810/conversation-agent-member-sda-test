"""
guards.py — ConversationGuardsMixin: per-turn safety checks.

Runs ordered safety checks each turn in run_conversation_guards():
  1. TRANSFER_REQUEST — caller wants a human agent
  2. ABUSE            — hostile language detected
  3. SELF_HARM        — self-harm or suicidal ideation detected
  4. INTERRUPTION     — caller interrupted mid-flow           → LLM 2
  5. OFFTOPIC_GLOBAL  — utterance has zero healthcare relevance → static response
  6. OFFTOPIC_AGENT   — valid healthcare topic, wrong agent   → LLM 2

Primary detection is LLM-based (result.guard + result.guard_confidence).
Keyword/regex fallback fires when LLM confidence is below 0.7 or result is None.

TRANSFER, ABUSE, SELF_HARM: always static — never LLM 2.
INTERRUPTION, OFFTOPIC_AGENT: LLM 2 generates the response sentence.
OFFTOPIC_GLOBAL: static response from MSG_OFFTOPIC_GLOBAL pool.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from agent.core.constants import ABUSE_PATTERNS, INTERRUPTION_PATTERNS, MAX_SLOT_ATTEMPTS, SELF_HARM_PATTERNS
from agent.responses.static import (
    MSG_ABUSE_ESCALATION,
    MSG_OFFTOPIC_GLOBAL,
    MSG_REPEATED_REQUEST_ESCALATE,
    MSG_SELF_HARM_ESCALATION,
    MSG_TRANSFER_REQUEST,
)
from agent.state import State
from agent.utils import detect_transfer_request, pick

if TYPE_CHECKING:
    from agent.llm.schema import WorkerResult

_NON_MEMBER_ROUTING: dict[str, tuple[str, str]] = {
    "provider": ("providers", "1-740-660-3977"),
    "employer_group": ("employer groups", "1-800-555-0202"),
    "other_carrier": ("insurance carriers", "1-800-555-0203"),
}

_NON_MEMBER_FALLBACK_LABEL = "callers with this type of enquiry"
_NON_MEMBER_FALLBACK_NUMBER = "1-800-555-0200"


def _normalize_request_key(text: str) -> str:
    """Stable counter key for an ignored/deflected caller request."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:48]


_NON_MEMBER_MSG_TEMPLATES = [
    (
        "Thank you for letting me know. Our dedicated line for {label} "
        "is {number} — they'll be able to assist you directly. "
        "I will connect you with them now."
    ),
    (
        "I appreciate you letting me know. For {label}, the right team "
        "to speak with can be reached at {number}. "
        "Let me transfer you to them now."
    ),
    ("Thanks for that — we have a dedicated team for {label} at {number}. Let me transfer you to them now."),
]


class ConversationGuardsMixin:
    SUPPORTED_TOPICS: set = set()

    async def _generate_guard_response(
        self, state: State, guard: str, *, attempt_override: int | None = None
    ) -> str:
        from agent.llm.redaction import _is_reportable_slot, mask_confirmed
        from agent.llm.response_generator import generate_recovery_message
        from agent.utils import _last_user_msg

        awaiting = state.get("awaiting_slot") or ""
        slot_state = (state.get("slot_attempts") or {}).get(awaiting, {})
        attempt = (
            attempt_override
            if attempt_override is not None
            else (slot_state.get("attempt_count", 0) if isinstance(slot_state, dict) else 0)
        )
        messages = list(state.get("messages") or [])
        # Values only — never the attempt dicts. Counter/flag pseudo-slots
        # (name_confirmed, update_*, *_cycles …) are not reportable values.
        confirmed = {
            k: attempt_rec.get("last_value") or str(state.get(k) or "").strip() or "confirmed"
            for k, attempt_rec in (state.get("slot_attempts") or {}).items()
            if isinstance(attempt_rec, dict) and attempt_rec.get("confirmed") and _is_reportable_slot(k)
        }
        return await generate_recovery_message(
            slot_name=awaiting,
            attempt=attempt,
            guard=guard,
            last_messages=messages[-4:],
            user_utterance=_last_user_msg(messages),
            confirmed_slots=mask_confirmed(confirmed),
        )

    def _handle_non_member_caller(
        self,
        state: State,
        caller_type: str,
    ) -> dict:
        """
        Called when a non-member explicitly identifies themselves.

        Delivers a message with the correct dedicated number and ends
        the call immediately. No yes/no question, no loop.

        Routing is a simple dict lookup — no LLM call, no token cost.
        """
        import random

        label, number = _NON_MEMBER_ROUTING.get(
            caller_type,
            (_NON_MEMBER_FALLBACK_LABEL, _NON_MEMBER_FALLBACK_NUMBER),
        )

        msg = random.choice(_NON_MEMBER_MSG_TEMPLATES).format(
            label=label,
            number=number,
        )

        result = self.ask_member(state, msg)
        result["caller_type"] = caller_type
        result["caller_type_handled"] = True
        result["next_node"] = "END"
        result["is_interrupt"] = False
        result["awaiting_slot"] = ""
        return result

    async def run_conversation_guards(  # noqa: C901
        self,
        state: State,
        *,
        user_text: str,
        result: Optional["WorkerResult"] = None,
    ) -> Optional[dict]:
        # ── Passive caller type detection ─────────────────────────────
        # Fires at any point in the conversation when caller explicitly
        # identifies themselves as a non-member.
        # result.extracted is already populated by each agent's LLM call —
        # no extra LLM call needed here.
        if result and result.extracted and not state.get("caller_type_handled"):
            detected = result.extracted.get("caller_type", "")
            if detected and detected not in ("member", "unknown", ""):
                return self._handle_non_member_caller(state, detected)

        if result is not None and result.guard_confidence >= 0.7:
            guard = result.guard
            # When the caller provides the awaiting slot value AND asks an
            # off-topic question in the same turn, the OFFTOPIC guard should
            # not intercept — the slot pipeline's FOLLOWUP_DECLINE path
            # confirms the value and declines the side question with the
            # correct next-slot ask appended. Without this, the guard fires
            # first and member_id (or whichever slot) is never confirmed.
            if guard in ("OFFTOPIC_GLOBAL", "OFFTOPIC_AGENT"):
                awaiting = state.get("awaiting_slot") or ""
                if awaiting and result.extracted and result.extracted.get(awaiting):
                    self.logger.info(
                        "%s: %s suppressed — slot %r value present in extraction; deferring to slot pipeline",
                        self.AGENT_NAME,
                        guard,
                        awaiting,
                    )
                    return None
            if guard == "TRANSFER_REQUEST":
                self.logger.info("%s: transfer requested", self.AGENT_NAME)
                return self.signal_escalate(
                    state,
                    pick(MSG_TRANSFER_REQUEST),
                    f"Transfer requested during {self.AGENT_NAME}",
                    initiator="Caller",
                )
            if guard == "ABUSE":
                self.logger.warning(f"{self.AGENT_NAME}: abuse detected")
                return self.signal_escalate(
                    state, pick(MSG_ABUSE_ESCALATION), "abuse_detected", initiator="Agent"
                )
            if guard == "SELF_HARM":
                self.logger.warning(f"{self.AGENT_NAME}: self-harm signal detected")
                return self.signal_escalate(
                    state,
                    pick(MSG_SELF_HARM_ESCALATION),
                    "self_harm_detected",
                    initiator="Agent",
                )
            if guard == "INTERRUPTION":
                msg = await self._generate_guard_response(state, "INTERRUPTION")
                return self.ask_member(state, msg)
            if guard == "OFFTOPIC_GLOBAL":
                self.logger.info("%s: global offtopic — static response", self.AGENT_NAME)
                offtopic_count = (state.get("offtopic_global_count") or 0) + 1
                if offtopic_count >= MAX_SLOT_ATTEMPTS:
                    return self.signal_escalate(
                        state,
                        pick(MSG_TRANSFER_REQUEST),
                        "Repeated off-topic requests",
                        initiator="Agent",
                    )
                if self.AGENT_NAME != "intake_agent":
                    awaiting = state.get("awaiting_slot") or ""
                    if awaiting:
                        slot_state = (state.get("slot_attempts") or {}).get(awaiting, {})
                        attempt_count = (
                            slot_state.get("attempt_count", 0) if isinstance(slot_state, dict) else 0
                        )
                        if attempt_count > 0:
                            self.slot_fail(awaiting)
                            if self.get_slot(awaiting).is_exhausted():
                                from agent.responses.static import build_slot_exhausted_message

                                return self.signal_escalate(
                                    state,
                                    build_slot_exhausted_message(awaiting),
                                    f"{awaiting}_exhausted_offtopic",
                                    initiator="Agent",
                                )
                        msg = await self._generate_guard_response(state, "OFFTOPIC_AGENT")
                        result = self.ask_member(state, msg)
                        result["offtopic_global_count"] = offtopic_count
                        return result
                    return None
                result = self.ask_member(state, pick(MSG_OFFTOPIC_GLOBAL))
                result["offtopic_global_count"] = offtopic_count
                return result
            if guard == "OFFTOPIC_AGENT":
                if escalation := self._repeated_ignored_request(state, user_text):
                    return escalation
                msg = await self._generate_guard_response(state, "OFFTOPIC_AGENT")
                return self.ask_member(state, msg)
            # guard == "NONE"
            return None

        self.logger.debug(
            "guards: falling back to keyword detection — "
            f"result={'none' if result is None else f'confidence={result.guard_confidence:.2f}'}"
        )
        if detect_transfer_request(state):
            self.logger.info("%s: transfer requested", self.AGENT_NAME)
            return self.signal_escalate(
                state,
                pick(MSG_TRANSFER_REQUEST),
                f"Transfer requested during {self.AGENT_NAME}",
                initiator="Caller",
            )
        if self._detect_abuse(user_text):
            self.logger.warning(f"{self.AGENT_NAME}: abuse detected")
            return self.signal_escalate(
                state, pick(MSG_ABUSE_ESCALATION), "abuse_detected", initiator="Agent"
            )
        if self._detect_self_harm(user_text):
            self.logger.warning(f"{self.AGENT_NAME}: self-harm signal detected (keyword fallback)")
            return self.signal_escalate(
                state,
                pick(MSG_SELF_HARM_ESCALATION),
                "self_harm_detected",
                initiator="Agent",
            )
        if self._detect_interruption(user_text):
            msg = await self._generate_guard_response(state, "INTERRUPTION")
            return self.ask_member(state, msg)
        return None

    def _repeated_ignored_request(self, state: State, user_text: str) -> Optional[dict]:
        """Repeated-ignored-request guard (Phase 4): the second time the caller
        repeats the same deflected request, stop re-asking the same thing
        verbatim — escalate honestly instead."""
        request_key = _normalize_request_key(user_text)
        if not request_key:
            return None
        return self.guard_loop_limit(
            state,
            f"ignored_request_{request_key}",
            2,
            escalate_message=pick(MSG_REPEATED_REQUEST_ESCALATE),
            escalate_reason="repeated_ignored_request_offtopic",
        )

    def _detect_abuse(self, text: str) -> bool:
        t = (text or "").lower().strip()
        return any(re.search(p, t) for p in ABUSE_PATTERNS)

    def _detect_self_harm(self, text: str) -> bool:
        t = (text or "").lower().strip()
        return any(re.search(p, t) for p in SELF_HARM_PATTERNS)

    def _detect_interruption(self, text: str) -> bool:
        t = (text or "").lower()
        # LLM guard handles broad interruption detection via guard == "INTERRUPTION".
        # INTERRUPTION_PATTERNS contains only unambiguous keyword fallbacks.
        return any(p in t for p in INTERRUPTION_PATTERNS)
