"""
agent.py — Warm transfer. Uses pick() for varied templates.
"""

from __future__ import annotations

import random

from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

# Escalation handoff templates — delivered at moment of transfer
_REASON_TEMPLATES: dict[str, list[str]] = {
    "requested": [
        (
            "Of course. Let me connect you to one of our specialists now. Your "
            "reference number is {ref}. Please hold."
        ),
        (
            "Absolutely — I'm transferring you to a live representative right away. "
            "Reference: {ref}. Please hold."
        ),
    ],
    "wants": [
        ("Absolutely. I'm transferring you to a live representative. Reference number {ref}. Please hold."),
        "Of course — connecting you now. Your reference is {ref}. Please hold.",
    ],
    "asked for": [
        ("Sure thing. Connecting you to our team. Your reference number is {ref}. Please stay on the line."),
        "Right away — reference: {ref}. Please hold.",
    ],
    "max retries": [
        (
            "I wasn't able to fully assist with your request. I'm transferring you to "
            "a specialist who can help. Reference: {ref}. Please hold."
        ),
        "Let me get you to a specialist who can take it from here. Reference: {ref}. Please hold.",
    ],
    "exhausted": [
        (
            "I wasn't able to capture your information after several attempts. "
            "Transferring you to a specialist. Reference: {ref}. Please hold."
        ),
        "Let me get someone who can help with this directly. Reference: {ref}. Please hold.",
    ],
    "not verified": [
        (
            "For account security, I'm connecting you with a representative who can "
            "assist directly. Reference: {ref}. Please hold."
        ),
        ("I'll connect you with a specialist who can verify your account. Reference: {ref}. Please hold."),
    ],
    "technical": [
        (
            "I'm experiencing a technical issue. Connecting you to a live agent. "
            "Reference: {ref}. Please hold."
        ),
        (
            "I'm running into a technical difficulty — let me get you to a live "
            "agent. Reference: {ref}. Please hold."
        ),
    ],
    "abuse": [
        "I'm transferring this conversation to a live representative. Reference: {ref}. Please hold.",
    ],
    "complaint": [
        ("I understand your concern. Transferring you to our support team. Reference: {ref}. Please hold."),
        "I hear you — let me get you to the right team. Reference: {ref}. Please hold.",
    ],
}

_INTENT_TEMPLATES: dict[str, list[str]] = {
    "provider_services": [
        "Let me connect you with a provider services specialist. Reference: {ref}. Please hold.",
        "Transferring you to our provider team now. Reference: {ref}. Please hold.",
    ],
    "claim_services": [
        "I'm transferring you to our claims team. Reference: {ref}. Please hold.",
        "Connecting you with a claims specialist. Reference: {ref}. Please hold.",
    ],
    "benefits_inquiry": [
        "Connecting you with a benefits specialist. Reference: {ref}. Please hold.",
        "Let me get you to our benefits team. Reference: {ref}. Please hold.",
    ],
    "care_wellness": [
        "Let me connect you with our Care Coach team. Reference: {ref}. Please hold.",
        "Transferring you to our wellness team. Reference: {ref}. Please hold.",
    ],
    "rewards": [
        "Transferring you to our wellness rewards team. Reference: {ref}. Please hold.",
    ],
}

_DEFAULT_TEMPLATE = [
    "Let me connect you to one of our representatives. Your reference number is {ref}. Please hold.",
    "I'll get you to the right person. Reference: {ref}. Please hold.",
    "Connecting you with our team now. Reference: {ref}. Please hold.",
]

logger = get_logger(__name__)


class EscalationAgent(BaseAgent):
    AGENT_NAME = "escalation_agent"

    async def run(self, state: State) -> dict:
        ref_no = (
            state.get("ref_no")
            or state.get("escalation_reference_number")
            or f"REF{random.randint(100000000, 999999999)}"
        )

        pre_message = (state.get("escalation_pre_message") or "").strip()
        # reason = (state.get("escalation_reason") or "").lower()

        if pre_message:
            # A contextual message was already shown to the member by the calling agent.
            # Only append the reference number and warm sign-off — do NOT generate
            # a second standalone sentence.
            clean = pre_message.rstrip(". ")
            message = (
                f"{clean}. "
                f"Your reference number for this call is {ref_no}. "
                f"Thank you for calling Sagility Health. Have a great day!"
            )
        else:
            # No pre-message — build the full standalone escalation message as before.
            message = self._build_message(state, ref_no)

        logger.info("EscalationAgent: transfer", extra={"ref_no": ref_no})

        existing_event = next(
            (
                e
                for e in (state.get("metadata_events") or [])
                if e.get("eventType") == "AgentCallEvent"
                and e.get("data", {}).get("eventName") == "AgentCallTransfer"
            ),
            None,
        )
        transfer_event = {  # noqa: F841
            "eventType": "AgentCallEvent",
            "data": (
                {**existing_event["data"], "referenceNumber": ref_no}
                if existing_event
                else {
                    "eventName": "AgentCallTransfer",
                    "transferInitiator": "Agent",
                    "detail": state.get("escalation_reason", "Transfer initiated"),
                    "referenceNumber": ref_no,
                }
            ),
        }
        result = self.signal_complete(
            state,
            message=message,
            resolved_intents=["escalation"],
            context_updates={"escalation_reference_number": ref_no, "ref_no": ref_no},
            reasoning=f"Member transferred — ref {ref_no}",
        )
        # result["metadata_events"] = result.get("metadata_events", []) + [transfer_event]

        result["metadata_events"] = []
        result["next_node"] = "END"
        return result

    def _build_message(self, state: State, ref_no: str) -> str:
        reason = (state.get("escalation_reason") or "").lower()
        for key, templates in _REASON_TEMPLATES.items():
            if key in reason:
                return pick(templates).format(ref=ref_no)
        intent = state.get("call_intent", "")
        if intent in _INTENT_TEMPLATES:
            return pick(_INTENT_TEMPLATES[intent]).format(ref=ref_no)
        return pick(_DEFAULT_TEMPLATE).format(ref=ref_no)


async def escalation_agent(state: State) -> dict:
    return await EscalationAgent.from_state(state).execute(state)
