"""
agent.py — CareWellnessAgent: sends Care Coach program details using the
already-confirmed delivery contact from session context.
Never re-collects delivery information.
"""

from __future__ import annotations

import random

from agent.agents.care_wellness.constants import (
    AGENT_NAME,
    CARE_COACH_INTRO_TEMPLATES,
    CARE_COACH_NOOFFER_TEMPLATES,
    LOG_DETAILS_SENT,
    LOG_ENTERED,
    MSG_NO_CONTACT,
)
from agent.agents.care_wellness.handlers import _resolve_delivery_contact, dispatch_care_coach
from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State
from agent.utils import pick, speak_email

logger = get_logger(__name__)


class CareWellnessAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:
        # Early exit: already handled (re-entry guard)
        if state.get("care_coach_details_sent") or state.get("care_coach_nooffer_sent"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["care_wellness"],
                context_updates=self._completion_context(state),
            )

        # proactive_offer_available was set by benefits_agent:
        #   True  = member accepted Care Coach → dispatch + confirm
        #   False = member declined Care Coach → no-offer message, no dispatch
        care_coach_accepted = bool(state.get("proactive_offer_available"))

        if care_coach_accepted:
            return await self._handle_yes(state)
        else:
            return self._handle_no(state)

    async def _handle_yes(self, state: State) -> dict:
        method, contact = _resolve_delivery_contact(state)
        if not method or not contact:
            return self.signal_escalate(
                state, pick(MSG_NO_CONTACT), reason="no_delivery_contact_for_care_coach"
            )

        if fail := await dispatch_care_coach(self, state, method, contact):
            return fail

        logger.info(LOG_DETAILS_SENT, extra={"method": method, "contact_tail": contact[-4:]})
        # Spoken-form requirement: emails in AI messages are fully spelled
        # out in words — "@" → "at" AND "." → "dot".
        if method == "email":
            contact = speak_email(contact)
        message = random.choice(CARE_COACH_INTRO_TEMPLATES).format(method=method, contact=contact)
        result = self.ask_member(state, message)  # is_interrupt=True — user sees message and can respond
        result["next_node"] = "follow_up_agent"  # their response goes to follow_up_agent
        result.update(self._completion_context(state, dispatched=True))  # persist care coach flags into state
        return result

    def _handle_no(self, state: State) -> dict:
        message = pick(CARE_COACH_NOOFFER_TEMPLATES)
        result = self.ask_member(state, message)
        result["next_node"] = "follow_up_agent"
        result["care_coach_offered"] = True
        result["care_coach_nooffer_sent"] = True
        result["care_coach_details_sent"] = False
        return result

    @staticmethod
    def _completion_context(state: State, dispatched: bool = False) -> dict:
        return {
            "care_coach_offered": True,
            "care_coach_details_sent": dispatched,
            "care_coach_nooffer_sent": not dispatched,
            "rewards_portal_shared": state.get("rewards_portal_shared", False),
            "delivery_method": state.get("delivery_method", ""),
            "fax": state.get("fax", ""),
            "email": state.get("email", ""),
        }


async def care_wellness_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await CareWellnessAgent.from_state(state).execute(state)
