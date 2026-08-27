"""
agent.py — Greeting and intent classification.

Flow:
  Turn 1: No messages yet → send greeting, wait for member
  Turn 2+: Run guards → classify intent → route to verification

Adding new intents:
  1. Add a value to IntentTag in models.py
  2. Update the intake.md prompt to describe the new intent
  3. Update SUPPORTED_TOPICS in constants.py if needed
"""

from __future__ import annotations

import random
import uuid

from agent.agents.intake.constants import (
    GREETING,
    INTENT_BRIDGE_MSGS,
    LOG_DIFFERENT_MEMBER,
    LOG_INTAKE_GREETING,
    LOG_INTENT_CLASSIFIED,
    LOG_SAME_MEMBER_AMBIGUOUS,
    LOG_SAME_MEMBER_CHECK,
    LOG_SAME_MEMBER_CONFIRMED,
    MAX_CLARIFICATION_ATTEMPTS,
    OFFTOPIC_ESCALATION,
    OFFTOPIC_REASON,
    SAME_MEMBER_CHECK_QUESTION,
    SAME_MEMBER_CLARIFICATION_MSGS,
)
from agent.agents.intake.handlers import (
    _get_clarification_attempts,
    handle_out_of_scope_intent,
    handle_unclear_intent,
    handle_unsupported_provider_type,
)
from agent.agents.intake.llm import extract_intake_intent, extract_same_member_decision
from agent.agents.intake.models import IntentTag
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.llm.schema import EventType
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.slots.normalizers import normalize_provider_type
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_core,
)

logger = get_logger(__name__)


class IntakeAgent(BaseAgent):
    AGENT_NAME = "intake_agent"

    def get_system_prompt(self, state: State) -> str:
        return build_extraction_prompt_core("extraction/intake.md")

    async def run(self, state: State) -> dict:
        app_run_id = state.get("app_run_id") or str(uuid.uuid4())

        # ── Same-member disambiguation re-entry ───────────────────────────────
        # Must come before the call_intent guard: call_intent is already set in
        # state from the previous turn that triggered the disambiguation question.
        if state.get("same_member_check_pending"):
            return await self._handle_same_member_check(state, app_run_id)

        # ── Immediate same-member trigger (follow-up re-entry, intent pre-known) ──
        # follow_up already classified the new intent; skip LLM re-classification
        # (which can misfire on a long conversation history) and ask the
        # same-vs-different-member question directly.
        if state.get("saved_member_context") and state.get("call_intent"):
            msgs = list(state.get("messages") or [])
            intent_value = state["call_intent"]
            provider_type = state.get("provider_type", "") or normalize_provider_type(
                _last_user_msg(msgs) or ""
            )
            return self._start_same_member_check(state, intent_value, provider_type, app_run_id)

        # Guard: if intent already classified in a prior turn, skip
        # re-classification and hand off to verification immediately.
        # This prevents re-entry from sending the bridge message again.
        if state.get("call_intent"):
            logger.info(
                LOG_INTENT_CLASSIFIED,
                extra={"intent": state["call_intent"], "app_run_id": app_run_id},
            )
            return self.signal_complete(
                state=state,
                message="",
                resolved_intents=["intake"],
                context_updates={"app_run_id": app_run_id, "call_intent": state["call_intent"]},
                reasoning=f"Intent already classified as {state['call_intent']}",
            )

        # Turn 1: no messages yet — send greeting
        if not state.get("messages"):
            logger.info(LOG_INTAKE_GREETING, extra={"app_run_id": app_run_id})
            result = self.ask_member(state, GREETING)
            result["app_run_id"] = app_run_id
            return result

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        attempts = _get_clarification_attempts(state)

        result = await extract_intake_intent(
            get_extraction_llm(),
            self.get_system_prompt(state),
            last_agent_message=last_agent,
            last_user_message=last_user,
            pending_slots=["intent"],
            attempt=attempts,
            recent_messages=messages,
        )

        if interrupt := await self.run_conversation_guards(
            state,
            user_text=last_user,
            result=result,
        ):
            if getattr(result, "guard", "") == "OFFTOPIC_AGENT":
                if attempts >= MAX_CLARIFICATION_ATTEMPTS:
                    return self.signal_escalate(
                        state, OFFTOPIC_ESCALATION, OFFTOPIC_REASON, initiator="Agent"
                    )
            return interrupt

        intent_value = (result.extracted or {}).get("intent", IntentTag.UNCLEAR.value)

        # ── Unsupported provider type — escalate immediately at intake ────────
        # Fires before verification so the member is never put through identity
        # collection for a provider type the system cannot serve.
        if intent_value == IntentTag.PROVIDER_TYPE_UNSUPPORTED.value:
            return await handle_unsupported_provider_type(agent=self, state=state, result=result)

        # ── Deterministic unsupported-type guard ───────────────────────────────
        # The extraction LLM occasionally misclassifies an explicitly-unsupported
        # specialty (e.g. neurologist) as provider_services. Cross-check with the
        # same keyword list used by handle_unsupported_provider_type — if the
        # utterance names a known unsupported type, override and escalate now.
        if intent_value == IntentTag.PROVIDER_SERVICES.value:
            from agent.agents.intake.handlers import _extract_provider_type_from_utterance

            if _extract_provider_type_from_utterance(last_user) != "this provider type":
                logger.info(
                    "IntakeAgent: deterministic guard overriding provider_services "
                    "to provider_type_unsupported",
                    extra={"utterance": last_user},
                )
                return await handle_unsupported_provider_type(agent=self, state=state, result=result)

        if intent_value == IntentTag.OUT_OF_SCOPE.value:
            return await handle_out_of_scope_intent(agent=self, state=state, result=result)

        if intent_value == IntentTag.UNCLEAR.value:
            return await handle_unclear_intent(agent=self, state=state, result=result)

        provider_type = ""
        if intent_value == IntentTag.PROVIDER_SERVICES.value:
            provider_type = normalize_provider_type((result.extracted or {}).get("provider_type", ""))
            if provider_type:
                logger.info(
                    "IntakeAgent: provider_type extracted at intake — propagating to state",
                    extra={"provider_type": provider_type, "app_run_id": app_run_id},
                )

        # ── Same-member disambiguation (follow-up re-entry) ──────────────────
        # When the previous agent was follow_up and a verified member context was
        # preserved, ask whether this new PCP/Claims request is for the same member
        # rather than routing blindly to verification.
        if self._should_trigger_same_member_check(state, intent_value):
            return self._start_same_member_check(state, intent_value, provider_type, app_run_id)

        # Intent is classified — check if caller also said something extra
        # that needs acknowledging before we route to verification.
        # Phase 6: routed through the same disposition mapping as _collect_slot
        # (Phase 4). Intake has no confirmed slots yet, so the extraction prompt
        # emits park/decline (answer_now only for repeat/confirmation of the
        # just-stated intent); missing/none defaults to decline. PARK queues the
        # side question in parked_followups for follow_up_agent to answer.
        # Option A applies here too: Gemini only acknowledges — Python appends
        # the first-name bridge ask and routes straight to verification, exactly
        # like the clean answered path below.
        if result.event_type == EventType.ANSWERED_WITH_FOLLOWUP:
            from agent.conversation.context import ConversationContext
            from agent.core.slot_manager import _DISPOSITION_GUARDS, _mk_session_ctx

            disposition = getattr(result, "followup_disposition", None)
            disposition_value = str(getattr(disposition, "value", disposition) or "none")
            guard = _DISPOSITION_GUARDS.get(disposition_value, "FOLLOWUP_RESPOND")
            followup_query = (getattr(result, "followup_query", None) or "").strip()
            logger.info(
                "IntakeAgent: answered_with_followup — disposition routing",
                extra={"intent": intent_value, "guard": guard, "app_run_id": app_run_id},
            )

            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state,
                slot_name="intent",
                ctx=ctx,
                messages=messages,
                guard=guard,
                session_context=_mk_session_ctx(followup_query=followup_query),
                extracted_this_turn=intent_value,
            )
            bridge = self.ask_member(state, msg.rstrip() + " " + random.choice(INTENT_BRIDGE_MSGS))
            bridge["call_intent"] = intent_value
            bridge["app_run_id"] = app_run_id
            bridge["resolved_intents"] = ["intake"]
            bridge["next_node"] = AgentNode.VERIFICATION.value
            bridge["metadata_events"] = []
            if guard == "FOLLOWUP_PARK" and followup_query:
                from agent.state import normalize_parked_followups

                parked = normalize_parked_followups(state.get("parked_followups"))
                parked.append({"query": followup_query, "kind": "question", "target": ""})
                bridge["parked_followups"] = parked
            if provider_type:
                bridge["provider_type"] = provider_type
            return bridge

        # Clean answered path — fire bridge and route to verification
        logger.info(LOG_INTENT_CLASSIFIED, extra={"intent": intent_value, "app_run_id": app_run_id})
        bridge = self.ask_member(state, random.choice(INTENT_BRIDGE_MSGS))
        bridge["call_intent"] = intent_value
        bridge["app_run_id"] = app_run_id
        bridge["resolved_intents"] = ["intake"]
        bridge["next_node"] = AgentNode.VERIFICATION.value
        # bridge["metadata_events"] = [
        #     {
        #         "eventType": "CallAgentField",
        #         "data": {"field": "call_intent", "value": intent_value},
        #     }
        # ]
        bridge["metadata_events"] = []
        if provider_type:
            bridge["provider_type"] = provider_type
        return bridge

    # ── Same-member disambiguation helpers ───────────────────────────────────

    @staticmethod
    def _should_trigger_same_member_check(state: State, intent_value: str) -> bool:
        """Return True when the same-member disambiguation question should be asked.

        Conditions (all must hold):
          * The resolved intent is PCP (provider_services) or Claims (claim_services).
          * The previous agent was follow_up_agent (active_agent still reflects the
            agent that routed to intake, which is follow_up when it calls
            _reroute_through_intake).
          * A saved verified member context exists in state (placed there by
            follow_up's _reroute_through_intake when the member was verified).
        """
        return (
            intent_value in (IntentTag.PROVIDER_SERVICES.value, IntentTag.CLAIM_SERVICES.value)
            and state.get("active_agent") == "follow_up_agent"
            and bool(state.get("saved_member_context"))
        )

    def _start_same_member_check(
        self, state: State, intent_value: str, provider_type: str, app_run_id: str
    ) -> dict:
        """Ask the disambiguation question and pause for the member's reply."""
        logger.info(
            LOG_SAME_MEMBER_CHECK,
            extra={"intent": intent_value, "app_run_id": app_run_id},
        )
        result = self.ask_member(state, SAME_MEMBER_CHECK_QUESTION)
        result["call_intent"] = intent_value
        result["app_run_id"] = app_run_id
        result["same_member_check_pending"] = True
        result["metadata_events"] = []
        if provider_type:
            result["provider_type"] = provider_type
        return result

    async def _handle_same_member_check(self, state: State, app_run_id: str) -> dict:
        """Process the member's reply to the same-vs-different-member question.

        Classification is LLM-first (same_member_check.md prompt → WorkerResult
        extracted["same_member"] = "yes" | "no" | "unclear").  A keyword-based
        fallback fires only when the LLM call fails entirely, ensuring robustness
        while keeping natural-language understanding as the primary path.

        Ambiguous replies trigger a single clarification question; if still unclear
        after that, we default to routing through verification (the safe path).
        """
        import re

        messages = list(state.get("messages") or [])
        last_user = (_last_user_msg(messages) or "").strip()
        last_agent = _last_assistant_msg(messages)
        call_intent = state.get("call_intent", "")
        provider_type = state.get("provider_type", "")

        # ── LLM classification ────────────────────────────────────────────────
        system_prompt = build_extraction_prompt_core("extraction/same_member_check.md")
        llm_result = await extract_same_member_decision(
            get_extraction_llm(),
            system_prompt=system_prompt,
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-6:],
        )

        same_member_value = (llm_result.extracted or {}).get("same_member", "")

        # ── Keyword fallback (only when LLM extraction yields nothing) ────────
        if not same_member_value:
            logger.info(
                "IntakeAgent: same-member LLM yielded no result — applying keyword fallback",
                extra={"app_run_id": app_run_id},
            )
            lowered = last_user.lower()
            _SAME_KW = frozenset(
                {
                    "same",
                    "yes",
                    "yeah",
                    "yep",
                    "yup",
                    "correct",
                    "that's right",
                    "thats right",
                    "same member",
                    "same person",
                    "the same",
                    "for the same",
                }
            )
            _DIFF_KW = frozenset(
                {
                    "different",
                    "no",
                    "nope",
                    "nah",
                    "another",
                    "new",
                    "someone else",
                    "other member",
                    "different member",
                    "different person",
                    "new member",
                    "a different",
                    "not the same",
                }
            )
            kw_same = any(kw in lowered for kw in _SAME_KW)
            kw_diff = any(kw in lowered for kw in _DIFF_KW)
            if kw_same and re.search(r"\bnot\b.{0,10}\bsame\b", lowered):
                kw_same = False
                kw_diff = True
            if kw_same and not kw_diff:
                same_member_value = "yes"
            elif kw_diff and not kw_same:
                same_member_value = "no"
            else:
                same_member_value = "unclear"

        # ── Route on classification result ────────────────────────────────────
        if same_member_value == "yes":
            logger.info(
                LOG_SAME_MEMBER_CONFIRMED,
                extra={"intent": call_intent, "app_run_id": app_run_id},
            )
            saved = state.get("saved_member_context") or {}
            context_updates: dict = dict(saved)
            context_updates["member_status_verify"] = True
            context_updates["saved_member_context"] = None
            context_updates["same_member_check_pending"] = False
            context_updates["pending_intent"] = ""  # consumed — fast-path dispatches via call_intent
            context_updates["app_run_id"] = app_run_id

            # For provider_services, relationship must be re-asked even when the
            # member is the same — the subscriber/dependent may differ between
            # requests. Clear any carried-over value and route to verification so
            # it collects relationship before handing off to provider_search.
            if call_intent == "provider_services":
                context_updates["relationship"] = None
                result = self.ask_member(state, "Are you the subscriber or dependent?")
                result.update(context_updates)
                result["next_node"] = AgentNode.VERIFICATION.value
                result["awaiting_slot"] = "relationship"
                result["metadata_events"] = []
                return result

            return self.signal_complete(
                state=state,
                message="",
                resolved_intents=["intake"],
                context_updates=context_updates,
                new_intent_detected=call_intent,
                reasoning="same member confirmed — restoring verification context, skipping verification",
            )

        if same_member_value == "no":
            logger.info(
                LOG_DIFFERENT_MEMBER,
                extra={"intent": call_intent, "app_run_id": app_run_id},
            )
            bridge = self.ask_member(state, random.choice(INTENT_BRIDGE_MSGS))
            bridge["call_intent"] = call_intent
            bridge["app_run_id"] = app_run_id
            bridge["resolved_intents"] = ["intake"]
            bridge["next_node"] = AgentNode.VERIFICATION.value
            bridge["metadata_events"] = []
            bridge["saved_member_context"] = None
            bridge["same_member_check_pending"] = False
            if provider_type:
                bridge["provider_type"] = provider_type
            return bridge

        # ── Unclear — ask one clarification question ──────────────────────────
        logger.info(
            LOG_SAME_MEMBER_AMBIGUOUS,
            extra={"intent": call_intent, "app_run_id": app_run_id, "utterance": last_user},
        )
        result = self.ask_member(state, random.choice(SAME_MEMBER_CLARIFICATION_MSGS))
        result["call_intent"] = call_intent
        result["app_run_id"] = app_run_id
        result["same_member_check_pending"] = True
        result["metadata_events"] = []
        if provider_type:
            result["provider_type"] = provider_type
        return result


async def intake_agent(state: State) -> dict:
    return await IntakeAgent.from_state(state).execute(state)
