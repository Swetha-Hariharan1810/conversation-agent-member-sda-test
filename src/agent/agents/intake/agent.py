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
    INTENT_BRIDGE_ASKS,
    INTENT_BRIDGE_MSGS,
    LOG_DIFFERENT_MEMBER,
    LOG_INTAKE_GREETING,
    LOG_INTENT_CLASSIFIED,
    LOG_SAME_MEMBER_AMBIGUOUS,
    LOG_SAME_MEMBER_CHECK,
    LOG_SAME_MEMBER_CONFIRMED,
    LOG_SAME_MEMBER_GIVE_UP,
    LOG_SAME_MEMBER_WITHDRAWN,
    MAX_CLARIFICATION_ATTEMPTS,
    OFFTOPIC_ESCALATION,
    OFFTOPIC_REASON,
    SAME_MEMBER_CHECK_QUESTION,
    SAME_MEMBER_CLARIFICATION_MSGS,
    SAME_MEMBER_MAX_CLARIFICATIONS,
)
from agent.agents.intake.handlers import (
    _get_clarification_attempts,
    _phrase_hit,
    handle_out_of_scope_intent,
    handle_unclear_intent,
    handle_unsupported_provider_type,
    screen_out_of_scope,
    screen_request_withdrawn,
    screen_unsupported_provider_type,
)
from agent.agents.intake.llm import extract_intake_intent, extract_same_member_decision
from agent.agents.intake.models import IntentTag
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
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

        extracted = result.extracted or {}
        intent_value = (extracted.get("intent") or "").strip()
        if not intent_value:
            # No classification came back at all. That is not the same thing as
            # classifying the caller "unclear" — "unclear" is a reading of what
            # they said, and a missing key is the absence of one. Both take the
            # clarification path, because there is nothing else to do with the
            # turn, but they have different fixes: one is a caller who has not
            # said what they need, the other is the extraction contract failing
            # on a caller who said it plainly. Distinguishing them in the log is
            # the difference between reading a transcript and guessing at it.
            logger.warning(
                "IntakeAgent: extraction reported no intent key — treating as unclear",
                extra={
                    "utterance": last_user,
                    "turn_intent": getattr(result.turn_intent, "value", ""),
                    "extracted_keys": sorted(extracted),
                    "app_run_id": app_run_id,
                },
            )
            intent_value = IntentTag.UNCLEAR.value

        # ── Deterministic screens ──────────────────────────────────────────────
        # Both read tables that already knew the answer, and both used to run
        # only after the classification they were rescuing had come back right.
        # An appeal named in the caller's words is an appeal, and a specialty
        # named in the caller's words is a specialty, whatever the tag says —
        # so they are checked before the tag is branched on. See the screens in
        # handlers.py for the two transcripts that made this necessary.
        #
        # Appeals first: "appeal my neurologist's denial" belongs to the appeals
        # team, not to the unsupported-specialty handoff.
        if screen_out_of_scope(intent_value, last_user):
            logger.info(
                "IntakeAgent: deterministic screen overriding %s to out_of_scope",
                intent_value,
                extra={"utterance": last_user},
            )
            return await handle_out_of_scope_intent(agent=self, state=state, result=result)

        if screened := screen_unsupported_provider_type(intent_value, last_user):
            logger.info(
                "IntakeAgent: deterministic screen overriding %s to provider_type_unsupported",
                intent_value,
                extra={"utterance": last_user, "provider_type": screened},
            )
            return await handle_unsupported_provider_type(agent=self, state=state, result=result)

        # ── Unsupported provider type — escalate immediately at intake ────────
        # Fires before verification so the member is never put through identity
        # collection for a provider type the system cannot serve.
        if intent_value == IntentTag.PROVIDER_TYPE_UNSUPPORTED.value:
            return await handle_unsupported_provider_type(agent=self, state=state, result=result)

        if intent_value == IntentTag.OUT_OF_SCOPE.value:
            return await handle_out_of_scope_intent(agent=self, state=state, result=result)

        if intent_value == IntentTag.UNCLEAR.value:
            return await handle_unclear_intent(agent=self, state=state, result=result)

        provider_type = ""
        if intent_value == IntentTag.PROVIDER_SERVICES.value:
            provider_type = normalize_provider_type(extracted.get("provider_type", ""))
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
        # (Phase 4). Intake has no confirmed slots yet, so a side question here
        # is answered from call scope or declined — in the sentence it is asked.
        # Nothing parks: intake routes no updates, and a parked question
        # is a promise follow_up does not keep. Missing/none defaults to
        # FOLLOWUP_RESPOND, which self-triages.
        # Option A applies here too: Gemini only acknowledges — Python appends
        # the first-name bridge ask and routes straight to verification, exactly
        # like the clean answered path below.
        # A side question alongside the intent: both halves are real, so the
        # turn answers the question and bridges into the flow in one sentence.
        #
        # This used to test an ANSWERED_WITH_FOLLOWUP label as well, and had to
        # test the question too, because the label arrived without one: "I want
        # to check my claim status. Can you help me with that today?" came back
        # labelled that way with followup_query null — the courtesy question is
        # part of the request, not a side question. Generating on the label
        # alone handed FOLLOWUP_RESPOND a payload with no "Followup:" line to
        # answer and it padded. The question is the whole condition now.
        if (getattr(result, "followup_query", None) or "").strip():
            from agent.conversation.context import ConversationContext
            from agent.core.call_stages import remaining_call_stages
            from agent.core.slot_manager import _mk_session_ctx

            # Intake collects one slot and routes no updates, so nothing here is
            # ever an action — a side question is answered or declined where it
            # is asked.
            guard = self.resolve_park_guard("FOLLOWUP_RESPOND", parks_as_action=False)
            followup_query = (getattr(result, "followup_query", None) or "").strip()
            logger.info(
                "IntakeAgent: answering a side question asked with the intent",
                extra={"intent": intent_value, "guard": guard, "app_run_id": app_run_id},
            )

            # The whole flow is ahead of intake, so a side question about any
            # of it ("will I get a text about this?") is answerable now. The
            # intent comes from this turn's extraction — state.call_intent is
            # only set on the bridge result below.
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state,
                slot_name="intent",
                ctx=ctx,
                messages=messages,
                guard=guard,
                session_context=_mk_session_ctx(
                    followup_query=followup_query,
                    coming_up=remaining_call_stages(
                        intent=intent_value, current_agent=self.AGENT_NAME, state=state
                    ),
                ),
                extracted_this_turn=intent_value,
                # Python appends the first-name ask below, so the generated
                # sentence must not ask for it too. Without this the model's own
                # ask survives and the caller hears the whole thing twice:
                #   "…check your claim status. Could I get your first name?
                #    I can definitely help with that. To get started, could I
                #    get your first name?"
                next_slot_label="first name",
                will_append_ask=True,
            )
            # The bare ask, not a full bridge: the sentence above has already
            # acknowledged, in words that answer what the caller actually said.
            bridge = self.ask_member(state, msg.rstrip() + " " + random.choice(INTENT_BRIDGE_ASKS))
            bridge["call_intent"] = intent_value
            bridge["app_run_id"] = app_run_id
            bridge["resolved_intents"] = ["intake"]
            bridge["next_node"] = AgentNode.VERIFICATION.value
            bridge["metadata_events"] = []
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
        # The intent field is reported by BaseAgent.stamp_metadata_events, which
        # reads call_intent off this dict — see core/metadata_events.py.
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
        result["same_member_clarify_attempts"] = 0
        result["metadata_events"] = []
        if provider_type:
            result["provider_type"] = provider_type
        return result

    async def _handle_same_member_check(self, state: State, app_run_id: str) -> dict:
        """Process the member's reply to the same-vs-different-member question.

        Three things can come back, and only two of them are answers:

          * an answer — same member, or a different one;
          * a withdrawal — the caller no longer wants the request they just
            made, and is closing the call;
          * neither — a hedge, a non-answer, silence dressed as words.

        Withdrawal is screened from the words before the model is asked
        (screen_request_withdrawn), because "that's everything, thanks" is a
        fact about the sentence and the classifier has no category for it: it
        answered "unclear", and "unclear" used to mean ask again, forever.

        Classification is otherwise LLM-first (same_member_check.md prompt →
        WorkerResult extracted["same_member"] = "yes" | "no" | "withdrawn" |
        "unclear"). A keyword-based fallback fires only when the LLM call fails
        entirely, ensuring robustness while keeping natural-language
        understanding as the primary path.

        Ambiguous replies are clarified at most SAME_MEMBER_MAX_CLARIFICATIONS
        times — each with a different sentence, so a caller never hears the same
        question twice — and then the flow stops asking and routes through
        verification, the safe path: a request whose member we could not
        establish gets a fresh identity check rather than the saved one.
        """
        import re

        messages = list(state.get("messages") or [])
        last_user = (_last_user_msg(messages) or "").strip()
        last_agent = _last_assistant_msg(messages)
        call_intent = state.get("call_intent", "")
        provider_type = state.get("provider_type", "")

        # ── Withdrawal screen — read from the words, before any LLM call ──────
        if screen_request_withdrawn(last_user):
            logger.info(
                LOG_SAME_MEMBER_WITHDRAWN,
                extra={"intent": call_intent, "app_run_id": app_run_id, "utterance": last_user},
            )
            return self._close_same_member_check(state, app_run_id)

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
            # Word-boundary matching, not `in`: "no" lives inside "nothing" and
            # "know", "new" inside "renew". Substring matching read "nope,
            # nothing else" as a different member and sent a caller who was
            # hanging up through identity verification.
            _SAME_KW = (
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
            )
            _DIFF_KW = (
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
            )
            kw_same = _phrase_hit(_SAME_KW, lowered)
            kw_diff = _phrase_hit(_DIFF_KW, lowered)
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
        if same_member_value == "withdrawn":
            logger.info(
                LOG_SAME_MEMBER_WITHDRAWN,
                extra={"intent": call_intent, "app_run_id": app_run_id, "utterance": last_user},
            )
            return self._close_same_member_check(state, app_run_id)

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
            context_updates["same_member_clarify_attempts"] = 0
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
            return self._same_member_to_verification(state, call_intent, provider_type, app_run_id)

        # ── Unclear — clarify, but a bounded number of times ──────────────────
        attempts = int(state.get("same_member_clarify_attempts") or 0)
        if attempts >= SAME_MEMBER_MAX_CLARIFICATIONS:
            # Asking a third time is the loop. The question exists to save the
            # caller a re-verification, so when it cannot be answered we spend
            # the re-verification rather than the caller's patience.
            logger.info(
                LOG_SAME_MEMBER_GIVE_UP,
                extra={"intent": call_intent, "app_run_id": app_run_id, "attempts": attempts},
            )
            return self._same_member_to_verification(state, call_intent, provider_type, app_run_id)

        logger.info(
            LOG_SAME_MEMBER_AMBIGUOUS,
            extra={
                "intent": call_intent,
                "app_run_id": app_run_id,
                "utterance": last_user,
                "attempt": attempts + 1,
            },
        )
        # Indexed, not random: the second clarification must not be the first
        # one again. Two phrasings, two attempts, then the cap above.
        msg = SAME_MEMBER_CLARIFICATION_MSGS[min(attempts, len(SAME_MEMBER_CLARIFICATION_MSGS) - 1)]
        result = self.ask_member(state, msg)
        result["call_intent"] = call_intent
        result["app_run_id"] = app_run_id
        result["same_member_check_pending"] = True
        result["same_member_clarify_attempts"] = attempts + 1
        result["metadata_events"] = []
        if provider_type:
            result["provider_type"] = provider_type
        return result

    def _close_same_member_check(self, state: State, app_run_id: str) -> dict:
        """The caller withdrew the request — end the call instead of asking again.

        Routed straight at closure_agent (see ``intake_routing``): the
        orchestrator's fast path would send this at verification, because
        ``reset_for_new_intent`` cleared member_status_verify when follow_up
        handed the call back here.
        """
        result = self.signal_complete(
            state=state,
            message="",
            resolved_intents=["intake"],
            context_updates={
                "app_run_id": app_run_id,
                "same_member_check_pending": False,
                "same_member_clarify_attempts": 0,
                "saved_member_context": None,
                "call_intent": "",
                "pending_intent": "",
            },
            closure_requested=True,
            reasoning="request withdrawn during same-member check — closing the call",
        )
        result["next_node"] = AgentNode.CLOSURE.value
        return result

    def _same_member_to_verification(
        self, state: State, call_intent: str, provider_type: str, app_run_id: str
    ) -> dict:
        """Take the request through verification — a different member, or one we
        could not establish. Either way the saved context must not be reused."""
        bridge = self.ask_member(state, random.choice(INTENT_BRIDGE_MSGS))
        bridge["call_intent"] = call_intent
        bridge["app_run_id"] = app_run_id
        bridge["resolved_intents"] = ["intake"]
        bridge["next_node"] = AgentNode.VERIFICATION.value
        bridge["metadata_events"] = []
        bridge["saved_member_context"] = None
        bridge["same_member_check_pending"] = False
        bridge["same_member_clarify_attempts"] = 0
        if provider_type:
            bridge["provider_type"] = provider_type
        return bridge


async def intake_agent(state: State) -> dict:
    return await IntakeAgent.from_state(state).execute(state)
