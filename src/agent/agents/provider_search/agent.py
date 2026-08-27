"""
agent.py — ProviderSearchAgent: collects provider_type and confirms ZIP,
then routes to delivery_management_agent.

NOTE: The ZIP read-back confirmation step has been removed. When the member
provides an updated ZIP code, it is validated, written to Salesforce
immediately, and the flow proceeds. The zip_code_updated flag is set in
state so delivery_management_agent can include the new ZIP in its
dispatch-confirmation message.
"""

from __future__ import annotations

import random

from agent.agents.provider_search.constants import (
    DELIVERY_BRIDGE_TEMPLATES,
    LOG_ENTERED,
    LOG_PROVIDER_TYPE,
    LOG_ZIP_CONFIRMED,
    LOG_ZIP_UPDATED,
    MSG_NOT_VERIFIED,
    MSG_PROVIDER_TYPE_UNSUPPORTED,
    MSG_ZIP_EXHAUST,
    PROVIDER_SEARCH_BRIDGE_MSGS,
    PROVIDER_SEARCH_SLOT_ORDER,
    ZIP_CONFIRM_TEMPLATES,
    ZIP_UPDATE_PROMPT,
)
from agent.agents.provider_search.handlers import update_zip_in_salesforce
from agent.agents.provider_search.llm import extract_provider_search_decision
from agent.agents.provider_search.pipelines import (
    build_provider_type_pipeline,
    build_zip_confirmation_pipeline,
)
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.llm.schema import EventType, RequestKind
from agent.logger import get_logger
from agent.slots.normalizers import normalize_provider_type, normalize_yes_no, normalize_zip_code
from agent.slots.validators import validate_zip_code
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt_extraction, pick

logger = get_logger(__name__)


class ProviderSearchAgent(BaseAgent):
    AGENT_NAME = "provider_search_agent"

    def __init__(self) -> None:
        super().__init__()
        self._provider_type_pipeline = build_provider_type_pipeline(self)
        self._zip_pipeline = build_zip_confirmation_pipeline(self)

    async def run(self, state: State) -> dict:  # noqa: C901
        # 1. Guard: member must be verified
        if not state.get("member_status_verify"):
            return self.signal_escalate(state, pick(MSG_NOT_VERIFIED), reason="member_not_verified")

        # 2. Early exit: both slots already collected
        # provider_type = (state.get("provider_type") or "").strip()
        provider_type = normalize_provider_type((state.get("provider_type") or "").strip())
        zip_code_used = (state.get("zip_code_used") or "").strip()
        if provider_type and zip_code_used:
            return self._signal_done(state, provider_type, zip_code_used)

        # 3. Resolve awaiting_slot
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        zip_on_file = (state.get("zip_code") or "").strip()
        # Capture the raw awaiting_slot BEFORE any mutation.
        # Empty raw_awaiting means this is the very first turn inside provider_search
        # (fresh entry from verification). The last user message belongs to verification
        # (e.g. "I'm the plan holder") and must NOT reach the extraction LLM — doing so
        # maps "plan holder" → provider_type, silently skipping the question.
        raw_awaiting = state.get("awaiting_slot", "")
        current_awaiting = raw_awaiting
        if not current_awaiting:
            if not provider_type:
                current_awaiting = "provider_type"
            elif zip_on_file:
                current_awaiting = "zip_confirmed"
            else:
                current_awaiting = "zip_code"
        state = {**state, "awaiting_slot": current_awaiting}

        # ── FIRST-ENTRY FAST PATH ─────────────────────────────────────────────────
        # raw_awaiting is empty → fresh entry from verification; skip all LLM work.
        # Every branch below returns immediately — the LLM extraction and pipeline
        # sections further down only run when raw_awaiting is non-empty.
        if not raw_awaiting:
            if not provider_type:
                # Normal case: ask for provider type with no LLM call.
                interrupt = self.ask_member(state, pick(PROVIDER_SEARCH_BRIDGE_MSGS))
                interrupt["awaiting_slot"] = "provider_type"
                return interrupt
            # Edge case: provider_type pre-populated but zip_code_used missing.
            # Ask for ZIP directly without extraction.
            if zip_on_file:
                r = self.ask_member(
                    state,
                    random.choice(ZIP_CONFIRM_TEMPLATES).format(zip_code=zip_on_file),
                )
                r["awaiting_slot"] = "zip_confirmed"
                r["provider_type"] = provider_type
                return r
            r = self.ask_member(state, ZIP_UPDATE_PROMPT)
            r["awaiting_slot"] = "zip_code"
            r["provider_type"] = provider_type
            return r

        # 4. LLM extraction
        confirmed_slots: dict = {}
        if provider_type:
            confirmed_slots["provider_type"] = provider_type
        if zip_on_file:
            confirmed_slots["zip_code"] = zip_on_file

        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_provider_search_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/provider_search.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            pending_slots=[s for s in PROVIDER_SEARCH_SLOT_ORDER if s not in confirmed_slots],
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        # 5. Conversation guards
        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        # ── fast-fail: valid medical type not supported by this system ──
        # Fires only when the LLM extracted a non-empty provider_type value
        # that the normalizer cannot map (e.g. "radiologist"). This means the
        # caller answered correctly but we cannot serve the request.
        # (non-medical like "electrician") is handled by normal retry
        # exhaustion because the LLM returns empty extraction for those.
        if not provider_type:
            extracted_raw = (result.extracted or {}).get("provider_type", "") if result else ""
            if extracted_raw and not normalize_provider_type(extracted_raw):
                msg = random.choice(MSG_PROVIDER_TYPE_UNSUPPORTED).format(provider_type=extracted_raw.strip())
                return self.signal_escalate(
                    state,
                    msg,
                    reason="provider_type_unsupported",
                )

        # 6. Collect provider_type via pipeline
        collected = {"provider_type": provider_type}
        if interrupt := await self._provider_type_pipeline.collect(
            state, messages, collected, decision=result
        ):
            return interrupt
        provider_type = collected["provider_type"]
        logger.info(LOG_PROVIDER_TYPE, extra={"provider_type": provider_type})

        # 7a. Collecting new ZIP after member declined existing one.
        # The read-back confirmation step has been removed: a valid new ZIP is
        # written to Salesforce immediately and the flow proceeds to delivery.
        if current_awaiting == "zip_code":
            zip_state = {**state, "zip_code": ""}
            collected_zip: dict = {"zip_code": ""}
            if interrupt := await self._zip_pipeline.collect(
                zip_state, messages, collected_zip, decision=result
            ):
                interrupt["provider_type"] = provider_type
                return interrupt
            new_zip = collected_zip["zip_code"]
            if fail := await update_zip_in_salesforce(self, state, new_zip):
                return fail
            logger.info(LOG_ZIP_UPDATED, extra={"zip_code": new_zip})
            done = self._signal_done(state, provider_type, new_zip)
            done["zip_code"] = new_zip
            done["zip_code_updated"] = True
            return done

        # 7b. Processing response to ZIP confirmation question
        if current_awaiting == "zip_confirmed":
            extracted = (result.extracted or {}) if result else {}
            new_zip_raw = extracted.get("zip_code", "")
            zip_conf_raw = extracted.get("zip_confirmed", "")

            zip_conf = normalize_yes_no(zip_conf_raw) if zip_conf_raw else ""
            # Extraction contract: a replacement ZIP and zip_confirmed are mutually
            # exclusive ("if caller declines AND provides a new ZIP, omit
            # zip_confirmed"). If both arrive, zip_code is an echo of the
            # Confirmed: context line — discard it so the yes/no is honored.
            if zip_conf in ("yes", "no"):
                new_zip_raw = ""

            if new_zip_raw:
                normalized = normalize_zip_code(str(new_zip_raw))
                if normalized and validate_zip_code(normalized).valid:
                    if normalized == normalize_zip_code(zip_on_file):
                        # Member repeated the ZIP we already have on file
                        logger.info(LOG_ZIP_CONFIRMED, extra={"zip_code": zip_on_file})
                        return self._signal_done(state, provider_type, zip_on_file)
                    # New ZIP provided inline — accept it directly, write to
                    # Salesforce, and proceed. No read-back confirmation.
                    if fail := await update_zip_in_salesforce(self, state, normalized):
                        return fail
                    logger.info(LOG_ZIP_UPDATED, extra={"zip_code": normalized})
                    done = self._signal_done(state, provider_type, normalized)
                    done["zip_code"] = normalized
                    done["zip_code_updated"] = True
                    return done
                # Provided ZIP was invalid — ask for a proper one
                ask_result = self.ask_member(state, ZIP_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "zip_code"
                ask_result["provider_type"] = provider_type
                ask_result["zip_code"] = zip_on_file
                return ask_result

            if zip_conf == "yes":
                logger.info(LOG_ZIP_CONFIRMED, extra={"zip_code": zip_on_file})
                return self._signal_done(state, provider_type, zip_on_file)
            if zip_conf == "no":
                ask_result = self.ask_member(state, ZIP_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "zip_code"
                ask_result["provider_type"] = provider_type
                ask_result["zip_code"] = zip_on_file
                return ask_result

            # Implicit "no": LLM signalled an update/correction intent targeting
            # zip_confirmed or zip_code but didn't extract zip_confirmed="no" directly.
            # Covers indirect denials like "No. I want to update." or "I'd like to
            # change it" where the caller's intent is unambiguous even though the
            # LLM emits a corrected+update signal instead of a yes/no extraction.
            if (
                result is not None
                and result.event_type == EventType.CORRECTED
                and result.request_kind == RequestKind.UPDATE
                and (result.update_target or "").strip() in ("zip_confirmed", "zip_code", "")
            ):
                ask_result = self.ask_member(state, ZIP_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "zip_code"
                ask_result["provider_type"] = provider_type
                ask_result["zip_code"] = zip_on_file
                return ask_result

            # No clear yes/no — retry or exhaust
            self.slot_fail("zip_confirmed")
            slot = self.get_slot("zip_confirmed")
            if slot.is_exhausted():
                return self.signal_escalate(state, pick(MSG_ZIP_EXHAUST), reason="zip_confirmed_exhausted")
            from agent.llm.response_generator import generate_recovery_message

            spoken_zip = " ".join(zip_on_file)
            ctx = ConversationContext.from_state(state)
            retry_msg = await generate_recovery_message(
                slot_name="zip_confirmed",
                attempt=slot.attempt_count,
                guard="RETRY",
                last_messages=messages[-4:],
                slot_label_override=(
                    f"whether the ZIP code {spoken_zip} on file is correct (yes or no) — "
                    f"if they say their address changed, ask for their current ZIP"
                ),
                caller_name=ctx.caller_first_name,
                confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                user_utterance=last_user,
            )
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "zip_confirmed"
            retry_result["provider_type"] = provider_type
            retry_result["zip_code"] = zip_on_file
            return retry_result

        # 7c. First time asking ZIP confirmation
        if zip_on_file:
            confirm_msg = random.choice(ZIP_CONFIRM_TEMPLATES).format(zip_code=zip_on_file)
            confirm_result = self.ask_member(state, confirm_msg)
            confirm_result["awaiting_slot"] = "zip_confirmed"
            confirm_result["provider_type"] = provider_type
            return confirm_result

        # No ZIP on file — collect a new one
        collect_result = self.ask_member(state, ZIP_UPDATE_PROMPT)
        collect_result["awaiting_slot"] = "zip_code"
        collect_result["provider_type"] = provider_type
        return collect_result

    def _signal_done(self, state: State, provider_type: str, zip_code_used: str) -> dict:
        """
        Provider search complete — ask how the member wants the list delivered.

        Uses ask_member (is_interrupt=True) so the graph pauses for the user's
        fax/email answer. human_node reads next_node="delivery_management_agent"
        from state and routes there after the user responds. delivery_management_agent
        then receives the answer as last_user and extracts the delivery method in
        one LLM call — no double-ask.

        The old signal_complete(is_interrupt=False) caused conditional_routing to
        jump directly to delivery_management_agent with no user pause, so that
        agent had to ask fax/email all over again.

        Routed-update return (Phase 4, Bug C): when pending_cross_agent_request
        marks this run as a ZIP-update detour from another agent, signal
        COMPLETE (no interrupt, no delivery bridge ask) so the orchestrator
        fast-path returns to the request's return_to_agent in the same
        super-step with the refreshed zip_code_used — the search re-runs from
        the new ZIP at dispatch time, which reads zip_code_used.
        """
        from agent.state import normalize_cross_agent_request

        pending = normalize_cross_agent_request(state)
        if pending.get("return_to_agent"):
            result = self.signal_complete(
                state,
                message="",
                resolved_intents=["provider_search"],
            )
            result["provider_type"] = provider_type
            result["zip_code"] = zip_code_used
            result["zip_code_used"] = zip_code_used
            return result

        msg = pick(DELIVERY_BRIDGE_TEMPLATES)
        result = self.ask_member(state, msg)
        result["next_node"] = "delivery_management_agent"  # human_node reads this
        result["awaiting_slot"] = ""  # delivery_management_agent starts fresh
        result["provider_type"] = provider_type
        result["zip_code"] = zip_code_used
        result["zip_code_used"] = zip_code_used
        return result


async def provider_search_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await ProviderSearchAgent.from_state(state).execute(state)
