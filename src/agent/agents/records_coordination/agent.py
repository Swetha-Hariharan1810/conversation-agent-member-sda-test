"""
agent.py — RecordsCoordinationAgent (Sub-Agent 6a)

Four-branch decision tree for obtaining medical records:
  Branch A (member_upload):  generate upload link → send to email → confirm sent
  Branch B (doctor_direct):  acknowledge → offer upload link → if accepted send it
                              → offer Personal Guide
  Branch C (personal_guide): with explicit consent → trigger SF workflow
  Branch D (decline):        signal_escalate after all options exhausted

State machine phases (tracked via awaiting_slot):
  ""                    → initial turn: present Member upload/doctor option
  "upload_method"       → waiting for member's initial records preference
  "upload_consent"      → waiting for yes/no to upload link offer
  "email_confirmed"     → waiting for email confirmation before sending link
  "email"               → waiting for corrected email address
  "personal_guide_consent" → waiting for explicit consent to trigger Personal Guide
"""

from __future__ import annotations

import random

from agent.agents.records_coordination.constants import (
    AGENT_NAME,
    EMAIL_READBACK_FOR_UPLOAD,
    LOG_DOCTOR_DIRECT,
    LOG_ENTERED,
    LOG_GUIDE_TRIGGERED,
    LOG_UPLOAD_LINK_SENT,
    MAX_CONTACT_CHANGE_CYCLES,
    MSG_DECLINE_ESCALATE,
    MSG_DOCTOR_DIRECT_ACK,
    MSG_EMAIL_UPDATE_PROMPT,
    MSG_GUIDE_SCHEDULED,
    MSG_NOTIFICATION_BRIDGE,
    MSG_PERSONAL_GUIDE_OFFER,
    MSG_UPLOAD_OFFER,
    MSG_UPLOAD_SENT,
    RECORDS_SLOT_ORDER,
)
from agent.agents.records_coordination.handlers import dispatch_personal_guide, dispatch_upload_link
from agent.agents.records_coordination.llm import extract_records_decision
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.core.request_detection import reconcile_worker_result
from agent.llm.config import get_extraction_llm
from agent.llm.extractor import remaining_slots
from agent.logger import get_logger
from agent.slots.normalizers import normalize_email, normalize_yes_no
from agent.slots.validators import validate_email
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    pick,
    speak_email,
)

logger = get_logger(__name__)

_MAX_GUIDE_CONSENT_ATTEMPTS = 3


class RecordsCoordinationAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        # ── Re-entry guard ────────────────────────────────────────────────────
        if state.get("records_branch_taken"):
            return self._signal_done(state)

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        current_awaiting = state.get("awaiting_slot", "")

        # ── RESUME after a routed slot update (Phase 7, mirrors delivery) ─────
        # The return hop restored awaiting_slot; re-ask the preserved question
        # — no extraction on the stale turn (the owner consumed it).
        if state.get("slot_update_resume") and current_awaiting:
            result = self._reask_awaiting(state, current_awaiting, prefix="All set — that's been updated. ")
            result["slot_update_resume"] = False
            return result

        # ── First entry fast-path ─────────────────────────────────────────────
        # The ClaimAdjustmentAgent already asked "Can you send it over?"
        # and the member's answer is now last_user. We go straight to extraction.
        if not current_awaiting:
            current_awaiting = "upload_method"
            state = {**state, "awaiting_slot": current_awaiting}

        # ── LLM extraction ────────────────────────────────────────────────────
        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_records_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/records_coordination.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            pending_slots=remaining_slots(RECORDS_SLOT_ORDER, current_awaiting),
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        # ── DETERMINISTIC RECONCILE (Phase 1) ────────────────────────────────
        # llm.py already reconciles on success, but extraction fallbacks (and
        # monkeypatched results) bypass it — re-running here is idempotent.
        result = reconcile_worker_result(result, last_user)

        # ── ROUTED SLOT UPDATE (Phase 7, mirrors delivery's Phase 4 block) ───
        # A ZIP or identity update voiced mid-records routes to its owner and
        # returns to the exact awaiting slot; in-flow targets (email) fall
        # through to the branches below.
        update_target = ((getattr(result, "update_target", None) or "").strip()) if result else ""
        if update_target:
            if route := self._route_foreign_update(state, update_target, return_awaiting=current_awaiting):
                return route

        extracted = (result.extracted or {}) if result else {}

        # ── BRANCH ROUTING ────────────────────────────────────────────────────

        # Phase: upload_method — initial intent from member
        if current_awaiting == "upload_method":
            upload_method = extracted.get("upload_method", "")

            if upload_method == "member_upload":
                # Member wants to upload themselves — offer the link
                offer_result = self.ask_member(state, pick(MSG_UPLOAD_OFFER))
                offer_result["awaiting_slot"] = "upload_consent"
                return offer_result

            if upload_method == "doctor_direct":
                # Doctor will send it — acknowledge, then offer upload link anyway
                logger.info(LOG_DOCTOR_DIRECT)
                ack = pick(MSG_DOCTOR_DIRECT_ACK)
                offer = pick(MSG_UPLOAD_OFFER)
                combined = f"{ack}\n\n{offer}"
                offer_result = self.ask_member(state, combined)
                offer_result["awaiting_slot"] = "upload_consent"
                return offer_result

            if upload_method == "personal_guide":
                # Member immediately wants Personal Guide
                return await self._handle_guide_consent_ask(state)

            if upload_method == "decline":
                # Member declined this step entirely → escalate
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="member_declined_all_records_options",
                )

            # No clear extraction — re-ask (retry once before moving on)
            # Never verbatim-repeat over an unhandled request (Phase 7).
            if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                return handled
            self.slot_fail("upload_method")
            if self.get_slot("upload_method").is_exhausted():
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="records_upload_method_exhausted",
                )
            # Forward any followup_query from the extraction (e.g. "what records
            # do you need exactly?") so the CLARIFY generation LLM can answer it
            # before re-asking — generalized path, no keyword detection needed.
            from agent.core.slot_manager import _mk_session_ctx

            followup_q = (getattr(result, "followup_query", None) or "").strip()
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state,
                "upload_method",
                ctx,
                messages,
                guard="CLARIFY",
                session_context=_mk_session_ctx(followup_query=followup_q) if followup_q else None,
            )
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "upload_method"
            return retry

        # Phase: upload_consent — did member agree to receive the link?
        if current_awaiting == "upload_consent":
            upload_consent = normalize_yes_no(extracted.get("upload_consent", ""))

            if upload_consent == "yes":
                # Confirm email before sending
                email_on_file = (state.get("email") or "").strip()
                if email_on_file:
                    # Spell out the email in words ("at"/"dot") for the spoken message
                    display_email = speak_email(email_on_file)
                    msg = random.choice(EMAIL_READBACK_FOR_UPLOAD).format(email=display_email)
                    confirm_result = self.ask_member(state, msg)
                    confirm_result["awaiting_slot"] = "email_confirmed"
                    return confirm_result
                else:
                    ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                    ask_result["awaiting_slot"] = "email"
                    return ask_result

            if upload_consent == "no":
                # Member declined link — offer Personal Guide
                return await self._handle_guide_consent_ask(state)

            # Ambiguous — retry
            # Never verbatim-repeat over an unhandled request (Phase 7).
            if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                return handled
            self.slot_fail("upload_consent")
            if self.get_slot("upload_consent").is_exhausted():
                return await self._handle_guide_consent_ask(state)
            from agent.core.slot_manager import _mk_session_ctx

            followup_q = (getattr(result, "followup_query", None) or "").strip()
            _guard = "FOLLOWUP_DECLINE" if followup_q else "RETRY"
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state,
                "upload_consent",
                ctx,
                messages,
                guard=_guard,
                session_context=_mk_session_ctx(followup_query=followup_q) if followup_q else None,
            )
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "upload_consent"
            return retry

        # Phase: email_confirmed — is email on file correct?
        if current_awaiting == "email_confirmed":
            new_email_raw = extracted.get("email", "")
            contact_conf_raw = extracted.get("email_confirmed", extracted.get("contact_confirmed", ""))
            email_on_file = (state.get("email") or "").strip()
            pending_email = (state.get("pending_email") or "").strip()

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            # Deterministic fallback (mirrors delivery_management): when the LLM
            # returns extracted:{} for a turn that is clearly a yes/no (e.g. it used
            # corrections:{email:"changed recently"} instead), normalize the raw
            # utterance so a plain "no" / "No, I changed it" advances on the first
            # attempt without burning a retry.
            if not contact_conf and not new_email_raw:
                contact_conf = normalize_yes_no(last_user) if last_user else ""
            # Extraction contract: a replacement email and email_confirmed are
            # mutually exclusive. If a "no" arrives alongside an email, the email is
            # an echo of the Confirmed: context line — discard it so the decline is
            # honored.
            if contact_conf == "no":
                new_email_raw = ""

            # Inline replacement: member declined AND provided new email in same utterance
            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    if normalized == normalize_email(email_on_file):
                        # Member repeated the email we already have on file
                        done = await self._send_link_and_proceed(state, email_on_file)
                        done["pending_email"] = ""
                        return done
                    if pending_email and normalized == normalize_email(pending_email):
                        # Member re-stated the pending (already read-back) email — treat as yes
                        done = await self._send_link_and_proceed(state, pending_email)
                        done["pending_email"] = ""
                        return done
                    # New email — hold as pending until the member confirms the
                    # read-back. Spoken form ("at"/"dot") is used for the spoken
                    # message only.
                    # Inline replacement = implicit rejection of the read-back.
                    # Bound the change cycle so valid-value churn cannot loop forever.
                    if escalation := self.guard_loop_limit(
                        state,
                        "email_change_cycles",
                        MAX_CONTACT_CHANGE_CYCLES,
                        escalate_message=pick(MSG_DECLINE_ESCALATE),
                        escalate_reason="email_change_loop_exceeded_in_records",
                    ):
                        return escalation
                    display_email = speak_email(normalized)
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your email address is {display_email}, correct?",
                    )
                    confirm["awaiting_slot"] = "email_confirmed"
                    confirm["pending_email"] = normalized
                    return confirm
                ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                return ask_result

            # Explicit yes → proceed
            if contact_conf == "yes":
                done = await self._send_link_and_proceed(state, pending_email or email_on_file)
                done["pending_email"] = ""
                return done

            # Explicit no → ask for new email
            if contact_conf == "no":
                if escalation := self.guard_loop_limit(
                    state,
                    "email_change_cycles",
                    MAX_CONTACT_CHANGE_CYCLES,
                    escalate_message=pick(MSG_DECLINE_ESCALATE),
                    escalate_reason="email_change_loop_exceeded_in_records",
                ):
                    return escalation
                ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                return ask_result

            # Ambiguous / no extraction → re-read the email, ask again (do NOT ask for new email)
            # Never verbatim-repeat over an unhandled request (Phase 7).
            if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                return handled
            self.slot_fail("email_confirmed")
            if self.get_slot("email_confirmed").is_exhausted():
                # FIX: escalate on exhaustion instead of silently pivoting to email collection.
                # Test B5 expects escalation after 3 consecutive ambiguous email_confirmed answers.
                esc = self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="email_confirmed_exhausted_in_records",
                )
                esc["pending_email"] = ""
                return esc

            # Re-read the email on file and ask again using CLARIFY (gentle tone)
            from agent.llm.response_generator import generate_recovery_message

            display_email = speak_email(pending_email or email_on_file)
            ctx = ConversationContext.from_state(state)
            retry_msg = await generate_recovery_message(
                slot_name="email_confirmed",
                attempt=self.get_slot("email_confirmed").attempt_count,
                guard="CLARIFY",
                last_messages=messages[-4:],
                slot_label_override=(
                    f"whether the email address {display_email} is correct "
                    f"for sending the upload link (yes or no)"
                ),
                caller_name=ctx.caller_first_name,
                confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                user_utterance=_last_user_msg(messages),
            )
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "email_confirmed"
            return retry_result

        # Phase: email — collecting a new / corrected email
        if current_awaiting == "email":
            new_email_raw = extracted.get("email", "")
            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    # Hold the new email as pending until the member confirms
                    display_email = speak_email(normalized)
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your email address is {display_email}, correct?",
                    )
                    confirm["awaiting_slot"] = "email_confirmed"
                    confirm["pending_email"] = normalized
                    return confirm
            # Never verbatim-repeat over an unhandled request (Phase 7).
            if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                return handled
            self.slot_fail("email")
            if self.get_slot("email").is_exhausted():
                return self.signal_escalate(
                    state,
                    "I wasn't able to capture your email after a few tries. "
                    "Let me connect you with a representative.",
                    reason="email_exhausted_in_records",
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "email", ctx, messages)
            ask_result = self.ask_member(state, msg)
            ask_result["awaiting_slot"] = "email"
            return ask_result

        # Phase: personal_guide_consent — explicit consent required
        if current_awaiting == "personal_guide_consent":
            guide_consent = normalize_yes_no(extracted.get("personal_guide_consent", ""))

            if guide_consent == "yes":
                return await self._trigger_guide_and_proceed(state)

            if guide_consent == "no":
                from agent.agents.follow_up.constants import MSG_FOLLOW_UP_ASK

                handoff = pick(MSG_FOLLOW_UP_ASK)
                result = self.ask_member(state, handoff)
                result["next_node"] = "follow_up_agent"
                result["records_branch_taken"] = "declined_personal_guide"
                # Mark the claim flow as complete so follow_up_agent's
                # is_new_intake_intent gate recognises a subsequent same-intent
                # request (e.g. "I have another adjustment") as a fresh intake.
                result["claim_flow_complete"] = True
                result["last_agent_signal"] = {
                    "status": "complete",
                    "resolved_intents": ["records_coordination"],
                    "closure_requested": False,
                    "context_updates": {},
                    "proactive_offer_available": False,
                    "escalation_reason": None,
                    "reasoning": "records_coordination_agent",
                }
                return result

            # Ambiguous
            # Never verbatim-repeat over an unhandled request (Phase 7).
            if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                return handled
            self.slot_fail("personal_guide_consent")
            if self.get_slot("personal_guide_consent").is_exhausted():
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="personal_guide_consent_exhausted",
                )
            from agent.core.slot_manager import _mk_session_ctx

            followup_q = (getattr(result, "followup_query", None) or "").strip()
            _guard = "FOLLOWUP_DECLINE" if followup_q else "RETRY"
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state,
                "personal_guide_consent",
                ctx,
                messages,
                guard=_guard,
                session_context=_mk_session_ctx(followup_query=followup_q) if followup_q else None,
            )
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "personal_guide_consent"
            return retry

        # Fallback re-ask
        fallback = self.ask_member(state, pick(MSG_UPLOAD_OFFER))
        fallback["awaiting_slot"] = "upload_consent"
        return fallback

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _reask_awaiting(self, state: State, awaiting: str, prefix: str = "") -> dict:
        """Re-ask the preserved awaiting question on a slot_update_resume hop."""
        if awaiting == "email_confirmed":
            email_on_file = (state.get("pending_email") or state.get("email") or "").strip()
            msg = random.choice(EMAIL_READBACK_FOR_UPLOAD).format(email=speak_email(email_on_file))
            result = self.ask_member(state, msg)
        elif awaiting == "email":
            result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
        elif awaiting == "personal_guide_consent":
            result = self.ask_member(state, pick(MSG_PERSONAL_GUIDE_OFFER))
        else:  # upload_method / upload_consent
            result = self.ask_member(state, pick(MSG_UPLOAD_OFFER))
            awaiting = awaiting or "upload_consent"
        result["awaiting_slot"] = awaiting
        if prefix:
            result["messages"]["content"] = prefix + result["messages"]["content"]
        return result

    async def _handle_guide_consent_ask(self, state: State) -> dict:
        """Offer Personal Guide outreach and wait for explicit consent."""
        result = self.ask_member(state, pick(MSG_PERSONAL_GUIDE_OFFER))
        result["awaiting_slot"] = "personal_guide_consent"
        return result

    async def _send_link_and_proceed(self, state: State, email: str) -> dict:
        """
        Dispatch the upload link to email, then offer Personal Guide outreach.
        Corresponds to Branch A (and the B flow that goes through upload).
        """
        if fail := await dispatch_upload_link(self, state, email):
            return fail

        logger.info(LOG_UPLOAD_LINK_SENT, extra={"email_tail": email[-8:]})

        # upload_link_sent=True means the upload_method decision is resolved —
        # mark the slot confirmed so no dangling unconfirmed slot remains.
        self.get_slot("upload_method").record_attempt("upload_link", success=True)

        sent_msg = pick(MSG_UPLOAD_SENT)
        guide_msg = pick(MSG_PERSONAL_GUIDE_OFFER)
        combined = f"{sent_msg}\n\n{guide_msg}"

        result = self.ask_member(state, combined)
        result["upload_link_sent"] = True
        result["email"] = email
        result["awaiting_slot"] = "personal_guide_consent"
        return result

    async def _trigger_guide_and_proceed(self, state: State) -> dict:
        """
        Trigger Personal Guide workflow and transition to Notification Setup.
        Corresponds to Branch C completion.
        """
        if fail := await dispatch_personal_guide(self, state):
            return fail

        logger.info(LOG_GUIDE_TRIGGERED)

        scheduled_msg = pick(MSG_GUIDE_SCHEDULED)
        notification_bridge = pick(MSG_NOTIFICATION_BRIDGE)
        combined = f"{scheduled_msg}\n\n{notification_bridge}"

        result = self.ask_member(state, combined)
        result["personal_guide_outreach_requested"] = True
        result["records_branch_taken"] = "personal_guide"
        result["next_node"] = "notification_setup_agent"
        result["awaiting_slot"] = ""
        return result

    def _signal_done(self, state: State) -> dict:
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["records_coordination"],
            context_updates={
                "records_branch_taken": state.get("records_branch_taken", ""),
                "upload_link_sent": state.get("upload_link_sent", False),
                "personal_guide_outreach_requested": state.get("personal_guide_outreach_requested", False),
            },
        )


async def records_coordination_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await RecordsCoordinationAgent.from_state(state).execute(state)
