"""
agent.py — DeliveryManagementAgent: confirms delivery contact, dispatches provider list,
makes proactive benefits offer.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timezone

from agent.agents.delivery_management.constants import (
    BENEFITS_OFFER_TEMPLATES,
    DELIVERY_SLOT_ORDER,
    DELIVERY_WINDOW_MSG,
    DELIVERY_WINDOW_MSG_ZIP_UPDATED,
    EMAIL_READBACK_TEMPLATES,
    EMAIL_UPDATE_PROMPTS,
    FAX_READBACK_TEMPLATES,
    FAX_UPDATE_PROMPTS,
    LOG_CONTACT_CONFIRMED,
    LOG_CONTACT_UPDATED,
    LOG_ENTERED,
    LOG_LIST_DISPATCHED,
    LOG_METHOD_COLLECTED,
    MAX_CONTACT_CHANGE_CYCLES,
    MSG_CONTACT_EXHAUST,
    MSG_PHONE_DELIVERY_NOT_SUPPORTED,
)
from agent.agents.delivery_management.handlers import (
    dispatch_provider_list,
    update_email_in_salesforce,
    update_fax_in_salesforce,
)
from agent.agents.delivery_management.llm import extract_delivery_management_decision
from agent.agents.delivery_management.pipelines import (
    build_delivery_method_pipeline,
    build_email_pipeline,
    build_fax_pipeline,
)
from agent.core.agent import BaseAgent
from agent.core.request_detection import detect_request, reconcile_worker_result
from agent.core.slot_ownership import canonical_capability_topic
from agent.llm.config import get_extraction_llm
from agent.llm.extractor import remaining_slots
from agent.logger import get_logger
from agent.slots.normalizers import normalize_email, normalize_fax_number, normalize_yes_no
from agent.slots.validators import validate_email, validate_fax_number
from agent.state import State, normalize_cross_agent_request
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    pick,
    speak_email,
)

logger = get_logger(__name__)


class DeliveryManagementAgent(BaseAgent):
    AGENT_NAME = "delivery_management_agent"

    # Caller requests phone/verbal delivery — only fax and email are supported.
    # Patterns are intentionally narrow to avoid catching side questions that
    # merely mention the word "phone" in a different context.
    _PHONE_DELIVERY_RE = re.compile(
        r"\bby\s+phone\b"
        r"|\bover\s+(?:the\s+)?phone\b"
        r"|\bvia\s+phone\b"
        r"|\bon\s+(?:the\s+)?phone\s+now\b"
        r"|\bphone\s+now\b"
        r"|\bcall\s+me\s+(?:and|to)\s+tell\b"
        r"|\bverbally\b"
        r"|\btell\s+me\s+(?:the\s+)?(?:details|information|names|list|providers?)\b"
        r"|\btell\s+me\s+now\b"
        r"|\bjust\s+(?:tell|call)\s+me\b"
        r"|\bprovider\s+(?:list|names?|information?)\s+by\s+phone\b",
        re.IGNORECASE,
    )
    # Stale-value and update-intent signals for contact confirmation slots.
    # When these appear in the utterance the caller is signaling the contact on
    # file is wrong/outdated — treat as a decline even when the message starts
    # with a leading affirmative like "yeah" or "yes".
    _STALE_CONTACT_RE = re.compile(
        r"\bold\s+(?:fax|email|number|address)\b"  # "old fax number", "old email"
        r"|\boutdated\b"
        r"|\bgive\s+you\s+a\s+new\b"  # "I'll give you a new number"
        r"|\bhas\s+changed\b"
        r"|\bno\s+longer\s+(?:work|active|current|valid|right|correct)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__()
        self._delivery_method_pipeline = build_delivery_method_pipeline(self)
        self._fax_pipeline = build_fax_pipeline(self)
        self._email_pipeline = build_email_pipeline(self)

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        delivery_method = (state.get("delivery_method") or "").strip()
        fax_on_file = (state.get("fax") or "").strip()
        email_on_file = (state.get("email") or "").strip()

        current_awaiting = state.get("awaiting_slot", "") or "delivery_method"

        # ── CROSS-AGENT RE-ENTRY (Phase 6): redo / replay aimed at us ────────
        # A pending request with a delivery target means we are re-dispatching
        # (kind redo/update) or recapping (kind replay) — the completed-flow
        # early exits below must NOT fire while it is unresolved. redo_active
        # is checked on the raw pending request (not consume_cross_agent_
        # request) because the in-flow re-dispatch marker names US as the
        # requester and must still gate the early exits on later turns.
        pending_request = normalize_cross_agent_request(state)
        redo_active = pending_request.get("kind") in ("redo", "update") and pending_request.get("target") in (
            "delivery",
            "delivery_method",
        )
        replay_request = self.consume_cross_agent_request(
            state, kinds=("replay",), targets=("provider_list",)
        )
        if replay_request and state.get("provider_list_sent"):
            return self._replay_provider_list(state, replay_request)

        # ── EARLY EXIT: already dispatched + benefits offered (re-entry) ─────
        # Check before LLM extraction to avoid unnecessary calls.
        if (
            state.get("provider_list_sent")
            and state.get("benefits_offer_made")
            and current_awaiting != "benefits_response"
            and not redo_active
        ):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(state, delivery_method, False),
            )

        # ── RE-DISPATCH (Phase 6, redo): re-collect the delivery method ──────
        # The list was already sent; the caller wants it again by another
        # method/destination. Keep provider_list_sent history, never repeat
        # the benefits offer — _proceed_to_dispatch closes the redo instead.
        # The state copy clears delivery_method so the pipeline re-collects
        # instead of short-circuiting on the previous method.
        if redo_active and current_awaiting == "delivery_method":
            delivery_method = ""
            state = {**state, "delivery_method": ""}

        # ── RECOVERY: dispatched but offer not yet made ──────────────────────
        if (
            state.get("provider_list_sent")
            and not state.get("benefits_offer_made")
            and current_awaiting != "benefits_response"
            and not redo_active
        ):
            provider_type = (state.get("provider_type") or "provider").strip()
            benefits_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
            offer_result = self.ask_member(state, benefits_msg)
            offer_result["awaiting_slot"] = "benefits_response"
            offer_result["benefits_offer_made"] = True
            return offer_result

        state = {**state, "awaiting_slot": current_awaiting}

        # ── RESUME after a routed slot update (Phase 4, Bug C) ───────────────
        # The orchestrator fast-path sent us back here after the owning agent
        # (provider_search) finished the ZIP update. Acknowledge it and re-ask
        # the contact question we were on — no extraction on this hop (the
        # last user utterance was the new ZIP, already consumed by the owner).
        if state.get("slot_update_resume"):
            zip_used = (state.get("zip_code_used") or state.get("zip_code") or "").strip()
            prefix = (
                f"All set — I've updated your ZIP code to {zip_used} "
                "and refreshed your provider list for that area. "
                if zip_used
                else ""
            )
            if not delivery_method:
                # Routed away before fax/email was chosen — ask it now.
                from agent.agents.provider_search.constants import DELIVERY_BRIDGE_TEMPLATES

                result = self.ask_member(state, prefix + pick(DELIVERY_BRIDGE_TEMPLATES))
                result["awaiting_slot"] = "delivery_method"
            else:
                result = self._ask_contact_confirmation(
                    state, delivery_method, fax_on_file, email_on_file, prefix=prefix
                )
            result["slot_update_resume"] = False
            return result

        # LLM extraction
        confirmed_slots: dict = {}
        if delivery_method:
            confirmed_slots["delivery_method"] = delivery_method

        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_delivery_management_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/delivery_management.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            pending_slots=remaining_slots(DELIVERY_SLOT_ORDER, current_awaiting),
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        # Conversation guards
        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        # ── DETERMINISTIC RECONCILE (Phase 1) ────────────────────────────────
        # llm.py already reconciles on success, but extraction fallbacks (and
        # monkeypatched results) bypass it — re-running here is idempotent and
        # guarantees update_target/request_kind before any branch logic.
        result = reconcile_worker_result(result, last_user)

        # ── ROUTED SLOT UPDATE (Phase 4, Bug C) ──────────────────────────────
        # "Actually my ZIP changed" mid-delivery: zip_code is owned by
        # provider_search (route_to_owner) — hand off NOW instead of repeating
        # the fax/email question over the caller's request.
        update_target = ((getattr(result, "update_target", None) or "").strip()) if result else ""
        _kind_raw = getattr(result, "request_kind", None) if result else None
        _request_kind = str(getattr(_kind_raw, "value", _kind_raw) or "").strip().lower()
        _corrections = (
            {k: v for k, v in ((getattr(result, "corrections", None) or {}).items()) if v} if result else {}
        )
        if not update_target and "zip_code" in _corrections:
            update_target = "zip_code"
        if update_target:
            # ── PRE-DISPATCH CONTACT UPDATE: "sorry I need to update this" ───────
            # When the caller asks to update the fax or email contact BEFORE the
            # delivery method has been selected and the list has not yet been
            # sent, infer the delivery method from the update target and go
            # directly to contact confirmation. This handles "sorry I need to
            # update this" / "I need to update my fax" voiced mid-delivery-method
            # question: the agent reads back the on-file contact and the caller
            # can then correct the ZIP or confirm the contact as part of that flow.
            if (
                not delivery_method
                and update_target in ("fax", "email")
                and not state.get("provider_list_sent")
            ):
                logger.info(
                    "delivery_management: pre-dispatch contact update request — "
                    "inferring delivery_method from update_target",
                    extra={"update_target": update_target},
                )
                return self._ask_contact_confirmation(state, update_target, fax_on_file, email_on_file)
            # ── LIVE REDO (Phase 6): "send it by email instead" post-dispatch.
            # We own the delivery capability, so this resolves in-flow — no
            # orchestrator hop. The pending marker (requester = us) keeps the
            # completed-flow early exits open across the re-collection turns.
            # canonical_capability_topic maps list-phrased redo targets
            # (provider_list) onto delivery; live replays recap, never re-send.
            if (
                not redo_active
                and state.get("provider_list_sent")
                and _request_kind != "replay"
                and canonical_capability_topic("redo", update_target) == "delivery"
            ):
                return self._begin_redispatch(state, current_awaiting)
            # ── REPLAY: caller asked to hear back the current value of a slot ─────
            # "can you tell me my ZIP code again?" — answer with the current value
            # on file and re-ask the pending delivery question.
            # A replay is NOT an update: do NOT route to the owning agent.
            if _request_kind == "replay" and update_target == "zip_code":
                zip_current = (state.get("zip_code_used") or state.get("zip_code") or "").strip()
                prefix = f"Your current ZIP code is {zip_current}. " if zip_current else ""
                if not delivery_method:
                    from agent.agents.provider_search.constants import DELIVERY_BRIDGE_TEMPLATES

                    r = self.ask_member(state, prefix + pick(DELIVERY_BRIDGE_TEMPLATES))
                    r["awaiting_slot"] = "delivery_method"
                else:
                    r = self._ask_contact_confirmation(
                        state, delivery_method, fax_on_file, email_on_file, prefix=prefix
                    )
                return r

            from agent.conversation.context import ConversationContext

            ctx = ConversationContext.from_state(state)
            delivery_slots = {"delivery_method": None, "fax": None, "email": None}
            if self.resolve_update_target(update_target, ctx, state, delivery_slots) == "route":
                return self._route_slot_update(state, update_target, ctx, return_awaiting=current_awaiting)
            # Post-dispatch contact update while the benefits offer is still
            # pending (current_awaiting == "benefits_response"): the fax/email
            # branches further below are never reached in this phase, so we
            # must handle it here.
            if (
                state.get("provider_list_sent")
                and update_target in ("fax", "email")
                and current_awaiting == "benefits_response"
            ):
                if update_target == delivery_method:
                    # Same-channel update: caller wants to change the contact
                    # they already chose — ask for the new contact directly
                    # without re-asking the delivery method.
                    prompts = FAX_UPDATE_PROMPTS if update_target == "fax" else EMAIL_UPDATE_PROMPTS
                    r = self.ask_member(state, pick(prompts))
                    r["awaiting_slot"] = update_target
                    r[f"pending_{update_target}"] = ""
                    r[update_target] = fax_on_file if update_target == "fax" else email_on_file
                    r["delivery_method"] = delivery_method
                    r["pending_cross_agent_request"] = {
                        "kind": "redo",
                        "target": "delivery",
                        "return_to_agent": self.AGENT_NAME,
                        "return_awaiting": current_awaiting,
                    }
                    return r
                # Channel switch (e.g. "send by email instead"): re-ask delivery method.
                return self._begin_redispatch(state, current_awaiting)
            # allow/decline: fall through — the fax/email branches below handle
            # in-flow contact updates (including pre-dispatch delivery-method
            # switches, which stay in-flow per the capability registry);
            # human-only targets decline downstream.

        # ── BENEFITS RESPONSE PHASE ──────────────────────────────────────────
        if current_awaiting == "benefits_response":
            return await self._handle_benefits_response(state, result)

        # ── COLLECT DELIVERY METHOD ──────────────────────────────────────────
        if not delivery_method:
            collected: dict = {"delivery_method": ""}
            if interrupt := await self._delivery_method_pipeline.collect(
                state, messages, collected, decision=result
            ):
                # Redo re-entry guard: the pipeline may return a FOLLOWUP_DECLINE
                # interrupt because the LLM misclassifies "instead of fax" as a
                # decline-worthy redo side-question rather than the plain
                # delivery_method answer it is. When redo is active and the slot
                # was nonetheless confirmed (collected dict is populated), bypass
                # the decline and proceed to contact confirmation.
                if redo_active and collected.get("delivery_method"):
                    delivery_method = collected["delivery_method"]
                    logger.info(
                        "delivery_management: redo — bypassed FOLLOWUP_DECLINE, delivery_method confirmed",
                        extra={"delivery_method": delivery_method},
                    )
                    # Fall through to _ask_contact_confirmation below
                else:
                    return interrupt
            delivery_method = collected["delivery_method"]
            logger.info(LOG_METHOD_COLLECTED, extra={"delivery_method": delivery_method})
            # Immediately proceed to contact confirmation in the same turn
            return self._ask_contact_confirmation(state, delivery_method, fax_on_file, email_on_file)

        # ── FAX CONFIRMATION ─────────────────────────────────────────────────
        if current_awaiting == "fax_confirmed":
            if switch := self._maybe_switch_method(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return switch
            extracted = (result.extracted or {}) if result else {}
            new_fax_raw = extracted.get("fax", "")
            contact_conf_raw = extracted.get("fax_confirmed", "")
            pending_fax = (state.get("pending_fax") or "").strip()

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            # Deterministic fallback: the LLM sometimes puts descriptive text in
            # corrections (e.g. corrections:{fax:"changed recently"}) instead of
            # extracted:{fax_confirmed:"no"}, leaving contact_conf empty. When
            # extracted is empty AND no replacement fax was given this turn,
            # normalize the raw utterance so a plain "no" / "No, I changed it"
            # advances on the first attempt without burning a retry.
            # Guard: utterances that start with a leading affirmative ("yeah")
            # but also contain stale-value or new-number signals must be treated
            # as a decline — the correction intent takes priority.
            if not contact_conf and not new_fax_raw:
                if last_user and self._STALE_CONTACT_RE.search(last_user):
                    contact_conf = "no"
                elif last_user:
                    contact_conf = normalize_yes_no(last_user)
            # Extraction contract: a replacement fax and fax_confirmed are mutually
            # exclusive per the prompt. When the LLM sets both and the fax differs
            # from the on-file value it is a genuine replacement alongside the decline
            # — let the new_fax_raw block below handle it. Only clear new_fax_raw
            # when it matches the on-file value (Confirmed: context echo).
            if contact_conf == "no" and (
                not new_fax_raw or normalize_fax_number(str(new_fax_raw)) == normalize_fax_number(fax_on_file)
            ):
                new_fax_raw = ""

            if new_fax_raw:
                normalized = normalize_fax_number(str(new_fax_raw))
                if normalized and validate_fax_number(normalized).valid:
                    if normalized == normalize_fax_number(fax_on_file):
                        # Member repeated the fax we already have on file
                        logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "fax"})
                        done = await self._proceed_to_dispatch(state, delivery_method, fax_on_file)
                        done["pending_fax"] = ""
                        return done
                    if pending_fax and normalized == normalize_fax_number(pending_fax):
                        # Member re-stated the pending (already read-back) fax — treat as yes
                        if fail := await update_fax_in_salesforce(self, state, pending_fax):
                            return fail
                        logger.info(LOG_CONTACT_UPDATED, extra={"fax_tail": pending_fax[-4:]})
                        done = await self._proceed_to_dispatch(state, delivery_method, pending_fax)
                        done["pending_fax"] = ""
                        return done
                    # New fax — hold as pending until the member confirms the
                    # read-back. fax stays on the on-file value because ask_member
                    # would otherwise persist the pipeline-confirmed slot.
                    # Inline replacement = implicit rejection of the read-back.
                    # Bound the change cycle so valid-value churn cannot loop forever.
                    if escalation := self.guard_loop_limit(
                        state,
                        "fax_change_cycles",
                        MAX_CONTACT_CHANGE_CYCLES,
                        escalate_message=pick(MSG_CONTACT_EXHAUST),
                        escalate_reason="fax_change_loop_exceeded",
                    ):
                        return escalation
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your fax number is "
                        f"{normalized[:3]}-{normalized[3:6]}-{normalized[6:]}, correct?",
                    )
                    confirm["awaiting_slot"] = "fax_confirmed"
                    confirm["pending_fax"] = normalized
                    confirm["fax"] = fax_on_file
                    return confirm
                ask_result = self.ask_member(state, pick(FAX_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "fax"
                ask_result["pending_fax"] = ""
                ask_result["fax"] = fax_on_file
                return ask_result

            if contact_conf == "yes":
                if pending_fax:
                    if fail := await update_fax_in_salesforce(self, state, pending_fax):
                        return fail
                    logger.info(LOG_CONTACT_UPDATED, extra={"fax_tail": pending_fax[-4:]})
                    done = await self._proceed_to_dispatch(state, delivery_method, pending_fax)
                    done["pending_fax"] = ""
                    return done
                logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "fax"})
                done = await self._proceed_to_dispatch(state, delivery_method, fax_on_file)
                done["pending_fax"] = ""
                return done
            if contact_conf == "no":
                if escalation := self.guard_loop_limit(
                    state,
                    "fax_change_cycles",
                    MAX_CONTACT_CHANGE_CYCLES,
                    escalate_message=pick(MSG_CONTACT_EXHAUST),
                    escalate_reason="fax_change_loop_exceeded",
                ):
                    return escalation
                ask_result = self.ask_member(state, pick(FAX_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "fax"
                ask_result["pending_fax"] = ""
                ask_result["fax"] = fax_on_file
                return ask_result

            # No clear yes/no — before burning a retry, make sure the turn is
            # not an unhandled request (never verbatim-repeat over one).
            if handled := self._reroute_unhandled_request(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return handled
            self.slot_fail("fax_confirmed")
            if self.get_slot("fax_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="fax_confirmed_exhausted"
                )
            live_fax = pending_fax or fax_on_file
            retry_msg = random.choice(FAX_READBACK_TEMPLATES).format(fax=live_fax)
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "fax_confirmed"
            retry_result["fax"] = fax_on_file
            return retry_result

        # ── FAX UPDATE ───────────────────────────────────────────────────────
        if current_awaiting == "fax":
            if switch := self._maybe_switch_method(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return switch
            # Phone delivery intercept: caller refuses fax and wants phone instead.
            # Reset delivery_method so the caller can choose fax or email fresh.
            if self._PHONE_DELIVERY_RE.search(last_user or ""):
                r = self.ask_member(state, pick(MSG_PHONE_DELIVERY_NOT_SUPPORTED))
                r["awaiting_slot"] = "delivery_method"
                r["delivery_method"] = ""
                return r
            fax_state = {**state, "fax": ""}
            collected_fax: dict = {"fax": ""}
            if interrupt := await self._fax_pipeline.collect(
                fax_state, messages, collected_fax, decision=result
            ):
                return interrupt
            new_fax = collected_fax["fax"]
            # Hold the new fax as pending — no Salesforce write until confirmed
            confirm = self.ask_member(
                state,
                f"Just to be sure I have it right — your fax number is "
                f"{new_fax[:3]}-{new_fax[3:6]}-{new_fax[6:]}, correct?",
            )
            confirm["awaiting_slot"] = "fax_confirmed"
            confirm["pending_fax"] = new_fax
            confirm["fax"] = fax_on_file
            return confirm

        # ── EMAIL CONFIRMATION ───────────────────────────────────────────────
        if current_awaiting == "email_confirmed":
            if switch := self._maybe_switch_method(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return switch
            extracted = (result.extracted or {}) if result else {}
            new_email_raw = extracted.get("email", "")
            contact_conf_raw = extracted.get("email_confirmed", "")
            pending_email = (state.get("pending_email") or "").strip()

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            # Deterministic fallback (mirrors fax_confirmed): when the LLM returns
            # extracted:{} for a turn that is clearly a yes/no, normalize the raw
            # utterance so a plain "no" / "no, use a different one" advances on the
            # first attempt without burning a retry.
            # Guard: same stale-contact check as fax_confirmed — a leading
            # affirmative followed by stale-value content is still a decline.
            if not contact_conf and not new_email_raw:
                if last_user and self._STALE_CONTACT_RE.search(last_user):
                    contact_conf = "no"
                elif last_user:
                    contact_conf = normalize_yes_no(last_user)
            # Extraction contract: a replacement email and email_confirmed are
            # mutually exclusive per the prompt. When the LLM sets both and the email
            # differs from the on-file value it is a genuine replacement alongside
            # the decline — let the new_email_raw block below handle it. Only clear
            # new_email_raw when it matches the on-file value (Confirmed: context echo).
            if contact_conf == "no" and (
                not new_email_raw or normalize_email(str(new_email_raw)) == normalize_email(email_on_file)
            ):
                new_email_raw = ""

            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    if normalized == normalize_email(email_on_file):
                        # Member repeated the email we already have on file
                        logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "email"})
                        done = await self._proceed_to_dispatch(state, delivery_method, email_on_file)
                        done["pending_email"] = ""
                        return done
                    if pending_email and normalized == normalize_email(pending_email):
                        # Member re-stated the pending (already read-back) email — treat as yes
                        if fail := await update_email_in_salesforce(self, state, pending_email):
                            return fail
                        logger.info(LOG_CONTACT_UPDATED, extra={"method": "email"})
                        done = await self._proceed_to_dispatch(state, delivery_method, pending_email)
                        done["pending_email"] = ""
                        return done
                    # New email — hold as pending until the member confirms the
                    # read-back. Spoken form ("at"/"dot") is used for the spoken
                    # message only; the raw value stays in pending_email.
                    # Inline replacement = implicit rejection of the read-back.
                    # Bound the change cycle so valid-value churn cannot loop forever.
                    if escalation := self.guard_loop_limit(
                        state,
                        "email_change_cycles",
                        MAX_CONTACT_CHANGE_CYCLES,
                        escalate_message=pick(MSG_CONTACT_EXHAUST),
                        escalate_reason="email_change_loop_exceeded",
                    ):
                        return escalation
                    display_email = speak_email(normalized)
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your email address is {display_email}, correct?",
                    )
                    confirm["awaiting_slot"] = "email_confirmed"
                    confirm["pending_email"] = normalized
                    confirm["email"] = email_on_file
                    return confirm
                ask_result = self.ask_member(state, pick(EMAIL_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                ask_result["email"] = email_on_file
                return ask_result

            if contact_conf == "yes":
                if pending_email:
                    if fail := await update_email_in_salesforce(self, state, pending_email):
                        return fail
                    logger.info(LOG_CONTACT_UPDATED, extra={"method": "email"})
                    done = await self._proceed_to_dispatch(state, delivery_method, pending_email)
                    done["pending_email"] = ""
                    return done
                logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "email"})
                done = await self._proceed_to_dispatch(state, delivery_method, email_on_file)
                done["pending_email"] = ""
                return done
            if contact_conf == "no":
                if escalation := self.guard_loop_limit(
                    state,
                    "email_change_cycles",
                    MAX_CONTACT_CHANGE_CYCLES,
                    escalate_message=pick(MSG_CONTACT_EXHAUST),
                    escalate_reason="email_change_loop_exceeded",
                ):
                    return escalation
                ask_result = self.ask_member(state, pick(EMAIL_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                ask_result["email"] = email_on_file
                return ask_result

            # No clear yes/no — before burning a retry, make sure the turn is
            # not an unhandled request (never verbatim-repeat over one).
            if handled := self._reroute_unhandled_request(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return handled
            self.slot_fail("email_confirmed")
            if self.get_slot("email_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="email_confirmed_exhausted"
                )
            # FIX: spell out email in words ("at"/"dot") before writing into message history
            live_email = pending_email or email_on_file
            display_email = speak_email(live_email)
            retry_msg = random.choice(EMAIL_READBACK_TEMPLATES).format(email=display_email)
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "email_confirmed"
            retry_result["email"] = email_on_file
            return retry_result

        # ── EMAIL UPDATE ─────────────────────────────────────────────────────
        if current_awaiting == "email":
            if switch := self._maybe_switch_method(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            ):
                return switch
            # Phone delivery intercept: caller refuses email and wants phone instead.
            # Reset delivery_method so the caller can choose fax or email fresh.
            if self._PHONE_DELIVERY_RE.search(last_user or ""):
                r = self.ask_member(state, pick(MSG_PHONE_DELIVERY_NOT_SUPPORTED))
                r["awaiting_slot"] = "delivery_method"
                r["delivery_method"] = ""
                return r
            email_state = {**state, "email": ""}
            collected_email: dict = {"email": ""}
            if interrupt := await self._email_pipeline.collect(
                email_state, messages, collected_email, decision=result
            ):
                return interrupt
            new_email = collected_email["email"]
            # Hold the new email as pending — no Salesforce write until confirmed
            display_email = speak_email(new_email)
            confirm = self.ask_member(
                state,
                f"Just to be sure I have it right — your email address is {display_email}, correct?",
            )
            confirm["awaiting_slot"] = "email_confirmed"
            confirm["pending_email"] = new_email
            confirm["email"] = email_on_file
            return confirm

        # ── FALLBACK: delivery_method known but awaiting_slot not matched ────
        # Re-ask for contact confirmation (handles unexpected re-entry)
        return self._ask_contact_confirmation(state, delivery_method, fax_on_file, email_on_file)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _maybe_switch_method(  # noqa: C901
        self,
        state: State,
        result,
        current_awaiting: str,
        delivery_method: str,
        fax_on_file: str,
        email_on_file: str,
    ) -> dict | None:
        """Detect and honor a channel switch during contact confirmation.

        Triggers (checked in order):
          a. extraction produced a valid delivery_method different from the
             current one ("actually email is better");
          b. the caller answered a fax question with a valid email value (or
             vice versa) — giving the other channel's contact IS the switch;
             the value is carried through as the new pending contact;
          c. update_target / detect_request says redo|update on the delivery
             topic ("send it to my email instead", "use the other method") —
             with only two channels, the other one is implied unless the
             caller named ONLY the current channel (that is a same-channel
             redirect, e.g. "use a different fax" — the branch handles it).

        Pre-dispatch: switch delivery_method, drop the abandoned channel's
        pending value and change-cycle counters, and ask the new channel's
        confirmation (or the pending-value read-back when the new contact
        arrived in the same utterance). Post-dispatch with no redo already in
        flight: the switch is a re-send — delegate to _begin_redispatch.
        Returns None when no switch is requested.
        """
        extracted = (getattr(result, "extracted", None) or {}) if result else {}
        last_user = _last_user_msg(list(state.get("messages") or []))
        lowered = last_user.lower()

        old_method = (delivery_method or "").strip().lower()
        other = {"fax": "email", "email": "fax"}.get(old_method, "")
        if not other:
            return None

        # (a) explicit method in this turn
        new_method = (extracted.get("delivery_method") or "").strip().lower()
        if new_method not in ("fax", "email") or new_method == old_method:
            new_method = ""

        # (b) the other channel's value answered this channel's question
        if not new_method:
            awaiting_channel = "fax" if current_awaiting in ("fax_confirmed", "fax") else "email"
            if awaiting_channel == "fax":
                candidate = normalize_email(str(extracted.get("email") or ""))
                if candidate and validate_email(candidate).valid:
                    new_method = "email"
            else:
                candidate = normalize_fax_number(str(extracted.get("fax") or ""))
                if candidate and validate_fax_number(candidate).valid:
                    new_method = "fax"

        # (c) delivery-topic redo/update with no explicit method. Live replays
        # recap what was sent — they never switch the method or re-send.
        if not new_method:
            target = ((getattr(result, "update_target", None) or "").strip()) if result else ""
            kind_raw = getattr(result, "request_kind", None) if result else None
            request_kind = str(getattr(kind_raw, "value", kind_raw) or "").strip().lower()
            detected = detect_request(last_user)
            delivery_request = (
                request_kind != "replay" and canonical_capability_topic("redo", target) == "delivery"
            ) or (
                detected is not None
                and detected.kind in ("redo", "update")
                and detected.target in ("delivery", "delivery_method")
            )
            if delivery_request and (
                re.search(rf"\b{other}\b", lowered) or not re.search(rf"\b{old_method}\b", lowered)
            ):
                new_method = other

        if not new_method:
            return None

        # Post-dispatch: the list already went out — the switch is a re-send.
        pending_request = normalize_cross_agent_request(state)
        redo_active = pending_request.get("kind") in ("redo", "update") and pending_request.get("target") in (
            "delivery",
            "delivery_method",
        )
        if state.get("provider_list_sent") and not redo_active:
            return self._begin_redispatch(state, current_awaiting)

        logger.info(
            LOG_METHOD_COLLECTED,
            extra={"delivery_method": new_method, "switched_from": old_method},
        )

        # Abandon the old channel cleanly: its pending value and change-cycle /
        # confirmation counters must not leak into the new channel's flow.
        self.get_slot(f"{old_method}_change_cycles").reset()
        self.get_slot(f"{old_method}_confirmed").reset()

        # New contact value in the same utterance → straight to its read-back.
        carried_value = ""
        if new_method == "email":
            candidate = normalize_email(str(extracted.get("email") or ""))
            if candidate and validate_email(candidate).valid:
                carried_value = candidate
        else:
            candidate = normalize_fax_number(str(extracted.get("fax") or ""))
            if candidate and validate_fax_number(candidate).valid:
                carried_value = candidate

        if carried_value:
            if new_method == "email":
                confirm = self.ask_member(
                    state,
                    f"Just to be sure I have it right — your email address is "
                    f"{speak_email(carried_value)}, correct?",
                )
                confirm["awaiting_slot"] = "email_confirmed"
                confirm["pending_email"] = carried_value
                confirm["email"] = email_on_file
            else:
                confirm = self.ask_member(
                    state,
                    f"Just to be sure I have it right — your fax number is "
                    f"{carried_value[:3]}-{carried_value[3:6]}-{carried_value[6:]}, correct?",
                )
                confirm["awaiting_slot"] = "fax_confirmed"
                confirm["pending_fax"] = carried_value
                confirm["fax"] = fax_on_file
            confirm["delivery_method"] = new_method
            confirm[f"pending_{old_method}"] = ""
            return confirm

        switch = self._ask_contact_confirmation(state, new_method, fax_on_file, email_on_file)
        switch[f"pending_{old_method}"] = ""
        return switch

    def _reroute_unhandled_request(
        self,
        state: State,
        result,
        current_awaiting: str,
        delivery_method: str,
        fax_on_file: str,
        email_on_file: str,
    ) -> dict | None:
        """Last-resort request check before a verbatim confirmation retry.

        A "no clear yes/no" turn that is actually a request must be honored,
        never retried over: a routable slot update ("my ZIP changed") hands
        off to its owner; a delivery switch goes through _maybe_switch_method.
        Returns None only for genuinely unclassifiable turns — those may
        burn a slot_fail retry.
        """
        last_user = _last_user_msg(list(state.get("messages") or []))
        detected = detect_request(last_user)
        if detected is None:
            return None

        # Delivery switch (redo, or an update aimed at the delivery topic).
        if detected.kind == "redo" or detected.target in ("delivery", "delivery_method"):
            return self._maybe_switch_method(
                state, result, current_awaiting, delivery_method, fax_on_file, email_on_file
            )
        if detected.kind != "update":
            return None

        # Same-channel update ("change my fax number" over the fax read-back)
        # is a decline of the value on file — ask for the new one.
        channel = "fax" if current_awaiting.startswith("fax") else "email"
        if detected.target == channel:
            if escalation := self.guard_loop_limit(
                state,
                f"{channel}_change_cycles",
                MAX_CONTACT_CHANGE_CYCLES,
                escalate_message=pick(MSG_CONTACT_EXHAUST),
                escalate_reason=f"{channel}_change_loop_exceeded",
            ):
                return escalation
            prompts = FAX_UPDATE_PROMPTS if channel == "fax" else EMAIL_UPDATE_PROMPTS
            ask_result = self.ask_member(state, pick(prompts))
            ask_result["awaiting_slot"] = channel
            ask_result[f"pending_{channel}"] = ""
            ask_result[channel] = fax_on_file if channel == "fax" else email_on_file
            return ask_result

        # Foreign slot update ("my ZIP changed") — route to its owner NOW.
        from agent.conversation.context import ConversationContext

        ctx = ConversationContext.from_state(state)
        delivery_slots = {"delivery_method": None, "fax": None, "email": None}
        if self.resolve_update_target(detected.target, ctx, state, delivery_slots) == "route":
            return self._route_slot_update(state, detected.target, ctx, return_awaiting=current_awaiting)
        return None

    async def _handle_benefits_response(self, state: State, result) -> dict:
        """Process the member's yes/no response to the benefits offer."""
        extracted = (result.extracted or {}) if result else {}
        benefits_raw = extracted.get("benefits_response", "")
        benefits_conf = normalize_yes_no(benefits_raw) if benefits_raw else ""

        if benefits_conf in ("yes", "no"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(
                    state,
                    state.get("delivery_method", ""),
                    benefits_conf == "yes",
                ),
                proactive_offer_available=(benefits_conf == "yes"),
            )

        # No clear yes/no — retry or exhaust gracefully
        self.slot_fail("benefits_response")
        if self.get_slot("benefits_response").is_exhausted():
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(state, state.get("delivery_method", ""), False),
                proactive_offer_available=False,
            )

        provider_type = (state.get("provider_type") or "provider").strip()
        retry_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
        retry_result = self.ask_member(state, retry_msg)
        retry_result["awaiting_slot"] = "benefits_response"
        retry_result["provider_list_sent"] = True
        retry_result["benefits_offer_made"] = True
        return retry_result

    def _blocking_list_invalidator(self, state: State) -> str:
        """Target of an unresolved update that invalidates the provider list.

        Checks pending_cross_agent_request and parked kind="action" items
        against the ownership registry: any target whose entry invalidates
        provider_list_sent (i.e. zip_code) blocks dispatch until resolved —
        never send a list generated from a ZIP the caller has disputed.
        """
        from agent.core.slot_ownership import get_ownership
        from agent.state import normalize_cross_agent_request, normalize_parked_followups

        candidates = [normalize_cross_agent_request(state).get("target", "")]
        candidates += [
            p.get("target", "")
            for p in normalize_parked_followups(state.get("parked_followups"))
            if p.get("kind") == "action"
        ]
        for target in candidates:
            own = get_ownership(target)
            if own and "provider_list_sent" in own.invalidates:
                return target
        return ""

    async def _proceed_to_dispatch(
        self, state: State, delivery_method: str, confirmed_destination: str
    ) -> dict:
        """Dispatch the provider list then make the benefits offer."""
        # ── DISPATCH PRECONDITION (Phase 4, Bug C) ───────────────────────────
        # A pending/parked update that invalidates the provider list (ZIP) must
        # be resolved BEFORE dispatch — route it to its owner now.
        if blocker := self._blocking_list_invalidator(state):
            from agent.conversation.context import ConversationContext
            from agent.state import normalize_parked_followups

            logger.info(
                "delivery_management: dispatch blocked by pending %s update — routing first",
                blocker,
            )
            ctx = ConversationContext.from_state(state)
            route = self._route_slot_update(
                state, blocker, ctx, return_awaiting=state.get("awaiting_slot") or "fax_confirmed"
            )
            # The routed update consumes the parked action item, if any.
            route["parked_followups"] = [
                p
                for p in normalize_parked_followups(state.get("parked_followups"))
                if not (p.get("kind") == "action" and p.get("target") == blocker)
            ]
            return route

        if fail := await dispatch_provider_list(self, state, delivery_method, confirmed_destination):
            return fail

        logger.info(
            LOG_LIST_DISPATCHED,
            extra={"method": delivery_method, "dest_tail": confirmed_destination[-4:]},
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        provider_type = (state.get("provider_type") or "provider").strip()

        # ── REDO COMPLETION (Phase 6): announce the re-send, never repeat ────
        # the benefits offer — benefits_offer_made stays True from the first
        # dispatch. How control returns depends on who asked for the redo.
        redo_request = normalize_cross_agent_request(state)
        if redo_request.get("kind") in ("redo", "update") and redo_request.get("target") in (
            "delivery",
            "delivery_method",
        ):
            return self._finish_redispatch(
                state, redo_request, delivery_method, confirmed_destination, timestamp
            )

        # When the member updated their ZIP earlier in this call
        # (provider_search sets zip_code_updated=True), the dispatch
        # confirmation explicitly includes the new ZIP so the member hears
        # it was applied to the in-network search.
        zip_used = (state.get("zip_code_used") or state.get("zip_code") or "").strip()
        if state.get("zip_code_updated") and zip_used:
            window_msg = random.choice(DELIVERY_WINDOW_MSG_ZIP_UPDATED).format(zip_code=zip_used)
        else:
            window_msg = pick(DELIVERY_WINDOW_MSG)

        benefits_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
        combined_msg = f"{window_msg} {benefits_msg}"

        offer_result = self.ask_member(state, combined_msg)
        offer_result["awaiting_slot"] = "benefits_response"
        offer_result["provider_list_sent"] = True
        offer_result["benefits_offer_made"] = True
        offer_result["delivery_method"] = delivery_method
        offer_result["delivery_timestamp"] = timestamp
        if delivery_method == "fax":
            offer_result["fax"] = confirmed_destination
        else:
            offer_result["email"] = confirmed_destination
        return offer_result

    def _begin_redispatch(self, state: State, return_awaiting: str) -> dict:
        """Enter the re-dispatch branch (Phase 6 redo, in-flow).

        The provider list was already sent and the caller asked for it again
        by another method/destination while WE were active. Ask for the new
        delivery method; the pending marker (requester = us) keeps the
        completed-flow early exits open until _finish_redispatch closes it.
        return_awaiting: "benefits_response" when the still-unanswered
        benefits offer must be re-asked after the re-send; "" otherwise.
        """
        logger.info("delivery_management: re-dispatch requested — re-collecting delivery method")
        result = self.ask_member(
            state,
            "Of course — I can send that same list again. Would you like it by fax or email?",
        )
        result["awaiting_slot"] = "delivery_method"
        result["pending_cross_agent_request"] = {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": self.AGENT_NAME,
            "return_awaiting": return_awaiting if return_awaiting == "benefits_response" else "",
        }
        return result

    def _finish_redispatch(
        self,
        state: State,
        request: dict,
        delivery_method: str,
        confirmed_destination: str,
        timestamp: str,
    ) -> dict:
        """Close a re-dispatch: announce the re-send and hand control back.

        - Requester is a slot-collecting agent (return_awaiting set, e.g.
          benefits mid care-coach): silent COMPLETE with the request still
          set — the orchestrator return hop restores the awaiting slot and
          arms slot_update_resume; the requester speaks the acknowledgement.
        - Requester is us (in-flow redo over the unanswered benefits offer):
          announce + re-ask the pending benefits offer, clear the request.
        - Requester collects nothing (follow_up): announce here, hand back
          via next_node, clear the request.
        """
        provider_type = (state.get("provider_type") or "provider").strip()
        spoken_dest = (
            speak_email(confirmed_destination) if delivery_method == "email" else confirmed_destination
        )
        announce = (
            f"All set — I've sent that same {provider_type} list to your "
            f"{delivery_method} at {spoken_dest} as well. {pick(DELIVERY_WINDOW_MSG)}"
        )
        common = {
            "provider_list_sent": True,
            "benefits_offer_made": True,  # never re-offered on a redo
            "delivery_method": delivery_method,
            "delivery_timestamp": timestamp,
            ("fax" if delivery_method == "fax" else "email"): confirmed_destination,
        }
        logger.info(
            "delivery_management: re-dispatch complete",
            extra={"method": delivery_method, "return_to": request.get("return_to_agent", "")},
        )

        if request.get("return_to_agent") == self.AGENT_NAME:
            # In-flow: the benefits offer was interrupted by the redo and is
            # still unanswered — re-ask it now (not a repeat: never answered).
            if request.get("return_awaiting") == "benefits_response":
                benefits_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
                result = self.ask_member(state, f"{announce} {benefits_msg}")
                result["awaiting_slot"] = "benefits_response"
            else:
                result = self.ask_member(state, announce)
                result["next_node"] = "follow_up_agent"
                result["awaiting_slot"] = ""
            result.update(common)
            result["pending_cross_agent_request"] = {}
            result["pending_slot_update"] = {}  # legacy key
            return result

        if request.get("return_awaiting"):
            # Routed from a slot-collecting agent — COMPLETE with the request
            # kept; the orchestrator consumes it and the requester announces.
            result = self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(state, delivery_method, False),
            )
            result.update(common)
            return result

        result = self.ask_member(state, announce)
        result["next_node"] = request.get("return_to_agent") or "follow_up_agent"
        result["awaiting_slot"] = ""
        result.update(common)
        result["pending_cross_agent_request"] = {}
        result["pending_slot_update"] = {}  # legacy key
        return result

    def _replay_provider_list(self, state: State, request: dict) -> dict:
        """Replay capability (Phase 6): re-state what was sent, where, and the
        delivery window — answerable purely from state, no re-dispatch."""
        provider_type = (state.get("provider_type") or "provider").strip()
        method = (state.get("delivery_method") or "").strip()
        contact = (state.get("fax") if method == "fax" else state.get("email")) or ""
        spoken_contact = speak_email(contact) if method == "email" else contact
        zip_used = (state.get("zip_code_used") or state.get("zip_code") or "").strip()
        parts = [f"Of course — I sent your {provider_type} provider list"]
        if zip_used:
            parts.append(f"for ZIP code {zip_used}")
        if method and spoken_contact:
            parts.append(f"by {method} to {spoken_contact}")
        elif method:
            parts.append(f"by {method}")
        summary = " ".join(parts) + f". {pick(DELIVERY_WINDOW_MSG)}"
        logger.info(
            "delivery_management: provider_list replay",
            extra={"return_to": request.get("return_to_agent", "")},
        )
        result = self.ask_member(state, summary)
        result["next_node"] = request.get("return_to_agent") or "follow_up_agent"
        result["awaiting_slot"] = request.get("return_awaiting", "")
        result["pending_cross_agent_request"] = {}
        result["pending_slot_update"] = {}  # legacy key
        return result

    def _ask_contact_confirmation(
        self,
        state: State,
        delivery_method: str,
        fax_on_file: str,
        email_on_file: str,
        prefix: str = "",
    ) -> dict:
        """Ask for confirmation of the contact details on file (or collect new ones).

        prefix: optional acknowledgement spoken before the question (e.g. the
        ZIP-updated confirmation on a slot_update_resume hop).
        """
        if delivery_method == "fax":
            if fax_on_file:
                msg = prefix + random.choice(FAX_READBACK_TEMPLATES).format(fax=fax_on_file)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "fax_confirmed"
                result["delivery_method"] = delivery_method
            else:
                result = self.ask_member(state, prefix + pick(FAX_UPDATE_PROMPTS))
                result["awaiting_slot"] = "fax"
                result["delivery_method"] = delivery_method
        elif delivery_method == "email":
            if email_on_file:
                # Spell out the email in words ("at"/"dot") for the spoken
                # message. Also prevents Azure content filter triggering on
                # email addresses in conversation history.
                display_email = speak_email(email_on_file)
                msg = prefix + random.choice(EMAIL_READBACK_TEMPLATES).format(email=display_email)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "email_confirmed"
                result["delivery_method"] = delivery_method
            else:
                result = self.ask_member(state, prefix + pick(EMAIL_UPDATE_PROMPTS))
                result["awaiting_slot"] = "email"
                result["delivery_method"] = delivery_method
        else:
            result = self.signal_escalate(state, pick(MSG_CONTACT_EXHAUST), reason="invalid_delivery_method")
        return result

    @staticmethod
    def _completion_context(state: State, delivery_method: str, proactive: bool) -> dict:
        return {
            "provider_list_sent": True,
            "delivery_method": delivery_method,
            "delivery_timestamp": state.get("delivery_timestamp", ""),
            "fax": state.get("fax", ""),
            "email": state.get("email", ""),
            "benefits_offer_made": True,
            "proactive_offer_available": proactive,
        }


async def delivery_management_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await DeliveryManagementAgent.from_state(state).execute(state)
