"""
agent.py — ClaimAdjustmentAgent

Flow:
  PHASE 0: Re-entry guard — if claim_flow_complete, signal_complete
  PHASE 1: Collect reference_number directly (first-entry fast-path: skip LLM)
           Fallback sub-flow when member cannot provide reference_number:
             Stage "claim_number_ask"  — ask for / collect claim_number
             Stage "dos_billed_ask"    — ask for / collect DOS + billed_amount
  PHASE 2: Salesforce lookup — find_adjustment(reference_number, member_id)
           (skipped when a fallback lookup already returned a record)
           Not found → signal_escalate with MSG_REF_NOT_FOUND
  PHASE 3: Report status and last_update_date to member
  PHASE 4: If records_required → signal_complete to route to records_coordination_agent
           If not records_required → signal_complete to route to notification_setup_agent
"""

from __future__ import annotations

import random
import re

from agent.agents.claim_adjustment.constants import (
    AGENT_NAME,
    LOG_ENTERED,
    LOG_REF_COLLECTED,
    LOG_STATUS_REPORTED,
    MSG_RECORDS_NEEDED,
    MSG_REF_EXHAUST,
    MSG_REF_FALLBACK_CLAIM_NUMBER_ASK,
    MSG_REF_FALLBACK_CLAIM_NUMBER_HAVE_IT,
    MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY,
    MSG_REF_FALLBACK_DOS_BILLED_ASK,
    MSG_REF_FALLBACK_DOS_BILLED_HAVE_IT,
    MSG_REF_FALLBACK_DOS_BILLED_RETRY,
    MSG_REF_FALLBACK_NOT_FOUND,
    MSG_REF_NOT_FOUND,
    REFERENCE_NUMBER_BRIDGE_MSGS,
    STATUS_REPORT_TEMPLATES,
)
from agent.agents.claim_adjustment.handlers import (
    lookup_adjustment,
    lookup_adjustment_by_claim_number,
    lookup_adjustment_by_dos_and_billed,
)
from agent.agents.claim_adjustment.llm import extract_claim_adjustment_decision
from agent.agents.verification.constants import MAX_LOOKUP_ATTEMPTS
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.core.request_detection import reconcile_worker_result
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.responses.static import MSG_WAIT_ACK
from agent.slots.normalizers import (
    normalize_billed_amount,
    normalize_claim_number,
    normalize_date_of_service,
    normalize_reference_number,
)
from agent.slots.validators import (
    validate_billed_amount,
    validate_claim_number,
    validate_date_of_service,
    validate_reference_number,
)
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    detect_cannot_provide,
    detect_wait_request,
    pick,
)

logger = get_logger(__name__)

# Simple "no" check for fallback branching (user says "No" to "do you have the claim number?")
_FALLBACK_NO_PHRASES = frozenset({"no", "nope", "nah", "negative", "i don't", "i dont"})

# Bare affirmatives — member is confirming they HAVE the number, not providing it
_AFFIRMATIVE_PHRASES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "correct",
        "right",
        "i do",
        "i have it",
        "i have one",
        "i have the number",
    }
)

# Spoken digit words used to detect digit content in an utterance
_DIGIT_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "oh",
    }
)


def _is_no_response(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in _FALLBACK_NO_PHRASES


class ClaimAdjustmentAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        # ── CROSS-AGENT RE-ENTRY (Phase 7): claim-status replay aimed at us ───
        # "what's happening with my claim again?" after the flow completed —
        # re-state the adjustment status from state (idempotent read), gated
        # on the status actually existing. Checked BEFORE the completed-flow
        # early exit, exactly like delivery's provider_list replay.
        replay_request = self.consume_cross_agent_request(state, kinds=("replay",), targets=("claim_status",))
        if replay_request and state.get("claim_status"):
            return self._replay_claim_status(state, replay_request)

        # ── PHASE 0: Re-entry guard ────────────────────────────────────────────
        if state.get("claim_flow_complete"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["claim_services"],
                context_updates=self._completion_context(state),
            )

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        raw_awaiting = state.get("awaiting_slot", "")
        reference_number = (state.get("reference_number") or "").strip()

        # ── Defensive: inconsistent fallback state guard ───────────────────────
        # If a fallback sub-flow is active (ref_no_fallback_stage set), the
        # reference_number must be empty so Phase 1 enters the fallback branch.
        # An inconsistency here (both set) causes Phase 2 to reuse a stale ref#
        # instead of routing to the fallback — clear it defensively.
        _cur_fallback_stage = (state.get("ref_no_fallback_stage") or "").strip()
        if _cur_fallback_stage and reference_number:
            logger.warning(
                "claim_adjustment_agent: ref_no_fallback_stage=%r with non-empty "
                "reference_number=%r — clearing to enter fallback correctly",
                _cur_fallback_stage,
                reference_number,
            )
            reference_number = ""

        # ── RESUME after a routed slot update (Phase 7, mirrors delivery) ─────
        # The return hop restored awaiting_slot; re-ask the preserved question
        # — no extraction on the stale turn (the owner consumed it).
        if state.get("slot_update_resume") and raw_awaiting:
            result = self.ask_member(
                state, "All set — that's been updated. " + pick(REFERENCE_NUMBER_BRIDGE_MSGS)
            )
            result["awaiting_slot"] = raw_awaiting
            result["slot_update_resume"] = False
            return result

        # ── PHASE 1 FAST PATH: first entry — ask without LLM ─────────────────
        if not raw_awaiting and not reference_number:
            interrupt = self.ask_member(state, pick(REFERENCE_NUMBER_BRIDGE_MSGS))
            interrupt["awaiting_slot"] = "reference_number"
            return interrupt

        # ── PHASE 1: Collect reference_number directly (no pipeline) ──────────
        adjustment_record_from_fallback: dict | None = None
        if not reference_number:
            current_awaiting = raw_awaiting or "reference_number"
            state = {**state, "awaiting_slot": current_awaiting}

            ref_no_fallback_stage = (state.get("ref_no_fallback_stage") or "").strip()

            # ── Guard: stale fallback stage — awaiting_slot says reference_number
            # but ref_no_fallback_stage still holds a leftover value from a prior
            # member's flow that was not cleared (e.g. NEW_INTENT_CLEAR_FIELDS
            # gap).  Reset the stage so we collect the reference number normally.
            if ref_no_fallback_stage and current_awaiting == "reference_number":
                logger.info(
                    "claim_adjustment_agent: stale ref_no_fallback_stage=%r cleared "
                    "(awaiting_slot=reference_number)",
                    ref_no_fallback_stage,
                )
                ref_no_fallback_stage = ""
                state = {**state, "ref_no_fallback_stage": ""}

            # ── Fallback sub-flow: member cannot provide reference number ─────
            if ref_no_fallback_stage == "claim_number_ask":
                fallback_record, interrupt = await self._collect_claim_number_fallback(
                    state, messages, last_user, last_agent
                )
                if interrupt is not None:
                    return interrupt
                # claim_number lookup succeeded — bridge into Phase 2/3
                adjustment_record_from_fallback = fallback_record
                reference_number = (fallback_record or {}).get("reference_number", "ref_fallback")
                state = {**state, "reference_number": reference_number, "ref_no_fallback_stage": ""}
            elif ref_no_fallback_stage == "dos_billed_ask":
                fallback_record, interrupt = await self._collect_dos_billed_fallback(
                    state, messages, last_user, last_agent
                )
                if interrupt is not None:
                    return interrupt
                # dos+billed lookup succeeded — bridge into Phase 2/3
                adjustment_record_from_fallback = fallback_record
                reference_number = (fallback_record or {}).get("reference_number", "ref_fallback")
                state = {**state, "reference_number": reference_number, "ref_no_fallback_stage": ""}
            else:
                # ── Cannot-provide check BEFORE any LLM call ─────────────────
                # "I don't have it" → start the claim_number fallback instead of
                # escalating immediately.
                if detect_cannot_provide(last_user):
                    logger.info(
                        "claim_adjustment_agent: cannot-provide for reference_number "
                        "— starting fallback sub-flow"
                    )
                    result = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_ASK))
                    result["ref_no_fallback_stage"] = "claim_number_ask"
                    result["awaiting_slot"] = "fallback_claim_number"
                    return result

                # Member switched to offering a claim number — start that fallback
                _user_lower_ref = (last_user or "").lower()
                if "claim number" in _user_lower_ref or "claim #" in _user_lower_ref:
                    logger.info(
                        "claim_adjustment_agent: claim_number offered during reference_number "
                        "collection — starting claim_number fallback"
                    )
                    result = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY))
                    result["ref_no_fallback_stage"] = "claim_number_ask"
                    result["awaiting_slot"] = "fallback_claim_number"
                    return result

                attempts_dict = state.get("slot_attempts") or {}
                current_attempt = attempts_dict.get(current_awaiting, {})
                attempt_count = (
                    current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0
                )

                result = await extract_claim_adjustment_decision(
                    get_extraction_llm(),
                    build_extraction_prompt_extraction("extraction/claim_adjustment.md"),
                    awaiting_slot=current_awaiting,
                    last_agent_message=last_agent,
                    last_user_message=last_user,
                    confirmed_slots={},
                    pending_slots=["reference_number"],
                    attempt=attempt_count,
                    recent_messages=messages[-6:],
                )

                if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
                    self.slot_fail("reference_number")
                    interrupt["slot_attempts"] = self.slots_dict()
                    if self.get_slot("reference_number").is_exhausted():
                        return self.signal_escalate(
                            state,
                            pick(MSG_REF_EXHAUST),
                            reason="reference_number_exhausted",
                        )
                    return interrupt

                result = reconcile_worker_result(result, last_user)

                update_target = (getattr(result, "update_target", None) or "").strip()
                if update_target:
                    if route := self._route_foreign_update(
                        state, update_target, return_awaiting=current_awaiting
                    ):
                        return route

                # LLM-driven pivot: mid-sentence correction or hesitation to a different identifier
                _pivot_ref = (getattr(result, "fallback_pivot", None) or "").strip()
                if _pivot_ref == "claim_number":
                    logger.info("claim_adjustment_agent: LLM pivot → claim_number during ref# collection")
                    _r = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY))
                    _r["ref_no_fallback_stage"] = "claim_number_ask"
                    _r["awaiting_slot"] = "fallback_claim_number"
                    return _r
                if _pivot_ref == "dos_billed":
                    logger.info("claim_adjustment_agent: LLM pivot → dos_billed during ref# collection")
                    _r = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
                    _r["ref_no_fallback_stage"] = "dos_billed_ask"
                    _r["awaiting_slot"] = "fallback_dos_billed"
                    return _r

                extracted_raw = (result.extracted or {}).get("reference_number", "") if result else ""
                normalized = normalize_reference_number(extracted_raw) if extracted_raw else ""

                if normalized and validate_reference_number(normalized).valid:
                    reference_number = normalized
                    self.slot_ok("reference_number", reference_number)
                    logger.info(LOG_REF_COLLECTED)
                    state = {**state, "reference_number": reference_number}
                else:
                    if handled := self._reroute_detected_update(state, return_awaiting=current_awaiting):
                        return handled
                    self.slot_fail("reference_number")
                    if self.get_slot("reference_number").is_exhausted():
                        return self.signal_escalate(
                            state,
                            pick(MSG_REF_EXHAUST),
                            reason="reference_number_exhausted",
                        )
                    msg = await self._generate_slot_retry_response(
                        state,
                        "reference_number",
                        ConversationContext.from_state(state),
                        messages,
                        extracted_this_turn=normalized if normalized else extracted_raw,
                        guard="RETRY",
                    )
                    retry = self.ask_member(state, msg)
                    retry["awaiting_slot"] = "reference_number"
                    return retry

        # ── PHASE 2: Salesforce lookup ─────────────────────────────────────────
        # Use the record already found by the fallback sub-flow (claim_number or
        # DOS+billed lookup) when available, skipping a redundant SF call.
        adjustment_record = adjustment_record_from_fallback
        if not state.get("claim_status") and not adjustment_record:
            adjustment_record, interrupt = await lookup_adjustment(self, state)
            if interrupt:
                return interrupt
            if not adjustment_record:
                self.slot_fail("ref_lookup_fail", reference_number)
                count = self.get_slot("ref_lookup_fail").attempt_count
                if count >= MAX_LOOKUP_ATTEMPTS:
                    return self.signal_escalate(
                        state,
                        pick(MSG_REF_NOT_FOUND),
                        "adjustment_reference_not_found",
                    )
                # Under the limit — start the claim_number fallback sub-flow.
                result = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_ASK))
                result["reference_number"] = ""  # clear so PHASE 1 enters fallback
                result["ref_no_fallback_stage"] = "claim_number_ask"
                result["awaiting_slot"] = "fallback_claim_number"
                return result

        # ── PHASE 3: Report status ─────────────────────────────────────────────
        if not state.get("claim_status") and adjustment_record:
            claim_status = adjustment_record.get("claim_status", "open for Review from our adjustment team")
            last_update_date = adjustment_record.get("claim_update_date", "")
            records_required = bool(adjustment_record.get("records_required", True))

            logger.info(LOG_STATUS_REPORTED)

            status_msg = random.choice(STATUS_REPORT_TEMPLATES).format(
                status=claim_status,
                last_update_date=last_update_date,
            )

            if records_required:
                records_msg = pick(MSG_RECORDS_NEEDED)
                full_msg = f"{status_msg}\n{records_msg}"
                result = self.ask_member(state, full_msg)
                result["reference_number"] = reference_number
                result["claim_status"] = claim_status
                result["last_update_date"] = last_update_date
                result["records_required"] = True
                result["next_node"] = "records_coordination_agent"
                result["awaiting_slot"] = ""
                result["ref_no_fallback_stage"] = ""
                result["fallback_claim_number"] = ""
                result["fallback_dos"] = ""
                result["fallback_billed_amount"] = ""
                return result
            else:
                result = self.ask_member(state, status_msg)
                result["reference_number"] = reference_number
                result["claim_status"] = claim_status
                result["last_update_date"] = last_update_date
                result["records_required"] = False
                result["next_node"] = "notification_setup_agent"
                result["awaiting_slot"] = ""
                result["ref_no_fallback_stage"] = ""
                result["fallback_claim_number"] = ""
                result["fallback_dos"] = ""
                result["fallback_billed_amount"] = ""
                return result

        # ── PHASE 4: signal_complete — both sub-agents already ran ─────────────
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["claim_services"],
            context_updates=self._completion_context(state),
        )

    def _replay_claim_status(self, state: State, request: dict) -> dict:
        """Replay capability (Phase 7): re-state the adjustment status from
        state — an idempotent read, exactly like delivery's
        _replay_provider_list. No lookup, no flow re-entry."""
        status = (state.get("claim_status") or "").strip()
        last_update = (state.get("last_update_date") or "").strip()
        ref = (state.get("reference_number") or "").strip()
        parts = ["Of course — your claim adjustment"]
        if ref:
            parts.append(f"(reference {ref})")
        parts.append(f"is currently {status}")
        if last_update:
            parts.append(f"as of {last_update}")
        summary = (
            " ".join(parts) + ". The resolution timeline is 5 to 10 business days "
            "from receipt of the required information."
        )
        logger.info(
            "claim_adjustment_agent: claim_status replay",
            extra={"return_to": request.get("return_to_agent", "")},
        )
        result = self.ask_member(state, summary)
        result["next_node"] = request.get("return_to_agent") or "follow_up_agent"
        result["awaiting_slot"] = request.get("return_awaiting", "")
        result["pending_cross_agent_request"] = {}
        result["pending_slot_update"] = {}  # legacy key
        return result

    # ── Fallback sub-flow helpers ─────────────────────────────────────────────

    async def _collect_claim_number_fallback(
        self,
        state: State,
        messages: list,
        last_user: str,
        last_agent: str,
    ) -> tuple[dict | None, dict | None]:
        """Collect claim_number from the member.

        Returns (adjustment_record, None) when lookup succeeds.
        Returns (None, interrupt_dict) when still collecting or escalating.
        """
        _user_lower_cn = (last_user or "").lower()

        # Bare negation only — unambiguous single-word denials with no qualifying clause.
        # detect_cannot_provide is intentionally NOT here: it matches multi-clause sentences
        # like "No. I don't have the claim number, but I have the reference number." where
        # the LLM must see the full context to detect the pivot intent.
        if _is_no_response(last_user):
            logger.info("claim_adjustment_agent: no claim_number — moving to dos_billed fallback")
            result = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
            result["ref_no_fallback_stage"] = "dos_billed_ask"
            result["awaiting_slot"] = "fallback_dos_billed"
            return None, result

        # If user is asking for more time — acknowledge and stay on this stage
        if detect_wait_request(last_user):
            logger.info("claim_adjustment_agent: WAIT detected during claim_number fallback")
            wait_result = self.ask_member(state, pick(MSG_WAIT_ACK))
            wait_result["ref_no_fallback_stage"] = "claim_number_ask"
            wait_result["awaiting_slot"] = "fallback_claim_number"
            return None, wait_result

        # If user gave a bare affirmative ("Yes", "Sure", "Yep, I do") with no digits, they are
        # confirming they HAVE the claim number — re-ask without burning a retry.
        _user_stripped = (last_user or "").strip().lower().rstrip(".!?,")
        _has_digits = bool(
            re.search(r"\d", last_user or "") or _DIGIT_WORDS.intersection(_user_stripped.split())
        )
        _first_word_cn = re.sub(r"[^\w]", "", _user_stripped.split()[0]) if _user_stripped.split() else ""
        _is_affirmative_cn = _user_stripped in _AFFIRMATIVE_PHRASES or _first_word_cn in {
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
        }
        if not _has_digits and _is_affirmative_cn:
            logger.info("claim_adjustment_agent: affirmative-only response — re-asking for claim_number")
            reask = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_HAVE_IT))
            reask["ref_no_fallback_stage"] = "claim_number_ask"
            reask["awaiting_slot"] = "fallback_claim_number"
            return None, reask

        # Member offered DOS+billed info instead of claim number — forward to that stage
        if not _has_digits and (
            "date of service" in _user_lower_cn
            or "service date" in _user_lower_cn
            or "billed amount" in _user_lower_cn
            or "billing amount" in _user_lower_cn
        ):
            logger.info(
                "claim_adjustment_agent: dos/billed info offered during claim_number fallback "
                "— advancing to dos_billed_ask"
            )
            result = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
            result["ref_no_fallback_stage"] = "dos_billed_ask"
            result["awaiting_slot"] = "fallback_dos_billed"
            return None, result

        # Member says "claim number" without digits — they're announcing they have it.
        # Skip this gate when the utterance also mentions reference number or signals
        # cannot-provide: those need the LLM to detect the qualifying pivot clause.
        _mentions_ref = "reference number" in _user_lower_cn or "ref number" in _user_lower_cn
        if not _has_digits and ("claim number" in _user_lower_cn or "claim #" in _user_lower_cn):
            if not _mentions_ref and not detect_cannot_provide(last_user):
                logger.info(
                    "claim_adjustment_agent: "
                    "'claim number' mention without digits — re-asking for claim_number"
                )
                reask = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_HAVE_IT))
                reask["ref_no_fallback_stage"] = "claim_number_ask"
                reask["awaiting_slot"] = "fallback_claim_number"
                return None, reask

        # Try to extract claim_number via LLM.
        # Use awaiting_slot="claim_number" (not "fallback_claim_number") so the
        # extraction prompt context matches a field the model knows.
        extraction = await extract_claim_adjustment_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/claim_adjustment.md"),
            awaiting_slot="claim_number",
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            pending_slots=["claim_number"],
            attempt=0,
            recent_messages=messages[-6:],
        )
        extraction = reconcile_worker_result(extraction, last_user)

        # LLM-based WAIT check: detect_wait_request can miss wait phrases that are
        # followed by meta-commentary ("I need to look this up") because the
        # continuation guard fires.  Honor the LLM's own WAIT label as a fallback.
        _evt_cn = getattr(extraction, "event_type", None) if extraction else None
        if str(getattr(_evt_cn, "value", _evt_cn) or "").strip().lower() == "wait":
            logger.info("claim_adjustment_agent: LLM WAIT detected during claim_number fallback")
            wait_result = self.ask_member(state, pick(MSG_WAIT_ACK))
            wait_result["ref_no_fallback_stage"] = "claim_number_ask"
            wait_result["awaiting_slot"] = "fallback_claim_number"
            return None, wait_result

        # LLM-driven pivot: handles mid-sentence corrections and hesitations keywords miss
        _pivot_cn = (getattr(extraction, "fallback_pivot", None) or "").strip()
        if _pivot_cn == "reference_number":
            logger.info("claim_adjustment_agent: LLM pivot → reference_number during claim_number fallback")
            # Do NOT reset ref_lookup_fail here — Phase 2 owns that logic.
            self.get_slot(
                "reference_number"
            ).reset()  # un-confirm stale value so ask_member doesn't re-inject it
            _r = self.ask_member(state, pick(REFERENCE_NUMBER_BRIDGE_MSGS))
            _r["ref_no_fallback_stage"] = ""
            _r["reference_number"] = ""
            _r["awaiting_slot"] = "reference_number"
            return None, _r
        if _pivot_cn == "dos_billed":
            logger.info("claim_adjustment_agent: LLM pivot → dos_billed during claim_number fallback")
            _r = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
            _r["ref_no_fallback_stage"] = "dos_billed_ask"
            _r["awaiting_slot"] = "fallback_dos_billed"
            return None, _r

        raw_cn = (extraction.extracted or {}).get("claim_number", "") if extraction else ""
        normalized_cn = normalize_claim_number(raw_cn) if raw_cn else ""

        if normalized_cn and validate_claim_number(normalized_cn).valid:
            logger.info("claim_adjustment_agent: fallback claim_number collected → SF lookup")
            record, interrupt = await lookup_adjustment_by_claim_number(self, state, normalized_cn)
            if interrupt is not None:
                return None, interrupt
            if record is not None:
                state_update = {**state, "fallback_claim_number": normalized_cn}
                _ = state_update  # state updates propagated via returned record path
                return record, None
            # Not found — escalate
            logger.info("claim_adjustment_agent: fallback claim_number not found in SF")
            return None, self.signal_escalate(
                state,
                pick(MSG_REF_FALLBACK_NOT_FOUND),
                reason="fallback_claim_number_not_found",
            )

        # LLM found no pivot and no claim number. Now safe to apply cannot-provide:
        # multi-clause denials ("I don't have it") route to DOS/billed only after the
        # LLM confirmed there is no qualifying pivot hint in the same utterance.
        if detect_cannot_provide(last_user):
            logger.info("claim_adjustment_agent: cannot-provide (post-LLM) → dos_billed fallback")
            result = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
            result["ref_no_fallback_stage"] = "dos_billed_ask"
            result["awaiting_slot"] = "fallback_dos_billed"
            return None, result

        # Could not extract — retry once then move to dos_billed
        attempts_dict = state.get("slot_attempts") or {}
        cn_attempts = attempts_dict.get("fallback_claim_number") or {}
        attempt_count = cn_attempts.get("attempt_count", 0) if isinstance(cn_attempts, dict) else 0

        if attempt_count >= 1:
            # One retry already used — move to DOS+billed
            logger.info("claim_adjustment_agent: claim_number retry exhausted → dos_billed fallback")
            result = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_ASK))
            result["ref_no_fallback_stage"] = "dos_billed_ask"
            result["awaiting_slot"] = "fallback_dos_billed"
            return None, result

        retry = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY))
        retry["ref_no_fallback_stage"] = "claim_number_ask"
        retry["awaiting_slot"] = "fallback_claim_number"
        # Manually bump attempt count for this slot
        updated_attempts = {**attempts_dict, "fallback_claim_number": {"attempt_count": attempt_count + 1}}
        retry["slot_attempts"] = updated_attempts
        return None, retry

    async def _collect_dos_billed_fallback(
        self,
        state: State,
        messages: list,
        last_user: str,
        last_agent: str,
    ) -> tuple[dict | None, dict | None]:
        """Collect date_of_service + billed_amount from the member.

        Returns (adjustment_record, None) when lookup succeeds.
        Returns (None, interrupt_dict) when still collecting or escalating.
        """
        # If user is asking for more time — acknowledge and stay on this stage
        if detect_wait_request(last_user):
            logger.info("claim_adjustment_agent: WAIT detected during dos_billed fallback")
            wait_result = self.ask_member(state, pick(MSG_WAIT_ACK))
            wait_result["ref_no_fallback_stage"] = "dos_billed_ask"
            wait_result["awaiting_slot"] = "fallback_dos_billed"
            return None, wait_result

        # Bare affirmative with no date/amount content — re-ask without burning a retry.
        _user_stripped_db = (last_user or "").strip().lower().rstrip(".!?,")
        _has_value_content = bool(
            re.search(r"\d", last_user or "") or _DIGIT_WORDS.intersection(_user_stripped_db.split())
        )
        _first_word_db = (
            re.sub(r"[^\w]", "", _user_stripped_db.split()[0]) if _user_stripped_db.split() else ""
        )
        _is_affirmative_db = _user_stripped_db in _AFFIRMATIVE_PHRASES or _first_word_db in {
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
        }
        if not _has_value_content and _is_affirmative_db:
            logger.info("claim_adjustment_agent: affirmative-only response — re-asking for dos+billed")
            reask = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_HAVE_IT))
            reask["ref_no_fallback_stage"] = "dos_billed_ask"
            reask["awaiting_slot"] = "fallback_dos_billed"
            return None, reask

        # Member now has the reference number — pivot back to collecting it
        _user_lower_db = (last_user or "").lower()
        if "reference number" in _user_lower_db or "ref number" in _user_lower_db:
            logger.info(
                "claim_adjustment_agent: reference_number offered during dos_billed fallback "
                "— resetting collection"
            )
            # Do NOT reset ref_lookup_fail here — Phase 2 owns that logic.
            self.get_slot(
                "reference_number"
            ).reset()  # un-confirm stale value so ask_member doesn't re-inject it
            result = self.ask_member(state, pick(REFERENCE_NUMBER_BRIDGE_MSGS))
            result["ref_no_fallback_stage"] = ""
            result["reference_number"] = ""
            result["awaiting_slot"] = "reference_number"
            return None, result

        # Member now has the claim number — pivot back to claim_number collection
        if not _has_value_content and ("claim number" in _user_lower_db or "claim #" in _user_lower_db):
            logger.info(
                "claim_adjustment_agent: claim_number offered during dos_billed fallback "
                "— resetting to claim_number_ask"
            )
            # Member is announcing they have it, not retrying a failed attempt.
            result = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_HAVE_IT))
            result["ref_no_fallback_stage"] = "claim_number_ask"
            result["awaiting_slot"] = "fallback_claim_number"
            return None, result

        # Try to extract dos and billed_amount via LLM.
        # Use awaiting_slot="dos" (not "fallback_dos_billed") so the extraction
        # prompt context matches known fields.
        extraction = await extract_claim_adjustment_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/claim_adjustment.md"),
            awaiting_slot="dos",
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            pending_slots=["dos", "billed_amount", "claim_number"],
            attempt=0,
            recent_messages=messages[-6:],
        )
        extraction = reconcile_worker_result(extraction, last_user)

        # LLM-based WAIT check (mirrors claim_number fallback above).
        _evt_db = getattr(extraction, "event_type", None) if extraction else None
        if str(getattr(_evt_db, "value", _evt_db) or "").strip().lower() == "wait":
            logger.info("claim_adjustment_agent: LLM WAIT detected during dos_billed fallback")
            wait_result = self.ask_member(state, pick(MSG_WAIT_ACK))
            wait_result["ref_no_fallback_stage"] = "dos_billed_ask"
            wait_result["awaiting_slot"] = "fallback_dos_billed"
            return None, wait_result

        # LLM-driven pivot: handles mid-sentence corrections and hesitations keywords miss
        _pivot_db = (getattr(extraction, "fallback_pivot", None) or "").strip()
        if _pivot_db == "reference_number":
            logger.info("claim_adjustment_agent: LLM pivot → reference_number during dos_billed fallback")
            # Do NOT reset ref_lookup_fail here — Phase 2 owns that logic.
            self.get_slot(
                "reference_number"
            ).reset()  # un-confirm stale value so ask_member doesn't re-inject it
            _r = self.ask_member(state, pick(REFERENCE_NUMBER_BRIDGE_MSGS))
            _r["ref_no_fallback_stage"] = ""
            _r["reference_number"] = ""
            _r["awaiting_slot"] = "reference_number"
            return None, _r
        if _pivot_db == "claim_number":
            logger.info("claim_adjustment_agent: LLM pivot → claim_number during dos_billed fallback")
            _r = self.ask_member(state, pick(MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY))
            _r["ref_no_fallback_stage"] = "claim_number_ask"
            _r["awaiting_slot"] = "fallback_claim_number"
            return None, _r

        extracted = (extraction.extracted or {}) if extraction else {}

        # Member offered a claim number instead of DOS+billed — rescue: switch
        # back to the claim_number stage and try the claim_number lookup.
        raw_cn_rescue = extracted.get("claim_number", "")
        normalized_cn_rescue = normalize_claim_number(raw_cn_rescue) if raw_cn_rescue else ""
        if normalized_cn_rescue and validate_claim_number(normalized_cn_rescue).valid:
            logger.info("claim_adjustment_agent: claim_number offered during dos_billed stage — rescuing")
            record, interrupt = await lookup_adjustment_by_claim_number(self, state, normalized_cn_rescue)
            if interrupt is not None:
                return None, interrupt
            if record is not None:
                return record, None
            return None, self.signal_escalate(
                state,
                pick(MSG_REF_FALLBACK_NOT_FOUND),
                reason="fallback_claim_number_rescue_not_found",
            )

        # Merge with any already-collected dos/billed from state
        raw_dos = extracted.get("dos", "") or (state.get("fallback_dos") or "")
        raw_billed = extracted.get("billed_amount", "") or (state.get("fallback_billed_amount") or "")

        normalized_dos = normalize_date_of_service(raw_dos) if raw_dos else ""
        normalized_billed = normalize_billed_amount(raw_billed) if raw_billed else ""

        dos_valid = bool(normalized_dos and validate_date_of_service(normalized_dos).valid)
        billed_valid = bool(normalized_billed and validate_billed_amount(normalized_billed).valid)

        if dos_valid and billed_valid:
            logger.info("claim_adjustment_agent: fallback dos+billed collected → SF lookup")
            record, interrupt = await lookup_adjustment_by_dos_and_billed(
                self, state, normalized_dos, normalized_billed
            )
            if interrupt is not None:
                return None, interrupt
            if record is not None:
                return record, None
            # Not found — escalate
            logger.info("claim_adjustment_agent: fallback dos+billed not found in SF")
            return None, self.signal_escalate(
                state,
                pick(MSG_REF_FALLBACK_NOT_FOUND),
                reason="fallback_dos_billed_not_found",
            )

        # Missing one or both values — retry once then escalate
        attempts_dict = state.get("slot_attempts") or {}
        db_attempts = attempts_dict.get("fallback_dos_billed") or {}
        attempt_count = db_attempts.get("attempt_count", 0) if isinstance(db_attempts, dict) else 0

        if attempt_count >= 1:
            return None, self.signal_escalate(
                state,
                pick(MSG_REF_FALLBACK_NOT_FOUND),
                reason="fallback_dos_billed_exhausted",
            )

        retry = self.ask_member(state, pick(MSG_REF_FALLBACK_DOS_BILLED_RETRY))
        retry["ref_no_fallback_stage"] = "dos_billed_ask"
        retry["awaiting_slot"] = "fallback_dos_billed"
        # Persist any partial values already collected
        if normalized_dos and dos_valid:
            retry["fallback_dos"] = normalized_dos
        if normalized_billed and billed_valid:
            retry["fallback_billed_amount"] = normalized_billed
        updated_attempts = {**attempts_dict, "fallback_dos_billed": {"attempt_count": attempt_count + 1}}
        retry["slot_attempts"] = updated_attempts
        return None, retry

    @staticmethod
    def _completion_context(state: State) -> dict:
        return {
            "claim_flow_complete": True,
            "reference_number": state.get("reference_number", ""),
            "claim_status": state.get("claim_status", ""),
            "last_update_date": state.get("last_update_date", ""),
            "records_required": state.get("records_required", False),
            "records_branch_taken": state.get("records_branch_taken", ""),
            "upload_link_sent": state.get("upload_link_sent", False),
            "personal_guide_outreach_requested": state.get("personal_guide_outreach_requested", False),
            "notification_channel": state.get("notification_channel", "not_set"),
            "claim_notification_contact": state.get("claim_notification_contact", ""),
        }


async def claim_adjustment_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await ClaimAdjustmentAgent.from_state(state).execute(state)
