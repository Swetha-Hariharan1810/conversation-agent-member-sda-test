"""
verification_agent.py — Identity verification orchestrator.

run() is the only logic here. Everything else is delegated:
  pipelines.py                         — slot collection configuration (what to collect)
  handlers.py                          — SF lookup, post-lookup, corrections, off-topic redirect
  llm.py                               — LLM extraction
  constants.py                         — slot ordering, keywords, log names
  src/agent/responses/message_builders — verification prompt builders
"""

from __future__ import annotations

import random

from agent.agents.intake.constants import INTENT_BRIDGE_MSGS
from agent.agents.verification.constants import (
    IDENTITY_SLOT_ORDER,
    LOG_ENTERED,
    LOG_NAME_CONFIRM_EXHAUST,
    LOG_NAME_CONFIRMED,
    LOG_NAME_CORRECTED,
    LOG_NAME_READBACK,
    LOG_VERIFIED,
    MAX_NAME_CONFIRM_ATTEMPTS,
    MAX_SSN_ATTEMPTS,
    MEMBER_ID_DENIAL_PHRASES,
    MEMBER_ID_PIVOT_PHRASES,
    MSG_NAME_CONFIRM_EXHAUST,
    MSG_REASK_DOB,
    MSG_REASK_FIRST_NAME,
    MSG_REASK_GENERIC,
    MSG_REASK_LAST_NAME,
    MSG_SSN_ASK,
    MSG_SSN_ASK_EXHAUSTED,
    MSG_SSN_BACK_TO_MID,
    MSG_SSN_COLLECT,
    MSG_SSN_DOB,
    MSG_SSN_EITHER,
    MSG_SSN_ESCALATE,
    MSG_SSN_INVALID,
    MSG_SSN_NEED_FIELD,
    MSG_SSN_RETRY_EXHAUSTED,
    MSG_SSN_SUCCESS,
    NAME_CORRECTION_PROMPTS,
    NAME_READBACK_TEMPLATES,
    NO_PHRASES,
    NO_SSN_AVAILABLE_PHRASES,
    YES_PHRASES,
)
from agent.agents.verification.handlers import (
    _NORMALIZERS,
    _VALIDATORS,
    apply_corrections,
    collect_post_lookup,
    lookup_and_verify,
    redirect_off_topic,
)
from agent.agents.verification.llm import (
    extract_name_confirmation,
    extract_ssn_decision,
    extract_verification_decision,
)
from agent.agents.verification.pipelines import (
    build_claims_pipeline,
    build_identity_pipeline,
    build_provider_pipeline,
)
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.core.request_detection import reconcile_worker_result
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.slots.normalizers import normalize_member_id, normalize_name, normalize_ssn
from agent.slots.validators import validate_member_id, validate_name, validate_ssn
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt,
    build_extraction_prompt_extraction,
    detect_cannot_provide,
    pick,
)

logger = get_logger(__name__)

_NAME_CONFIRM_SLOT = "name_confirmed"
_NAME_CORRECTION_SLOT = "name_correction"

# Mid-call intent switch → domain node. When follow_up stages a new intent via
# reset_for_new_intent, verification consumes pending_intent on success and
# dispatches straight to that intent's node. Keys are the intent-tag vocabulary
# (call_intent / detected_intent), values are the registered graph node names.
_PENDING_INTENT_NODE = {
    "provider_services": AgentNode.PROVIDER_SEARCH.value,
    "claim_services": AgentNode.CLAIM_ADJUSTMENT.value,
}


def _spell_name(first: str, last: str) -> str:
    """
    Emily Carter -> "Emily Carter, E-M-I-L-Y C-A-R-T-E-R"
    """

    def _spell(word: str) -> str:
        return "-".join(ch.upper() for ch in word if ch.isalpha() or ch == "-")

    full_name = " ".join(part for part in [first, last] if part).strip()
    combined_name = "".join(part for part in [first, last] if part)
    spelled_name = _spell(combined_name)

    return f"{full_name}. That's spelled {spelled_name}"


def _build_name_readback_message(first: str, last: str) -> str:
    """Pick a random readback template and fill in the spelled name."""
    spelled = _spell_name(first, last)
    return random.choice(NAME_READBACK_TEMPLATES).format(spelled=spelled)


_SSN_FIELD_LABELS = {
    "first_name": "first name",
    "last_name": "last name",
    "ssn": "Social Security Number",
    "dob": "date of birth",
}


class VerificationAgent(BaseAgent):
    AGENT_NAME = "verification_agent"

    def __init__(self) -> None:
        super().__init__()
        self._identity_pipeline = build_identity_pipeline(self)
        self._claims_pipeline = build_claims_pipeline(self)
        self._provider_pipeline = build_provider_pipeline(self)

    # ── SSN fallback helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_ssn_from_text(text: str) -> str:
        """Return normalized SSN if a 9-digit pattern is found in text, else ''."""
        import re

        m = re.search(r"\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b", text)
        if m:
            digits = m.group(1) + m.group(2) + m.group(3)
            candidate = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
            result = validate_ssn(candidate)
            return candidate if result.valid else ""
        return ""

    @staticmethod
    def _is_yes(text: str) -> bool:
        t = text.strip().lower()
        return t in YES_PHRASES or any(p in t for p in YES_PHRASES)

    @staticmethod
    def _is_no(text: str) -> bool:
        return text.strip().lower() in NO_PHRASES

    @staticmethod
    def _is_no_ssn_available(text: str) -> bool:
        t = text.lower()
        return any(phrase in t for phrase in NO_SSN_AVAILABLE_PHRASES)

    @staticmethod
    def _is_member_id_denial(text: str) -> bool:
        t = text.lower()
        return any(phrase in t for phrase in MEMBER_ID_DENIAL_PHRASES)

    @staticmethod
    def _pivots_to_member_id(extraction: object | None, text: str) -> bool:
        """Caller wants their Member ID instead of the SSN.

        The extraction model decides (SsnIntent.HAS_MEMBER_ID, or the shared
        fallback_pivot field); the phrase list is only the backstop for when it
        misses or the extraction call failed outright.
        """
        intent = getattr(getattr(extraction, "ssn_intent", None), "value", None)
        if intent == "has_member_id":
            return True
        if (getattr(extraction, "fallback_pivot", None) or "").strip() == "member_id":
            return True
        return VerificationAgent._wants_member_id(text)

    @staticmethod
    def _wants_member_id(text: str) -> bool:
        """Caller is pivoting back to their Member ID mid-SSN-fallback.

        Deterministic backstop for SsnIntent.HAS_MEMBER_ID: the SSN stages are
        a narrow yes/no gate, and without this a "I have the member id now"
        lands in ambiguous and re-asks the SSN question forever.
        """
        t = (text or "").lower()
        if not any(phrase in t for phrase in MEMBER_ID_PIVOT_PHRASES):
            return False
        # "I don't have my member ID" contains "have my member id" — a denial
        # is never a pivot, so let the denial phrasing win.
        return not detect_cannot_provide(t)

    @staticmethod
    def _extract_member_id_from_text(text: str) -> str:
        """Return a validated Member ID found inline in free text, else "".

        Only the written form (M907503, m 907 503) — the spoken form is left to
        the normal member_id collection path, which already handles digit words.
        """
        import re

        m = re.search(r"\b[mn][\s.-]*(?:\d[\s.-]*){6}", text or "", re.IGNORECASE)
        if not m:
            return ""
        candidate = normalize_member_id(m.group(0))
        return candidate if candidate and validate_member_id(candidate).valid else ""

    def _resume_member_id_collection(self, state: State, last_user: str) -> dict:
        """Leave the SSN fallback and go back to collecting the Member ID.

        Clearing ssn_fallback_stage drops the next turn back into the normal
        verification flow. When the caller already said the ID, take it now
        rather than making them repeat it.
        """
        inline = self._extract_member_id_from_text(last_user)
        if inline:
            logger.info("VerificationAgent SSN fallback: member_id given inline on pivot — resuming")
            self.slot_ok("member_id", inline)
            resumed = self.ask_member(state, "")
            resumed["member_id"] = inline
            resumed["ssn_fallback_stage"] = ""
            resumed["is_interrupt"] = False
            return resumed

        logger.info("VerificationAgent SSN fallback: caller pivoted back to member_id — re-asking")
        result = self.ask_member(state, pick(MSG_SSN_BACK_TO_MID))
        result["ssn_fallback_stage"] = ""
        result["awaiting_slot"] = "member_id"
        return result

    async def _handle_ssn_fallback(self, state: State, last_user: str, messages: list) -> dict:
        """Route SSN fallback sub-stages. LLM-backed for ask/collecting/dob; deterministic for retry."""
        stage = state.get("ssn_fallback_stage") or ""

        if stage == "ssn_ask":
            return await self._ssn_ask_stage(state, last_user, messages)
        if stage == "ssn_collecting":
            return await self._ssn_collecting_stage(state, last_user, messages)
        if stage == "ssn_dob_collecting":
            return await self._ssn_dob_collecting_stage(state, last_user, messages)
        if stage == "ssn_or_mid_retry":
            return await self._ssn_or_mid_retry_stage(state, last_user, messages)
        if stage == "ssn_recheck":
            return await self._ssn_recheck_stage(state, last_user, messages)

        # Unknown stage — escalate defensively
        return self.signal_escalate(state, MSG_SSN_ESCALATE, reason="SSN fallback unknown stage")

    async def _ssn_ask_stage(self, state: State, last_user: str, messages: list) -> dict:
        """User was asked 'Do you have the SSN?' — use LLM to extract intent + optional SSN."""
        from agent.llm.config import get_extraction_llm

        last_agent = _last_assistant_msg(messages)
        extraction = await extract_ssn_decision(
            get_extraction_llm(),
            stage="ssn_ask",
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-4:],
        )

        intent = extraction.ssn_intent.value if extraction else "ambiguous"
        ssn_raw = (extraction.ssn or "").strip() if extraction else ""

        if intent == "yes_with_ssn" and ssn_raw:
            normalized = normalize_ssn(ssn_raw)
            if normalized and validate_ssn(normalized).valid:
                logger.info("VerificationAgent SSN fallback: SSN provided inline — asking for DOB")
                return self._accept_ssn(state, normalized)
            intent = "ambiguous"

        if intent == "no_ssn_available":
            logger.info("VerificationAgent SSN fallback: no SSN available — escalating")
            return self.signal_escalate(state, MSG_SSN_ESCALATE, reason="no identifier available")

        if intent == "no":
            logger.info("VerificationAgent SSN fallback: soft no — prompt for either identifier")
            result = self.ask_member(state, MSG_SSN_EITHER)
            result["ssn_fallback_stage"] = "ssn_or_mid_retry"
            return result

        if intent == "yes":
            logger.info("VerificationAgent SSN fallback: yes — collecting SSN")
            result = self.ask_member(state, pick(MSG_SSN_COLLECT))
            result["ssn_fallback_stage"] = "ssn_collecting"
            return result

        # Caller pivoted back to the Member ID — honour it instead of re-asking
        # a question they have stopped answering.
        if self._pivots_to_member_id(extraction, last_user):
            return self._resume_member_id_collection(state, last_user)

        # Ambiguous — re-ask with the same hardcoded question, but only so many
        # times: an unresolvable gate used to repeat verbatim forever.
        # Do NOT call LLM 2 here: ssn_ask is a yes/no gate and LLM 2 hallucinates
        # irrelevant follow-up questions (e.g. date of birth) from pipeline context.
        slot_attempts = dict(state.get("slot_attempts") or {})
        ask_attempt = slot_attempts.get("ssn_ask")
        if not isinstance(ask_attempt, dict):
            ask_attempt = {}
        count = ask_attempt.get("attempt_count", 0) + 1
        slot_attempts["ssn_ask"] = {**ask_attempt, "attempt_count": count}

        if count >= MAX_SSN_ATTEMPTS:
            logger.warning("VerificationAgent SSN fallback: ssn_ask gate unresolved — escalating")
            escalation = self.signal_escalate(state, MSG_SSN_ASK_EXHAUSTED, reason="ssn_ask_unresolved")
            escalation["slot_attempts"] = slot_attempts
            return escalation

        result = self.ask_member(state, MSG_SSN_ASK)
        result["ssn_fallback_stage"] = "ssn_ask"
        result["slot_attempts"] = slot_attempts
        return result

    async def _ssn_collecting_stage(self, state: State, last_user: str, messages: list) -> dict:
        """User was asked to provide their SSN — extract, validate, then ask for DOB."""
        from agent.llm.config import get_extraction_llm
        from agent.llm.response_generator import generate_recovery_message

        last_agent = _last_assistant_msg(messages)
        extraction = await extract_ssn_decision(
            get_extraction_llm(),
            stage="ssn_collecting",
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-4:],
        )

        intent = extraction.ssn_intent.value if extraction else "ambiguous"
        ssn_raw = (extraction.ssn or "").strip() if extraction else ""

        if intent == "yes_with_ssn" and ssn_raw:
            normalized = normalize_ssn(ssn_raw)
            if normalized and validate_ssn(normalized).valid:
                logger.info("VerificationAgent SSN fallback: SSN collected — asking for DOB")
                return self._accept_ssn(state, normalized)

        if intent == "no_ssn_available":
            return self.signal_escalate(state, MSG_SSN_ESCALATE, reason="no identifier available")

        # Caller found their Member ID after all — take it over the SSN.
        if self._pivots_to_member_id(extraction, last_user):
            return self._resume_member_id_collection(state, last_user)

        # Ambiguous or failed normalization — retry with LLM 2 re-ask
        slot_attempts = dict(state.get("slot_attempts") or {})
        ssn_attempt = slot_attempts.get("ssn") or {}
        if not isinstance(ssn_attempt, dict):
            ssn_attempt = {}
        count = ssn_attempt.get("attempt_count", 0) + 1
        slot_attempts["ssn"] = {**ssn_attempt, "attempt_count": count}

        if count >= MAX_SSN_ATTEMPTS:
            return self.signal_escalate(state, MSG_SSN_RETRY_EXHAUSTED, reason="SSN validation failed")

        logger.info(
            "VerificationAgent SSN fallback: invalid/ambiguous SSN (attempt %d) — calling LLM 2", count
        )
        guard = "CLARIFY" if count == 1 else "RETRY"
        reask = await generate_recovery_message(
            slot_name="ssn",
            attempt=count,
            guard=guard,
            last_messages=messages[-6:],
            user_utterance=last_user,
        )
        result = self.ask_member(state, reask or MSG_SSN_INVALID)
        result["ssn_fallback_stage"] = "ssn_collecting"
        result["slot_attempts"] = slot_attempts
        return result

    async def _ssn_dob_collecting_stage(self, state: State, last_user: str, messages: list) -> dict:
        """SSN is set — collect DOB via existing verification LLM, then do the lookup."""
        from agent.llm.config import get_extraction_llm
        from agent.llm.response_generator import generate_recovery_message
        from agent.slots.normalizers import normalize_dob
        from agent.slots.validators import validate_dob

        call_intent = state.get("call_intent", "")
        _prompt_file = (
            "extraction/verification_claims.md"
            if call_intent == "claim_services"
            else "extraction/verification_provider.md"
        )
        system_prompt = build_extraction_prompt(_prompt_file)
        last_agent = _last_assistant_msg(messages)

        confirmed = {
            k: v
            for k, v in {
                "first_name": state.get("first_name", ""),
                "last_name": state.get("last_name", ""),
            }.items()
            if v
        }

        dob_result = await extract_verification_decision(
            get_extraction_llm(),
            system_prompt,
            awaiting_slot="dob",
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed,
            pending_slots=["dob"],
            attempt=0,
            recent_messages=messages[-4:],
        )

        dob_raw = (dob_result.extracted or {}).get("dob", "") if dob_result else ""

        if dob_raw:
            dob_normalized = normalize_dob(dob_raw)
            if dob_normalized and validate_dob(dob_normalized).valid:
                logger.info("VerificationAgent SSN fallback: DOB collected — proceeding to SF lookup")
                # slot_ok, not just a local dict write: ask_member persists
                # confirmed slots into every interrupt it builds, and that is
                # the only way this value reaches graph state. Without it the
                # DOB lived in one call frame, and the next turn looked up an
                # identity with an empty date.
                self.slot_ok("dob", dob_normalized)
                state = {**state, "dob": dob_normalized, "ssn_fallback_stage": "ssn_lookup"}
                return await self._finish_after_ssn(state, messages, call_intent)

        # DOB invalid or absent — retry with LLM 2
        slot_attempts = dict(state.get("slot_attempts") or {})
        dob_attempt = slot_attempts.get("ssn_dob") or {}
        if not isinstance(dob_attempt, dict):
            dob_attempt = {}
        count = dob_attempt.get("attempt_count", 0) + 1
        slot_attempts["ssn_dob"] = {"attempt_count": count}

        if count >= 3:
            return self.signal_escalate(
                state,
                "I wasn't able to collect your date of birth. "
                "Let me connect you with a representative who can assist.",
                reason="DOB collection failed in SSN verification path",
            )

        guard = "CLARIFY" if count == 1 else "RETRY"
        reask = await generate_recovery_message(
            slot_name="dob",
            attempt=count,
            guard=guard,
            last_messages=messages[-6:],
            user_utterance=last_user,
        )
        r = self.ask_member(state, reask or "Could you provide your date of birth?")
        r["ssn_fallback_stage"] = "ssn_dob_collecting"
        r["slot_attempts"] = slot_attempts
        return r

    async def _ssn_or_mid_retry_stage(self, state: State, last_user: str, messages: list) -> dict:
        """User given second chance to provide Member ID or SSN — LLM-backed extraction."""
        from agent.llm.config import get_extraction_llm
        from agent.slots.normalizers import normalize_member_id, normalize_ssn
        from agent.slots.validators import validate_member_id, validate_ssn

        # Use the full verification LLM to extract member_id (handles spoken digits,
        # prefix text like "Oh wait, I found it — M nine zero seven five zero three.")
        call_intent = state.get("call_intent", "")
        _prompt_file = (
            "extraction/verification_claims.md"
            if call_intent == "claim_services"
            else "extraction/verification_provider.md"
        )
        system_prompt = build_extraction_prompt(_prompt_file)
        last_agent = _last_assistant_msg(messages)

        confirmed = {
            k: v
            for k, v in {
                "first_name": state.get("first_name", ""),
                "last_name": state.get("last_name", ""),
            }.items()
            if v
        }

        llm_result = await extract_verification_decision(
            get_extraction_llm(),
            system_prompt,
            awaiting_slot="member_id",
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed,
            pending_slots=["member_id"],
            attempt=0,
            recent_messages=messages[-4:],
        )

        # Check for member_id first
        mid_raw = (llm_result.extracted or {}).get("member_id", "") if llm_result else ""
        if mid_raw:
            mid = normalize_member_id(mid_raw)
            if mid and validate_member_id(mid).valid:
                logger.info("VerificationAgent SSN fallback: member_id found in retry — resuming normal flow")
                self.slot_ok("member_id", mid)
                r = self.ask_member(state, "")
                r["member_id"] = mid
                r["ssn_fallback_stage"] = ""
                r["is_interrupt"] = False
                return r

        # Check for inline SSN
        ssn_raw = self._extract_ssn_from_text(last_user)
        if ssn_raw:
            normalized = normalize_ssn(ssn_raw)
            if normalized and validate_ssn(normalized).valid:
                return self._accept_ssn(state, normalized)

        # Hard no → escalate
        if self._is_no_ssn_available(last_user):
            return self.signal_escalate(state, MSG_SSN_ESCALATE, reason="no identifier available")

        # Caller says they have the Member ID but the extraction found no digits
        # — ask for it plainly rather than repeating the either/or prompt.
        if self._pivots_to_member_id(llm_result, last_user):
            return self._resume_member_id_collection(state, last_user)

        # Bounded, like every other stage — this prompt used to repeat forever.
        slot_attempts = dict(state.get("slot_attempts") or {})
        retry_attempt = slot_attempts.get("ssn_or_mid")
        if not isinstance(retry_attempt, dict):
            retry_attempt = {}
        count = retry_attempt.get("attempt_count", 0) + 1
        slot_attempts["ssn_or_mid"] = {**retry_attempt, "attempt_count": count}

        if count >= MAX_SSN_ATTEMPTS:
            logger.warning("VerificationAgent SSN fallback: no identifier after retries — escalating")
            escalation = self.signal_escalate(state, MSG_SSN_ESCALATE, reason="no identifier available")
            escalation["slot_attempts"] = slot_attempts
            return escalation

        r = self.ask_member(state, MSG_SSN_EITHER)
        r["ssn_fallback_stage"] = "ssn_or_mid_retry"
        r["slot_attempts"] = slot_attempts
        return r

    def _accept_ssn(self, state: State, ssn: str) -> dict:
        """SSN collected and valid — ask for DOB to complete the SSN-path verification."""
        logger.info("VerificationAgent SSN fallback: SSN accepted, requesting DOB")
        result = self.ask_member(state, MSG_SSN_DOB)
        result["ssn"] = ssn
        result["ssn_fallback_stage"] = "ssn_dob_collecting"
        return result

    @staticmethod
    def _identity_fingerprint(state: State) -> str:
        """The exact tuple the SSN-path lookup matches on."""
        return "|".join(
            str(state.get(k) or "").strip().lower() for k in ("first_name", "last_name", "dob", "ssn")
        )

    async def _handle_ssn_lookup_miss(self, state: State, messages: list) -> dict:
        """Diagnose which field actually missed, then re-ask only that field.

        Mirrors lookup_and_verify on the Member ID path. That one asks the store
        whether the Member ID exists: if it does not, everything is re-asked; if
        it does, compare_identity_fields names the fields that differ and only
        those are re-collected. The SSN path now does the same with the SSN as
        the key, instead of blaming a field it had not checked.

          SSN not on file        → the SSN is wrong; name, SSN and DOB all go
          SSN found, name differs → re-ask that name field only
          SSN found, DOB differs  → re-ask the DOB only
        """
        from agent.storage.queries.members import compare_identity_fields, find_member_by_ssn

        slot_attempts = dict(state.get("slot_attempts") or {})
        entry = slot_attempts.get("ssn_lookup") or {}
        rounds = (entry.get("attempt_count", 0) if isinstance(entry, dict) else 0) + 1
        slot_attempts["ssn_lookup"] = {"attempt_count": rounds}

        if rounds > MAX_SSN_ATTEMPTS:
            logger.warning("VerificationAgent SSN fallback: lookup still missing after re-checks")
            escalation = self.signal_escalate(
                state,
                "I wasn't able to match your details to an account. "
                "Let me connect you with a representative who can assist.",
                reason="SSN+DOB lookup failed after max attempts",
            )
            escalation["slot_attempts"] = slot_attempts
            return escalation

        ssn_record = None
        try:
            ssn_record = await find_member_by_ssn(ssn=state.get("ssn", ""))
        except Exception:
            logger.exception("VerificationAgent SSN fallback: SSN lookup raised — re-asking everything")

        if not ssn_record:
            # No account holds this SSN, so it is the SSN that is wrong. Nothing
            # else can be trusted either — the name and DOB were only ever
            # checked together with it.
            logger.info("VerificationAgent SSN fallback: SSN not on file — re-asking name, SSN and DOB")
            return self._ask_ssn_recheck(state, ["first_name", "last_name", "ssn", "dob"], slot_attempts)

        field_matches = compare_identity_fields(
            ssn_record,
            first_name=state.get("first_name", ""),
            last_name=state.get("last_name", ""),
            dob=state.get("dob", ""),
        )
        mismatched = [f for f in ("first_name", "last_name", "dob") if field_matches.get(f) is False]

        if not mismatched:
            # The SSN is on file and every field agrees, yet the combined query
            # missed. Nothing is safe to single out — re-ask the lot.
            logger.warning("VerificationAgent SSN fallback: no field mismatch found — re-asking everything")
            return self._ask_ssn_recheck(state, ["first_name", "last_name", "ssn", "dob"], slot_attempts)

        logger.info(
            "VerificationAgent SSN fallback: targeted re-ask of mismatched fields",
            extra={"mismatched": mismatched},
        )
        return self._ask_ssn_recheck(state, mismatched, slot_attempts)

    _SSN_RECHECK_PROMPTS = {  # noqa: RUF012
        "first_name": MSG_REASK_FIRST_NAME,
        "last_name": MSG_REASK_LAST_NAME,
        "dob": MSG_REASK_DOB,
        "ssn": MSG_SSN_COLLECT,
    }

    def _ask_ssn_recheck(
        self, state: State, fields: list[str], slot_attempts: dict, *, missing: bool = False
    ) -> dict:
        """Ask for the first field to re-check and queue the rest.

        ``missing`` distinguishes "we never got this" from "this did not match":
        telling a caller their date of birth did not match, when in fact it was
        dropped along the way, sends them looking for a problem that is ours.
        """
        field = fields[0]
        if missing:
            message = pick(MSG_SSN_NEED_FIELD).format(field_label=_SSN_FIELD_LABELS[field])
        elif len(fields) > 1:
            # Multi-field re-asks open with the non-disclosing generic prompt:
            # reading back several wrong details to a caller we have not
            # verified is not something to do out loud.
            message = pick(MSG_REASK_GENERIC)
        else:
            message = pick(self._SSN_RECHECK_PROMPTS[field])
        r = self.ask_member(state, message)
        # Carry forward every identity field being kept. ask_member only
        # persists slot-confirmed values, so anything that reached this frame
        # in the state dict alone would silently vanish from the next turn.
        for keep in self._SSN_IDENTITY_FIELDS:
            if keep not in fields and str(state.get(keep) or "").strip():
                r[keep] = state[keep]
        for stale in fields:
            r[stale] = ""
            self.get_slot(stale).reset()
        if "first_name" in fields or "last_name" in fields:
            r["name_confirmed"] = False
            r["name_confirm_attempts"] = 0
        r["ssn_recheck_fields"] = ",".join(fields)
        r["ssn_fallback_stage"] = "ssn_recheck"
        r["slot_attempts"] = slot_attempts
        return r

    # The SSN-path lookup matches on all of these; a hole in any one of them
    # means there is nothing to look up yet.
    _SSN_IDENTITY_FIELDS = ("first_name", "last_name", "ssn", "dob")

    @classmethod
    def _missing_ssn_identity(cls, state: State) -> list[str]:
        """Which fields the SSN lookup still needs, in the order they are asked."""
        return [f for f in cls._SSN_IDENTITY_FIELDS if not str(state.get(f) or "").strip()]

    async def _ssn_recheck_stage(self, state: State, last_user: str, messages: list) -> dict:
        """Re-collect the fields the diagnosis flagged, then run the lookup again."""
        from agent.responses.static import build_slot_exhausted_message

        pending = [f for f in (state.get("ssn_recheck_fields") or "").split(",") if f]
        call_intent = state.get("call_intent", "")
        if not pending:
            # The queue is gone but fields may still be cleared. Re-deriving it
            # from state is what keeps a lost queue from reaching the store with
            # a hole in the identity.
            missing = self._missing_ssn_identity(state)
            if missing:
                logger.warning(
                    "VerificationAgent SSN fallback: recheck queue lost — rebuilding from state",
                    extra={"missing": missing},
                )
                return self._ask_ssn_recheck(
                    state, missing, dict(state.get("slot_attempts") or {}), missing=True
                )
            return await self._finish_after_ssn(dict(state), messages, call_intent)

        field = pending[0]
        value = await self._extract_recheck_value(field, state, last_user, messages)

        if not value:
            slot_attempts = dict(state.get("slot_attempts") or {})
            entry = slot_attempts.get(f"recheck_{field}") or {}
            count = (entry.get("attempt_count", 0) if isinstance(entry, dict) else 0) + 1
            slot_attempts[f"recheck_{field}"] = {"attempt_count": count}
            if count >= MAX_SSN_ATTEMPTS:
                escalation = self.signal_escalate(
                    state,
                    build_slot_exhausted_message(field),
                    reason=f"ssn_recheck_{field}_exhausted",
                )
                escalation["slot_attempts"] = slot_attempts
                return escalation
            retry = self.ask_member(state, pick(self._SSN_RECHECK_PROMPTS[field]))
            retry["ssn_fallback_stage"] = "ssn_recheck"
            retry["ssn_recheck_fields"] = ",".join(pending)
            retry["slot_attempts"] = slot_attempts
            return retry

        updated = {**state, field: value}
        self.slot_ok(field, value)
        remaining = pending[1:]

        if remaining:
            nxt = self.ask_member(updated, pick(self._SSN_RECHECK_PROMPTS[remaining[0]]))
            nxt[field] = value
            nxt["ssn_recheck_fields"] = ",".join(remaining)
            nxt["ssn_fallback_stage"] = "ssn_recheck"
            return nxt

        logger.info("VerificationAgent SSN fallback: re-checked fields collected — retrying lookup")
        updated["ssn_recheck_fields"] = ""
        updated["ssn_fallback_stage"] = "ssn_lookup"
        return await self._finish_after_ssn(updated, messages, call_intent)

    async def _extract_recheck_value(self, field: str, state: State, last_user: str, messages: list) -> str:
        """Extract and validate one re-checked field. Returns "" when unusable."""
        from agent.llm.config import get_extraction_llm
        from agent.slots.normalizers import normalize_dob
        from agent.slots.validators import validate_dob

        if field == "ssn":
            inline = self._extract_ssn_from_text(last_user)
            if inline:
                return inline
            extraction = await extract_ssn_decision(
                get_extraction_llm(),
                stage="ssn_collecting",
                last_agent_message=_last_assistant_msg(messages),
                last_user_message=last_user,
                recent_messages=messages[-4:],
            )
            raw = (getattr(extraction, "ssn", None) or "").strip()
            normalized = normalize_ssn(raw) if raw else ""
            return normalized if normalized and validate_ssn(normalized).valid else ""

        call_intent = state.get("call_intent", "")
        system_prompt = build_extraction_prompt(
            "extraction/verification_claims.md"
            if call_intent == "claim_services"
            else "extraction/verification_provider.md"
        )
        extraction = await extract_verification_decision(
            get_extraction_llm(),
            system_prompt,
            awaiting_slot=field,
            last_agent_message=_last_assistant_msg(messages),
            last_user_message=last_user,
            confirmed_slots={},
            pending_slots=[field],
            attempt=0,
            recent_messages=messages[-4:],
        )
        values = {**(extraction.extracted or {}), **(extraction.corrections or {})} if extraction else {}
        raw = (values.get(field) or "").strip()
        if not raw:
            return ""
        if field == "dob":
            normalized = normalize_dob(raw)
            return normalized if normalized and validate_dob(normalized).valid else ""
        normalized = normalize_name(raw)
        return normalized if normalized and validate_name(normalized).valid else ""

    async def _finish_after_ssn(self, state: State, messages: list, call_intent: str) -> dict:
        """Lookup by SSN + DOB + names — reuses find_member_by_identity, no new tool needed."""
        from agent.storage.queries.members import find_member_by_identity

        missing = self._missing_ssn_identity(state)
        if missing:
            # Reaching the store with a cleared field is how an empty DOB got
            # into a SOQL date filter and took down the run. There is nothing to
            # match on until the hole is filled, so ask for it instead.
            logger.warning(
                "VerificationAgent SSN fallback: lookup skipped — identity incomplete",
                extra={"missing": missing},
            )
            return self._ask_ssn_recheck(state, missing, dict(state.get("slot_attempts") or {}), missing=True)

        ssn = state.get("ssn", "")
        dob = state.get("dob", "")
        first_name = state.get("first_name", "")
        last_name = state.get("last_name", "")

        record = await find_member_by_identity(
            member_id="",  # no member_id in SSN path
            first_name=first_name,
            last_name=last_name,
            dob=dob,
            ssn=ssn,
        )

        if not record:
            return await self._handle_ssn_lookup_miss(state, messages)

        # Success — populate state from SF record.
        # Prefer values already in state (normalizer-formatted) over the SF record's
        # raw storage format. E.g. state["dob"] = "04/12/1988" (MM/DD/YYYY from
        # normalize_dob) must NOT be overwritten by SF's ISO "1988-04-12", which
        # would cause validate_dob to fail on the next pipeline pass.
        for key in (
            "first_name",
            "last_name",
            "member_id",
            "dob",
            "relationship",
            "zip_code",
            "phone_number",
            "fax",
            "email",
        ):
            sf_val = record.get(key)
            if not sf_val:
                continue
            existing = state.get(key)
            value_to_use = existing if existing else sf_val
            state[key] = value_to_use
            self.slot_ok(key, value_to_use)

        state["member_status_verify"] = True
        state["ssn_fallback_stage"] = ""
        collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}

        state["success_message"] = MSG_SSN_SUCCESS
        if interrupt := await collect_post_lookup(
            self,
            state,
            messages,
            collected,
            call_intent,
            record,
            None,
            self._claims_pipeline,
            self._provider_pipeline,
        ):
            interrupt["ssn_fallback_stage"] = ""
            return interrupt

        return self._signal_verified(state, collected, record)

    async def run(self, state: State) -> dict:  # noqa: C901
        # Early exit: member already fully verified on re-entry — skip all slot collection.
        # Guard requires awaiting_slot to be empty: if a post-lookup slot (relationship,
        # phone_confirmed) is still being collected, the pipeline must run to completion
        # or escalation. Firing here while awaiting_slot is set skips exhaustion checks
        # and incorrectly signals verification complete mid-pipeline.
        if state.get("member_status_verify") and not state.get("awaiting_slot"):
            collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}
            if state.get("phone_confirmed") is True:
                collected["phone_confirmed"] = True
            member_record = self._member_record_from_state(state)
            return self._signal_verified(state, collected, member_record)

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        call_intent = state.get("call_intent", "")

        # ── SSN fallback — route active sub-stages before any other logic ────
        ssn_fallback_stage = state.get("ssn_fallback_stage") or ""
        if ssn_fallback_stage == "ssn_lookup":
            # Guard: if lookup already ran this turn (member_status_verify set),
            # clear the stale stage and fall through to normal post-lookup flow.
            if state.get("member_status_verify"):
                state = {**state, "ssn_fallback_stage": ""}
            else:
                return await self._finish_after_ssn(state, messages, call_intent)
        elif ssn_fallback_stage:
            return await self._handle_ssn_fallback(state, last_user, messages)

        # ── Mid-call re-verification first-name bridge (one-shot) ────────────
        # reset_for_new_intent sets reverify_bridge_pending=True when a fresh
        # intent is detected mid-call. The utterance that triggered the switch
        # carries no identity data, so the extraction LLM call below would be
        # wasted. Instead deliver the same deterministic first-name bridge intake
        # uses and pause for the member's reply. The flag is one-shot — keyed off
        # reverify_bridge_pending, NOT pending_intent (which persists across every
        # re-verification turn and would re-fire the bridge each turn).
        # Edge case (acceptable tradeoff): if the trigger utterance happened to
        # include a name, we still re-ask for first name via the bridge rather
        # than extracting it — identity is re-collected from scratch by design.
        if state.get("reverify_bridge_pending"):
            msg = random.choice(INTENT_BRIDGE_MSGS)
            result = self.ask_member(state, msg)  # sets is_interrupt=True, next_node=verification_agent
            result["reverify_bridge_pending"] = False  # one-shot: clear so next turn extracts normally
            result["awaiting_slot"] = "first_name"  # correct slot context for next-turn extraction
            logger.info("VerificationAgent: re-verify first-name bridge delivered (LLM call skipped)")
            return result

        _prompt_file = (
            "extraction/verification_claims.md"
            if call_intent == "claim_services"
            else "extraction/verification_provider.md"
        )
        system_prompt = build_extraction_prompt(_prompt_file)
        collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}

        if call_intent == "claim_services":
            slot_order = ["first_name", "last_name", "member_id", "dob", "phone_confirmed"]
        else:
            slot_order = ["first_name", "last_name", "member_id", "dob", "relationship"]
        awaiting_slot = state.get("awaiting_slot") or next(
            (s for s in slot_order if not str(state.get(s) or "").strip()),
            IDENTITY_SLOT_ORDER[-1],  # "dob" — all identity slots collected; accurate
        )  # context so the LLM detects corrections correctly
        # Write computed awaiting_slot into state so _collect_slot can route CORRECTED/
        # AMBIGUOUS events correctly when the slot was not explicitly set by a prior turn.
        state = {**state, "awaiting_slot": awaiting_slot}

        # ── NAME CONFIRMATION GATE ────────────────────────────────────────────
        # Fires once both names are in state and before member_id is collected.
        # The name_confirmed flag prevents re-entry on subsequent turns.
        _fn = (state.get("first_name") or "").strip()
        _ln = (state.get("last_name") or "").strip()
        if _fn and _ln and not state.get("name_confirmed"):
            return await self._handle_name_confirmation(state, messages, last_user)
        # ─────────────────────────────────────────────────────────────────────

        # ── Member ID denial → SSN fallback ──────────────────────────────────
        # Fires after names are confirmed but before the LLM extraction so the
        # denial is caught deterministically rather than by the extraction model.
        if (
            awaiting_slot == "member_id"
            and not state.get("member_id")
            and not state.get("ssn")
            and last_user
            # The phrase list is a fast path; detect_cannot_provide covers the
            # phrasings it does not ("I do not have a member ID", "I never
            # received one"). Without it those escalated straight to a rep
            # instead of ever offering the SSN alternative.
            and (self._is_member_id_denial(last_user) or detect_cannot_provide(last_user))
        ):
            logger.info("VerificationAgent: member_id denial detected — starting SSN fallback")
            result = self.ask_member(state, MSG_SSN_ASK)
            result["ssn_fallback_stage"] = "ssn_ask"
            return result
        # ─────────────────────────────────────────────────────────────────────

        last_agent = _last_assistant_msg(messages)
        confirmed_slots = {k: v for k, v in collected.items() if v and v.strip()}
        current_attempt = (state.get("slot_attempts") or {}).get(awaiting_slot, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0
        restart_index = state.get("verification_restart_index") or 0
        if restart_index:
            # Include up to 2 pre-restart messages as context so the extraction
            # LLM can re-extract slots the caller already stated in round 1.
            pre_restart_context = messages[max(0, restart_index - 2) : restart_index]
            post_restart = messages[restart_index:]
            recent_messages = (pre_restart_context + post_restart)[-8:]
        else:
            recent_messages = messages[-6:]
        result = await extract_verification_decision(
            get_extraction_llm(),
            system_prompt,
            awaiting_slot=awaiting_slot,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            pending_slots=[s for s in slot_order if not str(state.get(s) or "").strip()],
            attempt=attempt_count,
            recent_messages=recent_messages,
        )

        # Phase 4 (follow-up disposition routing): ANSWERED_WITH_FOLLOWUP is no
        # longer flattened to ANSWERED here. _collect_slot now routes the
        # follow-up disposition itself (answer_now/park/decline → FOLLOWUP_*
        # guards) and appends the next static ask, so the event must reach the
        # pipeline intact.

        if interrupt := await self.run_conversation_guards(
            state,
            user_text=last_user,
            result=result,
        ):
            if getattr(result, "guard", "") == "OFFTOPIC_AGENT":
                return redirect_off_topic(self, state, collected, self._identity_pipeline)
            return interrupt

        # ── DETERMINISTIC RECONCILE (Phase 1) ────────────────────────────────
        # llm.py already reconciles on success, but extraction fallbacks (and
        # monkeypatched results) bypass it — re-running here is idempotent and
        # guarantees a mid-verification update request ("also I need to update
        # my last name") reaches the pipeline as update_target.
        result = reconcile_worker_result(result, last_user)

        # ── Member ID denial → SSN fallback (semantic) ───────────────────────
        # The pre-extraction gate above is a keyword fast path. This is the one
        # that actually generalises: the extraction model judges meaning, so
        # phrasings no list anticipates ("that's in my wallet at home") still
        # reach the SSN offer instead of the escalation in _collect_slot.
        # fallback_pivot == "ssn" is the caller naming the alternative outright.
        if (
            awaiting_slot == "member_id"
            and not state.get("member_id")
            and not state.get("ssn")
            and not (result.extracted or {}).get("member_id")
            and (
                getattr(result, "cannot_provide", False)
                or (getattr(result, "fallback_pivot", None) or "").strip() == "ssn"
            )
        ):
            logger.info(
                "VerificationAgent: member_id unavailable (extraction) — starting SSN fallback",
                extra={"fallback_pivot": getattr(result, "fallback_pivot", None)},
            )
            ssn_offer = self.ask_member(state, MSG_SSN_ASK)
            ssn_offer["ssn_fallback_stage"] = "ssn_ask"
            return ssn_offer
        # ─────────────────────────────────────────────────────────────────────

        corrected_fields: list[str] = []
        _was_verified = bool(state.get("member_status_verify"))
        if last_user:
            corrected_fields = apply_corrections(self, collected, state, result) or []
        # apply_corrections clears member_status_verify in-place when an identity
        # slot is corrected post-verification (so lookup_and_verify re-runs this
        # turn). Remember that it happened so interrupts returned below persist
        # the cleared flag to LangGraph state.
        _reverify_cleared = _was_verified and not state.get("member_status_verify")

        # sync corrected first_name into state and context immediately
        if "first_name" in corrected_fields:
            ctx = ConversationContext.from_state(state)
            ctx.update_caller_name(collected["first_name"])
            state = {
                **state,
                "first_name": collected["first_name"],
                "conversation_context": ctx.to_dict(),
            }

        # Cascade clears: apply_corrections zeroes collected["last_name"] when first_name
        # is corrected and collected["dob"] when member_id is corrected. But _collect_slot
        # reads from state (not collected) for the "already valid" check, so we must also
        # clear the relevant state keys to prevent the old value from being silently reused.
        # only cascade-clear if value not provided in same utterance
        _corr = getattr(result, "corrections", {}) or {}
        _extracted_this_turn = (result.extracted or {}) if result else {}
        _cascade_cleared: list[str] = []
        if (
            _corr.get("first_name")
            and not collected.get("last_name")
            and not _extracted_this_turn.get("last_name")
        ):
            state = {**state, "last_name": ""}
            _cascade_cleared.append("last_name")
        if _corr.get("member_id") and not collected.get("dob") and not _extracted_this_turn.get("dob"):
            state = {**state, "dob": ""}
            _cascade_cleared.append("dob")
        # Reset the cleared slots' records so ask_member's confirmed-value
        # persistence cannot resurrect the stale value in the interrupt.
        for _cleared in _cascade_cleared:
            self.get_slot(_cleared).reset()

        # ── correction_return_to: set when correcting a field other than awaiting slot ──
        if corrected_fields and awaiting_slot not in corrected_fields:
            # Caller corrected a confirmed slot; pipeline must return to awaiting_slot after.
            state = {**state, "correction_return_to": awaiting_slot}
        elif not corrected_fields and state.get("correction_return_to"):
            # No correction this turn — preserve existing pointer from prior turn.
            pass  # state already has the right value

        # ── ambiguous_counts: carry forward so the counter accumulates across turns ──
        # _collect_slot writes updated counts into the interrupt dict each turn.
        # On re-entry, those counts arrive in state — no additional wiring needed here,
        # but confirm ambiguous_counts is present with a safe default if state is fresh:
        if "ambiguous_counts" not in state:
            state = {**state, "ambiguous_counts": {}}

        # ── Pre-save bonus extractions ────────────────────────────────────────
        # The pipeline processes slots IN ORDER and stops at the first failure.
        # If the user provided a valid value for a slot that comes AFTER the
        # currently failing slot (e.g., gave DOB while awaiting member_id), that
        # value is in result.extracted but will be discarded when the pipeline
        # returns early for the failing slot.
        # Pre-populating collected here ensures the pipeline skips those slots
        # on the next iteration instead of re-asking.
        if result and result.extracted:
            for _bonus_slot, _bonus_raw in result.extracted.items():
                if (
                    _bonus_slot in IDENTITY_SLOT_ORDER
                    and _bonus_slot != awaiting_slot
                    and not collected.get(_bonus_slot)
                    and _bonus_raw
                ):
                    _norm_fn = _NORMALIZERS.get(_bonus_slot)
                    _val_fn = _VALIDATORS.get(_bonus_slot)
                    if _norm_fn and _val_fn:
                        _normalized = _norm_fn(str(_bonus_raw))
                        if _normalized and _val_fn(_normalized).valid:
                            collected[_bonus_slot] = _normalized
                            self.slot_ok(_bonus_slot, _normalized)
                            logger.info(
                                "VerificationAgent: bonus extraction saved",
                                extra={"slot": _bonus_slot, "awaiting": awaiting_slot},
                            )

        # ── NAME-PAIR INTERCEPT — fire readback before collecting member_id ──
        # The identity pipeline collects first_name → last_name → member_id → dob
        # in one pass. On the turn the member supplies the last name (first name
        # already on file), the pipeline confirms last_name and continues straight
        # to asking for member_id, skipping the name-confirmation readback. Resolve
        # the name pair here — including last_name from THIS turn's extraction,
        # which bonus-extraction skips because it equals the awaiting slot — and
        # deliver the readback before the pipeline can advance.
        if not state.get("name_confirmed"):
            _extracted_now = (result.extracted or {}) if result else {}

            def _resolve_name(slot: str) -> str:
                v = (collected.get(slot) or state.get(slot) or "").strip()
                if v:
                    return v
                raw = _extracted_now.get(slot, "")
                if raw:
                    norm = normalize_name(raw)
                    if norm and validate_name(norm).valid:
                        return norm
                return ""

            _fn_pair = _resolve_name("first_name")
            _ln_pair = _resolve_name("last_name")

            # ── FALLBACK: LLM extracted only first_name but last_name was
            # also in the utterance (e.g. "emily carter", "John Smith").
            # When the utterance is exactly two whitespace-separated tokens
            # and the first token matches the extracted first name, treat
            # the second token as the last name. Zero LLM cost; prevents
            # the spurious "What's your last name?" re-ask when both names
            # were already given in a single utterance.
            if _fn_pair and not _ln_pair and last_user:
                _tokens = last_user.strip().split()
                if len(_tokens) == 2:
                    _candidate_last = normalize_name(_tokens[1])
                    if (
                        _candidate_last
                        and validate_name(_candidate_last).valid
                        and normalize_name(_tokens[0]).lower() == _fn_pair.lower()
                    ):
                        _ln_pair = _candidate_last
                        logger.info(
                            "VerificationAgent: last_name recovered from two-token utterance fallback",
                            extra={"first": _fn_pair, "last": _candidate_last},
                        )
            # ── END FALLBACK ──────────────────────────────────────────────

            if _fn_pair and _ln_pair:
                self.slot_ok("first_name", _fn_pair)
                self.slot_ok("last_name", _ln_pair)
                collected["first_name"] = _fn_pair
                collected["last_name"] = _ln_pair
                state = {**state, "first_name": _fn_pair, "last_name": _ln_pair}
                return await self._handle_name_confirmation(state, messages, last_user)

        # Collect first_name → last_name → member_id → dob
        if interrupt := await self._identity_pipeline.collect(state, messages, collected, decision=result):
            if _reverify_cleared and "member_status_verify" not in interrupt:
                interrupt["member_status_verify"] = False
            # Persist cascade clears: the pipeline only carries non-empty
            # collected values forward, so the cleared keys must be written
            # explicitly or LangGraph keeps the stale pre-correction value.
            for _cleared in _cascade_cleared:
                if not interrupt.get(_cleared):
                    interrupt[_cleared] = ""
            return interrupt

        # Both names just confirmed this turn — fire name readback immediately.
        _fn_now = (collected.get("first_name") or state.get("first_name") or "").strip()
        _ln_now = (collected.get("last_name") or state.get("last_name") or "").strip()
        if _fn_now and _ln_now and not state.get("name_confirmed"):
            state = {**state, "first_name": _fn_now, "last_name": _ln_now}
            return await self._handle_name_confirmation(state, messages, last_user)

        # Salesforce lookup
        #
        # ── Partial re-ask round-trip (wrong DOB only) ───────────────────────────
        # Trace of the targeted re-ask path, e.g. caller's DOB is wrong but name +
        # Member ID match:
        #
        #   Turn N (all four slots collected):
        #     pipeline.collect() completes → lookup_and_verify() runs →
        #     full match fails → lookup_member returns member_id_found=True with
        #     field_matches={first_name:T, last_name:T, dob:F} →
        #     handlers._partial_reask(mismatched=["dob"]) returns an interrupt that:
        #       • clears ONLY dob (""); keeps first_name, last_name, member_id
        #       • leaves name_confirmed=True untouched (no name field mismatched)
        #       • sets awaiting_slot="dob" and verification_restart_index=len(msgs)
        #       • asks MSG_REASK_DOB  ("…date of birth once more?")
        #     member_status_verify is NOT set, so the loop stays open.
        #
        #   Turn N+1 (caller restates DOB):
        #     • name gate (line ~136) skipped: name_confirmed is True
        #     • both readback intercepts (lines ~260, ~316) skipped: name_confirmed True
        #       → NO spelled-name read-back, NO name/Member-ID re-ask
        #     • awaiting_slot="dob" gives the extractor the correct slot context
        #     • pipeline.collect() skips the still-populated first_name / last_name /
        #       member_id and collects only dob (first empty slot in identity order)
        #     • member_status_verify still falsy → lookup_and_verify() runs AGAIN
        #       with the corrected dob → fresh full match → _signal_verified().
        # ─────────────────────────────────────────────────────────────────────────
        final = await self._finish_after_identity(state, collected, messages, call_intent, result)
        # Persist the re-verify clear unless the path already decided the flag
        # (a fresh successful lookup sets True via _signal_verified; post-lookup
        # interrupts set it via collect_post_lookup).
        if _reverify_cleared and isinstance(final, dict) and "member_status_verify" not in final:
            final["member_status_verify"] = False
        return final

    async def _finish_after_identity(
        self, state: State, collected: dict, messages: list, call_intent: str, decision
    ) -> dict:
        """Salesforce lookup → post-lookup slot → verified signal.

        Shared by run()'s main path and _name_confirmed_proceed's all-slots-present
        branch (name-only partial re-ask), so a corrected name flows straight to the
        lookup instead of re-asking an already-known Member ID.
        """
        if not state.get("member_status_verify"):
            member_record, interrupt = await lookup_and_verify(self, state, collected)
            if interrupt:
                return interrupt
        else:
            member_record = self._member_record_from_state(state)

        # Eagerly merge SF lookup fields into a local state snapshot so they are
        # readable during post-lookup slot collection retries (relationship label,
        # phone_confirmed label). _signal_verified also writes these at the end,
        # but only when the node returns — they must be readable mid-execution.
        if member_record:
            state = {
                **state,
                "phone_number": member_record.get("phone_number") or state.get("phone_number", ""),
                "relationship": member_record.get("relationship") or state.get("relationship", ""),
            }

        # Phone confirmation (claims) or relationship (provider)
        if interrupt := await collect_post_lookup(
            self,
            state,
            messages,
            collected,
            call_intent,
            member_record,
            decision,
            self._claims_pipeline,
            self._provider_pipeline,
        ):
            return interrupt

        return self._signal_verified(state, collected, member_record)

    # =========================================================================
    # Name confirmation phase
    # =========================================================================

    async def _handle_name_confirmation(self, state: State, messages: list, last_user: str) -> dict:
        """
        Router for the name readback → confirm / correct loop.

        Entry routing by awaiting_slot:
          ""                      → first entry: deliver readback
          _NAME_CONFIRM_SLOT      → member just responded to a readback
          _NAME_CORRECTION_SLOT   → member just gave the corrected name (after bare no)
        """
        current_awaiting = state.get("awaiting_slot", "")
        if current_awaiting == _NAME_CONFIRM_SLOT:
            return await self._process_name_readback_response(state, messages, last_user)
        if current_awaiting == _NAME_CORRECTION_SLOT:
            return await self._collect_name_correction(state, messages, last_user)
        # First entry or any unrecognised slot — deliver the readback.
        return self._deliver_name_readback(state)

    def _deliver_name_readback(self, state: State) -> dict:
        """Send the spelled-out name readback and set awaiting_slot=name_confirmed."""
        first = (state.get("first_name") or "").strip()
        last = (state.get("last_name") or "").strip()

        ctx = ConversationContext.from_state(state)
        ctx.update_caller_name(first)
        msg = _build_name_readback_message(first, last)

        logger.info(LOG_NAME_READBACK, extra={"first": first, "last": last})
        result = self.ask_member(state, msg)
        result["awaiting_slot"] = _NAME_CONFIRM_SLOT
        result["first_name"] = first
        result["last_name"] = last
        result["name_confirm_attempts"] = state.get("name_confirm_attempts") or 0
        result["conversation_context"] = ctx.to_dict()
        return result

    async def _process_name_readback_response(self, state: State, messages: list, last_user: str) -> dict:
        """
        Extract the member's response to the spelled-out name readback.

        Three outcomes — see name_confirmation.md for the extraction contract:
          1. name_confirmed="yes"            → proceed to member_id
          2. first_name / last_name present  → inline correction; re-read back
          3. name_confirmed="no", no names   → ask for correct name separately
          4. ambiguous                       → slot_fail → retry readback or escalate
        """
        last_agent = _last_assistant_msg(messages)
        attempt_count = self.get_slot(_NAME_CONFIRM_SLOT).attempt_count

        result = await extract_name_confirmation(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/name_confirmation.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            pending_slots=[s for s in IDENTITY_SLOT_ORDER if not str(state.get(s) or "").strip()],
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}
        name_conf_raw = extracted.get("name_confirmed", "")
        corrected_first_raw = extracted.get("first_name", "")
        corrected_last_raw = extracted.get("last_name", "")

        # ── OUTCOME 1: confirmed ─────────────────────────────────────────────
        if name_conf_raw == "yes":
            logger.info(LOG_NAME_CONFIRMED)
            # When the caller confirmed AND asked a side question, address it
            # before moving to member_id — never silently skip a caller's question.
            _event_raw = getattr(result, "event_type", None)
            _event_str = str(getattr(_event_raw, "value", _event_raw) or "").lower()
            _followup_q = (getattr(result, "followup_query", None) or "").strip()
            if _event_str == "answered_with_followup" and _followup_q:
                return await self._name_confirmed_with_followup(state, messages, _followup_q)
            return await self._name_confirmed_proceed(state, messages)

        # ── OUTCOME 2: inline correction ─────────────────────────────────────
        corrected_first = normalize_name(corrected_first_raw) if corrected_first_raw else ""
        corrected_last = normalize_name(corrected_last_raw) if corrected_last_raw else ""
        first_ok = bool(corrected_first) and validate_name(corrected_first).valid
        last_ok = bool(corrected_last) and validate_name(corrected_last).valid

        if first_ok or last_ok:
            new_first = corrected_first if first_ok else (state.get("first_name") or "").strip()
            new_last = corrected_last if last_ok else (state.get("last_name") or "").strip()
            logger.info(LOG_NAME_CORRECTED, extra={"new_first": new_first, "new_last": new_last})

            attempts = (state.get("name_confirm_attempts") or 0) + 1
            if attempts >= MAX_NAME_CONFIRM_ATTEMPTS:
                logger.warning(LOG_NAME_CONFIRM_EXHAUST)
                return self.signal_escalate(
                    state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
                )
            new_state = {
                **state,
                "first_name": new_first,
                "last_name": new_last,
                "name_confirm_attempts": attempts,
            }
            return self._deliver_name_readback(new_state)

        # ── OUTCOME 3: bare no ───────────────────────────────────────────────
        if name_conf_raw == "no":
            attempts = (state.get("name_confirm_attempts") or 0) + 1
            if attempts >= MAX_NAME_CONFIRM_ATTEMPTS:
                logger.warning(LOG_NAME_CONFIRM_EXHAUST)
                return self.signal_escalate(
                    state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
                )
            ask = self.ask_member(state, pick(NAME_CORRECTION_PROMPTS))
            ask["awaiting_slot"] = _NAME_CORRECTION_SLOT
            ask["name_confirm_attempts"] = attempts
            return ask

        # ── OUTCOME 4: no "yes" + no inline correction → ask for correction ────
        # Re-delivering the readback is only ever useful after a "yes".  Any other
        # response (ambiguous, mis-recognised rejection) has the same right next
        # step: ask what the correct name is.
        attempts = (state.get("name_confirm_attempts") or 0) + 1
        if attempts >= MAX_NAME_CONFIRM_ATTEMPTS:
            logger.warning(LOG_NAME_CONFIRM_EXHAUST)
            return self.signal_escalate(
                state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
            )
        ask = self.ask_member(state, pick(NAME_CORRECTION_PROMPTS))
        ask["awaiting_slot"] = _NAME_CORRECTION_SLOT
        ask["name_confirm_attempts"] = attempts
        return ask

    async def _collect_name_correction(self, state: State, messages: list, last_user: str) -> dict:
        """
        The member said bare 'no' to the readback. We asked for the correct name.
        Extract the corrected name, then re-deliver the readback with that name.
        """
        last_agent = _last_assistant_msg(messages)
        attempt_count = self.get_slot(_NAME_CORRECTION_SLOT).attempt_count

        result = await extract_name_confirmation(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/name_confirmation.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            pending_slots=[s for s in IDENTITY_SLOT_ORDER if not str(state.get(s) or "").strip()],
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}
        corrected_first_raw = extracted.get("first_name", "")
        corrected_last_raw = extracted.get("last_name", "")

        corrected_first = normalize_name(corrected_first_raw) if corrected_first_raw else ""
        corrected_last = normalize_name(corrected_last_raw) if corrected_last_raw else ""
        first_ok = bool(corrected_first) and validate_name(corrected_first).valid
        last_ok = bool(corrected_last) and validate_name(corrected_last).valid

        name_conf_raw = extracted.get("name_confirmed", "")

        if first_ok or last_ok:
            new_first = corrected_first if first_ok else (state.get("first_name") or "").strip()
            new_last = corrected_last if last_ok else (state.get("last_name") or "").strip()
            logger.info(
                LOG_NAME_CORRECTED,
                extra={"new_first": new_first, "new_last": new_last, "source": "correction_slot"},
            )
            # name_confirm_attempts was already incremented when we entered the
            # correction slot — do not increment again.
            new_state = {**state, "first_name": new_first, "last_name": new_last}
            return self._deliver_name_readback(new_state)

        # LLM returned name_confirmed="yes" (user repeated the name that was just
        # read back, which the model interprets as confirmation rather than a fresh
        # correction).  Accept the current state names and re-deliver the readback
        # so the member can confirm explicitly on the next turn.
        if name_conf_raw == "yes":
            current_first = (state.get("first_name") or "").strip()
            current_last = (state.get("last_name") or "").strip()
            if current_first and current_last:
                logger.info(
                    LOG_NAME_CORRECTED,
                    extra={
                        "new_first": current_first,
                        "new_last": current_last,
                        "source": "correction_slot_reconfirm",
                    },
                )
                return self._deliver_name_readback(state)

        # Nothing extractable — retry the "what is the correct name?" question.
        from agent.slots.types import SlotType

        self.slot_fail(_NAME_CORRECTION_SLOT)
        if self.get_slot(_NAME_CORRECTION_SLOT).is_exhausted():
            logger.warning(LOG_NAME_CONFIRM_EXHAUST)
            return self.signal_escalate(
                state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
            )
        ctx = ConversationContext.from_state(state)
        retry_msg = await self._generate_slot_retry_response(
            state,
            _NAME_CORRECTION_SLOT,
            ctx,
            messages,
            guard="RETRY",
            decision=result,
            slot_type=SlotType.FULL_NAME,
        )
        retry = self.ask_member(state, retry_msg)
        retry["awaiting_slot"] = _NAME_CORRECTION_SLOT
        return retry

    async def _name_confirmed_with_followup(self, state: State, messages: list, followup_query: str) -> dict:
        """Name confirmed with a side question — address the question then ask for member_id.

        Mirrors the FOLLOWUP_RESPOND path in _collect_slot._handle_answered_followup.
        Called when event_type==ANSWERED_WITH_FOLLOWUP on the name_confirmed turn.
        """
        from agent.responses.builder import build_transition_prompt
        from agent.slots.types import SlotType

        first = (state.get("first_name") or "").strip()
        last = (state.get("last_name") or "").strip()

        self.slot_ok("first_name", first)
        self.slot_ok("last_name", last)

        ctx = ConversationContext.from_state(state)
        ctx.update_caller_name(first)
        state = {
            **state,
            "first_name": first,
            "last_name": last,
            "name_confirmed": True,
            "name_confirm_attempts": 0,
            "conversation_context": ctx.to_dict(),
        }

        next_slot = next((s for s in IDENTITY_SLOT_ORDER if not str(state.get(s) or "").strip()), None)

        if next_slot is None:
            # All identity slots already present — go straight to lookup
            collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}
            call_intent = state.get("call_intent", "")
            lookup_result = await self._finish_after_identity(state, collected, messages, call_intent, None)
            if isinstance(lookup_result, dict) and "name_confirmed" not in lookup_result:
                lookup_result["name_confirmed"] = True
                lookup_result["name_confirm_attempts"] = 0
            return lookup_result

        next_slot_label = next_slot.replace("_", " ")
        followup_msg = await self._generate_slot_retry_response(
            state,
            _NAME_CONFIRM_SLOT,
            ctx,
            messages,
            guard="FOLLOWUP_RESPOND",
            session_context={"followup_query": followup_query},
            extracted_this_turn=f"{first} {last}",
            next_slot_label=next_slot_label,
            will_append_ask=True,
        )
        slot_type = SlotType.MEMBER_ID if next_slot == "member_id" else SlotType.DOB
        next_ask = build_transition_prompt(slot_type, ctx)
        combined_msg = followup_msg.rstrip() + " " + next_ask

        result = self.ask_member(state, combined_msg)
        result["awaiting_slot"] = next_slot
        result["name_confirmed"] = True
        result["name_confirm_attempts"] = 0
        result["first_name"] = first
        result["last_name"] = last
        result["conversation_context"] = ctx.to_dict()
        return result

    async def _name_confirmed_proceed(self, state: State, messages: list) -> dict:
        """
        Mark the name confirmed and continue identity collection.

        Normal first-time flow: member_id / dob are still empty, so deliver the
        next-slot transition prompt and pause (is_interrupt=True) so the next
        human turn is the real member_id/dob answer. We must not re-enter run()
        with is_interrupt=False on this path, because that reprocesses the same
        stale "yes": run() would fire a second extraction LLM call (classified
        against member_id) and the pipeline would treat the "yes" as a non-answer,
        firing a recovery-message LLM call — so the member hears a retry prompt
        instead of the clean member_id ask.

        Name-only partial re-ask: member_id AND dob were retained, so there is no
        empty identity slot to ask. Re-asking would produce a spurious Member-ID
        prompt, so instead proceed straight to the Salesforce lookup with the
        corrected name. (No stale-"yes" hazard here: with every slot filled, the
        pipeline has nothing to misclassify the "yes" against.)
        """
        from agent.responses.builder import build_transition_prompt
        from agent.slots.types import SlotType

        first = (state.get("first_name") or "").strip()
        last = (state.get("last_name") or "").strip()

        # Persist confirmed names so the gate never re-fires.
        self.slot_ok("first_name", first)
        self.slot_ok("last_name", last)

        ctx = ConversationContext.from_state(state)
        ctx.update_caller_name(first)
        state = {
            **state,
            "first_name": first,
            "last_name": last,
            "name_confirmed": True,
            "name_confirm_attempts": 0,
            "conversation_context": ctx.to_dict(),
        }

        # Next identity slot to collect (member_id, then dob), or None if every
        # identity slot is already present (name-only partial re-ask).
        next_slot = next((s for s in IDENTITY_SLOT_ORDER if not str(state.get(s) or "").strip()), None)

        if next_slot is None:
            # All identity slots present → re-run the lookup with the corrected
            # name rather than re-asking an already-known slot.
            collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}
            call_intent = state.get("call_intent", "")
            result = await self._finish_after_identity(state, collected, messages, call_intent, None)
            # Persist the just-confirmed name on the RETURNED dict. On success
            # _finish_after_identity returns the post-lookup interrupt
            # (relationship / phone) or the COMPLETE signal — neither carries
            # name_confirmed, so without this the gate would re-fire the read-back
            # on the next (post-lookup) turn. If the lookup instead returned a
            # re-ask that deliberately set name_confirmed (full restart, or a
            # fresh name mismatch → False), respect that value.
            if isinstance(result, dict) and "name_confirmed" not in result:
                result["name_confirmed"] = True
                result["name_confirm_attempts"] = 0
            return result

        slot_type = SlotType.MEMBER_ID if next_slot == "member_id" else SlotType.DOB
        msg = build_transition_prompt(slot_type, ctx)

        result = self.ask_member(state, msg)
        result["awaiting_slot"] = next_slot
        result["name_confirmed"] = True
        result["name_confirm_attempts"] = 0
        result["first_name"] = first
        result["last_name"] = last
        result["conversation_context"] = ctx.to_dict()
        return result

    # -------------------------------------------------------------------------
    # Private helpers (state loading + final signal)
    # -------------------------------------------------------------------------

    def _member_record_from_state(self, state: State) -> dict:
        """Reconstruct a minimal member record from already-verified state fields."""
        return {k: state.get(k, "") for k in ["phone_number", "zip_code", "fax", "email", "relationship"]}

    def _signal_verified(self, state: State, collected: dict, member_record: dict | None) -> dict:
        """Emit COMPLETE signal with all verified identity fields as context updates."""
        context_updates = {"member_status_verify": True, "verification_restart_index": 0, **collected}
        if member_record:
            for field in ["zip_code", "phone_number", "fax", "email", "relationship"]:
                if val := member_record.get(field):
                    context_updates[field] = val
            # Pass prefetched benefits fields into state so benefits_agent can skip
            # its own Salesforce call. Fields are only written if non-empty strings
            # to avoid overwriting existing state values with empty placeholders.
            for field in [
                "individual_deductible",
                "family_deductible",
                "coinsurance_percent",
                "individual_oop_max",
                "family_oop_max",
            ]:
                val = member_record.get(field, "")
                if val:
                    context_updates[field] = val
        logger.info(LOG_VERIFIED)
        result = self.signal_complete(
            state,
            # message=(
            #     ""
            #     if state.get("call_intent") in ("provider_services", "claim_services")
            #     else random.choice(VERIFIED_MSG_TEMPLATES).format(first_name=collected["first_name"])
            # ),
            message="",
            resolved_intents=["verification"],
            context_updates=context_updates,
            reasoning="Identity verified — routing to domain agent",
        )

        # ── Mid-call intent switch dispatch ──────────────────────────────────
        # follow_up stages a new intent via reset_for_new_intent, which sets
        # pending_intent. On successful re-verification, route straight to that
        # intent's domain node and consume pending_intent. First-ever verification
        # has no pending_intent → keep next_node="orchestrator" so the fast-path
        # routes by call_intent (existing behavior).
        pending = (state.get("pending_intent") or "").strip()
        if pending:
            result["pending_intent"] = ""  # consumed
            domain_node = _PENDING_INTENT_NODE.get(pending)
            if domain_node:
                result["next_node"] = domain_node
                logger.info(
                    "verification: pending_intent dispatch",
                    extra={"pending_intent": pending, "next_node": domain_node},
                )
        return result


async def verification_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await VerificationAgent.from_state(state).execute(state)
