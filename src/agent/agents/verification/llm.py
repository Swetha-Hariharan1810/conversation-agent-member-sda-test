"""
verification_llm.py — LLM extraction for identity verification.

Public API:
    extract_verification_decision(llm, system_prompt, awaiting_slot,
                                  last_agent_message, last_user_message,
                                  confirmed_slots, attempt, recent_messages)
        → WorkerResult

Extracted slot values live in result.extracted:
  first_name, last_name, member_id, dob, relationship, phone_confirmation

Corrections live in result.corrections (dict[str, str]).
"""

from __future__ import annotations

import re as _re

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import SsnFallbackResult, WorkerResult
from agent.logger import get_logger
from agent.utils import build_extraction_prompt

logger = get_logger(__name__)


async def extract_verification_decision(
    llm,
    system_prompt: str,
    awaiting_slot: str,
    last_agent_message: str,
    last_user_message: str,
    *,
    confirmed_slots: dict | None = None,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract identity slots from the latest user utterance.

    confirmed_slots: already-confirmed slot values to include as context so
        the LLM can classify corrections for slots it has seen before.
    pending_slots: slots still to be collected later this call, so the LLM
        can classify parkable follow-up questions.
    attempt: how many collection attempts have been made for awaiting_slot.
    recent_messages: recent conversation turns (dicts with "role"/"content")
        passed through to build_worker_input for history context.

    Falls back to an empty WorkerResult on any exception — the slot
    collection loop handles the missing values gracefully.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot=awaiting_slot,
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=confirmed_slots,
        pending_slots=pending_slots,
        attempt=attempt,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        return result
    except Exception as _exc:
        _code = getattr(_exc, "code", None) or getattr(getattr(_exc, "error", None), "code", None)
        if _code == "content_filter":
            logger.warning(
                "extract_verification_decision: Azure content filter"
                "blocked the request (jailbreak pattern in caller utterance)"
            )
        else:
            logger.exception("extract_verification_decision: LLM extraction failed")
        return WorkerResult()


async def extract_name_confirmation(
    llm,
    system_prompt: str,
    *,
    last_agent_message: str,
    last_user_message: str,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract the member's response to the name readback.

    Uses name_confirmation.md which handles three outcomes:
      - name_confirmed="yes"               → member confirmed the spelled name
      - first_name / last_name extracted   → inline correction provided
      - name_confirmed="no", no names      → bare no, correction needed separately

    Falls back to an empty WorkerResult on any exception.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot="name_confirmed",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=None,
        pending_slots=pending_slots,
        attempt=attempt,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        return result
    except Exception:
        logger.exception("extract_name_confirmation: LLM extraction failed")
        return WorkerResult()


_SSN_FALLBACK_PROMPT: str | None = None


def _get_ssn_fallback_prompt() -> str:
    global _SSN_FALLBACK_PROMPT
    if _SSN_FALLBACK_PROMPT is None:
        _SSN_FALLBACK_PROMPT = build_extraction_prompt("extraction/ssn_fallback.md")
    return _SSN_FALLBACK_PROMPT


async def extract_ssn_decision(
    llm,
    *,
    stage: str,
    last_agent_message: str,
    last_user_message: str,
    recent_messages: list | None = None,
) -> "SsnFallbackResult":
    """
    Run one LLM call to extract the caller's intent and SSN from their
    response during the SSN fallback flow.

    stage:
        "ssn_ask"        — agent asked "Do you have the SSN?"
        "ssn_collecting" — agent asked "Please provide your SSN."

    Returns SsnFallbackResult with ssn_intent and optionally ssn.
    Falls back to SsnFallbackResult(ssn_intent="ambiguous") on any error.
    """
    from agent.llm.schema import SsnFallbackResult

    system_prompt = _get_ssn_fallback_prompt()

    messages = build_worker_input(
        system_prompt,
        awaiting_slot=stage,
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=None,
        pending_slots=None,
        attempt=0,
        recent_messages=recent_messages,
    )
    try:
        result: SsnFallbackResult = await llm.with_structured_output(SsnFallbackResult).ainvoke(messages)
        return result
    except Exception:
        logger.exception("extract_ssn_decision: LLM extraction failed — using keyword fallback")
        return _ssn_keyword_fallback(stage, last_user_message)


# ---------------------------------------------------------------------------
# Keyword fallback — used when the LLM extractor throws an exception
# ---------------------------------------------------------------------------


_SSN_PATTERN = _re.compile(r"\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b")

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_SPOKEN_DIGIT_RE = _re.compile(
    r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b", _re.IGNORECASE
)

_YES_KW = frozenset({"yes", "yeah", "yep", "sure", "okay", "ok", "yup"})
_NO_KW = frozenset({"no", "nope", "nah"})
_NO_SSN_KW = (
    "don't have it",
    "dont have it",
    "don't have my ssn",
    "dont have my ssn",
    "neither",
    "don't know it",
    "dont know it",
    "can't access",
    "cant access",
    "lost it",
    "i can't",
    "i cant",
    "i don't",
    "i dont",
    "no ssn",
)


def _kw_extract_ssn(text: str) -> str:
    """Try to extract a 9-digit SSN from text, including spoken-word digits."""
    # Direct pattern match first
    m = _SSN_PATTERN.search(text)
    if m:
        digits = m.group(1) + m.group(2) + m.group(3)
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"

    # Convert spoken digits and retry
    def _replace(match):
        return _DIGIT_WORDS[match.group(0).lower()]

    converted = _SPOKEN_DIGIT_RE.sub(_replace, text)
    digits = _re.sub(r"\D", "", converted)
    if len(digits) == 9:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"

    return ""


def _ssn_keyword_fallback(stage: str, text: str) -> "SsnFallbackResult":
    """Deterministic keyword-based fallback when LLM extraction fails."""
    t = text.strip().lower()

    ssn = _kw_extract_ssn(text)
    if ssn:
        return SsnFallbackResult(ssn_intent="yes_with_ssn", ssn=ssn)

    if any(phrase in t for phrase in _NO_SSN_KW):
        return SsnFallbackResult(ssn_intent="no_ssn_available")

    if t in _NO_KW:
        return SsnFallbackResult(ssn_intent="no")

    if t in _YES_KW or any(kw in t for kw in _YES_KW):
        return SsnFallbackResult(ssn_intent="yes")

    return SsnFallbackResult(ssn_intent="ambiguous")
