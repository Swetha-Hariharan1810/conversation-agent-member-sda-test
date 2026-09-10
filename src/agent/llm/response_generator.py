"""
response_generator.py — LLM 2: natural recovery response generation.

Called only on recovery turns. Uses Gemini (get_routing_llm).
Input is intentionally minimal. Output is one spoken sentence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from agent.llm.config import get_generation_llm
from agent.llm.redaction import mask_confirmed
from agent.logger import get_logger
from agent.utils import build_generation_prompt, build_history

logger = get_logger(__name__)

_SLOT_LABELS: dict[str, str] = {
    "first_name": "first name",
    "last_name": "last name",
    "member_id": (
        "Member ID — Must begin with m followed by 6 digit "
        "(you can find this on your insurance card — begin with m followed by 6 digits)"
    ),
    "dob": "date of birth — Must include year, month, and day",
    "relationship": "whether they are the subscriber or dependent",
    "phone_confirmed": "phone number on file — yes or no",
    "phone_confirmation": "phone number on file — yes or no",
    "caller_role": "relationship to the plan",
    "provider_type": (
        "type of provider they are looking for "
        "(accepted values are Primary Care Physicians, Pediatricians,"
        "Cardiologists, Dermatologists, and Orthopedic Specialists)"
    ),
    "zip_code": "five-digit ZIP code",
    "delivery_method": "fax or email",
    "intent": "what they need help with today — ask openly, never list options",
    "topic": "what they need help with",
    "reference_number": "reference number — should be 8 digits",
    "upload_method": (
        "how they want to provide the medical records for this adjustment — "
        "we need a complete copy (visit notes, test results, treatment history related to the claim) — "
        "options: upload themselves via secure link, have their doctor send them, "
        "or have us contact their provider on their behalf"
    ),
    "upload_consent": "whether they want to receive a secure upload link via email (yes or no)",
    "personal_guide_consent": "yes or no — whether they want a Personal Guide to "
    "contact their provider and request the medical records on their behalf",
    "email": "correct email address for the upload link",
    "notification_method": "preferred notification channel — SMS or email",
    "phone": "correct phone number",
    "n2_notification_method": "preferred channel for claim progress updates — SMS or email",
    "timeline_question": (
        "whether they have questions about the timeline — "
        "say yes to hear it, no to skip, or ask their question directly"
    ),
    # SSN fallback slots
    "ssn_ask": "whether they have their SSN available — yes or no",
    "ssn": (
        "Social Security Number — 9 digits in the format XXX-XX-XXXX "
        "(may be spoken digit by digit or given as a full number)"
    ),
}

# ── Recovery guard labels ────────────────────────────────────────────────────
# These are Python-internal routing labels passed to generate_recovery_message().
# They are NOT LLM extraction outputs (see llm/schema.py EventType for those).
#
# Label            | Produced by              | Meaning
# -----------------|--------------------------|-----------------------------------
# "RETRY"          | _collect_slot            | Genuine failed answer — attempt counted
# "CLARIFY"        | _collect_slot            | First AMBIGUOUS turn — no attempt cost;
#                  |                          | ask caller to repeat more clearly
# "CORRECTION"     | _generate_correction_ack | Caller corrected a confirmed slot
# "INTERRUPTION"   | guards.py                | Caller switched topic mid-collection
# "OFFTOPIC_AGENT" | guards.py                | Wrong-agent topic — steer back
# "OFFTOPIC"       | guards.py (fallback)     | Legacy alias for OFFTOPIC_AGENT
# "FOLLOWUP_ANSWER"| _collect_slot (Phase 4)  | Slot confirmed + side question that is
#                  |                          | answerable from Confirmed values now
# "FOLLOWUP_PARK"  | _collect_slot (Phase 4)  | Slot confirmed + side question parked
#                  |                          | for later in the call — acknowledge only
# "FOLLOWUP_DECLINE"| _collect_slot (Phase 4) | Slot confirmed + side question we cannot
#                  |                          | answer — acknowledge and move on
# "CORRECTION_ACK" | _handle_answered_followup| Slot confirmed + correction applied,
#                  | (Phase 2 hygiene)        | no side question — acknowledge both
#
# For the FOLLOWUP_* and CORRECTION_ACK labels the model must NOT ask for any
# slot — Python appends the next static ask (or a detour ask) after the
# generated sentence.
#
# The distinction between RETRY and CLARIFY matters:
#   RETRY   → attempt_count was just incremented; LLM should re-ask firmly
#   CLARIFY → attempt_count unchanged; LLM should re-ask gently, no implication
#             that the caller did anything wrong
_FALLBACKS: dict[str, str] = {
    "INTERRUPTION": "Of course — and I still need your {slot_label}.",
    "OFFTOPIC": "Let me finish this first — your {slot_label}?",
    "RETRY": "Could you try your {slot_label} once more?",
    "OFFTOPIC_AGENT": "Let's stay focused — could I get your {slot_label}?",
    "CORRECTION": "Got it — I've updated that. Now could I get your {slot_label}?",
    "CLARIFY": (
        "I just want to make sure I have that right — \ncould you say your {slot_label} one more time for me?"
    ),
    "ANSWERED_WITH_FOLLOWUP": "Got that — and to confirm, you said {slot_label}.",
    "FOLLOWUP_ANSWER": "Got it — {slot_label} noted.",
    "FOLLOWUP_RESPOND": "Got it — I'll keep that in mind.",
    "FOLLOWUP_PARK": "Got it — and I'll come back to your question shortly.",
    "CORRECTION_ACK": "Got it — I've updated your {slot_label}.",
}

# Guards fired AFTER this turn's value was captured — the model must never
# ask for or re-confirm a slot on these turns; Python appends the next ask.
_POST_CAPTURE_GUARDS = ("FOLLOWUP_ANSWER", "FOLLOWUP_RESPOND", "FOLLOWUP_PARK", "CORRECTION_ACK")

_COLLECTING_NOTHING = "(nothing — this turn's value was captured; do not ask for or re-confirm any slot)"


def _tone_hint(attempt: int) -> str:
    """Coarse Python-derived tone label — raw attempt counts never reach LLM 2."""
    if attempt == 0:
        return "first ask"
    if attempt <= 2:
        return "gentle retry"
    return "patient retry"


# ── Output sanitizer (Phase 2: single-ask invariant) ─────────────────────────
# Spoken phrasings the generation LLM uses for each slot, beyond the plain
# slot name with underscores replaced. Used only for fuzzy ask-detection in
# sanitize_generated — not for rendering.
SLOT_ASK_SYNONYMS: dict[str, tuple[str, ...]] = {
    "dob": ("date of birth", "birth date", "birthdate"),
    "member_id": ("member id", "member id number", "member number"),
    "first_name": ("first name",),
    "last_name": ("last name",),
    "zip_code": ("zip code", "zip"),
    "phone": ("phone number",),
    "phone_confirmed": ("phone number",),
    "phone_confirmation": ("phone number",),
    "email": ("email address",),
    "reference_number": ("reference number",),
    "notification_method": ("notification channel", "notification method"),
    "delivery_method": ("delivery method", "fax or email"),
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _slot_match_terms(name_or_label: str) -> tuple[str, ...]:
    """Fuzzy-match terms for a slot: its name with _ → space (label qualifiers
    after an em-dash dropped) plus any SLOT_ASK_SYNONYMS entries."""
    key = (name_or_label or "").strip().lower()
    base = key.split("—")[0].strip().replace("_", " ")
    terms = {base} if base else set()
    terms.update(SLOT_ASK_SYNONYMS.get(key, ()))
    return tuple(terms)


def _mentions(sentence: str, terms: Sequence[str]) -> bool:
    lowered = sentence.lower()
    return any(t in lowered for t in terms)


def sanitize_generated(
    text: str,
    *,
    guard: str,
    next_slot_label: str | None = None,
    confirmed_labels: Sequence[str] = (),
    will_append_ask: bool = False,
    fallback_slot_label: str = "",
) -> str:
    """Enforce the single-ask invariant on LLM-2 output (Bug A).

    - Any sentence that asks for (contains "?" and fuzzy-matches) a slot in
      ``confirmed_labels`` is stripped — the model must never re-ask a
      confirmed slot.
    - When ``will_append_ask`` is True (Python appends _next_slot_ask after
      this text), sentences mentioning ``next_slot_label`` and any trailing
      question sentences are also stripped, so the appended ask is the one
      and only ask in the combined utterance.
    - If sanitization empties the text, the guard's _FALLBACKS entry is
      substituted (formatted with ``fallback_slot_label``).

    Every strip is logged at INFO with the guard and dropped sentence for
    eval visibility.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]
    confirmed_terms = [_slot_match_terms(label) for label in confirmed_labels]

    kept: list[str] = []
    for sentence in sentences:
        if "?" in sentence and any(_mentions(sentence, terms) for terms in confirmed_terms):
            logger.info("sanitize_generated: stripped confirmed-slot re-ask [guard=%s]: %r", guard, sentence)
            continue
        # FOLLOWUP_DECLINE post-capture: also strip statement-form slot
        # back-references that don't use "?". Pattern: "... but I can get
        # your Member ID for you". Truncate at " but " when what follows
        # mentions a confirmed slot; if the whole sentence is a slot mention
        # and we are in a post-capture guard, drop it entirely.
        if guard == "FOLLOWUP_DECLINE" and confirmed_terms:
            lower = sentence.lower()
            truncated = sentence
            for conj in (" but ", " however ", " although "):
                idx = lower.find(conj)
                if idx != -1:
                    tail = sentence[idx + len(conj) :]
                    if any(_mentions(tail, terms) for terms in confirmed_terms):
                        truncated = sentence[:idx].rstrip(" ,;")
                        logger.info(
                            "sanitize_generated: truncated slot back-ref clause [guard=%s]: %r → %r",
                            guard,
                            sentence,
                            truncated,
                        )
                        break
            kept.append(truncated if truncated.strip() else None)
            if truncated.strip():
                kept[-1] = truncated
            else:
                kept.pop()
                logger.info("sanitize_generated: dropped empty sentence after truncation [guard=%s]", guard)
            continue
        kept.append(sentence)

    if will_append_ask:
        if next_slot_label:
            next_terms = _slot_match_terms(next_slot_label)
            remaining = []
            for sentence in kept:
                if _mentions(sentence, next_terms):
                    logger.info(
                        "sanitize_generated: stripped next-slot mention [guard=%s]: %r", guard, sentence
                    )
                    continue
                remaining.append(sentence)
            kept = remaining
        while kept and kept[-1].rstrip().endswith("?"):
            logger.info("sanitize_generated: stripped trailing question [guard=%s]: %r", guard, kept[-1])
            kept.pop()

    result = " ".join(s.strip() for s in kept).strip()
    if not result:
        template = _FALLBACKS.get(guard, "Got it.")
        result = template.format(slot_label=fallback_slot_label or "that")
        logger.info("sanitize_generated: text emptied — substituting %s fallback", guard)
    return result


def _render_payload(
    *,
    slot_name: str,
    attempt: int,
    guard: str,
    last_messages: list[dict],
    slot_label_override: str | None = None,
    confirmed_slots: dict | None = None,
    user_utterance: str | None = None,
    extracted_value: str | None = None,
    followup_query: str | None = None,
    ask_for_new_value: bool = False,
    allow_followup_event: bool = False,
) -> str:
    """Render the LLM-2 user payload. Pure function — unit-testable without an LLM."""
    history_text = "\n".join(build_history(last_messages, n=4))

    # Use the live prompt text when provided (dynamic slots such as relationship
    # and phone_confirmed whose options are only known at runtime from SF).
    # Fall back to the static label dict for fixed slots.
    slot_label = slot_label_override or _SLOT_LABELS.get(
        slot_name,
        (slot_name or _SLOT_LABELS["intent"]).replace("_", " "),
    )
    # Post-confirmation guards with a captured value: nothing is being
    # collected this turn — the real label would invite a spurious re-ask.
    if guard in _POST_CAPTURE_GUARDS and extracted_value is not None:
        slot_label = _COLLECTING_NOTHING

    content_lines = [
        "Conversation:",
        history_text,
        "",
    ]
    if user_utterance:
        content_lines.append(f"Caller just said: {user_utterance}")
    content_lines += [
        f"Collecting: {slot_label}",
        f"Tone:       {_tone_hint(attempt)}",
    ]
    if confirmed_slots is not None and len(confirmed_slots) > 0:
        # Centralized masking: whatever a call site passes, masked slots (and
        # anything member_id/dob-shaped) render as "on file" — no site can leak.
        filled = ", ".join(f"{k}={v}" for k, v in mask_confirmed(confirmed_slots).items())
        content_lines.append(f"Confirmed:  {filled}")
    elif confirmed_slots is not None:
        content_lines.append("Confirmed:  nothing yet")
    if extracted_value is not None:
        content_lines.append(f"Extracted this turn: {extracted_value}")
    if followup_query:
        content_lines.append(f"Followup: {followup_query}")
    if ask_for_new_value:
        content_lines.append("Ask for new value: yes")
    _event_guards = (
        "CORRECTION",
        "CORRECTION_ACK",
        "CLARIFY",
        "OFFTOPIC_AGENT",
        "FOLLOWUP_ANSWER",
        "FOLLOWUP_RESPOND",
        "FOLLOWUP_PARK",
    )
    if allow_followup_event:
        _event_guards = _event_guards + ("ANSWERED_WITH_FOLLOWUP",)
    if guard in _event_guards:
        content_lines.append(f"Event:      {guard}")

    return "\n".join(content_lines)


async def generate_recovery_message(
    *,
    slot_name: str,
    attempt: int,
    guard: str,
    last_messages: list[dict],
    slot_label_override: str | None = None,
    caller_name: str | None = None,
    confirmed_slots: dict | None = None,
    user_utterance: str | None = None,
    extracted_value: str | None = None,
    followup_query: str | None = None,
    ask_for_new_value: bool = False,
    allow_followup_event: bool = False,
) -> str:
    """
    Generate a natural recovery response via LLM 2 (Gemini).

    guard: "CORRECTION" | "CLARIFY" | "INTERRUPTION" | "OFFTOPIC" | "RETRY" | "OFFTOPIC_AGENT"
    last_messages: recent conversation history (role/content dicts); up to 8 messages used.
    caller_name: caller's first name if already confirmed, for personalisation.
    confirmed_slots: dict of slot names → confirmed values for this session;
        masked centrally here (see llm.redaction) before reaching the LLM.
    user_utterance: the caller's most recent utterance, so the LLM knows what was said.
    extracted_value: value successfully extracted this turn, if any.
    followup_query: the caller's side question this turn (FOLLOWUP_* guards only).
    ask_for_new_value: FOLLOWUP_ANSWER update detours — the sentence must end
        by asking for the new value (renders "Ask for new value: yes").

    attempt is consumed in Python only (coarse Tone: hint); the raw count is
    never forwarded to the LLM.

    Falls back to static string on any exception.
    """
    user_content = _render_payload(
        slot_name=slot_name,
        attempt=attempt,
        guard=guard,
        last_messages=last_messages,
        slot_label_override=slot_label_override,
        confirmed_slots=confirmed_slots,
        user_utterance=user_utterance,
        extracted_value=extracted_value,
        followup_query=followup_query,
        ask_for_new_value=ask_for_new_value,
        allow_followup_event=allow_followup_event,
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_generation_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content=build_generation_prompt(guard)),
                HumanMessage(content=user_content),
            ]
        )
        text = (response.content or "").strip()
        if text:
            return text
    except Exception:
        logger.exception("generate_recovery_message: LLM 2 failed — using fallback")

    fallback_label = slot_label_override or _SLOT_LABELS.get(
        slot_name,
        (slot_name or _SLOT_LABELS["intent"]).replace("_", " "),
    )
    return _FALLBACKS.get(guard, "Could you try again?").format(slot_label=fallback_label)
