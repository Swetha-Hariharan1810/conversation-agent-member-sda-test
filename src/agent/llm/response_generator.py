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
    # ── Offers and confirmations ─────────────────────────────────────────────
    # The caller was asked a question here, not asked for a value. The label is
    # the only thing the generation LLM has to return to, so it has to read as
    # that question: "benefits response" is not something anyone can be asked
    # for, and a model told to redirect to it reaches for the nearest askable
    # thing it knows about instead (the notification channel, off the system
    # prompt's list of what this line does).
    "benefits_response": (
        "whether they want to hear the benefits for office visits with the "
        "provider type they asked about — yes or no"
    ),
    "care_coach_response": (
        "whether they want us to send details about the free Care Coach Guides — yes or no"
    ),
    "name_confirmed": "whether the name just read back to them is correct — yes or no",
    "name_correction": "the correct name on the account",
    "zip_confirmed": "whether the ZIP code on file is still the right one — yes or no",
    "fax_confirmed": "whether the fax number on file is the right one — yes or no",
    "email_confirmed": "whether the email address on file is the right one — yes or no",
    "fax": "correct fax number",
    "fallback_claim_number": "claim number for the claim they are calling about",
    "fallback_dos_billed": ("date of service and billed amount for the claim — both are needed to locate it"),
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
# "FOLLOWUP_PARK"  | _collect_slot (Phase 4)  | Slot confirmed + an update another flow
#                  |                          | owns, carried to it — acknowledge only.
#                  |                          | Side questions never park.
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

# No slot is being collected at all — the caller is between steps, e.g. on
# "is there anything else?". Falling back to the intent label here would
# invent a collection step and send the caller back to a question that is
# not being asked.
_COLLECTING_NOTHING_PENDING = (
    "(nothing — no slot is being collected; do not ask for or re-confirm any "
    "slot, and return to the question already on the table)"
)


# ── Static fast path (no LLM 2 call) ─────────────────────────────────────────
# RETRY and CLARIFY are the only guards whose whole job is "ask the same slot
# again". On those turns there is nothing for the generation LLM to add — and
# a great deal for it to get wrong, since a hallucinated ask lands on a
# different slot and the caller answers the wrong question. Every other guard
# has real content to convey (an answer, an acknowledgement, a redirect) and
# always goes to the LLM.
_STATIC_ELIGIBLE_GUARDS = frozenset({"RETRY", "CLARIFY"})


# "Sorry, I didn't catch that" is a claim about hearing, and it is only true of
# a short, contentless turn — silence, "uh", "what". A caller who says a whole
# clear sentence was heard perfectly well; telling them otherwise is both false
# and, said twice running, insulting:
#
#     AI    Sorry, I didn't catch that — could you say your first name again?
#     User  Please check my claim status today.
#     AI    Sorry, I didn't catch that — could you say your first name again?
#
# Four words is the line. Below it a canned re-ask is honest and costs nothing;
# at or above it the turn has content the template cannot answer, and the
# generation LLM gets to respond to what was actually said.
_STATIC_RETRY_MAX_WORDS = 4


def needs_freeform_response(
    *,
    guard: str,
    decision: object | None = None,
    followup_query: str | None = None,
    extracted_value: str | None = None,
    user_utterance: str | None = None,
) -> bool:
    """Does this turn need a generated sentence, or will a static re-ask do?

    True  → call LLM 2 (Gemini).
    False → the caller gave a plain non-answer with nothing to acknowledge;
            ``build_retry_prompt`` says the same thing deterministically.

    The decision is driven by ``WorkerResult.needs_freeform_response``, set by
    the extraction LLM that already read the utterance — no extra call. It is
    overridden to True whenever there is content the static template cannot
    carry (a side question, a value to name back, or simply a sentence long
    enough to have been heard), and defaults to True when no extraction result
    was passed, so un-wired call sites keep the old always-generate behaviour.

    The length rule is a safety net under the model's flag, not a replacement
    for it: the flag is set by a model that can be wrong about its own output,
    and it was wrong on "Please check my claim status today" — a clear request
    that got "I didn't catch that" twice.
    """
    if guard not in _STATIC_ELIGIBLE_GUARDS:
        return True
    if followup_query:
        return True
    if extracted_value:
        return True
    if decision is None:
        return True
    # Safety net: content the caller supplied that a canned re-ask cannot
    # carry always wins over the model's flag, however it was set.
    if any(v for v in (getattr(decision, "corrections", None) or {}).values()):
        return True
    if (getattr(decision, "update_target", None) or "").strip():
        return True
    if (getattr(decision, "followup_query", None) or "").strip():
        return True
    if len((user_utterance or "").split()) >= _STATIC_RETRY_MAX_WORDS:
        return True
    flag = getattr(decision, "needs_freeform_response", None)
    if flag is None:
        return True
    return bool(flag)


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
    # Only ask-shaped phrases belong here. "claim status" and "status updates"
    # were listed too, and they are the caller's TOPIC, not a way of asking for
    # the notification channel — so on a claim-services call any question that
    # mentioned the reason for the call was stripped as a foreign-slot ask:
    #
    #   "I'm doing well, thanks — shall we carry on with your claim status?"
    #
    # went out as a decline instead. "sms or email" is the ask, and it is what
    # actually caught the hallucination these entries were added for.
    "notification_method": ("notification channel", "notification method", "sms or email"),
    "n2_notification_method": ("notification channel", "notification method", "sms or email"),
    "delivery_method": ("delivery method", "fax or email"),
    "benefits_response": ("office visit benefits", "benefits for office visits"),
    "care_coach_response": ("care coach", "health and wellness coach"),
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ── Clause-level trimming (an answer and an ask in ONE sentence) ─────────────
# The strips below drop a sentence whole. That is right when the sentence is
# nothing but an ask this turn may not make, and wrong when the model packed
# the answer the caller is owed into the same sentence:
#
#   "Of course, I can repeat that — I can only ask for your information,
#    not look it up, so could you tell me your first name?"
#
# One sentence, so dropping it loses the answer as well as the ask, the text
# empties, and a canned fallback goes out in its place — "Got it, I'll keep
# that in mind." in front of a re-ask that already said it better. Cutting at
# the clause boundary keeps the answer and drops only the ask.
_ASK_CLAUSE_SPLIT_RE = re.compile(r"\s*[—–]\s*|\s*;\s*|,?\s+\b(?:so|but|and then|then)\b\s+|,\s+")

# What makes a trailing clause an ASK rather than the rest of the answer.
_ASK_TAIL_RE = re.compile(
    r"\b(?:could|can|would|will|may|shall)\s+(?:you|i|we)\b"
    r"|\b(?:what|which|who|when|where|how)\b"
    r"|\b(?:tell|give|say|repeat|confirm|provide)\s+(?:me|it|that|us)\b"
    r"|\bis\s+that\s+(?:right|correct)\b",
    re.IGNORECASE,
)

# What disqualifies a LEAD — deliberately tighter than the tail test, since a
# lead is kept and a false positive there throws the answer away. Only
# second-person request forms and an opening interrogative count; an ordinary
# statement that happens to contain "can" or "repeat" ("I can repeat that",
# "I can only ask for your information") is answer, not ask.
_ASK_LEAD_RE = re.compile(
    r"\b(?:could|can|would|will|may|shall)\s+(?:you|i|we)\b"
    r"|^(?:what|which|who|when|where|how)\b"
    r"|\bplease\s+(?:tell|give|say|repeat|confirm|provide)\b",
    re.IGNORECASE,
)

# Below this, the lead is a filler opener ("Of course", "Sure thing") and
# keeping it alone says nothing — drop the sentence as before.
_MIN_LEAD_WORDS = 4


def _declarative_lead(sentence: str) -> str:
    """The answer half of a one-sentence "answer + ask", or "" if there is none.

    Cuts at the LAST clause boundary that still leaves a substantial,
    question-free lead, so as much of the answer survives as possible.
    """
    text = (sentence or "").strip()
    for match in reversed(list(_ASK_CLAUSE_SPLIT_RE.finditer(text))):
        lead = text[: match.start()].strip(" ,;—–")
        tail = text[match.end() :].strip()
        if not tail.endswith("?") or "?" in lead:
            continue
        if len(lead.split()) < _MIN_LEAD_WORDS:
            continue
        if not _ASK_TAIL_RE.search(tail) or _ASK_LEAD_RE.search(lead):
            continue
        return lead if lead.endswith((".", "!")) else lead + "."
    return ""


def _strip_ask(
    sentence: str, *, guard: str, what: str, warn: bool = False, keep_lead: bool = True
) -> list[str]:
    """Drop an ask this turn may not make, keeping any answer in front of it.

    Returns the sentences to keep in place of ``sentence`` — the declarative
    lead when there is one, nothing when the sentence was only an ask.

    ``keep_lead`` is False where the turn has no other ask coming: a trimmed
    lead would leave the caller with a statement and no question, and the
    guard's fallback — which always ends in one — is the better substitute.
    """
    log = logger.warning if warn else logger.info
    lead = _declarative_lead(sentence) if keep_lead else ""
    if lead:
        log(
            "sanitize_generated: trimmed %s, kept the answer [guard=%s]: %r -> %r",
            what,
            guard,
            sentence,
            lead,
        )
        return [lead]
    log("sanitize_generated: stripped %s [guard=%s]: %r", what, guard, sentence)
    return []


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


def _foreign_slot_terms(
    collecting_slot: str,
    exempt_slots: Sequence[str] = (),
) -> list[tuple[str, ...]]:
    """Match terms for every known slot OTHER than the ones this turn may name.

    ``exempt_slots`` are slots the turn is legitimately allowed to mention —
    the fields a correction acknowledgement reads back, for instance.

    A foreign term that overlaps an allowed slot's wording is dropped (either
    direction of containment, or a shared leading word): ``phone`` /
    ``phone_confirmed`` / ``phone_confirmation`` all say "phone number", and
    ``email_confirmed`` asks about an "email address" — stripping the
    legitimate ask would be worse than the hallucination we guard against.

    ``collecting_slot`` of "" means nothing is being collected, so every known
    slot is foreign: on that turn there is no ask the model is entitled to make.
    """
    allowed = {collecting_slot, *exempt_slots}
    own = tuple(t for name in allowed for t in _slot_match_terms(name) if t)
    own_heads = {t.split()[0] for t in own if t.split()}
    terms: list[tuple[str, ...]] = []
    for name in set(_SLOT_LABELS) | set(SLOT_ASK_SYNONYMS):
        if name in allowed:
            continue
        candidate = tuple(
            t
            for t in _slot_match_terms(name)
            if t and not any(t in o or o in t for o in own) and t.split()[0] not in own_heads
        )
        if candidate:
            terms.append(candidate)
    return terms


def sanitize_generated(
    text: str,
    *,
    guard: str,
    next_slot_label: str | None = None,
    confirmed_labels: Sequence[str] = (),
    will_append_ask: bool = False,
    fallback_slot_label: str = "",
    collecting_slot: str | None = None,
    exempt_slots: Sequence[str] = (),
    fallback_text: str = "",
    allow_empty: bool = False,
) -> str:
    """Enforce the single-ask invariant on LLM-2 output (Bug A).

    - Any sentence that asks for (contains "?" and fuzzy-matches) a slot in
      ``confirmed_labels`` is stripped — the model must never re-ask a
      confirmed slot.
    - When ``collecting_slot`` is given, any sentence that asks for a *different*
      known slot is stripped too, whether or not that slot was confirmed. This
      is the cross-slot hallucination guard: on a re-ask turn the model has
      been seen to drop the slot it was told to collect and ask for the next
      one instead ("...and your Member ID?" while last_name is still missing),
      which silently skips a required field. ``exempt_slots`` widens what the
      turn may name — the fields a correction acknowledgement reads back.
      ``collecting_slot=""`` says nothing is being collected at all, and every
      known slot is foreign; ``None`` turns the check off.
    - When ``will_append_ask`` is True (Python appends _next_slot_ask after
      this text), sentences mentioning ``next_slot_label`` and any trailing
      question sentences are also stripped, so the appended ask is the one
      and only ask in the combined utterance.
    - When ``will_append_ask`` is True the turn's question is guaranteed to
      come from Python, so a stripped sentence that also carried the caller's
      answer is trimmed at the clause boundary instead of dropped (see
      ``_declarative_lead``) and the answer survives the ask. Without an
      appended ask a trimmed lead would leave the caller with no question at
      all, so the sentence is dropped whole and the fallback speaks.
    - If sanitization empties the text, ``fallback_text`` is substituted when
      given, otherwise the guard's _FALLBACKS entry (formatted with
      ``fallback_slot_label``). A caller with no askable slot passes its own
      text — the _FALLBACKS templates all end in an ask. ``allow_empty``
      returns "" instead: a caller whose text is only ever PREFIXED to a turn
      that already speaks (BaseAgent's side-question net) must contribute
      nothing rather than a canned line the model never generated.

    Every strip is logged at INFO with the guard and dropped sentence for
    eval visibility.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]
    confirmed_terms = [_slot_match_terms(label) for label in confirmed_labels]
    foreign_terms = _foreign_slot_terms(collecting_slot, exempt_slots) if collecting_slot is not None else []

    kept: list[str] = []
    for sentence in sentences:
        if "?" in sentence and any(_mentions(sentence, terms) for terms in confirmed_terms):
            kept.extend(
                _strip_ask(
                    sentence,
                    guard=guard,
                    what="confirmed-slot re-ask",
                    keep_lead=will_append_ask,
                )
            )
            continue
        if "?" in sentence and any(_mentions(sentence, terms) for terms in foreign_terms):
            kept.extend(
                _strip_ask(
                    sentence,
                    guard=guard,
                    what=f"foreign-slot ask (collecting={collecting_slot!r})",
                    warn=True,
                    keep_lead=will_append_ask,
                )
            )
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
            trimmed = _strip_ask(kept[-1], guard=guard, what="trailing question")
            kept.pop()
            kept.extend(trimmed)
            if trimmed:
                break

    result = " ".join(s.strip() for s in kept).strip()
    if not result:
        if allow_empty:
            logger.info("sanitize_generated: text emptied — contributing nothing [guard=%s]", guard)
            return ""
        if fallback_text:
            result = fallback_text
        else:
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
    coming_up: list[str] | None = None,
) -> str:
    """Render the LLM-2 user payload. Pure function — unit-testable without an LLM."""
    history_text = "\n".join(build_history(last_messages, n=4))

    # Use the live prompt text when provided (dynamic slots such as relationship
    # and phone_confirmed whose options are only known at runtime from SF).
    # Fall back to the static label dict for fixed slots.
    if slot_label_override:
        slot_label = slot_label_override
    elif (slot_name or "").strip():
        slot_label = _SLOT_LABELS.get(slot_name, slot_name.replace("_", " "))
    else:
        slot_label = _COLLECTING_NOTHING_PENDING
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
    # Only when there is a value to name. An empty string rendered the line
    # with nothing after it, which reads as "nothing was captured" to a model
    # that is also being told "Collecting: (nothing — this turn's value was
    # captured)" — and the contradiction came back as a re-ask on a turn whose
    # whole contract was to ask for nothing.
    if extracted_value:
        content_lines.append(f"Extracted this turn: {extracted_value}")
    if followup_query:
        content_lines.append(f"Followup: {followup_query}")
    if coming_up:
        # What the call still has to cover. A side question about one of these
        # is answerable now — "you'll choose fax or email in a moment" — which
        # is why such a question no longer has to be deferred to the end.
        content_lines.append(f"Coming up: {', '.join(coming_up)}")
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
    coming_up: list[str] | None = None,
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
        coming_up=coming_up,
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
