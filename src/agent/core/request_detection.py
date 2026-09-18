"""
request_detection.py — deterministic fallback layer for cross-call request
detection (update / redo / replay), and the boundary where an extraction
result is reconciled with what the pipeline already knows. Fixes the
instability root cause: the extraction LLM intermittently drops the request
intent or its target, or labels a correction turn WAIT, and the whole
downstream routing hinges on it.

The LLM stays PRIMARY. This module never overrides a concrete LLM detection
with a different target — it only
  1. fills gaps  — the LLM reported no request but the caller's words plainly
     contain one of the covered request shapes; and
  2. supplies state — which spoken values are corrections, which the model
     cannot know and the pipeline can (see _split_corrections).
When neither the LLM nor the regex detects anything, behavior is unchanged.

The veto half of this module is gone. It existed to repair results whose
fields contradicted each other — a WAIT label beside an update target, an
ANSWERED beside a bare request, an ANSWERED_WITH_FOLLOWUP beside no question.
WorkerResult reports one intent now and derives the rest from it, so those
states have no representation to repair.

Slot patterns are DERIVED from SLOT_OWNERSHIP, not hand-written: every
registry key gets "update/change/correct my <label>" and "<label> changed /
is wrong / is different" coverage automatically, so a future registry entry
is covered the day it is added. SLOT_LABEL_ALIASES adds the spoken variants
("date of birth", "member number", "postal code"). Hand-written patterns are
reserved for phrasings that don't name the slot ("I moved" → zip_code,
"instead of fax" → redo delivery).

Precedence: update beats redo beats replay; a concrete slot target beats a
capability topic (updates are checked first and target canonical slot names).

Dependency-light on purpose: stdlib re / dataclasses / logging, the
dependency-free slot_ownership registry, and llm.schema, which is a leaf.
NEVER import from agents/ — the few cannot-provide negatives needed to stay out
of detect_cannot_provide's territory are duplicated below.

This module speaks to a WorkerResult through the methods it offers —
adopt_request, report_cannot_supply, split_corrections — rather than assigning
fields. Each one carries the precedence that decides whether a detection is
allowed to land at all, so the rule lives with the data instead of being
restated at every call site here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.core.followup_grounding import (
    is_grounded_followup,
    quotes_the_caller,
    recover_side_question,
)
from agent.core.slot_ownership import SLOT_OWNERSHIP
from agent.llm.schema import TurnIntent
from agent.utils import detect_cannot_provide

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedRequest:
    kind: str  # "update" | "redo" | "replay" | ""
    target: str  # canonical slot name or capability topic, "" when unknown
    matched: str  # the phrase that matched, for logging


# ── Slot label aliases ────────────────────────────────────────────────────────
# Spoken variants of registry slot names. A slot absent here still gets its
# default label (slot_name with underscores → spaces).

SLOT_LABEL_ALIASES: dict[str, list[str]] = {
    "dob": ["date of birth", "birthday", "birth date"],
    "zip_code": ["zip", "zip code", "postal code"],
    "member_id": ["member id", "member number"],
    "fax": ["fax", "fax number"],
    "email": ["email", "email address", "e-mail"],
    "notification_method": ["notification", "notification preference", "notification method"],
    "phone_number": ["phone", "phone number"],
}


def _slot_labels(slot: str) -> list[str]:
    labels = {slot.replace("_", " ").strip()}
    labels.update(SLOT_LABEL_ALIASES.get(slot, []))
    return [lbl for lbl in labels if lbl]


def _build_update_patterns() -> dict[str, list[re.Pattern]]:
    """Per-slot update patterns derived from the ownership registry.

    Longest label alternatives first so "zip code" wins over "zip" inside one
    slot's own alternation (cross-slot ambiguity is resolved by registry
    order in detect_request).
    """
    patterns: dict[str, list[re.Pattern]] = {}
    for slot in SLOT_OWNERSHIP:
        labels = sorted(_slot_labels(slot), key=len, reverse=True)
        alt = "|".join(re.escape(lbl) for lbl in labels)
        patterns[slot] = [
            # "update / change / correct / fix (my) <label>"
            re.compile(
                rf"\b(?:update|change|correct|fix)\s+(?:(?:my|the|that|your)\s+)?(?:{alt})\b",
                re.IGNORECASE,
            ),
            # "<label> changed / is wrong / is different / is incorrect"
            re.compile(
                rf"\b(?:{alt})\s+(?:has\s+changed|changed|is\s+(?:wrong|different|incorrect|not\s+right))\b",
                re.IGNORECASE,
            ),
            # "new <label>" — "I have a new zip", "there's a new email"
            re.compile(rf"\bnew\s+(?:{alt})\b", re.IGNORECASE),
        ]
    return patterns


_UPDATE_PATTERNS: dict[str, list[re.Pattern]] = _build_update_patterns()

# Hand-written ONLY for phrasings that never name the slot. "address" maps to
# zip_code (a moved member's postal address drives the provider search); the
# fixed-width lookbehind keeps "email address changed" with the email slot.
_UPDATE_PATTERNS_EXTRA: dict[str, list[re.Pattern]] = {
    "zip_code": [
        # "I moved" / "I relocated" and we-forms
        re.compile(r"\bi(?:'ve| have)?\s+(?:just\s+|recently\s+)?(?:moved|relocated)\b", re.IGNORECASE),
        re.compile(r"\bwe(?:'ve| have)?\s+(?:just\s+|recently\s+)?(?:moved|relocated)\b", re.IGNORECASE),
        re.compile(
            r"(?<!mail\s)\baddress\s+(?:has\s+changed|changed|is\s+(?:wrong|different|incorrect))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:update|change|correct)\s+(?:my\s+|the\s+)?(?:home\s+)?address\b",
            re.IGNORECASE,
        ),
        # "postal code / zip [... you have / on file ...] is off/wrong/incorrect"
        # The slot label may be separated from the status phrase by intervening
        # words like "you have", "on file", "in your system".
        re.compile(
            r"\b(?:zip(?:\s+code)?|postal\s+code)\b[^.?!]{0,40}\b(?:is|are)\s+"
            r"(?:off|wrong|incorrect|outdated|stale|not\s+right|no\s+longer\s+(?:right|correct|valid))\b",
            re.IGNORECASE,
        ),
    ],
}

# ── redo: re-perform a completed action with a changed parameter ──────────────
# All current redo phrasings concern re-dispatching the provider list, so the
# canonical capability topic is "delivery" (see CAPABILITY_REGISTRY).

_REDO_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), "delivery")
    for p in (
        # "(send|resend|email|fax) (it|that|the list) ... instead"
        r"\b(?:send|re-?send|email|fax)\s+(?:it|that|them|the\s+list)\b[^.?!]*\binstead\b",
        r"\bto\s+my\s+(?:email|fax)\s+instead\b",
        r"\bby\s+(?:email|fax)\s+instead\b",
        r"\binstead\s+of\s+(?:the\s+|my\s+)?(?:fax|email)\b",
        r"\buse\s+(?:the\s+other|a\s+different)\s+(?:method|way|one)\b",
        r"\bactually,?\s+(?:the\s+)?(?:email|fax)\s+(?:is\s+better|works\s+better|would\s+be\s+better|instead)\b",
        r"\bre-?send\s+(?:it|that|the\s+list)\b",
        r"\bsend\s+(?:it|that|the\s+list)\s+(?:again|one\s+more\s+time)\b",
    )
]

# ── replay: re-state information already given this call ─────────────────────

_REPLAY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:repeat|re-?read|go\s+over)\s+(?:my\s+|the\s+)?benefits\b", re.IGNORECASE), "benefits"),
    (re.compile(r"\bwhat\s+(?:were|are)\s+my\s+benefits\b", re.IGNORECASE), "benefits"),
    (re.compile(r"\b(?:my\s+|the\s+)?benefits\s+again\b", re.IGNORECASE), "benefits"),
    (re.compile(r"\bwhat\s+(?:exactly\s+)?did\s+you\s+send\b", re.IGNORECASE), "provider_list"),
    (re.compile(r"\bread\s+that\s+back\b", re.IGNORECASE), "provider_list"),
    # ZIP code replay — "can you tell me my ZIP code", "what's my ZIP", "read back my ZIP"
    (
        re.compile(
            r"\b(?:tell\s+me|read\s+(?:back|me)|what\s+(?:is|was|'?s))\s+"
            r"(?:my\s+)?(?:zip(?:\s+code)?|postal\s+code)\b",
            re.IGNORECASE,
        ),
        "zip_code",
    ),
    (
        re.compile(
            r"\b(?:zip(?:\s+code)?|postal\s+code)\s+again\b",
            re.IGNORECASE,
        ),
        "zip_code",
    ),
    # Claims-path replays (Phase 7): re-state the adjustment status.
    (
        re.compile(r"\bwhat(?:'s| is)\s+(?:happening|going on)\s+with\s+my\s+claim\b", re.IGNORECASE),
        "claim_status",
    ),
    (re.compile(r"\b(?:status|update)\s+(?:of|on)\s+my\s+claim\b", re.IGNORECASE), "claim_status"),
    (re.compile(r"\bwhen\s+will\s+i\s+hear\b[^.?!]*\bclaim\b", re.IGNORECASE), "claim_status"),
]

# ── Negative guard ────────────────────────────────────────────────────────────
# Cannot-provide statements must yield None — they route to the cannot-provide
# escalation, never to update/redo/replay. Deliberately DUPLICATED from
# agent.utils.detect_cannot_provide (only the phrasings that could co-occur
# with our positive patterns) to keep this module dependency-free.

_NEGATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+(?:do\s+not|don'?t)\s+have\b",
        r"\bi\s+don'?t\s+know\b",
        r"\bi\s+don'?t\s+(?:remember|recall)\b",
        r"\b(?:i\s+)?can'?t\s+(?:remember|recall|find)\b",
        r"\bi\s+(?:lost|misplaced)\b",
        r"\bi\s+never\s+(?:received|got)\b",
        r"\bdon'?t\s+have\s+(?:it|that|access)\b",
        r"\bnot\s+with\s+me\b",
        # Meta-questions about an update's timing/status are not requests —
        # "when will you update my zip?" is answered from the parked promise
        # (_match_promised_item), never re-detected as a fresh update.
        r"\bwhen\s+(?:will|are|is|do|does|can)\s+you\b",
        r"\bhave\s+you\s+(?:already\s+)?(?:updated|changed|sent|fixed)\b",
        r"\bdid\s+you\s+(?:already\s+)?(?:update|change|fix)\b",
    )
]


def detect_request(text: str | None) -> DetectedRequest | None:
    """Deterministic detection of a cross-call request in the caller's words.

    Returns None for plain answers, bare yes/no, wait-only phrases, and
    cannot-provide statements — anything that is not one of the covered
    request shapes. Update beats redo beats replay when multiple match.
    """
    if not text or not text.strip():
        return None
    t = re.sub(r"\s+", " ", text.strip().lower())
    if any(p.search(t) for p in _NEGATIVE_PATTERNS):
        return None

    # 1. updates — concrete slot targets, registry order breaks ties
    for slot, pats in _UPDATE_PATTERNS.items():
        for pat in pats + _UPDATE_PATTERNS_EXTRA.get(slot, []):
            if m := pat.search(t):
                return DetectedRequest(kind="update", target=slot, matched=m.group(0))

    # 2. redo
    for pat, topic in _REDO_PATTERNS:
        if m := pat.search(t):
            return DetectedRequest(kind="redo", target=topic, matched=m.group(0))

    # 3. replay
    for pat, topic in _REPLAY_PATTERNS:
        if m := pat.search(t):
            return DetectedRequest(kind="replay", target=topic, matched=m.group(0))

    return None


# ── WorkerResult reconciliation (fallback + veto, called after extraction) ────


# Schema field names the extractor sometimes writes INTO extracted{} instead of
# alongside it — seen in production as
#     "extracted": {"care_coach_response": "yes",
#                   "turn_target": "ID card", "turn_intent": "update"}
# extracted{} is slot name → caller value, and every consumer treats it that
# way: note_side_question joins its values into the "Extracted this turn:" line
# the generation LLM reads back ("yes, ID card, update"), and the slot
# pipelines index it by slot name. A schema key in there is never a slot, so it
# is dropped rather than spoken.
#
# These are the six names the model is given; no prompt mentions any other, so
# there is no other name for it to leak.
_RESERVED_RESULT_KEYS = frozenset(
    {
        "extracted",
        "turn_intent",
        "turn_target",
        "guard",
        "guard_confidence",
        "followup_query",
    }
)


def _strip_reserved_keys(result: Any) -> Any:
    """Remove schema field names the model wrote into extracted{}."""
    for field in ("extracted", "corrections"):
        values = getattr(result, field, None)
        if not isinstance(values, dict):
            continue
        leaked = [k for k in values if k in _RESERVED_RESULT_KEYS]
        if not leaked:
            continue
        setattr(result, field, {k: v for k, v in values.items() if k not in _RESERVED_RESULT_KEYS})
        logger.info(
            "request_detection: dropped schema keys from %s",
            field,
            extra={"source": "schema_hygiene", "field": field, "llm_value": ", ".join(leaked)},
        )
    return result


def _recover_missed_followup(result: Any, last_user: str | None) -> Any:
    """Fill in a side question the caller asked and the extractor did not report.

    The veto below handles a question the model invented. This is the other
    half, and it is the same bug seen from the other side:

        Caller  No. But I lost my credit ID card. Can you help me with the
                new one?                        (awaiting benefits_response)
        →       followup_query null

        Caller  That sounds interesting, but I lost my ID card. Can you help
                me to get a new one?            (awaiting care_coach_response)
        →       followup_query "can you help me to get a new one"

    The same request, two turns apart, classified both ways. Not model
    variance: those slots run different prompt stacks. benefits_response is
    collected by delivery_management against header_extraction.md +
    delivery_management.md, 4,100 words with the follow-up rules a long way
    from the field definitions; care_coach_response by the benefits agent
    against header_core.md + benefits.md, 1,300 words with a request block
    directly under FIELDS. Whether the caller is heard depends on which prompt
    file the slot they are on happens to live in.

    A missed question is invisible downstream — BaseAgent.execute's safety net
    only fires when followup_query is set, so a dropped one looks exactly like
    a caller who asked nothing, on that turn and on every repeat of it. Nothing
    escalates it either: the guard layer needs guard_confidence >= 0.7 and such
    a turn carries 0.0.

    Only the clean "answer, then ask" shape is recovered, and only when the
    extractor found a value — so there is an answer half — and reported no
    question. recover_side_question is far stricter than the veto's cue test,
    because a false positive here puts words in the caller's mouth.

    Setting the question is the whole edit: an ANSWERED turn that carries a
    value and a question derives ANSWERED_WITH_FOLLOWUP by itself.
    """
    reported = (getattr(result, "followup_query", None) or "").strip()
    if not result.has_extracted_value:
        return result
    # Only the clean "answer, then ask" shape: a turn already classified as a
    # request, a pivot or a denial is not one the caller merely asked alongside.
    if result.turn_intent is not TurnIntent.ANSWERED:
        return result
    recovered = recover_side_question(last_user, result.extracted)
    if not recovered:
        return result
    # The LLM stays primary: a question it reported that is about what the
    # caller talked about is its call to make, paraphrase and all. Recovery
    # only overrides the reported question when it shares no topic word with
    # the turn at all — the signature of one lifted from the AI's own earlier
    # turns, which the cue-based veto cannot catch on a turn where the caller
    # genuinely did ask something ("help with claim status" reported against
    # "No. But I lost my credit ID card. Can you help me with the new one?").
    if reported and quotes_the_caller(reported, last_user):
        return result
    result.followup_query = recovered
    logger.info(
        "request_detection: grounding_fallback %s followup_query",
        "replaced" if reported else "recovered",
        extra={
            "source": "grounding_fallback",
            "field": "followup_query",
            "llm_value": reported,
            "final_value": recovered,
        },
    )
    return result


def _reconcile_followup_query(result: Any, last_user: str | None) -> Any:
    """Drop a side question the caller did not ask, and the event that carried it.

    `followup_query` routes the turn into FOLLOWUP_RESPOND, whose whole job is
    to answer the "Followup:" line. A phantom line has no answer, so the
    generator pads — usually by restating the sentence the caller just heard —
    and the same phantom reaches BaseAgent.execute's safety net, which
    generates a second sentence for the turn and prefixes it. One hallucinated
    field, two generation calls, two sentences saying the same thing.

    See core.followup_grounding for why the check lives in Python at all: three
    extraction headers already forbid synthesizing a followup_query from topics
    the AI raised, in capitals, and the field keeps coming back with one.

    Only the question is cleared. corrections{} and the request intent are the
    caller's own request shapes, reconciled below on their own evidence, and
    the pipelines read them directly — so clearing a phantom question cannot
    take an honest update down with it, and there is no turn label left to
    repair afterwards either.
    """
    query = (getattr(result, "followup_query", None) or "").strip()
    if not query or is_grounded_followup(query, last_user):
        return result
    result.followup_query = None
    logger.info(
        "request_detection: grounding_veto cleared followup_query",
        extra={
            "source": "grounding_veto",
            "field": "followup_query",
            "llm_value": query,
            "final_value": "",
        },
    )
    return result


# Slots a caller can never correct by saying a different value: system flags
# and the classified call intent. handlers.CALLER_LOCKED_SLOTS is the full
# list and drops them again downstream; these two are duplicated here because
# this module must not import from agents/ (see the module docstring), and
# because keeping them out of corrections{} in the first place is what stops a
# locked slot reaching an acknowledgement path at all.
_NEVER_CORRECTED: frozenset[str] = frozenset({"member_status_verify", "call_intent"})


def _split_corrections(result: Any, confirmed_slots: Mapping[str, Any] | None, awaiting_slot: str) -> Any:
    """File spoken values that replace a confirmed slot as corrections.

    This is the half of the extraction contract that was never the model's to
    answer. The headers asked it for corrections{} separately from extracted{},
    gave it three worked examples and a rule that the keys "MUST be a slot
    listed in Confirmed:" — a fact about the pipeline's state, handed to the
    model as a context line so it could hand the same fact back. It reads that
    line correctly most turns and not all of them, and a missed correction is
    an accepted value silently overwritten.

    The pipeline has the Confirmed: view already. Asking it instead costs
    nothing and cannot disagree with itself.
    """
    if confirmed_slots is None or not hasattr(result, "split_corrections"):
        return result
    before = dict(getattr(result, "extracted", None) or {})
    result.split_corrections(confirmed_slots, awaiting_slot, locked_slots=_NEVER_CORRECTED)
    moved = [k for k in before if k not in (getattr(result, "extracted", None) or {})]
    if moved:
        logger.info(
            "request_detection: state_split moved values into corrections",
            extra={
                "source": "state_split",
                "field": "corrections",
                "final_value": ", ".join(moved),
                "awaiting_slot": awaiting_slot,
            },
        )
    return result


def _reconcile_cannot_provide(result: Any, last_user: str | None) -> Any:
    """Regex backstop for a denial the extraction model did not report.

    The model is the primary source — it reads phrasings no pattern list will
    ever cover. This only fills in a miss, so an extraction failure (which
    returns an empty WorkerResult) still routes an "I don't have it" to the
    fallback offer rather than an escalation.

    The precedence the headers used to spell out — a pivot outranks a denial, a
    value outranks both — lives in report_cannot_supply, which declines on a
    result that already reports a value, a pivot or a request of its own.
    """
    if not detect_cannot_provide(last_user):
        return result
    if not result.report_cannot_supply():
        return result
    logger.info(
        "request_detection: regex_fallback reported a denial the model missed",
        extra={"source": "regex_fallback", "field": "turn_intent", "final_value": "cannot_provide"},
    )
    return result


def reconcile_worker_result(
    result: Any,
    last_user: str | None,
    *,
    confirmed_slots: Mapping[str, Any] | None = None,
    awaiting_slot: str = "",
) -> Any:
    """Fallback + veto pass over an extraction result (WorkerResult-shaped).

    ``confirmed_slots`` is the same Confirmed: view the extraction prompt was
    built from, and ``awaiting_slot`` the slot it was asked about. Together
    they are what splits the values the caller spoke into new ones and
    corrections — the judgement the model used to make from a context line, now
    made from the state itself. Omit them and the split is skipped, which is
    what the idempotent re-runs several agents do rely on.

    - LLM reported a request intent (update / redo / replay) → kept as-is; the
      regex never overrides a concrete LLM detection with a different target.
    - LLM reported none but detect_request fires → adopt the detected kind and
      target, which is the whole edit. A turn the model labelled WAIT becomes
      the update it is ("wait, actually my ZIP changed" is a correction, not a
      hold request), and a value extracted in the same turn still wins, because
      the pipelines ask about the value and the request separately.
    - the LLM reported a pivot or a denial → the regex stays out of it. Those
      are readings of the slot being collected, made on the caller's words; a
      pattern match about some other slot does not outrank one.
    - followup_query names a side question with no trace of a question or a
      request in the caller's words → clear it. See _reconcile_followup_query
      and core.followup_grounding.
    - the caller plainly answered AND then asked, and followup_query came back
      null → recover the question from their own words. See
      _recover_missed_followup.
    - extracted{} carries a schema field name as a key ("turn_target": "ID
      card") → drop it; that dict is slot → value and every consumer reads it
      that way. See _strip_reserved_keys.
    - detect_cannot_provide fires but the LLM reported no denial → report one.
      The call is semantic and the model is the primary source; this is the
      backstop for a missed call or an extraction that threw, so a caller who
      cannot supply a slot still reaches the fallback offer.
    - values the caller spoke for slots already confirmed → moved into
      corrections{}, when confirmed_slots was supplied. See
      WorkerResult.split_corrections.
    - Neither detects → result returned untouched.
    """
    result = _strip_reserved_keys(result)
    result = _split_corrections(result, confirmed_slots, awaiting_slot)
    result = _reconcile_cannot_provide(result, last_user)
    result = _reconcile_followup_query(result, last_user)
    result = _recover_missed_followup(result, last_user)

    detected = detect_request(last_user)
    if detected is None:
        return result

    # Each of the two facts the detector found — the kind and the target — used
    # to be filled on its own, because they were two fields and could arrive
    # half-set:
    #
    #     AI      …Would you like us to send you details about our Care
    #             Coach Guides?
    #     Caller  please change my email address?
    #     AI      One more thing — you're eligible for a complimentary health
    #             and wellness coach. Would you like me to…
    #
    # detect_request reads that utterance as update/email without difficulty.
    # The extractor returned update_target "email" and left request_kind
    # "none", the gap-filler declined to fill either, and benefits_agent's
    # block for exactly this case tested request_kind == "update" and never
    # fired. The request was dropped and the offer re-asked.
    #
    # One intent carries both now, so there is no half to fill: adopt_request
    # takes the pair or leaves the result alone, and it declines whenever the
    # model already reported a request of its own, a pivot, or a denial.
    reported_kind, reported_target = result.change_kind, result.change_target

    def _log_adopted(source: str) -> None:
        logger.info(
            "request_detection: %s adopted a %s request",
            source,
            detected.kind,
            extra={
                "source": source,
                "field": "turn_intent/turn_target",
                "llm_value": f"{reported_kind or 'none'}:{reported_target or 'none'}",
                "final_value": f"{detected.kind}:{detected.target}",
                "matched": detected.matched,
            },
        )

    if not reported_target:
        if result.adopt_request(detected.kind, detected.target):
            _log_adopted("regex_fallback")
    # Redo-over-contact-field veto: "send that list to my email instead of fax"
    # fires redo patterns but the model often reports the target as a contact
    # field (email/fax) because it sees that word in the text. When the regex
    # detects redo and the reported target is a raw contact field rather than a
    # capability topic, the redo detection is authoritative.
    elif (
        detected.kind == "redo"
        and reported_target in ("email", "fax", "phone_number")
        and reported_kind != "redo"
    ):
        if result.adopt_request(detected.kind, detected.target):
            _log_adopted("regex_redo_veto")

    # The two event vetoes that used to close this function — WAIT on a turn
    # that is really a correction, ANSWERED on a bare request that only the
    # correction path could honor — were both repairs to a label that could
    # disagree with the fields beside it. There is no label any more: the
    # intent adopted above is the whole classification, so both repairs
    # happened the moment it changed.
    return result
