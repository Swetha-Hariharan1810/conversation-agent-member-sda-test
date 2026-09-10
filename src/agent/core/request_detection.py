"""
request_detection.py — deterministic fallback + veto layer for cross-call
request detection (update / redo / replay). Fixes the instability root cause:
the extraction LLM intermittently drops update_target/request_kind or labels
a correction turn WAIT, and the whole downstream routing hinges on those
fields.

The LLM stays PRIMARY. This module never overrides a concrete LLM detection
with a different target — it only
  1. fills gaps  — the LLM returned no update_target/request_kind but the
     caller's words plainly contain one of the covered request shapes; and
  2. vetoes      — known misclassifications (event_type WAIT on a turn that
     is actually a correction/update request).
When neither the LLM nor the regex detects anything, behavior is unchanged.

Slot patterns are DERIVED from SLOT_OWNERSHIP, not hand-written: every
registry key gets "update/change/correct my <label>" and "<label> changed /
is wrong / is different" coverage automatically, so a future registry entry
is covered the day it is added. SLOT_LABEL_ALIASES adds the spoken variants
("date of birth", "member number", "postal code"). Hand-written patterns are
reserved for phrasings that don't name the slot ("I moved" → zip_code,
"instead of fax" → redo delivery).

Precedence: update beats redo beats replay; a concrete slot target beats a
capability topic (updates are checked first and target canonical slot names).

Dependency-light on purpose: stdlib re / dataclasses / logging plus the
dependency-free slot_ownership registry. NEVER import from agents/ or
agent.utils — the few cannot-provide negatives needed to stay out of
detect_cannot_provide's territory are duplicated below.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from agent.core.slot_ownership import SLOT_OWNERSHIP

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


def _coerce_like(sample: Any, value: str) -> Any:
    """Coerce ``value`` into ``sample``'s enum class when sample is an enum
    member (duck-typed via .value so this module never imports the schema)."""
    if sample is not None and hasattr(sample, "value"):
        try:
            return type(sample)(value)
        except ValueError:
            pass
    return value


def reconcile_worker_result(result: Any, last_user: str | None) -> Any:
    """Fallback + veto pass over an extraction result (WorkerResult-shaped).

    - LLM produced update_target/request_kind → kept as-is; the regex never
      overrides a concrete LLM detection with a different target.
    - LLM produced neither but detect_request fires → populate both fields.
    - LLM returned event_type WAIT but detect_request fires → clear WAIT:
      "wait, actually my ZIP changed" is a correction, not a hold request.
      With an extracted value in the same turn the event downgrades to
      ANSWERED_WITH_FOLLOWUP (value wins, request handled as Case B);
      otherwise to CORRECTED (bare request, C2).
    - LLM returned event_type ANSWERED on a bare request (update_target set,
      no extracted values, no corrections) → upgrade to CORRECTED: the
      extraction contract classifies bare cross-call requests as corrected,
      and only the CORRECTED path (C2) can honor a target with no value.
    - Neither detects → result returned untouched.
    """
    detected = detect_request(last_user)
    if detected is None:
        return result

    llm_target = (getattr(result, "update_target", None) or "").strip()
    kind_raw = getattr(result, "request_kind", None)
    llm_kind = str(getattr(kind_raw, "value", kind_raw) or "").strip().lower()
    if llm_kind == "none":
        llm_kind = ""

    # Decision provenance: every field this pass changes is logged with the
    # LLM's original value and the final value, so production variance
    # (how often the regex layer has to intervene) is directly measurable.
    def _log_change(source: str, field: str, llm_value: str, final_value: str) -> None:
        logger.info(
            "request_detection: %s changed %s",
            source,
            field,
            extra={
                "source": source,
                "field": field,
                "llm_value": llm_value,
                "final_value": final_value,
                "matched": detected.matched,
            },
        )

    if not llm_target and not llm_kind:
        result.update_target = detected.target
        result.request_kind = _coerce_like(kind_raw, detected.kind)
        _log_change("regex_fallback", "update_target", llm_target, detected.target)
        _log_change("regex_fallback", "request_kind", llm_kind or "none", detected.kind)

    # Redo-over-contact-field veto: "send that list to my email instead of fax"
    # fires redo patterns but the LLM often sets update_target to a contact field
    # (email/fax) because it sees that word in the text. When the regex detects
    # redo and the LLM's target is a raw contact field rather than a capability
    # topic, the redo detection is authoritative.
    elif detected.kind == "redo" and llm_target in ("email", "fax", "phone_number") and llm_kind != "redo":
        result.update_target = detected.target
        result.request_kind = _coerce_like(kind_raw, detected.kind)
        _log_change("regex_redo_veto", "update_target", llm_target, detected.target)
        _log_change("regex_redo_veto", "request_kind", llm_kind or "none", detected.kind)

    event_raw = getattr(result, "event_type", None)
    event = str(getattr(event_raw, "value", event_raw) or "").strip().lower()
    has_value = any(v for v in (getattr(result, "extracted", None) or {}).values())
    if event == "wait":
        # A correction turn, not a hold request.
        new_event = "answered_with_followup" if has_value else "corrected"
        result.event_type = _coerce_like(event_raw, new_event)
        _log_change("regex_veto", "event_type", "wait", new_event)
    elif event == "answered" and not has_value:
        has_corrections = any((getattr(result, "corrections", None) or {}).values())
        target_now = (getattr(result, "update_target", None) or "").strip()
        if target_now and not has_corrections:
            # Bare request labeled ANSWERED — only CORRECTED (C2) honors it.
            result.event_type = _coerce_like(event_raw, "corrected")
            _log_change("regex_veto", "event_type", "answered", "corrected")

    return result
