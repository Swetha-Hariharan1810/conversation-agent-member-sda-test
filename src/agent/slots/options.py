"""
options.py — the accepted answers for every closed-set slot, in one place.

A closed-set slot is one where the caller must pick from a short list the
system already knows: ``delivery_method`` is fax or email, ``upload_method`` is
one of four ways to get records to us. Today that list is written down in five
different places — the agent's Markdown extraction prompt, the normalizer's
synonym map, the validator's accept set, a per-agent regex screen, and the
README — and the extraction prompt never actually states it next to the slot
being collected. So the extraction model classifies by grammatical form rather
than by content, and a valid answer phrased as a question reads as a question:

    AI      I can also send a secure link to your email so you can upload the
            records yourself. Would you like me to send that over?
    Caller  Can I ask my doctor to send them over?
    AI      That's a great question, and a representative will need to make
            that change for you.

``doctor_direct`` is a defined value of ``upload_method`` and
``records_coordination.md`` lists "you can contact my doctor" as an example of
it. The system declined an option it supports natively.

This module is the single source of truth:

  * :func:`render_options` puts the list into the extraction prompt, right
    beside "Currently asking for:", where the model is actually looking.
  * :func:`match_option` reads an option out of the caller's words without an
    LLM. It is a GAP FILLER, never an override — see
    ``request_detection.reconcile_worker_result``.

Two deliberate omissions, recorded so they are not re-litigated:

  * **Yes/no slots are not registered.** "yes | no" carries no information a
    model needs told, and a forced-choice block on a confirmation turn is
    precisely where "Yeah. That's right. But I just realized my ZIP code's
    wrong" lives — nudging the model toward one of two values is how the
    second half of that turn gets dropped.
  * **Nothing here skips the LLM call.** A deterministic match on the whole
    utterance would also have to be right about guards, corrections and side
    questions, which is the bug class this work exists to remove. The matcher
    only runs when extraction came back with nothing for the slot.

Dependency-light on purpose: stdlib plus ``agent.slots.normalizers``. Imported
by ``agent.core.request_detection``, which must never reach into ``agents/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from agent.slots.normalizers import (
    normalize_caller_role,
    normalize_delivery_method,
    normalize_notification_method,
    normalize_provider_type,
)

__all__ = [
    "SlotOption",
    "SlotOptionSet",
    "SLOT_OPTIONS",
    "get_option_set",
    "is_closed_set",
    "option_values",
    "match_option",
    "render_options",
    "render_open_options",
]


@dataclass(frozen=True)
class SlotOption:
    """One accepted answer for a closed-set slot.

    ``value`` is what goes into state — it must satisfy the slot's validator.
    ``gloss`` is the one line the extraction model reads.
    ``patterns`` read this option out of the caller's words directly; they are
    tried in the order the options are declared, so a set whose options overlap
    must declare the most specific first.
    """

    value: str
    gloss: str
    patterns: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class SlotOptionSet:
    """The accepted answers for one slot.

    ``normalizer`` is consulted after every option's own patterns miss. It is
    the slot's production normalizer, so the synonym tables that already exist
    ("facts" → fax, "my doctor" → Primary Care Physician) are reused rather
    than copied. Its result is only accepted when it is one of ``values``.
    """

    slot: str
    options: tuple[SlotOption, ...]
    normalizer: Optional[Callable[[Optional[str]], str]] = None
    note: str = ""

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(opt.value for opt in self.options)


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# ── upload_method ────────────────────────────────────────────────────────────
# The patterns below were the per-agent screen in
# records_coordination/handlers.py; they live here now so the prompt block and
# the matcher cannot drift apart. Order is the whole design:
#   "can you get them from the provider" names the provider AND asks us to
#   chase it, so personal_guide has to be tested before doctor_direct; and
#   "can I ask my doctor to send it" opens with "I" while the sender is the
#   doctor, so doctor_direct has to be tested before member_upload.
_SEND_VERB = (
    r"(?:send|sends|sending|fax|faxes|forward|forwards|mail|mails|submit|submits"
    r"|share|shares|get|gets|pull|pulls|request|requests|obtain)"
)
_DOCTOR = (
    r"(?:doctor|doctor'?s|physician|provider|provider'?s|clinic|hospital"
    r"|surgery|office|specialist|gp)"
)

UPLOAD_METHOD = SlotOptionSet(
    slot="upload_method",
    note=(
        "Choose by WHO does the sending, not by whether the caller phrased it "
        "as a question. Asking permission is how people pick an option."
    ),
    options=(
        SlotOption(
            "personal_guide",
            "we contact the provider and collect the records on their behalf",
            _rx(
                r"\b(?:you|your\s+team|someone\s+there|somebody\s+there)\b[^.?!]{0,30}?"
                rf"\b(?:contact|call|reach\s+out|chase|{_SEND_VERB})\b",
                r"\bon\s+my\s+behalf\b",
            ),
        ),
        SlotOption(
            "doctor_direct",
            "their doctor, provider or office sends the records to us directly",
            _rx(
                rf"\b{_DOCTOR}\b[^.?!]{{0,30}}?\b(?:can|could|will|would|to|should)?\s*{_SEND_VERB}\b",
                rf"\b{_SEND_VERB}\b[^.?!]{{0,20}}?\bfrom\s+(?:my\s+|the\s+)?{_DOCTOR}\b",
                rf"\bhave\s+(?:my\s+|the\s+)?{_DOCTOR}\b",
                rf"\bask\s+(?:my\s+|the\s+)?{_DOCTOR}\b",
                rf"\b{_DOCTOR}\s+(?:will|can|could)\s+handle\b",
            ),
        ),
        SlotOption(
            "member_upload",
            "the caller uploads the records themselves, via a secure link we email",
            _rx(
                rf"\bi(?:'ll|\s+will|\s+can|\s+could)?\s+(?:just\s+)?(?:{_SEND_VERB}|upload|uploads|scan|scans|do\s+it)\b",
                r"\bupload\s+(?:it|them|those)\b",
                r"\bsend\s+me\s+(?:the\s+|a\s+)?link\b",
                r"\b(?:do|handle)\s+it\s+(?:online|myself)\b",
                r"\bmyself\b",
            ),
        ),
        # "decline" carries no patterns on purpose. Inferring one escalates the
        # call, which is the worst outcome to reach on a guess — a caller who
        # has not refused is handed to a representative they did not ask for.
        # It is rendered so the model can choose it; it is never matched here.
        SlotOption("decline", "the caller refuses the records step itself"),
    ),
)


# ── delivery_method ──────────────────────────────────────────────────────────
DELIVERY_METHOD = SlotOptionSet(
    slot="delivery_method",
    normalizer=normalize_delivery_method,
    options=(
        SlotOption("fax", "send it to their fax number"),
        SlotOption("email", "send it to their email address"),
        SlotOption("both", "send it to both fax and email"),
    ),
)


# ── notification_method ──────────────────────────────────────────────────────
# validate_notification_method accepts sms and email only, while
# normalize_notification_method can still return "both". The validator is what
# the system actually accepts, so "both" is not offered here — and because
# match_option only accepts a normalizer result that is one of these values, a
# normalized "both" is discarded rather than handed on to fail validation.
NOTIFICATION_METHOD = SlotOptionSet(
    slot="notification_method",
    normalizer=normalize_notification_method,
    options=(
        SlotOption("sms", "text message to their mobile number"),
        SlotOption("email", "email to their email address"),
    ),
)


# ── provider_type ────────────────────────────────────────────────────────────
PROVIDER_TYPE = SlotOptionSet(
    slot="provider_type",
    normalizer=normalize_provider_type,
    note="Extract the canonical name, not the caller's wording.",
    options=(
        SlotOption("Primary Care Physician", "PCP, family doctor, GP, 'my regular doctor'"),
        SlotOption("Pediatrician", "children's doctor"),
        SlotOption("Cardiologist", "heart doctor or heart specialist"),
        SlotOption("Dermatologist", "skin doctor"),
        SlotOption("Orthopedic Specialist", "bone or joint doctor, orthopedist"),
    ),
)


# ── caller_role / relationship ───────────────────────────────────────────────
# Two slot names, one question: who is the caller relative to the plan.
# validate_relationship also accepts "subscriber", but normalize_caller_role
# folds subscriber wording into plan_holder, so there is no utterance that
# reaches it — it is not offered as a separate answer.
_CALLER_ROLE_OPTIONS = (
    SlotOption("plan_holder", "the member themselves — the plan or account holder"),
    SlotOption("dependent", "a spouse, child or other dependent on the plan"),
)

CALLER_ROLE = SlotOptionSet(
    slot="caller_role",
    normalizer=normalize_caller_role,
    options=_CALLER_ROLE_OPTIONS,
)

RELATIONSHIP = SlotOptionSet(
    slot="relationship",
    normalizer=normalize_caller_role,
    options=_CALLER_ROLE_OPTIONS,
)


SLOT_OPTIONS: dict[str, SlotOptionSet] = {
    s.slot: s
    for s in (
        UPLOAD_METHOD,
        DELIVERY_METHOD,
        NOTIFICATION_METHOD,
        PROVIDER_TYPE,
        CALLER_ROLE,
        RELATIONSHIP,
    )
}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def get_option_set(slot: str | None) -> Optional[SlotOptionSet]:
    """The option set for ``slot``, or None when it is not a closed set."""
    if not slot:
        return None
    return SLOT_OPTIONS.get(slot.strip())


def is_closed_set(slot: str | None) -> bool:
    return get_option_set(slot) is not None


def option_values(slot: str | None) -> tuple[str, ...]:
    """Every accepted value for ``slot``; empty when it is not a closed set."""
    option_set = get_option_set(slot)
    return option_set.values if option_set else ()


# ---------------------------------------------------------------------------
# Deterministic reading
# ---------------------------------------------------------------------------


def match_option(slot: str | None, utterance: str | None) -> str:
    """The option ``utterance`` names for ``slot``, or "".

    Patterns first, in declared order; then the slot's own normalizer, whose
    result is accepted only when it is one of the slot's values.

    This is a gap filler. It runs on a turn where the extraction model returned
    no value for the awaiting slot, and it never contradicts one that did — see
    ``request_detection.reconcile_worker_result``.

    The normalizer fallback inherits the normalizers' substring matching, so it
    is loose read in isolation: ``match_option("notification_method", "calling
    for myself")`` is "sms", because "call" is in the synonym table. What keeps
    that harmless is the window it runs in — the slot has to be the one being
    asked, and extraction has to have returned nothing for it. Against a turn
    that would otherwise burn a retry and re-ask the same question, a loose
    read of a plausible answer is the better trade. Do not widen the window
    without replacing this with boundary-anchored patterns.
    """
    option_set = get_option_set(slot)
    text = (utterance or "").strip()
    if option_set is None or not text:
        return ""

    for option in option_set.options:
        for pattern in option.patterns:
            if pattern.search(text):
                return option.value

    if option_set.normalizer is not None:
        try:
            normalized = (option_set.normalizer(text) or "").strip()
        except Exception:  # a normalizer must never take a turn down with it
            return ""
        if normalized in option_set.values:
            return normalized

    return ""


# ---------------------------------------------------------------------------
# Rendering into the extraction prompt
# ---------------------------------------------------------------------------

# Long enough for the longest canonical value ("Orthopedic Specialist"), so the
# glosses line up and the block reads as a table rather than as prose.
_GLOSS_COLUMN = 22


def render_options(slot: str | None) -> str:
    """The "Accepted answers" block for ``slot``, or "" when it is not a closed set.

    Goes into the extraction model's user message next to "Currently asking
    for:", not into the agent's Markdown prompt — the agent prompts are already
    8k tokens and the model reads the end of the message.
    """
    option_set = get_option_set(slot)
    if option_set is None:
        return ""

    lines = [f"Accepted answers for {option_set.slot}:"]
    lines += [f"  {opt.value:<{_GLOSS_COLUMN}} — {opt.gloss}" for opt in option_set.options]
    if option_set.note:
        lines.append(f"  {option_set.note}")
    # Without this the block reads as "pick one", and a caller who asked about
    # something else gets a value invented for them. An empty slot is recoverable;
    # a wrong one that reaches confirmation is not.
    lines.append(f"  If the caller named none of these, leave {option_set.slot} empty — never force a fit.")
    return "\n".join(lines)


def render_open_options(slots: list[str] | tuple[str, ...] | None, *, exclude: str = "") -> str:
    """A one-line-per-slot summary of the closed sets still open in this call.

    A caller answers the question before last more often than the design
    assumes — an ASR cutoff pushes their real answer one turn late by default —
    and today a valid answer to a slot that is not ``awaiting_slot`` has
    nowhere to land. Naming those slots and their values gives it somewhere to
    go. Values only, no glosses: this is context, not the question being asked.
    """
    if not slots:
        return ""
    lines = []
    for slot in slots:
        if slot == exclude:
            continue
        option_set = get_option_set(slot)
        if option_set is None:
            continue
        lines.append(f"  {option_set.slot}: {' | '.join(option_set.values)}")
    if not lines:
        return ""
    return "Also answerable (asked earlier, still open):\n" + "\n".join(lines)
