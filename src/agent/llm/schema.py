"""
schema.py — the structured-output contracts for the extraction models.

`WorkerResult` is what LLM 1 fills in on every collection turn, so its shape is
prompt surface as much as it is a data structure: every field has to be
described in the extraction headers, and every field the model has to think
about is attention it is not spending on the caller's utterance.

It used to carry eleven. Six of them were the same fact written down more than
once — `corrections` restated `extracted` with a claim about state the model
cannot see, `cannot_provide`/`fallback_pivot`/`event_type:"ambiguous"` were
three ways of saying "no usable value" with their precedence spelled out in
prose, and `event_type`/`request_kind`/`update_target` were one intent and one
target split across three fields. Two more were not the model's to decide at
all: the headers told it to always emit `followup_disposition:"none"` ("the
system decides"), and `needs_freeform_response` asked a perception model for a
routing decision Python overrode in eight of its ten branches.

What the model reports is one classification and one target:

    extracted        — the values the caller spoke this turn
    turn_intent      — what the utterance DID (TurnIntent)
    turn_target      — the slot, topic or identifier that intent points at
    guard            — safety / routing guard (a separate concern, unchanged)
    guard_confidence
    followup_query   — the side question, in the caller's words

Plus one thing the pipeline knows and the model does not: which of the values
the caller spoke replace something already confirmed. `split_corrections` works
that out from the same Confirmed: view the prompt was built from, and files
them under `corrections`.

Everything else the pipelines used to read is now a question asked of the
intent — `cannot_supply`, `pivot_target`, `change_target`, `asked_for_time`,
`no_usable_value`. Each is one line, and each carries a precedence rule the
headers used to spend a paragraph on: a value the caller spoke outranks every
reading that says there is none.

There is no second vocabulary. The old field names are gone from the schema,
from every consumer, and from the prompts, so a turn has exactly one
description and no way to disagree with itself.
"""

from enum import Enum
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class TurnIntent(str, Enum):
    """What the caller's utterance did this turn — the one classification the
    extraction model makes.

    The members are mutually exclusive by construction, which is the point: the
    fields these replaced could all be set at once, so the headers carried
    precedence rules ("a pivot outranks a denial", "a value given in the same
    utterance always wins", "never set both") and the pipelines carried the
    code to enforce them. An enum enforces it for free.

    Named ``turn_intent`` rather than ``intent`` because "intent" is already a
    slot name — intake collects it — and ``extracted{"intent": ...}`` beside a
    top-level ``intent`` field is exactly the kind of collision that puts a
    schema key into a slot dict.
    """

    ANSWERED = "answered"
    """The caller responded to the question they were asked. Any value they
    spoke is in extracted{}, whether it is new or replaces one already on
    file — that distinction is Python's to make, not the model's."""

    WAIT = "wait"
    """Asked for time to find or think about the value: "give me a minute"."""

    UNUSABLE = "unusable"
    """Nothing usable this turn — garbled, uncertain, or a question asked
    INSTEAD of answering (which also carries followup_query)."""

    CANNOT_PROVIDE = "cannot_provide"
    """Says they cannot supply the slot being collected and names no
    alternative: "I never got a card", "that's in my wallet at home"."""

    PIVOT = "pivot"
    """Wants to identify themselves a different way and has not given the value
    yet. turn_target names the identifier ("ssn", "member_id", ...)."""

    UPDATE = "update"
    """Wants a stored value changed and did not say the new one.
    turn_target names the slot. A caller who DID say the new value is
    ANSWERED — the value is in extracted{} and Python files it as a
    correction if that slot is already confirmed."""

    REDO = "redo"
    """Wants a completed action re-performed with a changed parameter
    ("send it by email instead"). turn_target names what changes."""

    REPLAY = "replay"
    """Wants information already given re-stated ("repeat my benefits").
    turn_target names the topic."""


# Intents that point at something outside the current question. These are the
# three that used to be request_kind, and turn_target is what update_target was.
REQUEST_INTENTS = frozenset({TurnIntent.UPDATE, TurnIntent.REDO, TurnIntent.REPLAY})

# Intents that assert there is no usable value. A value extracted in the same
# turn overrides all three — the rule the headers spelled out three times.
NO_VALUE_INTENTS = frozenset({TurnIntent.UNUSABLE, TurnIntent.CANNOT_PROVIDE, TurnIntent.PIVOT})

# The three request shapes by name, for the deterministic layer that detects
# them from the caller's words (core.request_detection). See adopt_request.
_REQUEST_INTENT_BY_KIND: dict[str, TurnIntent] = {
    "update": TurnIntent.UPDATE,
    "redo": TurnIntent.REDO,
    "replay": TurnIntent.REPLAY,
}


class RequestKind(str, Enum):
    """Cross-call request shapes, as FollowUpResult reports them.

    WorkerResult has no equivalent field: a collection turn says what it did
    with TurnIntent, and UPDATE / REDO / REPLAY are three of its members. The
    follow-up agent runs with no slot being collected around it, so it
    classifies the request on its own.
    """

    UPDATE = "update"
    REDO = "redo"
    REPLAY = "replay"
    NONE = "none"


class GuardType(str, Enum):
    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    ABUSE = "ABUSE"
    SELF_HARM = "SELF_HARM"
    INTERRUPTION = "INTERRUPTION"
    OFFTOPIC_GLOBAL = "OFFTOPIC_GLOBAL"  # non-healthcare — static response
    OFFTOPIC_AGENT = "OFFTOPIC_AGENT"  # wrong agent — dynamic LLM response
    NONE = "NONE"


def _clean(value: Any) -> str:
    """A trimmed string from a field that may be None."""
    return str(value or "").strip()


class WorkerResult(BaseModel):
    """One extraction turn: six fields the model reports, and the questions
    the pipelines ask of them.

    ``extra="forbid"`` is the whole compatibility story. There is no
    translation layer under it and nothing accepts the shape this replaced —
    a payload carrying `event_type`, `corrections`, `update_target`,
    `request_kind`, `cannot_provide`, `fallback_pivot`,
    `followup_disposition` or `needs_freeform_response` is rejected where it
    is built, which is where the mistake is.
    """

    model_config = ConfigDict(extra="forbid")

    # Slot values the caller spoke this turn, slot name → value. New values and
    # replacements for values already on file both live here: which is which is
    # a fact about state, and split_corrections settles it below.
    extracted: Optional[Dict[str, str]] = None
    # What the utterance did — see TurnIntent.
    turn_intent: TurnIntent = TurnIntent.ANSWERED
    # What that intent points at: the slot to change (update), the parameter or
    # topic (redo / replay), or the identifier to switch to (pivot). Null for
    # every other intent.
    turn_target: Optional[str] = None
    # Safety / routing guard triggered this turn
    guard: GuardType = GuardType.NONE
    # LLM's confidence in the guard classification (0.0–1.0)
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # The side question, condensed, verbatim-ish
    followup_query: Optional[str] = None

    # ── Python-owned, never reported by the model ────────────────────────────
    # Which of the values in extracted{} replace something already confirmed.
    # Filled by split_corrections below — see its docstring for why this is not
    # the model's to answer.
    _corrections: Dict[str, str] = PrivateAttr(default_factory=dict)
    _corrections_split: bool = PrivateAttr(default=False)

    # ── Reading the turn ─────────────────────────────────────────────────────
    # Each of these is one question the pipelines ask, answered from the intent
    # and the values in one place. Between them they carry the precedence the
    # extraction headers used to state three times over: a value the caller
    # actually spoke outranks every reading that says there is none. "hold
    # on — okay, it's M451982" is an answer, not a wait; "I don't have my card
    # but my ID is M451982" is an answer, not a denial.

    @property
    def has_extracted_value(self) -> bool:
        """Did the caller speak a usable value this turn?"""
        return any(v for v in (self.extracted or {}).values())

    @property
    def corrections(self) -> Dict[str, str]:
        """Values that replace something already confirmed, slot → value."""
        return self._corrections

    @corrections.setter
    def corrections(self, value: Optional[Mapping[str, str]]) -> None:
        self._corrections = {k: v for k, v in (value or {}).items() if v}

    @property
    def change_target(self) -> str:
        """The slot or topic the caller asked to change, redo or replay.

        Empty unless they asked for one of those. A caller who gave the new
        value is not asking for a change — the value is in extracted{} and
        split_corrections decides what it replaces.
        """
        return _clean(self.turn_target) if self.turn_intent in REQUEST_INTENTS else ""

    @property
    def change_kind(self) -> str:
        """ "update", "redo" or "replay" — or "" when none was asked for."""
        return self.turn_intent.value if self.turn_intent in REQUEST_INTENTS else ""

    @property
    def pivot_target(self) -> str:
        """The identifier the caller offered instead of the one being collected."""
        if self.turn_intent is TurnIntent.PIVOT and not self.has_extracted_value:
            return _clean(self.turn_target)
        return ""

    @property
    def cannot_supply(self) -> bool:
        """Did the caller say they cannot give the slot being collected?"""
        return self.turn_intent is TurnIntent.CANNOT_PROVIDE and not self.has_extracted_value

    @property
    def asked_for_time(self) -> bool:
        """Did the caller ask for a moment rather than answering?"""
        return self.turn_intent is TurnIntent.WAIT and not self.has_extracted_value

    @property
    def no_usable_value(self) -> bool:
        """Is there nothing to take from this turn?

        True for a turn that was garbled, a denial, a pivot, or a question
        asked INSTEAD of answering — the four shapes that leave the slot
        uncollected and the caller owed a response rather than a retry.
        """
        if self.has_extracted_value:
            return False
        if self.turn_intent in NO_VALUE_INTENTS:
            return True
        return self.turn_intent is TurnIntent.ANSWERED and bool(_clean(self.followup_query))

    # ── Writing the turn: the reconcile layer's channel ──────────────────────

    def adopt_request(self, kind: str, target: str) -> bool:
        """Take a request the model did not report. Returns whether it was taken.

        The deterministic layer in core.request_detection reads the caller's
        words for request shapes the extraction model missed. It fills gaps and
        never overrides: a request the model already reported keeps its target,
        and a pivot or a denial is left alone entirely — those are readings of
        the slot in hand, made on the caller's words, and a pattern match about
        some other slot does not outrank one.
        """
        kind, target = _clean(kind), _clean(target)
        if kind not in _REQUEST_INTENT_BY_KIND or not target:
            return False
        if self.turn_intent in (TurnIntent.PIVOT, TurnIntent.CANNOT_PROVIDE):
            return False
        self.turn_intent = _REQUEST_INTENT_BY_KIND[kind]
        self.turn_target = target
        return True

    def report_cannot_supply(self) -> bool:
        """Record a denial the model missed. Returns whether it was taken.

        Never overrides a value, a pivot or a request: the same precedence as
        adopt_request, and for the same reason.
        """
        if self.turn_intent not in (TurnIntent.ANSWERED, TurnIntent.UNUSABLE, TurnIntent.WAIT):
            return False
        if self.has_extracted_value:
            return False
        self.turn_intent, self.turn_target = TurnIntent.CANNOT_PROVIDE, None
        return True

    # ── The one derivation that needs state ──────────────────────────────────

    def split_corrections(
        self,
        confirmed_slots: Optional[Mapping[str, Any]],
        awaiting_slot: str = "",
        *,
        locked_slots: frozenset[str] = frozenset(),
    ) -> "WorkerResult":
        """Move values that replace a confirmed slot out of extracted{}.

        A spoken value is a correction when the pipeline already holds a
        different value for that slot and the caller was not being asked for it
        — which is the whole of the rule the headers used to spend three worked
        examples on. The awaiting slot is never a correction: answering the
        question you were asked is an answer, whatever is on file. A locked
        slot is never either, which is the LOCKED FIELDS section the headers
        used to carry.

        Runs once per result. The reconcile pass is re-run idempotently by
        several agents, and a second split against an already-split extracted{}
        would find nothing to move and blank the corrections it made.
        """
        if self._corrections_split or confirmed_slots is None:
            return self
        self._corrections_split = True
        values = {k: v for k, v in (self.extracted or {}).items() if v}
        if not values:
            return self
        kept: Dict[str, str] = {}
        moved: Dict[str, str] = {}
        dropped: list[str] = []
        for slot, value in values.items():
            # A locked slot is a system flag or a classification, never a value
            # the caller states — so it is neither a correction nor an answer,
            # and it leaves entirely. Keeping it in extracted{} would hand a
            # caller's words to a field they cannot set.
            if slot in locked_slots:
                dropped.append(slot)
                continue
            prior = confirmed_slots.get(slot)
            is_correction = (
                slot != awaiting_slot
                and isinstance(prior, str)
                and prior.strip()
                and prior.strip().casefold() != str(value).strip().casefold()
            )
            (moved if is_correction else kept)[slot] = value
        if moved or dropped:
            self.extracted = kept
            self._corrections = {**self._corrections, **moved}
        return self


class FollowUpIntent(str, Enum):
    DONE = "done"
    QUESTION = "question"
    UNSURE = "unsure"
    UPDATE_REQUEST = "update_request"
    NEW_INTENT = "new_intent"
    WAIT = "wait"


class FollowUpResult(BaseModel):
    """Dedicated schema for follow_up_agent: WorkerResult + generated answer."""

    model_config = ConfigDict(extra="forbid")

    extracted: Optional[Dict[str, str]] = None
    guard: GuardType = GuardType.NONE
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    follow_up_intent: FollowUpIntent = FollowUpIntent.UNSURE
    answer: Optional[str] = None
    detected_intent: Optional[str] = None
    # Cross-call request classification: "redo"/"replay" requests route to the
    # owning agent via the capability registry; "update" requests route via slot
    # ownership. request_target names the slot or topic.
    request_kind: RequestKind = RequestKind.NONE
    request_target: Optional[str] = None


class SsnIntent(str, Enum):
    YES_WITH_SSN = "yes_with_ssn"
    YES = "yes"
    NO = "no"
    NO_SSN_AVAILABLE = "no_ssn_available"
    # Caller changed their mind and wants to use their Member ID after all
    # ("actually I found it", "I have the member id now, can you use that").
    # Without this the flow has no way back and re-asks for the SSN forever.
    HAS_MEMBER_ID = "has_member_id"
    AMBIGUOUS = "ambiguous"


class SsnFallbackResult(BaseModel):
    """Schema for ssn_fallback.md extraction — used by extract_ssn_decision()."""

    model_config = ConfigDict(extra="forbid")

    ssn_intent: SsnIntent = SsnIntent.AMBIGUOUS
    ssn: Optional[str] = Field(
        default=None,
        description="Extracted SSN in XXX-XX-XXXX format, or null if not provided",
    )
