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

What the model reports now is one classification and one target:

    extracted        — the values the caller spoke this turn
    turn_intent      — what the utterance DID (TurnIntent)
    turn_target      — the slot, topic or identifier that intent points at
    guard            — safety / routing guard (a separate concern, unchanged)
    guard_confidence
    followup_query   — the side question, in the caller's words

Everything downstream still reads `event_type`, `corrections`, `update_target`,
`request_kind`, `cannot_provide` and `fallback_pivot`. Those are now DERIVED
here, from the reported intent plus what Python already knows, and the
derivation is one-way by construction: combinations the old schema allowed and
no code could handle — a pivot that is also a denial, a CORRECTED with empty
corrections and no target, an ANSWERED_WITH_FOLLOWUP carrying no question —
cannot be represented at all, so the defensive branches that used to catch
them have nothing left to catch.

`corrections` is the one derived value that needs state rather than arithmetic:
whether a spoken value is new or a correction depends on what is already
confirmed, which the pipeline knows and the model does not. `split_corrections`
applies it at the extraction boundary (see core.request_detection).
"""

from enum import Enum
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class EventType(str, Enum):
    """Legacy turn classification. Derived from TurnIntent — see
    ``WorkerResult.event_type``. Nothing sets it on the model any more."""

    ANSWERED = "answered"
    ANSWERED_WITH_FOLLOWUP = "answered_with_followup"
    CORRECTED = "corrected"
    AMBIGUOUS = "ambiguous"
    WAIT = "wait"  # caller asked for time: "give me a minute", "hold on"
    NONE = "none"


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


class FollowupDisposition(str, Enum):
    """How a side question is handled. Python's to choose, never the model's —
    the headers already instructed it to emit "none" on every turn, so the
    field was pure prompt cost. Kept as a Python-side attribute because
    request_detection's grounding recovery and slot_manager's park path both
    set it; see ``WorkerResult.followup_disposition``."""

    ANSWER = "answer"  # answer from Confirmed: if possible; gracefully decline if not
    PARK = "park"  # an update another flow owns, carried to it; never a question
    NONE = "none"  # default
    # Legacy aliases kept for backward compat with cached extraction results
    ANSWER_NOW = "answer_now"
    DECLINE = "decline"


class RequestKind(str, Enum):
    """Cross-call request shapes. On WorkerResult this is derived from
    TurnIntent; FollowUpResult still reports it directly, because the follow-up
    agent classifies a request with no slot being collected around it."""

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


# ── Legacy payload translation ───────────────────────────────────────────────
# Extraction results cached before the schema narrowed, and the hand-built
# WorkerResults in tests and live_e2e, still arrive in the old shape. They are
# translated once, here, on the way in — which is also where the old precedence
# prose ends up as code: pivot beats denial, a named request beats an event
# label, and a value beats all of them.
_LEGACY_EVENT_INTENT: dict[str, TurnIntent] = {
    "answered": TurnIntent.ANSWERED,
    "answered_with_followup": TurnIntent.ANSWERED,
    "corrected": TurnIntent.ANSWERED,  # the corrections{} carry it
    "ambiguous": TurnIntent.UNUSABLE,
    "wait": TurnIntent.WAIT,
    "none": TurnIntent.UNUSABLE,  # a guard fired; guard fields carry that
}

_LEGACY_KIND_INTENT: dict[str, TurnIntent] = {
    "update": TurnIntent.UPDATE,
    "redo": TurnIntent.REDO,
    "replay": TurnIntent.REPLAY,
}

_LEGACY_KEYS = (
    "event_type",
    "corrections",
    "update_target",
    "request_kind",
    "cannot_provide",
    "fallback_pivot",
    "followup_disposition",
    "needs_freeform_response",
)


def _enum_value(raw: Any) -> str:
    """Normalize an enum member, a bare string or None to a lowercase string."""
    return str(getattr(raw, "value", raw) or "").strip().lower()


def _translate_legacy(data: dict) -> tuple[dict, dict, Optional[str]]:
    """Fold a pre-narrowing payload into (new fields, corrections, disposition)."""
    legacy = {k: data.pop(k) for k in _LEGACY_KEYS if k in data}
    if not legacy:
        return data, {}, None

    # corrections{} the old shape reported are taken as already split: the
    # payload carries the model's own verdict on them and there is no fresher
    # Confirmed: view to re-derive it from.
    corrections = {k: v for k, v in (legacy.get("corrections") or {}).items() if isinstance(v, str) and v}

    target = (legacy.get("update_target") or "").strip()
    pivot = (legacy.get("fallback_pivot") or "").strip()
    kind = _enum_value(legacy.get("request_kind"))
    event = _enum_value(legacy.get("event_type"))

    if pivot:
        data["turn_intent"], data["turn_target"] = TurnIntent.PIVOT, pivot
    elif legacy.get("cannot_provide"):
        data["turn_intent"], data["turn_target"] = TurnIntent.CANNOT_PROVIDE, None
    elif target or kind in _LEGACY_KIND_INTENT:
        data["turn_intent"] = _LEGACY_KIND_INTENT.get(kind, TurnIntent.UPDATE)
        data["turn_target"] = target or None
    elif event:
        data["turn_intent"] = _LEGACY_EVENT_INTENT.get(event, TurnIntent.ANSWERED)

    disposition = _enum_value(legacy.get("followup_disposition")) or None
    return data, corrections, disposition


class WorkerResult(BaseModel):
    """One extraction turn. Six reported fields; the rest is derived.

    ``extra="forbid"`` keeps the model honest about the narrow contract, and
    the wrap validator below is what lets a legacy payload through it.
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
    _corrections: Dict[str, str] = PrivateAttr(default_factory=dict)
    _corrections_split: bool = PrivateAttr(default=False)
    _followup_disposition: FollowupDisposition = PrivateAttr(default=FollowupDisposition.NONE)

    @model_validator(mode="wrap")
    @classmethod
    def _accept_legacy_payload(cls, data: Any, handler):
        if not isinstance(data, Mapping):
            return handler(data)
        payload, corrections, disposition = _translate_legacy(dict(data))
        obj = handler(payload)
        if corrections:
            obj._corrections = corrections
            obj._corrections_split = True
        if disposition:
            try:
                obj._followup_disposition = FollowupDisposition(disposition)
            except ValueError:
                pass
        return obj

    # ── Derived: what the pipelines read ─────────────────────────────────────

    @property
    def has_extracted_value(self) -> bool:
        return any(v for v in (self.extracted or {}).values())

    @property
    def event_type(self) -> EventType:
        """The legacy turn classification, derived.

        A value the caller spoke outranks every no-value intent — the rule the
        three headers each wrote out ("a value given in the same utterance
        always wins") — so that comparison happens once, here.
        """
        intent = self.turn_intent
        has_value = self.has_extracted_value
        if intent is TurnIntent.WAIT and not has_value:
            return EventType.WAIT
        if intent in NO_VALUE_INTENTS and not has_value:
            return EventType.AMBIGUOUS
        if intent in REQUEST_INTENTS:
            return EventType.ANSWERED_WITH_FOLLOWUP if has_value else EventType.CORRECTED
        if any(self._corrections.values()):
            return EventType.ANSWERED_WITH_FOLLOWUP if has_value else EventType.CORRECTED
        if (self.followup_query or "").strip():
            return EventType.ANSWERED_WITH_FOLLOWUP if has_value else EventType.AMBIGUOUS
        return EventType.ANSWERED

    @event_type.setter
    def event_type(self, value: Any) -> None:
        """Legacy write path — mapped back onto turn_intent.

        Kept for shims and cached results that still assign an event. The
        reconcile layer no longer uses it: it changes the intent, and the event
        follows.
        """
        event = _enum_value(value)
        if event == "wait":
            self.turn_intent = TurnIntent.WAIT
        elif event in ("ambiguous", "none"):
            if self.turn_intent not in NO_VALUE_INTENTS:
                self.turn_intent = TurnIntent.UNUSABLE
        elif event == "corrected":
            if self.turn_intent not in REQUEST_INTENTS and not any(self._corrections.values()):
                self.turn_intent = TurnIntent.UPDATE
        elif event in ("answered", "answered_with_followup"):
            if self.turn_intent in (TurnIntent.WAIT, TurnIntent.UNUSABLE):
                self.turn_intent = TurnIntent.ANSWERED

    @property
    def corrections(self) -> Dict[str, str]:
        """Values that replace something already confirmed.

        Filled by split_corrections against the same Confirmed: view the
        extraction prompt was given, so the answer is identical to the one the
        model used to guess at — and available on turns where it guessed wrong.
        """
        return self._corrections

    @corrections.setter
    def corrections(self, value: Optional[Mapping[str, str]]) -> None:
        self._corrections = {k: v for k, v in (value or {}).items() if v}

    @property
    def update_target(self) -> Optional[str]:
        return self.turn_target if self.turn_intent in REQUEST_INTENTS else None

    @update_target.setter
    def update_target(self, value: Optional[str]) -> None:
        target = (value or "").strip()
        if not target:
            if self.turn_intent in REQUEST_INTENTS:
                self.turn_intent, self.turn_target = TurnIntent.ANSWERED, None
            return
        # A pivot or a denial is a reading of THIS slot the model made on the
        # words; a request target is a gap-filler's guess about another one.
        # The specific reading wins, and does so structurally.
        if self.turn_intent in (TurnIntent.PIVOT, TurnIntent.CANNOT_PROVIDE):
            return
        if self.turn_intent not in REQUEST_INTENTS:
            self.turn_intent = TurnIntent.UPDATE
        self.turn_target = target

    @property
    def request_kind(self) -> RequestKind:
        if self.turn_intent in REQUEST_INTENTS:
            return RequestKind(self.turn_intent.value)
        return RequestKind.NONE

    @request_kind.setter
    def request_kind(self, value: Any) -> None:
        kind = _enum_value(value)
        if kind in _LEGACY_KIND_INTENT:
            if self.turn_intent not in (TurnIntent.PIVOT, TurnIntent.CANNOT_PROVIDE):
                self.turn_intent = _LEGACY_KIND_INTENT[kind]
        elif self.turn_intent in REQUEST_INTENTS:
            self.turn_intent, self.turn_target = TurnIntent.ANSWERED, None

    @property
    def cannot_provide(self) -> bool:
        return self.turn_intent is TurnIntent.CANNOT_PROVIDE and not self.has_extracted_value

    @cannot_provide.setter
    def cannot_provide(self, value: bool) -> None:
        if value:
            if self.turn_intent in (TurnIntent.ANSWERED, TurnIntent.UNUSABLE, TurnIntent.WAIT):
                self.turn_intent, self.turn_target = TurnIntent.CANNOT_PROVIDE, None
        elif self.turn_intent is TurnIntent.CANNOT_PROVIDE:
            self.turn_intent = TurnIntent.UNUSABLE

    @property
    def fallback_pivot(self) -> Optional[str]:
        if self.turn_intent is TurnIntent.PIVOT and not self.has_extracted_value:
            return self.turn_target
        return None

    @fallback_pivot.setter
    def fallback_pivot(self, value: Optional[str]) -> None:
        pivot = (value or "").strip()
        if pivot:
            self.turn_intent, self.turn_target = TurnIntent.PIVOT, pivot
        elif self.turn_intent is TurnIntent.PIVOT:
            self.turn_intent, self.turn_target = TurnIntent.UNUSABLE, None

    @property
    def followup_disposition(self) -> FollowupDisposition:
        return self._followup_disposition

    @followup_disposition.setter
    def followup_disposition(self, value: Any) -> None:
        try:
            self._followup_disposition = FollowupDisposition(_enum_value(value) or "none")
        except ValueError:
            self._followup_disposition = FollowupDisposition.NONE

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
