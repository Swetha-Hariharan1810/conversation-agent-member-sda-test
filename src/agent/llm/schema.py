from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    ANSWERED = "answered"
    ANSWERED_WITH_FOLLOWUP = "answered_with_followup"
    CORRECTED = "corrected"
    AMBIGUOUS = "ambiguous"
    WAIT = "wait"  # caller asked for time: "give me a minute", "hold on"
    NONE = "none"


class FollowupDisposition(str, Enum):
    ANSWER = "answer"  # answer from Confirmed: if possible; gracefully decline if not
    PARK = "park"  # answerable later in this call (maps to a pending slot or later stage)
    NONE = "none"  # default when event_type != answered_with_followup
    # Legacy aliases kept for backward compat with cached extraction results
    ANSWER_NOW = "answer_now"
    DECLINE = "decline"


class RequestKind(str, Enum):
    """Cross-call request shapes (Phase 6). See CROSS-CALL REQUESTS in the
    extraction headers. "update" = change a stored value; "redo" = re-perform
    a completed action with a changed parameter ("send it by email instead");
    "replay" = re-state information already given ("repeat my benefits")."""

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


class WorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Slot values extracted from the caller's utterance this turn
    extracted: Optional[Dict[str, str]] = None
    # Slot corrections detected (e.g. "actually my name is James")
    corrections: Optional[Dict[str, str]] = None
    # What the caller's utterance did relative to the awaiting slot
    event_type: EventType = EventType.ANSWERED
    # Safety / routing guard triggered this turn
    guard: GuardType = GuardType.NONE
    # LLM's confidence in the guard classification (0.0–1.0)
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # How to handle the side question when event_type == ANSWERED_WITH_FOLLOWUP
    followup_disposition: FollowupDisposition = FollowupDisposition.NONE
    # The side question, condensed, verbatim-ish
    followup_query: Optional[str] = None
    # Slot the caller wants to change when NO new value was given; for
    # redo/replay requests, the topic being redone/replayed
    update_target: Optional[str] = None
    # Shape of the cross-call request when update_target is set (Phase 6):
    # "update" (default for bare value changes), "redo", or "replay"
    request_kind: RequestKind = RequestKind.NONE
    # Claim fallback pivot: caller signals they want to use a different identifier
    # (no value yet). Values: "reference_number" | "claim_number" | "dos_billed" | None
    fallback_pivot: Optional[str] = None


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
    # Cross-call request classification (Phase 6): "redo"/"replay" requests
    # route to the owning agent via the capability registry; "update" requests
    # route via slot ownership. request_target names the slot or topic.
    request_kind: RequestKind = RequestKind.NONE
    request_target: Optional[str] = None


class SsnIntent(str, Enum):
    YES_WITH_SSN = "yes_with_ssn"
    YES = "yes"
    NO = "no"
    NO_SSN_AVAILABLE = "no_ssn_available"
    AMBIGUOUS = "ambiguous"


class SsnFallbackResult(BaseModel):
    """Schema for ssn_fallback.md extraction — used by extract_ssn_decision()."""

    model_config = ConfigDict(extra="forbid")

    ssn_intent: SsnIntent = SsnIntent.AMBIGUOUS
    ssn: Optional[str] = Field(
        default=None,
        description="Extracted SSN in XXX-XX-XXXX format, or null if not provided",
    )
