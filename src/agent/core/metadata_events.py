"""
metadata_events.py — CallAgentField metadata events.

Every business field a call captures is reported to the platform as a
CallAgentField event on ``State["metadata_events"]``::

    {"eventType": "CallAgentField", "data": {"field": "intent", "value": "provider_services"}}
    {"eventType": "CallAgentField", "data": {"field": "first_name", "value": "Emily"}}

Emission is central — BaseAgent.execute() stamps the events onto the finished
update dict of every agent turn — because two things make a per-agent approach
lose fields:

  * Most captured values never pass through ``slot_ok()``. Intake classifies the
    intent and writes ``call_intent`` straight into its bridge dict; the claim
    fallback values, the notification channel and contact, and the escalation
    reference are all plain state writes. Only pipeline-collected values are
    slots, so slot confirmations alone report a fraction of the call.
  * ``metadata_events`` has no reducer — every node overwrites it — so an agent
    that emits an event and then hands off to another agent inside the same turn
    loses it before the pause that would have delivered it. ``stamp_field_events``
    carries forward whatever earlier nodes staged this turn, and ``human_node``
    clears the list once the pause has delivered it.

``State["emitted_fields"]`` remembers field → last reported value, so each field
is reported once per distinct value: re-confirming a slot on a later turn emits
nothing, changing it (fax → email) emits the new value.

Sources, in the order their events are appended:
  1. slots confirmed on this turn (``SlotManagerMixin.slot_ok``), which is the
     only place a value that never reaches a state key shows up;
  2. a sweep of the merged call view (``{**state, **result}``) over
     FIELD_EVENT_NAMES, which catches every direct state write.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple

# State key (or slot name) → the field name reported to the platform.
# A slot whose name is not listed reports under its own name, so a new slot
# needs no entry here; list a key only to rename it or to report a value that
# is written straight to state and never collected as a slot.
FIELD_EVENT_NAMES: dict[str, str] = {
    # ── Intent ───────────────────────────────────────────────────────────────
    "intent": "intent",  # intake's slot name
    "call_intent": "intent",  # the state key the same value lands in
    "provider_type": "provider_type",
    # ── Caller identity ──────────────────────────────────────────────────────
    "first_name": "first_name",
    "last_name": "last_name",
    "member_id": "member_id",
    "ssn": "ssn",
    "dob": "dob",
    "relationship": "relationship",
    "caller_role": "caller_role",
    "caller_type": "caller_type",
    "member_status_verify": "member_verified",
    # ── Contact ──────────────────────────────────────────────────────────────
    "phone_number": "phone_number",
    "zip_code": "zip_code",
    "fax": "fax",
    "email": "email",
    # ── Provider list delivery ───────────────────────────────────────────────
    "delivery_method": "delivery_method",
    # ── Claim adjustment ─────────────────────────────────────────────────────
    "reference_number": "reference_number",
    "fallback_claim_number": "claim_number",
    "fallback_dos": "date_of_service",
    "fallback_billed_amount": "billed_amount",
    "claim_status": "claim_status",
    "records_branch_taken": "records_branch",
    "notification_channel": "notification_channel",
    "claim_notification_contact": "notification_contact",
    "claim_timeline_notification_channel": "timeline_notification_channel",
    "claim_timeline_notification_contact": "timeline_notification_contact",
    # ── Call reference ───────────────────────────────────────────────────────
    "ref_no": "ref_no",
    "escalation_reference_number": "escalation_reference_number",
}

# Reported as a field, but never with the raw value in it. The platform gets
# enough to match the record without the call metadata carrying a full SSN.
MASKED_FIELDS: frozenset[str] = frozenset({"ssn"})

EVENT_TYPE = "CallAgentField"


def _mask(field: str, value: str) -> str:
    """Mask a sensitive field's value, keeping the last four characters."""
    if field not in MASKED_FIELDS:
        return value
    digits = "".join(c for c in value if c.isalnum())
    return f"****{digits[-4:]}" if len(digits) > 4 else "****"


def _format_value(value: Any) -> str:
    """Render a state value as the event's string value, "" when not reportable.

    Falsy values (unset slots, flags still False) are not a capture, and the
    containers state uses for its own bookkeeping (slot_attempts, the parked
    follow-ups, saved member context) are not fields.
    """
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    if isinstance(value, bool):
        return "true" if value else ""
    return str(value).strip()


def field_event(field: str, value: str) -> dict:
    """Build one CallAgentField event."""
    return {"eventType": EVENT_TYPE, "data": {"field": field, "value": value}}


def build_field_events(
    already_emitted: Optional[Mapping[str, str]],
    captured: Iterable[Tuple[str, Any]],
) -> Tuple[list[dict], dict[str, str]]:
    """Events for the captured (key, value) pairs the call has not reported yet.

    ``captured`` is read in order and keyed by state key or slot name; the
    reported field name comes from FIELD_EVENT_NAMES. Returns the new events and
    the full field → value map to persist as ``emitted_fields`` (full, not a
    delta: the state key has no reducer).
    """
    emitted = dict(already_emitted or {})
    events: list[dict] = []
    for key, raw in captured:
        field = FIELD_EVENT_NAMES.get(key, key)
        value = _mask(field, _format_value(raw))
        if not value or emitted.get(field) == value:
            continue
        emitted[field] = value
        events.append(field_event(field, value))
    return events, emitted


def merge_events(*groups: Optional[Iterable[dict]]) -> list[dict]:
    """Concatenate event groups, dropping duplicates and keeping order."""
    merged: list[dict] = []
    for group in groups:
        for event in group or []:
            if event not in merged:
                merged.append(event)
    return merged
