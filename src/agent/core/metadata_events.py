"""
metadata_events.py — the metadata events a call reports to the platform.

Two event types share ``State["metadata_events"]``.

CallAgentField, one per business field the call captures::

    {"eventType": "CallAgentField", "data": {"field": "intent", "value": "provider_services"}}
    {"eventType": "CallAgentField", "data": {"field": "first_name", "value": "Emily"}}

AgentCallEvent, how the call finished — a goodbye, a hard END, or a transfer::

    {"eventType": "AgentCallEvent", "data": {"eventName": "AgentCallEnded", "detail": "complete"}}
    {"eventType": "AgentCallEvent", "data": {"eventName": "AgentCallTransfer",
     "transferInitiator": "Agent", "detail": "abuse_detected", "referenceNumber": "REF123456789"}}

The transfer event is raised by ``signal_escalate`` at the moment an agent gives
up the call, so it carries that agent's reason and whether the caller asked to
be transferred; ``escalation_agent`` re-reports it with the reference number it
mints. merge_events keeps one event per eventName — the later, fuller one — so
the two reports are one event. AgentCallEnded is stamped on whatever turn routes
to END, and a transferred call reports the transfer instead: it did not end, it
was handed to a representative.

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
    loses it before the pause that would have delivered it. ``stamp_metadata_events``
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

EVENT_TYPE = "CallAgentField"
AGENT_CALL_EVENT_TYPE = "AgentCallEvent"

CALL_ENDED = "AgentCallEnded"
CALL_TRANSFER = "AgentCallTransfer"

# AgentCallEnded's detail when nothing cut the call short.
CALL_COMPLETE = "complete"

# next_node values that mean the graph is done. "__end__" is langgraph's own END
# sentinel; the agents write the plain string.
_END_NODES = frozenset({"END", "__end__"})


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


def agent_call_event(
    event_name: str,
    detail: str,
    *,
    initiator: Optional[str] = None,
    reference_number: str = "",
) -> dict:
    """Build one AgentCallEvent (the call lifecycle event)."""
    data: dict[str, str] = {"eventName": event_name, "detail": detail or ""}
    if initiator:
        data["transferInitiator"] = initiator
    if reference_number:
        data["referenceNumber"] = reference_number
    return {"eventType": AGENT_CALL_EVENT_TYPE, "data": data}


def transfer_event(detail: str, *, initiator: str = "Agent", reference_number: str = "") -> dict:
    """The call is being handed to a representative."""
    return agent_call_event(CALL_TRANSFER, detail, initiator=initiator, reference_number=reference_number)


def call_ended_event(detail: str = CALL_COMPLETE) -> dict:
    """The call finished with the agent."""
    return agent_call_event(CALL_ENDED, detail)


def find_agent_call_event(events: Optional[Iterable[dict]], event_name: str) -> Optional[dict]:
    """The AgentCallEvent of this name already reported, if there is one."""
    for event in events or []:
        if not isinstance(event, dict) or event.get("eventType") != AGENT_CALL_EVENT_TYPE:
            continue
        if (event.get("data") or {}).get("eventName") == event_name:
            return event
    return None


def is_call_ending(result: Mapping[str, Any]) -> bool:
    """True when this update routes the graph to END."""
    return str((result or {}).get("next_node") or "") in _END_NODES


def call_ended_detail(merged: Mapping[str, Any]) -> str:
    """Why the call ended, for the AgentCallEnded event.

    A goodbye reports "complete". A hard END reports what sent the caller
    elsewhere, in the order the sources are trustworthy: ``call_end_detail``,
    which a handler sets to say so itself; then intake's out-of-scope routing,
    which leaves its reason in escalation_reason; then the non-member guard,
    read off caller_type because it routes to a dedicated number without raising
    a reason of its own.
    """
    merged = merged or {}
    stated = str(merged.get("call_end_detail") or "").strip()
    if stated:
        return stated
    reason = str(merged.get("escalation_reason") or "").strip()
    if reason:
        return reason
    caller_type = str(merged.get("caller_type") or "").strip()
    if merged.get("caller_type_handled") and caller_type and caller_type != "member":
        return f"non-member caller: {caller_type}"
    return CALL_COMPLETE


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
        value = _format_value(raw)
        if not value or emitted.get(field) == value:
            continue
        emitted[field] = value
        events.append(field_event(field, value))
    return events, emitted


def merge_events(*groups: Optional[Iterable[dict]]) -> list[dict]:
    """Concatenate event groups, dropping duplicates and keeping order.

    A call reports one AgentCallEvent per eventName, so a later one replaces the
    earlier in place: escalation_agent re-reports the transfer its escalating
    agent raised, adding the reference number it mints, and the caller gets one
    transfer event carrying both halves.
    """
    merged: list[dict] = []
    lifecycle: dict[str, int] = {}
    for group in groups:
        for event in group or []:
            if not isinstance(event, dict):
                continue
            if event.get("eventType") == AGENT_CALL_EVENT_TYPE:
                name = str((event.get("data") or {}).get("eventName") or "")
                if name in lifecycle:
                    merged[lifecycle[name]] = event
                else:
                    lifecycle[name] = len(merged)
                    merged.append(event)
            elif event not in merged:
                merged.append(event)
    return merged
