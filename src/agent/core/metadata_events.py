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

The list is cumulative: agents carry it forward, the pause does not clear it,
and emitted_fields keeps a field from being reported twice at the same value. So
every pause hands over the call so far, and the turn that ends the call hands
over the whole call — every field it captured, then the AgentCallEvent saying
how it finished. replay_field_events backs that guarantee for a caller that
cleared the list mid-call.

Sources, in the order their events are appended:
  1. what this turn captured — slots confirmed on it (``SlotManagerMixin.slot_ok``),
     which is the only place a value that never reaches a state key shows up,
     plus whatever an agent recorded by hand (``BaseAgent.field_captured``);
  2. a sweep of the merged call view (``{**state, **result}``) over
     FIELD_EVENT_NAMES, which catches every direct state write, minus the fields
     standing at what the member record had on file (see "On file vs captured").
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

# The placeholders this codebase writes for "no value here yet" — the
# notification channels start "not_set" (reset_for_new_intent puts them back),
# an unclassified caller is "unknown". A placeholder is not a capture, so it is
# not reported; the field simply stays absent until something real fills it.
SENTINEL_VALUES: frozenset[str] = frozenset({"not_set", "unknown"})

# next_node values that mean the graph is done. "__end__" is langgraph's own END
# sentinel; the agents write the plain string.
_END_NODES = frozenset({"END", "__end__"})


def _format_value(value: Any) -> str:
    """Render a state value as the event's string value, "" when not reportable.

    Falsy values (unset slots, flags still False) are not a capture, nor are the
    placeholders that stand in for one (SENTINEL_VALUES), and the containers
    state uses for its own bookkeeping (slot_attempts, the parked follow-ups,
    saved member context) are not fields.
    """
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    if isinstance(value, bool):
        return "true" if value else ""
    text = str(value).strip()
    return "" if text.lower() in SENTINEL_VALUES else text


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


def replay_field_events(emitted: Optional[Mapping[str, str]]) -> list[dict]:
    """Every field the call captured, as CallAgentField events.

    Built from emitted_fields, so it is the whole call in the order the fields
    were first captured, each at the value that ended up sticking. Stamped on the
    turn that ends the call, where it is normally a no-op — the list is
    cumulative — and where it is the whole record for anything that cleared the
    list mid-call.
    """
    return [field_event(field, value) for field, value in (emitted or {}).items() if value]


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


# ── On file vs captured ──────────────────────────────────────────────────────
# The member record carries contact fields the call may never ask about. The
# lookup hydrates them into state so an agent that needs one has it without a
# second Salesforce call — delivery reads the fax on file to read it back, the
# provider search starts from the ZIP on file, care coach reuses the email —
# and the sweep above then reported every one of them, because a state key with
# a value is indistinguishable from a capture. A claim call that asked for a
# name, a member ID, a date of birth and a phone confirmation reported a ZIP, a
# fax and an email it never mentioned, the email while its own flow was still
# several turns short of asking for one.
#
# A value read off the record is what the plan has on file, not something this
# call captured. So the hydrating agent marks those fields in
# ``State["fields_on_file"]`` and the sweep passes them over. The mark is
# released — and the field reported — the moment the call actually captures the
# value: the caller supplies or confirms it through a slot (``slot_ok``), or the
# agent that puts it to use records it (``BaseAgent.field_captured``) — the
# phone read-back the caller confirms, the fax the provider list actually went
# to, the email the upload link was sent to. Nothing else changes: a field the
# call writes to state on its own is still reported by the sweep, marked or not,
# because capture is what the mark is about, not the state key.
#
# Marks are keyed by reported field name, like emitted_fields.

# What the member lookup hydrates from the record. relationship is not here:
# the record's Relationship__c is the list of relationships the account allows
# ("plan holder, subscriber, spouse"), not this caller's — only the caller's own
# answer to "are you the subscriber or dependent?" belongs in that field.
ON_FILE_FIELDS: tuple[str, ...] = ("phone_number", "zip_code", "fax", "email")


def reported_field(key: str) -> str:
    """The field name a state key or slot reports under."""
    return FIELD_EVENT_NAMES.get(key, key)


def mark_on_file(
    existing: Optional[Iterable[str]],
    keys: Iterable[str],
    *,
    emitted: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Add ``keys`` to the on-file marks, keeping order and dropping repeats.

    A field the call has already reported (``emitted``) is never marked back on
    file. A verified call re-enters verification with its record rebuilt out of
    state, so the fax the caller gave would come back round as record data and
    be filed away as something the call never said.
    """
    reported = set(emitted or {})
    marks = list(existing or [])
    for key in keys:
        field = reported_field(key)
        if field in reported or field in marks:
            continue
        marks.append(field)
    return marks


def release_on_file(existing: Optional[Iterable[str]], captured: Iterable[str]) -> list[str]:
    """The marks that survive a turn — every field it captured is now the call's."""
    taken = {reported_field(key) for key in captured}
    return [field for field in (existing or []) if field not in taken]


def sweep_captured(merged: Mapping[str, Any], on_file: Optional[Iterable[str]]) -> list[Tuple[str, Any]]:
    """The reportable (key, value) pairs of the merged call view.

    Every field event name present in the call, minus the ones standing at the
    value the member record supplied and not yet captured by this call.
    """
    skip = set(on_file or [])
    return [
        (key, merged.get(key))
        for key in FIELD_EVENT_NAMES
        if key in merged and reported_field(key) not in skip
    ]


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
