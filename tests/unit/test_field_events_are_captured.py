"""Every field the call captures leaves a CallAgentField event behind.

    {"eventType": "CallAgentField", "data": {"field": "intent", "value": "provider_services"}}
    {"eventType": "CallAgentField", "data": {"field": "first_name", "value": "Emily"}}

Only confirmed slots were ever considered for emission, and even that was
commented out — every agent shipped `metadata_events = []`. Most of what a call
captures is not a slot: intake classifies the intent and writes `call_intent`
onto its bridge dict, the claim fallback values, the notification channel and
contact, and the escalation reference are all plain state writes.

So the events are stamped centrally, by BaseAgent.execute() on the finished
update dict, from both sources — the turn's confirmed slots and a sweep of the
merged call view. This test pins the four properties that make that trustworthy:
every captured field is reported, a field is reported once per distinct value, a
changed value is reported again, and events staged earlier in a turn survive the
hand-off to the next agent (metadata_events has no reducer).
"""

from __future__ import annotations

import asyncio

from agent.agents.escalation.agent import EscalationAgent
from agent.core.agent import BaseAgent
from agent.core.metadata_events import build_field_events, field_event

REF = "REF123456789"


class _Agent(BaseAgent):
    """Minimal agent whose turn is whatever update dict the test hands it."""

    AGENT_NAME = "test_agent"

    def __init__(self, result: dict, confirmed: dict | None = None) -> None:
        super().__init__()
        self._result = result
        for name, value in (confirmed or {}).items():
            self.slot_ok(name, value)

    async def run(self, state):  # noqa: D102 — the turn under test
        return dict(self._result)


def _turn(state: dict, result: dict, confirmed: dict | None = None) -> dict:
    return asyncio.run(_Agent(result, confirmed).execute(state))


def _fields(update: dict) -> list[tuple[str, str]]:
    return [
        (e["data"]["field"], e["data"]["value"])
        for e in update["metadata_events"]
        if e["eventType"] == "CallAgentField"
    ]


def _lifecycle(update: dict) -> list[str]:
    return [e["data"]["eventName"] for e in update["metadata_events"] if e["eventType"] == "AgentCallEvent"]


def _escalate(state: dict) -> dict:
    """One escalation_agent turn, on a call whose reference number is pinned."""
    return asyncio.run(EscalationAgent.from_state(state).execute({"ref_no": REF, **state}))


def test_slot_and_state_written_fields_are_both_reported():
    update = _turn(
        {},
        {"call_intent": "provider_services", "metadata_events": []},
        confirmed={"first_name": "Emily"},
    )
    assert _fields(update) == [("first_name", "Emily"), ("intent", "provider_services")]
    assert all(e["eventType"] == "CallAgentField" for e in update["metadata_events"])


def test_every_reportable_field_of_a_verified_call_is_reported():
    update = _turn(
        {"call_intent": "claim_services"},
        {
            "first_name": "Emily",
            "last_name": "Carter",
            "member_id": "M123456",
            "dob": "10/18/1990",
            "relationship": "subscriber",
            "member_status_verify": True,
            "phone_number": "5551234567",
            "zip_code": "06103",
            "email": "emily@example.com",
            "delivery_method": "email",
            "reference_number": "ADJ98765",
        },
    )
    assert dict(_fields(update)) == {
        "intent": "claim_services",
        "first_name": "Emily",
        "last_name": "Carter",
        "member_id": "M123456",
        "dob": "10/18/1990",
        "relationship": "subscriber",
        "member_verified": "true",
        "phone_number": "5551234567",
        "zip_code": "06103",
        "email": "emily@example.com",
        "delivery_method": "email",
        "reference_number": "ADJ98765",
    }


def test_a_field_is_reported_once_per_value_and_again_when_it_changes():
    first = _turn({}, {"delivery_method": "fax", "fax": "8605551234"})

    # Same values on a later turn — nothing new to report.
    carried = {"emitted_fields": first["emitted_fields"], "metadata_events": []}
    again = _turn({**carried, "delivery_method": "fax", "fax": "8605551234"}, {})
    assert again["metadata_events"] == []

    # The caller switches channel — the new value is reported.
    switched = _turn(
        {**carried, "delivery_method": "fax"},
        {"delivery_method": "email", "email": "emily@example.com"},
    )
    assert _fields(switched) == [("email", "emily@example.com"), ("delivery_method", "email")]


def test_events_survive_a_hand_off_inside_one_turn():
    """metadata_events has no reducer, so each node must carry the turn forward."""
    intake = _turn({}, {"call_intent": "provider_services"})
    verification = _turn(
        {"metadata_events": intake["metadata_events"], "emitted_fields": intake["emitted_fields"]},
        {"first_name": "Emily"},
    )
    assert _fields(verification) == [("intent", "provider_services"), ("first_name", "Emily")]


def test_ssn_is_reported_with_the_value_the_lookup_used():
    update = _turn({}, {"ssn": "123-45-6789"})
    assert _fields(update) == [("ssn", "123-45-6789")]


def test_unset_and_non_field_state_is_not_reported():
    update = _turn(
        {},
        {
            "first_name": "",
            "last_name": None,
            "member_status_verify": False,
            "slot_attempts": {"first_name": {"confirmed": True}},
            "messages": [{"role": "assistant", "content": "hi"}],
        },
    )
    assert update["metadata_events"] == []


def test_build_field_events_keys_dedupe_by_reported_name():
    """intake's "intent" slot and the call_intent state key are one field."""
    events, emitted = build_field_events(
        {}, [("intent", "provider_services"), ("call_intent", "provider_services")]
    )
    assert events == [field_event("intent", "provider_services")]
    assert emitted == {"intent": "provider_services"}


# ── AgentCallEvent — how the call finished ───────────────────────────────────
#
# The transfer event was written and then commented out, on both halves: the
# escalating agent's (reason + initiator) and escalation_agent's (the reference
# number it mints). AgentCallEnded was never raised at all, so a call that ran
# to its goodbye reported nothing about having ended.


def test_a_goodbye_reports_the_call_ended_complete():
    update = _turn({}, {"next_node": "END"})
    assert update["metadata_events"] == [
        {"eventType": "AgentCallEvent", "data": {"eventName": "AgentCallEnded", "detail": "complete"}}
    ]


def test_a_hard_end_reports_what_cut_the_call_short():
    out_of_scope = _turn({}, {"next_node": "END", "escalation_reason": "out_of_scope"})
    assert out_of_scope["metadata_events"][-1]["data"] == {
        "eventName": "AgentCallEnded",
        "detail": "out_of_scope",
    }

    non_member = _turn({}, {"next_node": "END", "caller_type": "provider", "caller_type_handled": True})
    assert non_member["metadata_events"][-1]["data"] == {
        "eventName": "AgentCallEnded",
        "detail": "non-member caller: provider",
    }

    # A handler that ends the call for a reason of its own says so directly.
    declined = _turn({}, {"next_node": "END", "call_end_detail": "phone_not_confirmed"})
    assert declined["metadata_events"][-1]["data"] == {
        "eventName": "AgentCallEnded",
        "detail": "phone_not_confirmed",
    }


def test_escalating_reports_a_transfer_with_its_reason_and_initiator():
    agent = _Agent({})
    escalation = agent.signal_escalate({}, "One moment.", "abuse_detected", initiator="Caller")
    assert escalation["metadata_events"] == [
        {
            "eventType": "AgentCallEvent",
            "data": {
                "eventName": "AgentCallTransfer",
                "detail": "abuse_detected",
                "transferInitiator": "Caller",
            },
        }
    ]


def test_the_reference_number_joins_the_transfer_already_reported():
    """One transfer event per call — escalation_agent re-reports the same one."""
    staged = _turn({}, _Agent({}).signal_escalate({}, "", "requested", initiator="Caller"))

    escalation = _escalate({"metadata_events": staged["metadata_events"]})
    transfers = [
        e for e in escalation["metadata_events"] if e["data"].get("eventName") == "AgentCallTransfer"
    ]
    assert transfers == [
        {
            "eventType": "AgentCallEvent",
            "data": {
                "eventName": "AgentCallTransfer",
                "detail": "requested",
                "transferInitiator": "Caller",
                "referenceNumber": REF,
            },
        }
    ]


def test_a_transferred_call_does_not_also_report_that_it_ended():
    """It did not end — it was handed to a representative."""
    escalation = _escalate({})
    assert escalation["next_node"] == "END"
    assert _lifecycle(escalation) == ["AgentCallTransfer"]


# ── The end of a call carries the whole call ─────────────────────────────────
#
# Delivery is per turn: a pause hands the platform what that turn captured and
# human_node clears the list. That leaves the end of the call — the one payload
# something reading the call afterwards actually looks at — holding the last
# turn's leftovers. The ending turn replays the record instead.


def test_the_ending_turn_replays_every_field_the_call_captured():
    captured = {}
    for turn in ({"call_intent": "claim_services"}, {"first_name": "Emily"}, {"member_id": "M123456"}):
        update = _turn({"emitted_fields": captured, "metadata_events": []}, turn)
        captured = update["emitted_fields"]  # each pause delivered and cleared

    goodbye = _turn({"emitted_fields": captured, "metadata_events": []}, {"next_node": "END"})
    assert _fields(goodbye) == [
        ("intent", "claim_services"),
        ("first_name", "Emily"),
        ("member_id", "M123456"),
    ]
    assert _lifecycle(goodbye) == ["AgentCallEnded"]


def test_a_replayed_field_is_not_reported_twice():
    """The ending turn's own captures are already in the replay."""
    update = _turn({}, {"call_intent": "claim_services", "next_node": "END"})
    assert _fields(update) == [("intent", "claim_services")]


def test_a_transferred_call_also_ends_with_the_whole_record():
    staged = _turn({}, _Agent({}).signal_escalate({}, "", "requested", initiator="Caller"))
    escalation = _escalate(
        {"metadata_events": staged["metadata_events"], "emitted_fields": {"intent": "claim_services"}}
    )
    assert ("intent", "claim_services") in _fields(escalation)
    assert _lifecycle(escalation) == ["AgentCallTransfer"]
