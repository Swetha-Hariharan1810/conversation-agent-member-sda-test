"""What the member record carried is not what the call captured.

A claim adjustment call — name, member ID, date of birth, a phone read-back the
caller confirmed, then the claim — reported this:

    zip_code   78701
    fax        512-555-6199
    email      james.wilson@gmail.com
    relationship  "plan holder, subscriber, spouse"

None of it was said on the call. The lookup hydrates the contact fields into
state so the agents downstream have them without a second Salesforce call, and
the metadata sweep could not tell a hydrated key from a captured one, so it
reported all of them — the email while the records flow was still several turns
short of asking for one. relationship was worse than early: the record's
Relationship__c is the list of relationships the account allows, not this
caller's, and a claims call never asks the question at all.

So the lookup marks what it carried (``State["fields_on_file"]``), the sweep
passes those over, and the turn that puts a value to the caller releases it.
relationship is not carried at all.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.agents.verification.handlers import collect_post_lookup
from agent.core.metadata_events import ON_FILE_FIELDS, mark_on_file, release_on_file, sweep_captured

RECORD = {
    "phone_number": "512-555-6101",
    "zip_code": "78701",
    "fax": "512-555-6199",
    "email": "james.wilson@gmail.com",
    # The account's allowed relationships — a menu, not an answer.
    "relationship": "plan holder, subscriber, spouse",
}

COLLECTED = {
    "first_name": "James",
    "last_name": "Wilson",
    "member_id": "M310188",
    "dob": "07/30/1977",
}


def _verified(state: dict | None = None, record: dict | None = None) -> dict:
    agent = VerificationAgent()
    base = {"call_intent": "claim_services", "messages": []}
    return agent._signal_verified({**base, **(state or {})}, dict(COLLECTED), record or dict(RECORD))


def test_the_lookup_marks_every_contact_field_it_carried():
    update = _verified()
    # The values are in state — delivery, provider search and care coach read them.
    assert update["zip_code"] == "78701"
    assert update["fax"] == "512-555-6199"
    # And every one of them is marked as the record's, not the call's.
    assert update["fields_on_file"] == list(ON_FILE_FIELDS)


def test_the_records_relationship_is_never_taken_as_the_callers_answer():
    """Only "are you the subscriber or dependent?" fills that field."""
    assert "relationship" not in ON_FILE_FIELDS
    assert not _verified().get("relationship")


def test_a_field_the_call_already_reported_is_not_filed_away_again():
    """A verified call re-enters verification with its record rebuilt from state."""
    captured_fax = {"emitted_fields": {"fax": "512-555-0000"}, "fax": "512-555-0000"}
    update = _verified(captured_fax, {**RECORD, "fax": "512-555-0000"})
    assert "fax" not in update["fields_on_file"]


def test_the_record_rebuilt_from_state_is_contact_fields_only():
    state = {**RECORD, "relationship": "plan_holder", "first_name": "James"}
    assert VerificationAgent()._member_record_from_state(state) == {k: RECORD[k] for k in ON_FILE_FIELDS}


@pytest.mark.parametrize("field", ON_FILE_FIELDS)
def test_the_sweep_passes_over_a_marked_field_and_reports_it_once_released(field):
    merged = {**RECORD, "call_intent": "claim_services"}
    marks = mark_on_file(None, ON_FILE_FIELDS)

    swept = dict(sweep_captured(merged, marks))
    assert field not in swept
    assert swept["call_intent"] == "claim_services"  # never on file, always reported

    assert field in dict(sweep_captured(merged, release_on_file(marks, [field])))


# ── The read-back the caller confirms is a capture ───────────────────────────


class _Pipeline:
    """A slot pipeline that answers its one slot and finishes."""

    def __init__(self, slot: str, answer: str) -> None:
        self.configs = {slot: SimpleNamespace(prompt="")}
        self._slot, self._answer = slot, answer

    async def collect(self, state, messages, post_collected, decision=None):
        post_collected[self._slot] = self._answer
        return None


def _post_lookup(call_intent: str, pipeline: _Pipeline) -> VerificationAgent:
    agent = VerificationAgent()
    claims = pipeline if call_intent == "claim_services" else _Pipeline("phone_confirmed", "yes")
    provider = pipeline if call_intent != "claim_services" else _Pipeline("relationship", "subscriber")
    asyncio.run(
        collect_post_lookup(
            agent,
            {"call_intent": call_intent, "messages": []},
            [],
            dict(COLLECTED),
            call_intent,
            dict(RECORD),
            None,
            claims,
            provider,
        )
    )
    return agent


def test_the_phone_the_caller_confirms_stops_being_merely_on_file():
    agent = _post_lookup("claim_services", _Pipeline("phone_confirmed", "yes"))
    assert agent._confirmed_this_turn["phone_number"] == RECORD["phone_number"]


def test_a_flow_that_never_asks_about_the_phone_never_reports_it():
    """The provider flow asks the relationship; the number on file stays on file."""
    agent = _post_lookup("provider_services", _Pipeline("relationship", "subscriber"))
    assert "phone_number" not in agent._confirmed_this_turn
