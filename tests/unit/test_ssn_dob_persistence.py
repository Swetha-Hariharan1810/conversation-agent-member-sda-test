"""The SSN-path DOB must survive the turn it was collected on.

_ssn_dob_collecting_stage wrote `state["dob"] = ...` on its local dict and
never called slot_ok. ask_member only persists slot-confirmed values into the
interrupt it builds, so the DOB never reached graph state: it lived in one call
frame and was gone by the next turn.

That single omission produced both reported failures:

  - the SOQL crash — the next lookup ran with dob="" and emitted
    `Date_of_Birth__c = ''`, which Salesforce rejects on a Date field
  - "the only detail that didn't match was the date of birth" — asked straight
    after the caller had given it, because by then it really was missing
"""

from __future__ import annotations

import pytest

from agent.agents.verification import agent as verification_agent
from agent.agents.verification.agent import VerificationAgent
from agent.llm.schema import WorkerResult


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    content = spoken.get("content", "") if isinstance(spoken, dict) else ""
    return str(content or result.get("escalation_pre_message") or "")


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("agent.llm.config.get_extraction_llm", lambda: object(), raising=False)
    return VerificationAgent()


ACCOUNT = {
    "member_id": "M907503",
    "first_name": "Emily",
    "last_name": "Carter",  # the caller said Parker
    "dob": "1988-04-12",
    "ssn": "527-41-3820",
}

AFTER_DOB = {
    "first_name": "Emily",
    "last_name": "Parker",
    "ssn": "527-41-3820",
    "dob": "",  # not yet collected
    "ssn_fallback_stage": "ssn_dob_collecting",
}


async def test_the_dob_reaches_the_next_turn(agent, monkeypatch):
    """Replays the transcript: DOB given, lookup misses on the name, DOB must survive."""

    async def _fake_extract(*_args, **_kwargs):
        return WorkerResult(extracted={"dob": "April 12 1988"})

    monkeypatch.setattr(verification_agent, "extract_verification_decision", _fake_extract)

    async def _no_identity_match(**_kwargs):
        return None

    async def _ssn_on_file(**_kwargs):
        return ACCOUNT

    monkeypatch.setattr(
        "agent.storage.queries.members.find_member_by_identity", _no_identity_match, raising=False
    )
    monkeypatch.setattr("agent.storage.queries.members.find_member_by_ssn", _ssn_on_file, raising=False)

    result = await agent._ssn_dob_collecting_stage(dict(AFTER_DOB), "April twelve nineteen eighty eight", [])

    # The lookup missed on the name, so the name is what gets re-asked...
    assert result["ssn_recheck_fields"] == "last_name"
    # ...and the DOB the caller just gave is carried into the next turn.
    assert result["dob"] == "04/12/1988", "the DOB must not vanish with the call frame"
    assert result["ssn"] == "527-41-3820"
    assert "date of birth" not in _text(result).lower()


async def test_the_dob_is_confirmed_as_a_slot(agent, monkeypatch):
    async def _fake_extract(*_args, **_kwargs):
        return WorkerResult(extracted={"dob": "April 12 1988"})

    monkeypatch.setattr(verification_agent, "extract_verification_decision", _fake_extract)

    async def _found(**_kwargs):
        return ACCOUNT

    monkeypatch.setattr("agent.storage.queries.members.find_member_by_identity", _found, raising=False)
    monkeypatch.setattr("agent.storage.tools.lookup_member", None, raising=False)

    await agent._ssn_dob_collecting_stage(dict(AFTER_DOB), "April twelve nineteen eighty eight", [])

    assert agent.get_slot("dob").confirmed is True
    assert agent.get_slot("dob").last_value == "04/12/1988"


# ── A re-ask keeps every field it is not re-asking ───────────────────────────


def test_re_ask_carries_forward_the_fields_it_keeps(agent):
    state = {
        "first_name": "Emily",
        "last_name": "Parker",
        "ssn": "527-41-3820",
        "dob": "04/12/1988",
    }
    result = agent._ask_ssn_recheck(state, ["last_name"], {})

    assert result["last_name"] == ""
    assert result["dob"] == "04/12/1988"
    assert result["ssn"] == "527-41-3820"
    assert result["first_name"] == "Emily"


# ── Missing is not the same as mismatched ────────────────────────────────────


def test_a_missing_field_is_not_reported_as_a_mismatch(agent):
    spoken = _text(agent._ask_ssn_recheck({"ssn": "527-41-3820"}, ["dob"], {}, missing=True)).lower()

    assert "didn't match" not in spoken and "did not match" not in spoken
    assert "date of birth" in spoken


def test_a_real_mismatch_still_says_so(agent):
    state = {"first_name": "Emily", "last_name": "Parker", "ssn": "527-41-3820", "dob": "04/12/1988"}
    spoken = _text(agent._ask_ssn_recheck(state, ["last_name"], {})).lower()

    assert "last name" in spoken
