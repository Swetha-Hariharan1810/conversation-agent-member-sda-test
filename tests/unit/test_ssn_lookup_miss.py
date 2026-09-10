"""An SSN-path lookup miss is diagnosed, then only the wrong field is re-asked.

From a production transcript: Emily Parker gave a correct SSN and DOB, the
lookup missed because the account was under a different name, and the agent
asked her to "double-check your SSN" — twice — before escalating.

find_member_by_identity ANDs first_name, last_name, dob and ssn, so a miss on
its own names no field. lookup_and_verify already solved this on the Member ID
path: ask the store whether the key exists, then compare the rest field by
field. The SSN path now does the same with the SSN as the key.

  SSN not on file         → the SSN is wrong; name, SSN and DOB all go
  SSN found, name differs → that name field only
  SSN found, DOB differs  → the DOB only
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


def _queued(result: dict) -> list[str]:
    return [f for f in (result.get("ssn_recheck_fields") or "").split(",") if f]


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("agent.llm.config.get_extraction_llm", lambda: object(), raising=False)
    return VerificationAgent()


# The transcript: name is wrong, SSN and DOB are right.
CALLER = {
    "first_name": "Emily",
    "last_name": "Parker",
    "dob": "04/12/1988",
    "ssn": "527-41-3820",
    "ssn_fallback_stage": "ssn_lookup",
}

ACCOUNT = {
    "member_id": "M907503",
    "first_name": "Emily",
    "last_name": "Carter",  # the caller said Parker
    "dob": "1988-04-12",
    "ssn": "527-41-3820",
}


def _stub_ssn_lookup(monkeypatch, record):
    async def _find(**_kwargs):
        return record

    monkeypatch.setattr("agent.storage.queries.members.find_member_by_ssn", _find, raising=False)


# ── Diagnosis drives which field is re-asked ─────────────────────────────────


async def test_name_mismatch_asks_only_for_the_name(agent, monkeypatch):
    """The transcript case: the SSN is on file, the last name is not."""
    _stub_ssn_lookup(monkeypatch, ACCOUNT)

    result = await agent._handle_ssn_lookup_miss(dict(CALLER), [])

    assert _queued(result) == ["last_name"]
    assert result["last_name"] == ""
    assert "ssn" not in result, "a correct SSN must not be cleared"
    assert "dob" not in result, "a correct DOB must not be cleared"
    assert "social security" not in _text(result).lower()


async def test_dob_mismatch_asks_only_for_the_dob(agent, monkeypatch):
    _stub_ssn_lookup(monkeypatch, {**ACCOUNT, "last_name": "Parker", "dob": "1990-01-05"})

    result = await agent._handle_ssn_lookup_miss(dict(CALLER), [])

    assert _queued(result) == ["dob"]
    assert result["dob"] == ""
    assert "ssn" not in result
    assert "last_name" not in result


async def test_first_name_mismatch_asks_only_for_the_first_name(agent, monkeypatch):
    _stub_ssn_lookup(monkeypatch, {**ACCOUNT, "first_name": "Emma", "last_name": "Parker"})

    result = await agent._handle_ssn_lookup_miss(dict(CALLER), [])

    assert _queued(result) == ["first_name"]
    assert result["first_name"] == ""


async def test_unknown_ssn_re_asks_name_ssn_and_dob(agent, monkeypatch):
    """Nothing holds this SSN — it is the SSN that is wrong, and nothing else is trustworthy."""
    _stub_ssn_lookup(monkeypatch, None)

    result = await agent._handle_ssn_lookup_miss(dict(CALLER), [])

    assert _queued(result) == ["first_name", "last_name", "ssn", "dob"]
    for field in ("first_name", "last_name", "ssn", "dob"):
        assert result[field] == ""


async def test_multi_field_re_ask_does_not_read_details_back(agent, monkeypatch):
    """Reading several wrong details to an unverified caller is not something to say out loud."""
    _stub_ssn_lookup(monkeypatch, None)

    spoken = _text(await agent._handle_ssn_lookup_miss(dict(CALLER), [])).lower()

    assert "527" not in spoken and "parker" not in spoken and "1988" not in spoken


async def test_a_name_mismatch_resets_name_confirmation(agent, monkeypatch):
    _stub_ssn_lookup(monkeypatch, ACCOUNT)

    result = await agent._handle_ssn_lookup_miss({**CALLER, "name_confirmed": True}, [])

    assert result["name_confirmed"] is False


async def test_a_failing_ssn_query_falls_back_to_asking_everything(agent, monkeypatch):
    async def _boom(**_kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr("agent.storage.queries.members.find_member_by_ssn", _boom, raising=False)

    result = await agent._handle_ssn_lookup_miss(dict(CALLER), [])

    assert _queued(result) == ["first_name", "last_name", "ssn", "dob"]


async def test_rounds_are_bounded(agent, monkeypatch):
    _stub_ssn_lookup(monkeypatch, ACCOUNT)

    state = {**CALLER, "slot_attempts": {"ssn_lookup": {"attempt_count": 3}}}
    result = await agent._handle_ssn_lookup_miss(state, [])

    assert result["next_node"] == "escalation_agent"
    assert "representative" in _text(result)


# ── Re-collecting the flagged field, then retrying the lookup ────────────────


async def test_corrected_field_retries_the_lookup_keeping_the_others(agent, monkeypatch):
    async def _fake_extract(*_args, **_kwargs):
        return WorkerResult(extracted={"last_name": "Carter"})

    monkeypatch.setattr(verification_agent, "extract_verification_decision", _fake_extract)

    seen = {}

    async def _fake_finish(state, _messages, _call_intent):
        seen.update(state)
        return {"ok": True}

    monkeypatch.setattr(agent, "_finish_after_ssn", _fake_finish)

    state = {
        **CALLER,
        "last_name": "",
        "ssn_fallback_stage": "ssn_recheck",
        "ssn_recheck_fields": "last_name",
    }
    await agent._ssn_recheck_stage(state, "Carter, C-A-R-T-E-R", [])

    assert seen["last_name"] == "Carter"
    assert seen["ssn"] == "527-41-3820", "re-collecting a name must not discard the SSN"
    assert seen["dob"] == "04/12/1988"


async def test_a_queue_is_worked_one_field_at_a_time(agent, monkeypatch):
    async def _fake_extract(*_args, **_kwargs):
        return WorkerResult(extracted={"first_name": "Emma"})

    monkeypatch.setattr(verification_agent, "extract_verification_decision", _fake_extract)

    async def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the lookup must wait until every queued field is back")

    monkeypatch.setattr(agent, "_finish_after_ssn", _must_not_run)

    state = {**CALLER, "ssn_fallback_stage": "ssn_recheck", "ssn_recheck_fields": "first_name,last_name"}
    result = await agent._ssn_recheck_stage(state, "Emma", [])

    assert result["first_name"] == "Emma"
    assert _queued(result) == ["last_name"]


async def test_an_unusable_answer_is_retried_then_escalates(agent, monkeypatch):
    async def _fake_extract(*_args, **_kwargs):
        return WorkerResult(extracted={})

    monkeypatch.setattr(verification_agent, "extract_verification_decision", _fake_extract)

    state = {**CALLER, "ssn_fallback_stage": "ssn_recheck", "ssn_recheck_fields": "last_name"}
    for _ in range(2):
        result = await agent._ssn_recheck_stage(state, "mmhm", [])
        state = {**state, "slot_attempts": result.get("slot_attempts") or {}}

    final = await agent._ssn_recheck_stage(state, "mmhm", [])
    assert final["next_node"] == "escalation_agent"
