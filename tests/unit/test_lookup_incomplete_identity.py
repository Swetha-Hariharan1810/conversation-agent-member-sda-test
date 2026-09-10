"""An incomplete identity must never reach the store.

Production crash: the SSN-path re-check cleared the DOB, the lookup ran anyway,
and the SOQL builder emitted `Date_of_Birth__c = ''` on a Salesforce Date field:

    SOQL query failed: 400 value of filter criterion for field
    'Date_of_Birth__c' must be of type date and should not be enclosed in quotes

That killed the whole graph run, not just the turn. The builder skipped None but
not "", so it was reachable by any caller passing an empty value for a date
field — the re-check flow only exposed it.

Guarded at three layers, because a crash-the-run bug should not depend on one
of them being right:
  1. the SOQL builder never emits an empty filter
  2. find_member_by_identity refuses an incomplete tuple instead of widening
  3. the agent collects the missing field instead of calling the lookup
"""

from __future__ import annotations

import pytest

from agent.agents.verification.agent import VerificationAgent

# ── Layer 1: the SOQL builder ────────────────────────────────────────────────


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_values_are_not_emitted_as_filters(empty, monkeypatch):
    """The exact shape that produced the 400."""
    captured = {}

    class _FakeSF:
        async def query(self, soql):
            captured["soql"] = soql
            return {"records": []}

    import agent.storage.db as db

    monkeypatch.setattr(db, "_get_sf", lambda: _FakeSF())

    import asyncio

    asyncio.run(
        db._sf_query_store(
            "find_one",
            entity="members",
            where={"first_name": "Emily", "last_name": "Carter", "dob": empty},
        )
    )

    where = captured["soql"].split(" WHERE ", 1)[1]
    assert "= ''" not in where, "an empty filter is what Salesforce rejected"
    assert "Date_of_Birth__c" not in where
    assert "Last_Name__c = 'Carter'" in where


# ── Layer 2: the identity query ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_name": "", "last_name": "Carter", "dob": "04/12/1988"},
        {"first_name": "Emily", "last_name": "", "dob": "04/12/1988"},
        {"first_name": "Emily", "last_name": "Carter", "dob": ""},
    ],
)
async def test_incomplete_identity_returns_no_match_without_querying(kwargs, monkeypatch):
    """Dropping a filter would match on less — that hands back an unproven account."""
    from agent.storage.queries import members

    async def _must_not_query(*_args, **_kwargs):
        raise AssertionError("an incomplete identity must not reach the store")

    monkeypatch.setattr(members, "query_store", _must_not_query)

    assert await members.find_member_by_identity(member_id="", ssn="527-41-3820", **kwargs) is None


# ── Layer 3: the agent ───────────────────────────────────────────────────────


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("agent.llm.config.get_extraction_llm", lambda: object(), raising=False)
    return VerificationAgent()


async def test_lookup_is_skipped_and_the_hole_is_collected(agent, monkeypatch):
    """The transcript state at the moment of the crash: name corrected, DOB cleared."""
    from agent.storage.queries import members

    async def _must_not_query(*_args, **_kwargs):
        raise AssertionError("the lookup must not run with a cleared field")

    monkeypatch.setattr(members, "find_member_by_identity", _must_not_query)

    state = {
        "first_name": "Emily",
        "last_name": "Carter",
        "ssn": "527-41-3820",
        "dob": "",
        "ssn_fallback_stage": "ssn_lookup",
    }
    result = await agent._finish_after_ssn(state, [], "")

    assert result["ssn_recheck_fields"] == "dob"
    assert result["ssn_fallback_stage"] == "ssn_recheck"


async def test_a_lost_recheck_queue_is_rebuilt_not_looked_up(agent, monkeypatch):
    async def _must_not_finish(*_args, **_kwargs):
        raise AssertionError("a lost queue must be rebuilt, not looked up")

    monkeypatch.setattr(agent, "_finish_after_ssn", _must_not_finish)

    state = {
        "first_name": "Emily",
        "last_name": "Carter",
        "ssn": "",
        "dob": "",
        "ssn_fallback_stage": "ssn_recheck",
        "ssn_recheck_fields": "",  # lost
    }
    result = await agent._ssn_recheck_stage(state, "", [])

    assert result["ssn_recheck_fields"] == "ssn,dob"


def test_missing_identity_is_reported_in_ask_order():
    assert VerificationAgent._missing_ssn_identity(
        {"first_name": "Emily", "last_name": "", "ssn": "", "dob": "04/12/1988"}
    ) == ["last_name", "ssn"]
    assert (
        VerificationAgent._missing_ssn_identity(
            {"first_name": "Emily", "last_name": "Carter", "ssn": "527-41-3820", "dob": "04/12/1988"}
        )
        == []
    )
