"""A ZIP given once must be collected once.

Transcript bug — the caller was asked for the same ZIP three times:

    ai    I have your ZIP code as 58797. Is that right?
    human No. I would like to change.
    ai    Of course — could you provide your current ZIP code for me?
    human Seven eight seven zero one.
    ai    No problem — what is your current 5-digit ZIP code?     ← asked again
    human Yeah. My current five digit ZIP code is seven eight seven zero one.

Two failures compound:

  1. "No. I would like to change." produced no zip_confirmed extraction, so it
     fell through to the RETRY branch. The recovery message it generated asked
     for the current ZIP — but the branch leaves awaiting_slot on zip_confirmed,
     so the message and the state disagreed about what was being collected.
  2. On the next turn the extraction LLM is told "Currently asking for:
     zip_confirmed", so it answered THAT question: zip_confirmed="no", no
     zip_code, for an utterance that was nothing but a ZIP. The agent read a
     second decline and asked for the ZIP again.
"""

from __future__ import annotations

import pytest

from agent.agents.provider_search import agent as provider_search
from agent.agents.provider_search.agent import ProviderSearchAgent
from agent.llm.schema import WorkerResult
from agent.slots.normalizers import find_zip_in_utterance, normalize_yes_no

ZIP_ON_FILE = "58797"
NEW_ZIP = "78701"


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


def _state(last_user: str, **overrides) -> dict:
    state = {
        "messages": [
            {"role": "assistant", "content": "I have your ZIP code as 58797. Is that right?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_status_verify": True,
        "member_id": "M310188",
        "provider_type": "Primary Care Physician",
        "zip_code": ZIP_ON_FILE,
        "zip_code_used": "",
        "awaiting_slot": "zip_confirmed",
    }
    state.update(overrides)
    return state


@pytest.fixture
def agent(monkeypatch):
    """A ProviderSearchAgent with the LLM and Salesforce edges stubbed out."""
    saved: dict = {}

    async def _no_guards(*_args, **_kwargs):
        return None

    async def _save_zip(_agent, _state, zip_code):
        saved["zip_code"] = zip_code
        return None

    monkeypatch.setattr(ProviderSearchAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(provider_search, "update_zip_in_salesforce", _save_zip)
    monkeypatch.setattr(provider_search, "get_extraction_llm", lambda: object())

    instance = ProviderSearchAgent()
    instance.saved = saved
    return instance


def _with_extraction(monkeypatch, result: WorkerResult):
    async def _fake_extract(*_args, **_kwargs):
        return result

    monkeypatch.setattr(provider_search, "extract_provider_search_decision", _fake_extract)


# ── the transcript, turn by turn ─────────────────────────────────────────────


async def test_an_indirect_decline_asks_for_the_zip_and_says_so_in_state(agent, monkeypatch):
    """Turn 1: "No. I would like to change." — one ask, awaiting_slot follows it."""
    # What the extractor actually returned: nothing usable.
    _with_extraction(monkeypatch, WorkerResult())

    result = await agent.run(_state("No. I would like to change."))

    assert "ZIP" in _text(result)
    # The message asks for a value, so the state must say a value is expected —
    # this is what tells the next turn's extraction what it is collecting.
    assert result["awaiting_slot"] == "zip_code"


async def test_a_spoken_zip_is_accepted_even_when_extraction_calls_it_a_decline(agent, monkeypatch):
    """Turn 2: the exact extraction from the transcript, on a bare spoken ZIP."""
    _with_extraction(
        monkeypatch,
        WorkerResult(extracted={"zip_confirmed": "no"}),  # verbatim from the report
    )

    result = await agent.run(_state("Seven eight seven zero one."))

    assert agent.saved.get("zip_code") == NEW_ZIP
    assert result["zip_code"] == NEW_ZIP
    assert result["zip_code_updated"] is True
    assert "5-digit ZIP" not in _text(result)


async def test_the_zip_is_never_asked_for_a_third_time(agent, monkeypatch):
    """Turn 3: the caller repeating themselves must not read as a bare "yes"."""
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "no"}))

    result = await agent.run(_state("Yeah. My current five digit ZIP code is seven eight seven zero one."))

    # normalize_yes_no sees the leading "Yeah." and says yes — taking that would
    # confirm 58797, the very ZIP the caller is replacing.
    assert normalize_yes_no("Yeah. My current five digit ZIP code is seven eight seven zero one.") == "yes"
    assert agent.saved.get("zip_code") == NEW_ZIP
    assert result["zip_code"] == NEW_ZIP


# ── the pieces, guarded individually ─────────────────────────────────────────


async def test_a_plain_yes_still_confirms_the_zip_on_file(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "yes"}))

    result = await agent.run(_state("Yes."))

    assert agent.saved == {}  # nothing written
    assert result.get("zip_code_updated") is not True


async def test_genuine_uncertainty_still_re_asks_the_confirmation(agent, monkeypatch):
    """ "I'm not sure" is ambiguous — it must not become a decline."""
    _with_extraction(monkeypatch, WorkerResult())

    recovery = "Sorry — is the ZIP code 5 8 7 9 7 still correct?"

    async def _fake_recovery(**_kwargs):
        return recovery

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _fake_recovery)

    result = await agent.run(_state("I'm not sure."))

    assert result["awaiting_slot"] == "zip_confirmed"
    assert agent.saved == {}


@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("Seven eight seven zero one.", NEW_ZIP),
        ("Yeah. My current five digit ZIP code is seven eight seven zero one.", NEW_ZIP),
        ("my zip is 78701", NEW_ZIP),
        ("no, it's 10001", "10001"),
        ("No. I would like to change.", ""),
        ("Yes", ""),
        ("I'm not sure", ""),
        # A phone or fax number is not a ZIP.
        ("you can fax it to 2155553299", ""),
        # Two different candidates — ambiguous, leave it to the extraction LLM.
        ("I think it is 78701 or maybe 78702", ""),
        # The same ZIP said twice is still one ZIP.
        ("seven eight seven zero one, that is seven eight seven zero one", NEW_ZIP),
        # A spoken date must not read as a ZIP.
        ("July thirty nineteen seventy seven", ""),
    ],
)
def test_find_zip_in_utterance(utterance, expected):
    assert find_zip_in_utterance(utterance) == expected


def test_find_zip_in_utterance_keeps_the_final_digit():
    """normalize_zip_code drops "one." — trailing punctuation hides the token."""
    from agent.slots.normalizers import normalize_zip_code

    assert normalize_zip_code("Seven eight seven zero one.") != NEW_ZIP
    assert find_zip_in_utterance("Seven eight seven zero one.") == NEW_ZIP
