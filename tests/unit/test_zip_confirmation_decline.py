"""A declined ZIP is handled without anyone naming the phrasing.

The ways of saying "that ZIP is wrong" do not end: "I moved", "I'd like to
change it", "that's my old one", "we relocated last spring", "my daughter
handles my mail now". The agent used to need one of them recognised — by the
extraction prompt's list, or by detect_request's phrase table — before it
would ask for the new value. Anything neither list carried fell through to the
retry, which re-asked a question the caller had already answered.

"No. I would like to change." is what fell through in the reported call:

    detect_request('No. I would like to change.')  ->  None
    detect_request('No, I moved.')                 ->  update / zip_code

So the branch no longer recognises declines at all. It recognises the two
closed things — an affirmation, and a ZIP, which is a shape — and treats
everything else as a decline by default. The open-ended side is the one that
needs no list.
"""

from __future__ import annotations

import pytest

from agent.agents.provider_search import agent as provider_search
from agent.agents.provider_search.agent import ProviderSearchAgent
from agent.agents.provider_search.constants import ZIP_UPDATE_PROMPT
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

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
    saved: dict = {}
    asked: dict = {}

    async def _no_guards(*_args, **_kwargs):
        return None

    async def _save_zip(_agent, _state, zip_code):
        saved["zip_code"] = zip_code
        return None

    async def _echo_recovery(**kwargs):
        asked["slot_label_override"] = kwargs.get("slot_label_override", "")
        return f"[generated] {asked['slot_label_override']}"

    monkeypatch.setattr(ProviderSearchAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(provider_search, "update_zip_in_salesforce", _save_zip)
    monkeypatch.setattr(provider_search, "get_extraction_llm", lambda: object())
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _echo_recovery)

    instance = ProviderSearchAgent()
    instance.saved = saved
    instance.asked = asked
    return instance


def _with_extraction(monkeypatch, result: WorkerResult):
    async def _fake_extract(*_args, **_kwargs):
        return result

    monkeypatch.setattr(provider_search, "extract_provider_search_decision", _fake_extract)


# ── a decline needs no phrase, and no extraction, to be honoured ─────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "No. I would like to change.",  # the reported call
        "That's my old one.",
        "We relocated last spring.",
        "My daughter handles my mail now.",
        "Nah, that hasn't been right since the divorce.",
        "I've been at the new place since March.",
    ],
)
async def test_a_decline_is_honoured_whatever_the_wording(agent, monkeypatch, utterance):
    """Not one of these is in any list, and the extraction returns nothing."""
    _with_extraction(monkeypatch, WorkerResult())

    result = await agent.run(_state(utterance))

    assert _text(result) == ZIP_UPDATE_PROMPT
    assert result["awaiting_slot"] == "zip_code"
    assert agent.asked == {}  # no retry was generated
    assert result["slot_attempts"] == {}  # and no retry was burned


async def test_an_explicit_no_still_asks_for_the_new_zip(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "no"}))

    result = await agent.run(_state("No."))

    assert _text(result) == ZIP_UPDATE_PROMPT
    assert result["awaiting_slot"] == "zip_code"


# ── the two things that ARE recognised ───────────────────────────────────────


async def test_an_affirmation_confirms_the_zip_on_file(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "yes"}))

    result = await agent.run(_state("Yes, that's right."))

    assert agent.saved == {}  # nothing written
    assert result.get("zip_code_updated") is not True


async def test_a_new_zip_is_taken_and_written(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_code": NEW_ZIP}))

    result = await agent.run(_state("Seven eight seven zero one."))

    assert agent.saved.get("zip_code") == NEW_ZIP
    assert result["zip_code"] == NEW_ZIP
    assert result["zip_code_updated"] is True


# ── turns that are NOT an answer still re-ask, and must not read as declines ─


@pytest.mark.parametrize(
    "utterance, extraction",
    [
        # Genuine uncertainty — the caller does not know.
        ("I'm not sure.", WorkerResult(event_type=EventType.AMBIGUOUS)),
        # Asking for time.
        ("Hold on a second.", WorkerResult(event_type=EventType.WAIT)),
        # A side question rides the turn.
        (
            "Will I get this by email?",
            WorkerResult(
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                followup_disposition=FollowupDisposition.PARK,
                followup_query="will I get this by email?",
            ),
        ),
        # They want to change a different slot entirely.
        (
            "Actually my last name is wrong.",
            WorkerResult(event_type=EventType.CORRECTED, update_target="last_name"),
        ),
    ],
)
async def test_a_non_answer_re_asks_the_confirmation(agent, monkeypatch, utterance, extraction):
    _with_extraction(monkeypatch, extraction)

    result = await agent.run(_state(utterance))

    assert result["awaiting_slot"] == "zip_confirmed"
    assert ZIP_UPDATE_PROMPT not in _text(result)
    assert agent.saved == {}


async def test_an_empty_turn_re_asks_rather_than_declining(agent, monkeypatch):
    """Nothing heard is not a decline."""
    _with_extraction(monkeypatch, WorkerResult())

    result = await agent.run(_state("   "))

    assert result["awaiting_slot"] == "zip_confirmed"


async def test_the_retry_asks_the_question_it_says_it_is_asking(agent, monkeypatch):
    """A branch that keeps awaiting_slot on zip_confirmed must ask a yes/no."""
    _with_extraction(monkeypatch, WorkerResult(event_type=EventType.AMBIGUOUS))

    await agent.run(_state("I'm not sure."))

    override = agent.asked["slot_label_override"].lower()
    assert "yes or no" in override
    assert not any(p in override for p in ("ask for", "provide", "what is your", "give me"))


async def test_a_repeated_zip_on_file_is_a_confirmation(agent, monkeypatch):
    """Saying the ZIP we read back is agreement, not a replacement."""
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_code": ZIP_ON_FILE}))

    result = await agent.run(_state("Five eight seven nine seven."))

    assert agent.saved == {}  # already on file — no write
    assert result.get("zip_code_updated") is not True


async def test_an_invalid_zip_asks_for_a_proper_one(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_code": "787"}))

    result = await agent.run(_state("Seven eight seven."))

    assert _text(result) == ZIP_UPDATE_PROMPT
    assert result["awaiting_slot"] == "zip_code"
    assert agent.saved == {}
