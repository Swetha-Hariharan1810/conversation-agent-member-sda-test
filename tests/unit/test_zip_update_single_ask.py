"""The ZIP question and awaiting_slot must describe the same thing.

Transcript bug — the caller was asked for the same ZIP twice:

    ai    I have your ZIP code as 58797. Is that right?
    human No. I would like to change.
    ai    Of course — could you provide your current ZIP code for me?
    human Seven eight seven zero one.
    ai    No problem — what is your current 5-digit ZIP code?     ← asked again

That third line is generated, not a constant, and it came from the retry
branch: its slot_label_override told the generator "...— if they say their
address changed, ask for their current ZIP", so the message asked for a value
while the branch left awaiting_slot on "zip_confirmed".

The next turn's extraction is told what it is collecting. Asked for
zip_confirmed, it answered THAT question — zip_confirmed="no", no zip_code —
for an utterance that was nothing but a ZIP, and the agent read a second
decline. The caller had already given the value the agent was still asking
for, because the agent's own record of what it asked was wrong.
"""

from __future__ import annotations

import pytest

from agent.agents.provider_search import agent as provider_search
from agent.agents.provider_search.agent import ProviderSearchAgent
from agent.agents.provider_search.constants import ZIP_UPDATE_PROMPT
from agent.llm.schema import WorkerResult

ZIP_ON_FILE = "58797"
NEW_ZIP = "78701"

# Instructions that send the generator off to collect a value instead of a
# yes/no. The original override ended "— if they say their address changed,
# ask for their current ZIP", and the generator did exactly that.
_ASKS_FOR_A_VALUE = ("ask for", "provide", "what is your", "give me")


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
    """A ProviderSearchAgent with the LLM and Salesforce edges stubbed out.

    The recovery generator echoes the instruction it is handed, so a test can
    see what the retry branch actually asks for.
    """
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


# ── the invariant ────────────────────────────────────────────────────────────


async def test_a_retry_asks_the_confirmation_question_it_says_it_is_asking(agent, monkeypatch):
    """The retry branch keeps awaiting_slot on zip_confirmed, so it must ask a yes/no.

    This is the turn that broke the transcript: the extraction came back with
    nothing usable, and the branch asked for the ZIP anyway.
    """
    _with_extraction(monkeypatch, WorkerResult())

    result = await agent.run(_state("No. I would like to change."))

    assert result["awaiting_slot"] == "zip_confirmed"
    # Whatever else it says, it must not send the caller off to supply a value
    # while the state still says a yes/no is expected.
    override = agent.asked["slot_label_override"].lower()
    assert "yes or no" in override  # it is still the confirmation question
    assert not any(phrase in override for phrase in _ASKS_FOR_A_VALUE)


async def test_asking_for_the_zip_always_says_so_in_state(agent, monkeypatch):
    """A clean decline asks for the value — and records that a value is expected."""
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "no"}))

    result = await agent.run(_state("No, I moved."))

    assert _text(result) == ZIP_UPDATE_PROMPT
    assert result["awaiting_slot"] == "zip_code"


async def test_a_zip_given_on_the_confirmation_turn_is_taken_not_re_asked(agent, monkeypatch):
    """With a coherent question the extraction returns the ZIP — and it is used."""
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_code": "78701"}))

    result = await agent.run(_state("Seven eight seven zero one."))

    assert agent.saved.get("zip_code") == NEW_ZIP
    assert result["zip_code"] == NEW_ZIP
    assert result["zip_code_updated"] is True
    assert ZIP_UPDATE_PROMPT not in _text(result)


# ── unchanged behaviour, guarded ─────────────────────────────────────────────


async def test_a_plain_yes_still_confirms_the_zip_on_file(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_confirmed": "yes"}))

    result = await agent.run(_state("Yes."))

    assert agent.saved == {}  # nothing written to Salesforce
    assert result.get("zip_code_updated") is not True


async def test_genuine_uncertainty_still_re_asks_the_confirmation(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult())

    result = await agent.run(_state("I'm not sure."))

    assert result["awaiting_slot"] == "zip_confirmed"
    assert agent.saved == {}


async def test_a_repeated_zip_on_file_is_a_confirmation(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"zip_code": ZIP_ON_FILE}))

    result = await agent.run(_state("Five eight seven nine seven."))

    assert agent.saved == {}  # already on file — no write
    assert result.get("zip_code_updated") is not True
