"""The caller said which half of their name was wrong, and was asked again.

    AI    And your last name?
    User  watson
    AI    Thank you. Just to confirm — is your name Emily Watson. That's
          spelled E-M-I-L-Y-W-A-T-S-O-N, correct?
    User  Actually, my surname is Carter, not Watson.
    AI    Sure, what is the correct name?

The correct name was in the sentence. The caller named the half that was wrong
("surname"), gave its replacement ("Carter"), and named the value it replaced
("not Watson") — and got NAME_CORRECTION_PROMPTS, the line written for a caller
who rejected the readback and offered nothing.

name_confirmation.md teaches the correction shape as "no, it's Jhon Doe": a
whole name, no part named. Neither of this caller's two moves is in the
contract, and the trailing "not Watson" is the shape of every rejection listed
under OUTCOME 3 ("that's not right", "no that's not me"), so the turn came back
a bare no. The nastier reading is last_name="Watson" — the name being corrected,
read back as though the caller had asked for it.

So which part a caller is correcting is now read from their words whenever they
name it, before the outcomes are decided.
"""

from __future__ import annotations

import pytest

from agent.agents.verification.handlers import recover_name_correction
from agent.llm.schema import WorkerResult

NAME_CORRECTION_ASKS = ("what is the correct name", "correct first and last name")


@pytest.fixture
def verification(monkeypatch):
    from agent.agents.verification import agent as va

    monkeypatch.setattr(va, "get_extraction_llm", lambda: object())
    return va


def _state() -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "And your last name?"},
            {"role": "user", "content": "watson"},
            {
                "role": "assistant",
                "content": (
                    "Thank you. Just to confirm — is your name Emily Watson. "
                    "That's spelled E-M-I-L-Y-W-A-T-S-O-N, correct?"
                ),
            },
            {"role": "user", "content": "Actually, my surname is Carter, not Watson."},
        ],
        "first_name": "Emily",
        "last_name": "Watson",
        "awaiting_slot": "name_confirmed",
        "name_confirm_attempts": 0,
        "slot_attempts": {},
        "app_run_id": "test-run",
        "call_intent": "provider_services",
    }


async def _readback_turn(verification, monkeypatch, extracted: dict, utterance: str | None = None) -> dict:
    """One turn through _process_name_readback_response with the LLM saying
    exactly `extracted` — the readings the model actually returns for this
    sentence."""

    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted=extracted)

    monkeypatch.setattr(verification, "extract_name_confirmation", _extract)
    state = _state()
    if utterance is not None:
        state["messages"][-1] = {"role": "user", "content": utterance}
    agent = verification.VerificationAgent.from_state(state)
    return await agent._process_name_readback_response(
        state, state["messages"], state["messages"][-1]["content"]
    )


def _said(result: dict) -> str:
    return (result.get("messages") or {}).get("content", "")


# ── the reported turn, under every reading the model gives it ────────────────

# What comes back for "Actually, my surname is Carter, not Watson":
#   the bare no      — "not Watson" read as the rejections it resembles
#   the inversion    — the name being corrected returned as the correction
#   nothing at all   — neither a confirmation nor a name
#   the right answer — which must survive the recovery unchanged
READINGS = [
    pytest.param({"name_confirmed": "no"}, id="read-as-bare-no"),
    pytest.param({"last_name": "Watson"}, id="read-as-the-old-name"),
    pytest.param({"name_confirmed": "no", "last_name": "Watson"}, id="read-as-both"),
    pytest.param({}, id="read-as-nothing"),
    pytest.param({"last_name": "Carter"}, id="read-correctly"),
]


@pytest.mark.parametrize("extracted", READINGS)
async def test_the_caller_is_not_asked_for_a_name_they_just_gave(verification, monkeypatch, extracted):
    said = _said(await _readback_turn(verification, monkeypatch, extracted)).lower()
    for ask in NAME_CORRECTION_ASKS:
        assert ask not in said


@pytest.mark.parametrize("extracted", READINGS)
async def test_the_correction_is_read_back_for_confirmation(verification, monkeypatch, extracted):
    result = await _readback_turn(verification, monkeypatch, extracted)
    assert result["last_name"] == "Carter"
    assert result["first_name"] == "Emily"
    assert "C-A-R-T-E-R" in _said(result)
    assert result["awaiting_slot"] == "name_confirmed"


@pytest.mark.parametrize("extracted", READINGS)
async def test_the_name_being_replaced_is_never_the_replacement(verification, monkeypatch, extracted):
    assert "W-A-T-S-O-N" not in _said(await _readback_turn(verification, monkeypatch, extracted))


async def test_a_misfiled_part_is_not_confirmed_twice(verification, monkeypatch):
    """The model, reading the same sentence, sometimes files the surname under
    first_name. Taking both would read back "Carter Carter"."""
    result = await _readback_turn(verification, monkeypatch, {"first_name": "Carter"})
    assert (result["first_name"], result["last_name"]) == ("Emily", "Carter")


# ── the shapes, read from the words ──────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("Actually, my surname is Carter, not Watson.", {"last_name": "Carter"}),
        ("my surname is Carter", {"last_name": "Carter"}),
        ("my last name is Carter", {"last_name": "Carter"}),
        ("the family name is Carter", {"last_name": "Carter"}),
        ("my maiden name is Carter", {"last_name": "Carter"}),
        ("my surname's Carter", {"last_name": "Carter"}),
        ("my first name is Emma", {"first_name": "Emma"}),
        ("my given name is Emma", {"first_name": "Emma"}),
        ("it's Carter, not Watson", {"last_name": "Carter"}),
        ("Carter not Watson", {"last_name": "Carter"}),
        ("Emma, not Emily", {"first_name": "Emma"}),
        (
            "my first name is Emma and my surname is Carter",
            {"first_name": "Emma", "last_name": "Carter"},
        ),
        ("my surname is van der berg", {"last_name": "Van Der Berg"}),
        ("my surname's O'Brien", {"last_name": "O'Brien"}),
        ("my last name is Smith-Jones", {"last_name": "Smith-Jones"}),
    ],
)
def test_the_part_the_caller_names_is_read_from_the_words(utterance, expected):
    assert recover_name_correction(utterance, "Emily", "Watson") == expected


@pytest.mark.parametrize(
    "utterance",
    [
        "yes that's correct",
        "yes",
        "no",
        "nope that's wrong",
        "no, my last name is wrong",  # names the part, gives no replacement
        "my surname is not Watson",  # says what it isn't, not what it is
        "no it's Jhon Doe",  # a whole name, no part named — the model's shape
        "it's Carter",  # no part named, nothing contrasted
        "Carter, not Wilson",  # replaces a name this call never held
        "I spoke to Sarah, not Watson, about the claim last week",
    ],
)
def test_the_words_claim_nothing_they_do_not_name(utterance):
    """Everything else is left exactly as the model read it — the recovery adds
    a reading, it does not take one over."""
    assert recover_name_correction(utterance, "Emily", "Watson") == {}


# ── the answers either side of it still work ─────────────────────────────────


async def test_a_plain_confirmation_still_proceeds(verification, monkeypatch):
    result = await _readback_turn(
        verification, monkeypatch, {"name_confirmed": "yes"}, utterance="yes that's correct"
    )
    assert result["last_name"] == "Watson"
    assert "C-A-R-T-E-R" not in _said(result)


async def test_a_bare_no_still_asks_what_the_name_is(verification, monkeypatch):
    result = await _readback_turn(verification, monkeypatch, {"name_confirmed": "no"}, utterance="no")
    said = _said(result).lower()
    assert any(ask in said for ask in NAME_CORRECTION_ASKS)
    assert result["awaiting_slot"] == "name_correction"


async def test_a_whole_replacement_name_still_works(verification, monkeypatch):
    result = await _readback_turn(
        verification,
        monkeypatch,
        {"first_name": "Emma", "last_name": "Carter"},
        utterance="no, it's Emma Carter",
    )
    assert (result["first_name"], result["last_name"]) == ("Emma", "Carter")


# ── the second place the same sentence lands ─────────────────────────────────


async def test_the_correction_slot_reads_the_named_part_too(verification, monkeypatch):
    """After a bare no we ask "what is the correct name?" — a caller who
    answers "my surname is Carter" must not be asked a third time."""

    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"name_confirmed": "no"})

    monkeypatch.setattr(verification, "extract_name_confirmation", _extract)
    state = _state()
    state["awaiting_slot"] = "name_correction"
    state["messages"][-1] = {"role": "user", "content": "my surname is Carter"}
    agent = verification.VerificationAgent.from_state(state)
    result = await agent._collect_name_correction(state, state["messages"], "my surname is Carter")
    assert result["last_name"] == "Carter"
    assert result["first_name"] == "Emily"
    assert "C-A-R-T-E-R" in _said(result)


# ── the contract the model is held to ────────────────────────────────────────


def test_the_prompt_names_the_parts_it_is_asked_about():
    from pathlib import Path

    body = Path("src/agent/prompts/extraction/name_confirmation.md").read_text().lower()
    for word in ("surname", "family name", "maiden name", "given name"):
        assert word in body
    assert "not watson" in body  # the contrastive shape is taught, not inferred
