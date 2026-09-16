"""The caller was told about the session snapshot.

    AI    Sending it to your fax at 6175554199 — you should receive it within
          30 minutes. Is there anything else I can help you with?
    User  can you go over my claim history again?
    AI    Claim history was not included in the session snapshot. Do you have
          any other questions about what we covered?

"The session snapshot" is the name of a block of text in a prompt. The caller
is on a phone call about a fax.

follow_up.md asks the model to answer from that block and to return null when
the answer is not in it — and says, in as many words, "answer=null is the
correct and complete response when data is missing. Do not offer to find the
information. Do not redirect." Told to answer from a thing, a model that cannot
find something in that thing sometimes says so instead, and names it. The rule
was there; keeping it was not guaranteed.

Two changes, either of which alone would have been enough, and neither of which
is sufficient on its own:

  * the block is now headed OUR CONVERSATION, so the sentence a leak produces —
    "I don't have that from our conversation today" — is one a member services
    call can contain, and the prompt reads as English while it says it
    ("answer only from OUR CONVERSATION");
  * an answer that names the machinery is treated as the non-answer it is, so
    the turn falls to MSG_CANNOT_ANSWER, which says the same thing in the
    caller's own words.

The second change also fixes what the leak was hiding: a refusal that arrives
as an `answer` resets the cannot-answer streak, so a caller could be told "not
in the snapshot" all day without the call ever escalating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.agents.follow_up.agent import _names_the_machinery
from agent.agents.follow_up.constants import MSG_CANNOT_ANSWER, MSG_CONTINUATION
from agent.agents.follow_up.llm import SNAPSHOT_HEADER
from agent.llm.schema import FollowUpIntent, FollowUpResult

# The reported answer, and the other shapes the same slip takes.
LEAKED_ANSWERS = [
    "Claim history was not included in the session snapshot.",
    "That information is not in the session snapshot.",
    "The snapshot does not include your claim history.",
    "I don't see that in the session context.",
    "That wasn't in the context provided to me.",
    "My instructions don't cover that.",
    "That isn't in the call log.",
    "I don't have a transcript of that.",
]


@pytest.fixture
def follow_up(monkeypatch):
    from agent.agents.follow_up import agent as fu

    monkeypatch.setattr(fu, "get_follow_up_llm", lambda: object())
    return fu


def _state() -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Sending it to your fax at 6175554199 — you should receive it within "
                    "30 minutes. Is there anything else I can help you with?"
                ),
            },
            {"role": "user", "content": "can you go over my claim history again?"},
        ],
        "first_name": "James",
        "last_name": "Wilson",
        "member_status_verify": True,
        "call_intent": "claim_services",
        "slot_attempts": {},
        "app_run_id": "test-run",
        "follow_up_turn_count": 1,
        "follow_up_cannot_answer_count": 0,
    }


async def _turn(follow_up, monkeypatch, answer: str, cannot_answer_count: int = 0) -> dict:
    async def _extract(*_args, **_kwargs):
        return FollowUpResult(follow_up_intent=FollowUpIntent.QUESTION, answer=answer)

    monkeypatch.setattr(follow_up, "extract_follow_up_decision", _extract)
    state = _state()
    state["follow_up_cannot_answer_count"] = cannot_answer_count
    return await follow_up.FollowUpAgent.from_state(state).execute(state)


def _said(result: dict) -> str:
    return (result.get("messages") or {}).get("content", "")


# ── the reported turn ────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", LEAKED_ANSWERS)
async def test_the_caller_never_hears_the_plumbing(follow_up, monkeypatch, answer):
    said = _said(await _turn(follow_up, monkeypatch, answer)).lower()
    for term in ("snapshot", "session context", "context provided", "my instructions", "call log"):
        assert term not in said


@pytest.mark.parametrize("answer", LEAKED_ANSWERS)
async def test_the_same_thing_is_said_in_the_callers_words(follow_up, monkeypatch, answer):
    """MSG_CANNOT_ANSWER already had the sentence — "that isn't something we
    covered during this call" — the turn just wasn't reaching it."""
    said = _said(await _turn(follow_up, monkeypatch, answer))
    assert any(msg in said for msg in MSG_CANNOT_ANSWER)
    assert any(msg in said for msg in MSG_CONTINUATION)


@pytest.mark.parametrize("answer", LEAKED_ANSWERS)
async def test_a_refusal_counts_as_one(follow_up, monkeypatch, answer):
    """It arrived as an `answer`, so it reset the streak: three of these in a
    row left the caller exactly where they started."""
    result = await _turn(follow_up, monkeypatch, answer)
    assert result["follow_up_cannot_answer_count"] == 1


async def test_three_of_them_still_escalate(follow_up, monkeypatch):
    result = await _turn(follow_up, monkeypatch, LEAKED_ANSWERS[0], cannot_answer_count=2)
    assert result["last_agent_signal"]["escalation_reason"] == "repeated_cannot_answer_in_follow_up"


# ── a real answer is left alone ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer",
    [
        "Your provider list went out to your fax at 6175554199.",
        "Your individual deductible is 1500 dollars and you've met 400 dollars of it.",
        "Rewards are at www dot mysagilityhealth dot com under the My Wellness section.",
        "We covered your claim adjustment and sent the consultation link to your fax.",
        "Your appointment link is on its way — it should arrive within 30 minutes.",
    ],
)
async def test_an_answer_that_answers_is_spoken_as_written(follow_up, monkeypatch, answer):
    result = await _turn(follow_up, monkeypatch, answer)
    assert answer in _said(result)
    assert result["follow_up_cannot_answer_count"] == 0


@pytest.mark.parametrize(
    "answer",
    [
        "Your deductible resets in January.",
        "We covered your benefits and your provider search today.",
        "That was sent to your email at james dot wilson at gmail dot com.",
        "Your claim is in process — the context of the review is a records request.",
    ],
)
def test_ordinary_sentences_are_not_mistaken_for_plumbing(answer):
    assert _names_the_machinery(answer) == ""


@pytest.mark.parametrize("answer", LEAKED_ANSWERS)
def test_every_reported_shape_is_caught(answer):
    assert _names_the_machinery(answer)


# ── the header, written to be harmless when it leaks ─────────────────────────


def test_the_header_reads_as_a_sentence_about_the_call():
    """There is no header that cannot leak. This one leaks into English."""
    assert SNAPSHOT_HEADER == "OUR CONVERSATION"
    leaked = f"I don't have your claim history from {SNAPSHOT_HEADER.lower()} today."
    assert not _names_the_machinery(leaked)


def test_the_prompts_no_longer_teach_the_word():
    """The model reaches for the label it was given; it should not be given one
    that has to be scrubbed. The word survives in exactly one place — the line
    forbidding it."""
    for name in ("follow_up.md", "follow_up_claims.md"):
        text = (Path("src/agent/prompts/extraction") / name).read_text()
        assert SNAPSHOT_HEADER in text
        assert "SESSION SNAPSHOT" not in text
        mentions = [line for line in text.splitlines() if "snapshot" in line.lower()]
        assert len(mentions) == 1
        assert "must never appear" in text
        assert 'Words like "snapshot"' in mentions[0]


def test_the_prompts_forbid_naming_the_source():
    for name in ("follow_up.md", "follow_up_claims.md"):
        text = (Path("src/agent/prompts/extraction") / name).read_text()
        assert "NEVER NAME THE SOURCE" in text


def test_the_header_the_model_sees_is_the_one_that_was_reviewed():
    """The injection builds its header from SNAPSHOT_HEADER, so the constant
    above is the whole truth about what the model is told."""
    import inspect

    from agent.agents.follow_up import llm as fu_llm

    body = inspect.getsource(fu_llm.extract_follow_up_decision)
    assert "SNAPSHOT_HEADER" in body
    assert "SESSION SNAPSHOT" not in body
