"""The same-member question asked itself until the caller hung up.

    AI    Is this request for the same member we've been discussing, or is
          this for a different member?
    User  no worries, that's fine. no, that's everything — thanks for the help
    AI    Could you clarify — is this for the member we already have on file,
          or a different person?
    User  really, that's all — thanks
    AI    Could you clarify — is this for the member we already have on file,
          or a different person?
    User  nope, nothing else
    AI    Could you clarify — is this for the member we already have on file,
          or a different person?

Two faults, one on top of the other.

The caller had dropped the request — they were saying goodbye — and the
classifier has no category for that, so every turn came back "unclear". And
"unclear" re-armed same_member_check_pending and re-asked, with no counter
anywhere, so "unclear" meant ask again for as many turns as the caller was
willing to spend. Its own docstring promised "a single clarification question;
if still unclear after that, we default to routing through verification"; the
code never counted.

So: a withdrawal is read from the words and ends the call, and clarification is
capped — and because the caller heard the identical sentence every time, the
two phrasings are now spent in order rather than drawn at random.
"""

from __future__ import annotations

import pytest

from agent.agents.intake.constants import (
    SAME_MEMBER_CLARIFICATION_MSGS,
    SAME_MEMBER_MAX_CLARIFICATIONS,
)
from agent.agents.intake.handlers import screen_request_withdrawn
from agent.llm.schema import WorkerResult
from agent.orchestration.orchestration import AgentNode

# Every caller turn from the reported transcript, after the question was asked.
WITHDRAWALS = [
    "no worries, that's fine. no, that's everything — thanks for the help",
    "really, that's all — thanks",
    "nope, nothing else",
    "I'm all set",
    "that's it, thank you",
]


@pytest.fixture
def intake(monkeypatch):
    """Intake with the same-member classifier answering "unclear" — what the
    prompt returned for every one of those turns before this change."""
    from agent.agents.intake import agent as ik

    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"same_member": "unclear"})

    monkeypatch.setattr(ik, "extract_same_member_decision", _extract)
    monkeypatch.setattr(ik, "get_extraction_llm", lambda: object())
    return ik


def _state(utterance: str, attempts: int = 0) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "could you check what my doctor billed for that visit?"},
            {
                "role": "assistant",
                "content": (
                    "Is this request for the same member we've been discussing, "
                    "or is this for a different member?"
                ),
            },
            {"role": "user", "content": utterance},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "call_intent": "claim_services",
        "same_member_check_pending": True,
        "same_member_clarify_attempts": attempts,
        "saved_member_context": {"first_name": "James", "member_id": "M310188"},
        "member_status_verify": False,
    }


async def _turn(intake, utterance: str, attempts: int = 0) -> dict:
    state = _state(utterance, attempts)
    return await intake.IntakeAgent.from_state(state).execute(state)


# ── the reported turns ───────────────────────────────────────────────────────


@pytest.mark.parametrize("utterance", WITHDRAWALS)
async def test_a_caller_saying_goodbye_is_not_asked_again(intake, utterance):
    result = await _turn(intake, utterance)
    assert "different person" not in (result.get("messages") or {}).get("content", "")
    assert not result.get("same_member_check_pending")


@pytest.mark.parametrize("utterance", WITHDRAWALS)
async def test_a_caller_saying_goodbye_reaches_closure(intake, utterance):
    result = await _turn(intake, utterance)
    assert result["next_node"] == AgentNode.CLOSURE.value
    assert result["last_agent_signal"]["closure_requested"] is True
    assert not result.get("is_interrupt")


@pytest.mark.parametrize("utterance", WITHDRAWALS)
def test_the_withdrawal_is_read_from_the_words(utterance):
    """No LLM call is spent deciding that "that's everything" ends the call."""
    assert screen_request_withdrawn(utterance)


async def test_the_saved_member_is_not_carried_into_the_goodbye(intake):
    result = await _turn(intake, WITHDRAWALS[0])
    updates = result["last_agent_signal"]["context_updates"]
    assert updates["saved_member_context"] is None
    assert updates["same_member_check_pending"] is False


# ── an answer that happens to end politely is still an answer ────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "no, it's for a different member",
        "that's all — it's for my wife",
        "same member, that's it",
        "different person, thanks",
    ],
)
def test_naming_a_member_outranks_the_closing_breath(utterance):
    assert not screen_request_withdrawn(utterance)


def test_nothing_else_is_not_a_different_member():
    """The keyword fallback matched "no" inside "nothing" and "nope", so a
    caller hanging up looked like a caller naming someone new."""
    from agent.agents.intake.handlers import _phrase_hit

    for word in ("nothing else", "I don't know", "let me know", "renew my card"):
        assert not _phrase_hit(("no", "new"), word)
    assert _phrase_hit(("no", "new"), "no, a new member")


# ── the cap ──────────────────────────────────────────────────────────────────


async def test_an_unanswerable_question_is_dropped_not_repeated(intake):
    """A hedge that never resolves stops being asked about."""
    result = await _turn(intake, "I think so, probably", attempts=SAME_MEMBER_MAX_CLARIFICATIONS)
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert not result["same_member_check_pending"]
    assert result["saved_member_context"] is None


async def test_the_first_hedge_is_still_clarified(intake):
    """The cap must not cost the clarification that does work — a caller who
    answers "I think so" on turn one usually settles it when asked again."""
    result = await _turn(intake, "I think so, probably")
    assert result["same_member_check_pending"] is True
    assert result["same_member_clarify_attempts"] == 1
    assert result["is_interrupt"] is True


async def test_the_caller_never_hears_the_same_sentence_twice(intake):
    """Two clarifications, two different phrasings — the transcript's complaint
    was as much the repetition as the looping."""
    spoken = [
        (await _turn(intake, "I think so, probably", attempts=n))["messages"]["content"]
        for n in range(SAME_MEMBER_MAX_CLARIFICATIONS)
    ]
    assert len(set(spoken)) == len(spoken)
    assert set(spoken) <= set(SAME_MEMBER_CLARIFICATION_MSGS)


async def test_the_clarification_budget_is_bounded(intake):
    """However many turns the caller spends, the question is asked a fixed
    number of times and then stops."""
    asked = 0
    attempts = 0
    for _ in range(10):
        result = await _turn(intake, "I think so, probably", attempts=attempts)
        if not result.get("same_member_check_pending"):
            break
        asked += 1
        attempts = result["same_member_clarify_attempts"]
    else:
        pytest.fail("the same-member question never stopped asking")
    assert asked == SAME_MEMBER_MAX_CLARIFICATIONS


# ── the answers still work ───────────────────────────────────────────────────


async def test_a_confirmed_same_member_still_skips_verification(monkeypatch):
    from agent.agents.intake import agent as ik

    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"same_member": "yes"})

    monkeypatch.setattr(ik, "extract_same_member_decision", _extract)
    monkeypatch.setattr(ik, "get_extraction_llm", lambda: object())

    state = _state("yes, same member")
    result = await ik.IntakeAgent.from_state(state).execute(state)
    updates = result["last_agent_signal"]["context_updates"]
    assert updates["member_status_verify"] is True
    assert updates["member_id"] == "M310188"
    assert updates["same_member_clarify_attempts"] == 0


async def test_a_different_member_still_goes_to_verification(monkeypatch):
    from agent.agents.intake import agent as ik

    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"same_member": "no"})

    monkeypatch.setattr(ik, "extract_same_member_decision", _extract)
    monkeypatch.setattr(ik, "get_extraction_llm", lambda: object())

    state = _state("no, it's for my wife")
    result = await ik.IntakeAgent.from_state(state).execute(state)
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["saved_member_context"] is None
    assert "first name" in result["messages"]["content"].lower()


# ── the route out of intake ──────────────────────────────────────────────────


def test_closure_is_reachable_from_intake():
    """Everything intake_routing does not name falls through to verification —
    the withdrawal has to be named or the goodbye becomes a re-verification."""
    from agent.app_graph import intake_routing

    assert intake_routing({"next_node": AgentNode.CLOSURE.value}) == AgentNode.CLOSURE.value
