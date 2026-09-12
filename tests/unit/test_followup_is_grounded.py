"""A side question the caller did not ask is not answered.

    AI      Thanks — and could I get your last name?
    Caller  Customer.
    AI      Got your last name as Customer, and I can certainly help you with
            your claim status today. Thanks — now, could I get your date of
            birth?

"Customer." asked nothing. `followup_query` came back as "help with claim
status" — a topic the AI itself had raised two turns earlier — and the turn
paid a generation call to answer a question nobody asked, in words that
restated what the caller had already been told.

The same phantom reaches BaseAgent.execute's safety net, which generates a
SECOND sentence for the turn and prefixes it to the first, so one hallucinated
field costs two generation calls and produces two sentences saying overlapping
things. That is the double-append.

Three extraction headers already tell the model, in capitals, that
followup_query "MUST be derived from the caller's current utterance" and to
"NEVER synthesize a followup_query from topics the AI raised in prior turns".
An instruction repeated three times is an instruction not being followed, and
nothing in Python was checking. Now two places are: reconcile_worker_result
(the funnel every extraction passes through) and note_side_question (the last
gate in front of the net).
"""

from __future__ import annotations

import pytest

from agent.core.agent import BaseAgent
from agent.core.followup_grounding import (
    carries_freeform_content,
    is_grounded_followup,
    request_cue,
)
from agent.core.request_detection import reconcile_worker_result
from agent.llm.schema import EventType, FollowupDisposition, RequestKind, WorkerResult

# ── the detector ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "Customer.",
        "M451982.",
        "Monique",
        "yes",
        "no, that's not right",  # a correction, not a question
        "November 5th 1992",
        "nine zero two one zero",
        "sms",
        "fax is fine",
    ],
)
def test_a_plain_answer_grounds_no_question(utterance):
    assert is_grounded_followup("help with claim status", utterance) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "It's 90210 — when will I get the list?",
        "Customer. Can you help me with a new ID card?",
        "M451982, and what's my deductible",
        "Smith. Sorry, could you say that again",
        "90210. I need to change my email too",
        "Monique — remind me what you already have",
        "yes, but my zip is wrong",
        "Customer, and I want a new card please",
        "sms. how long does that take",
    ],
)
def test_a_real_question_survives(utterance):
    assert is_grounded_followup("anything at all", utterance) is True


def test_no_query_is_never_a_followup():
    assert is_grounded_followup("", "what's my deductible?") is False
    assert is_grounded_followup(None, "what's my deductible?") is False


def test_no_transcript_leaves_the_extractor_trusted():
    """The veto fires on evidence the caller asked nothing, never on the
    absence of anything to check against."""
    assert is_grounded_followup("when will I get the list", "") is True
    assert is_grounded_followup("when will I get the list", None) is True


def test_the_cue_name_is_reported_for_logging():
    assert request_cue("what's my deductible") == "interrogative"
    assert request_cue("Customer.") == ""


# ── reconcile_worker_result: the upstream funnel ─────────────────────────────


def _awf(query: str, **extracted) -> WorkerResult:
    return WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.ANSWER,
        followup_query=query,
        extracted=extracted,
    )


def test_a_phantom_question_is_cleared_and_the_event_downgraded():
    result = reconcile_worker_result(_awf("help with claim status", last_name="Customer"), "Customer.")
    assert result.followup_query is None
    assert result.followup_disposition == FollowupDisposition.NONE
    assert result.event_type == EventType.ANSWERED
    assert result.extracted == {"last_name": "Customer"}, "the answer itself is untouched"


def test_a_real_question_is_left_alone():
    result = reconcile_worker_result(
        _awf("when will the list arrive", zip_code="90210"),
        "It's 90210 — when will I get the list?",
    )
    assert result.followup_query == "when will the list arrive"
    assert result.event_type == EventType.ANSWERED_WITH_FOLLOWUP


def test_an_update_request_keeps_its_event_when_the_question_goes():
    """corrections{} and update_target are the caller's own request shapes and
    are reconciled on their own evidence — only the question is dropped, and
    the event stays ANSWERED_WITH_FOLLOWUP so Case A / Case B still run."""
    result = WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.ANSWER,
        followup_query="claim status",
        extracted={"zip_code": "90210"},
        corrections={"email": "a@b.com"},
    )
    result = reconcile_worker_result(result, "90210 and a at b dot com")
    assert result.followup_query is None
    assert result.event_type == EventType.ANSWERED_WITH_FOLLOWUP
    assert result.corrections == {"email": "a@b.com"}


def test_a_bare_update_target_still_upgrades_to_corrected():
    """The existing ANSWERED-on-a-bare-request veto must keep working through
    the new pass: the downgrade only fires when nothing else is left."""
    result = WorkerResult(
        event_type=EventType.ANSWERED,
        update_target="email",
        request_kind=RequestKind.UPDATE,
        followup_query="change my email",
    )
    result = reconcile_worker_result(result, "I need to change my email")
    assert result.event_type == EventType.CORRECTED
    assert result.update_target == "email"


def test_a_clean_answer_is_untouched():
    result = reconcile_worker_result(WorkerResult(extracted={"last_name": "Customer"}), "Customer.")
    assert result.followup_query is None
    assert result.event_type == EventType.ANSWERED


# ── note_side_question: the last gate in front of the safety net ─────────────

ANSWER = "ID card replacements are handled on a different line."


class _Forgetful(BaseAgent):
    """A handler that never looks at the question — the shape the net exists
    for. See test_side_question_is_answered."""

    AGENT_NAME = "notification_setup_agent"

    def __init__(self, result, said):
        super().__init__()
        self._result = result
        self._said = said

    async def run(self, state):
        if interrupt := await self.run_conversation_guards(state, user_text=self._said, result=self._result):
            return interrupt
        return self.ask_member(state, "Got it — SMS it is.")


def _state(said: str) -> dict:
    return {
        "messages": [{"role": "user", "content": said}],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "notification_method",
        "call_intent": "claim_services",
    }


@pytest.fixture
def generation(monkeypatch):
    calls: list[dict] = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return ANSWER

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    return calls


async def test_the_net_does_not_answer_a_question_nobody_asked(generation):
    said = "SMS please."
    agent = _Forgetful(_awf("help with claim status", notification_method="sms"), said)

    result = await agent.execute(_state(said))

    assert result["messages"]["content"] == "Got it — SMS it is."
    assert generation == [], "a phantom question must not reach the generation LLM"


async def test_the_net_still_answers_a_real_one(generation):
    said = "SMS please. Can you help me with a new ID card?"
    agent = _Forgetful(_awf("new ID card", notification_method="sms"), said)

    result = await agent.execute(_state(said))

    assert result["messages"]["content"] == f"{ANSWER} Got it — SMS it is."
    assert len(generation) == 1


async def test_the_transcript_grounds_it_when_user_text_is_a_placeholder(generation):
    """Production passes the caller's utterance as user_text; a harness may
    pass something else. The transcript is checked too, so a real question is
    never dropped because one of the two renderings was a stand-in."""
    said = "SMS please. Can you help me with a new ID card?"

    class _Placeholder(_Forgetful):
        async def run(self, state):
            if interrupt := await self.run_conversation_guards(state, user_text="x", result=self._result):
                return interrupt
            return self.ask_member(state, "Got it — SMS it is.")

    result = await _Placeholder(_awf("new ID card", notification_method="sms"), said).execute(_state(said))

    assert result["messages"]["content"] == f"{ANSWER} Got it — SMS it is."


# ── carries_freeform_content: the static re-ask path ────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "um, I'm not really sure about that",  # 6 words, nothing to acknowledge
        "the the the yeah",
        "hmm sorry",
        "uh",
        "what?",  # a repeat request the canned re-ask answers exactly
        "",
    ],
)
def test_a_mumble_stays_on_the_canned_re_ask(utterance):
    assert carries_freeform_content(utterance) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "Please check my claim status today.",  # the reported turn
        "How are you doing today?",
        "Can you repeat that?",
        "I need to speak to somebody else",
        "why do you need that",
    ],
)
def test_a_request_always_gets_a_real_response(utterance):
    assert carries_freeform_content(utterance) is True
