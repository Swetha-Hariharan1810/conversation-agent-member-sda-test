"""A question asked alongside a slot answer must be answered, not dropped.

    AI      …would you like the benefits for office visits with your Pediatrician?
    Caller  No. But I lost my ID card. Can you help me with the new one?
    AI      By the way, you are eligible for a free health and wellness coach…

The extraction was right: event_type answered_with_followup, disposition
"answer", followup_query "I lost my credit ID card. Can you help me with the
new one?". _handle_benefits_response read extracted["benefits_response"], took
the "no", and returned — event_type and followup_query were never looked at.

Nothing else caught it either. The guard layer only acts at
guard_confidence >= 0.7 and this turn carries 0.0, and the
repeated-ignored-request escalation hangs off the OFFTOPIC_AGENT branch, so a
question arriving as answered_with_followup cannot reach it however many times
the caller repeats it — which they did, on the very next turn.

This was never a benefits bug. _collect_slot is the only implementation of
answer-plus-question; every slot collected by a hand-written handler
reimplemented the "answer" half alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.core.slot_manager import SlotManagerMixin
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

ANSWER = "ID card replacements are handled on a different line."
QUESTION = "I lost my credit ID card. Can you help me with the new one?"


@pytest.fixture
def generation(monkeypatch):
    """Stub LLM 2 and record what each call asked it for."""
    calls: list[dict] = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return ANSWER

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    return calls


def _answered_with(query: str | None, **extracted) -> WorkerResult:
    return WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP if query else EventType.ANSWERED,
        followup_disposition=FollowupDisposition.ANSWER if query else FollowupDisposition.NONE,
        followup_query=query,
        extracted=extracted,
    )


# ── the helper ───────────────────────────────────────────────────────────────


async def _ask(result, **state) -> str:
    """Driven through a real agent — the helper lives on the mixin every agent
    inherits, and the slot bookkeeping it touches comes from BaseAgent."""
    from agent.agents.delivery_management.agent import DeliveryManagementAgent

    base = {"call_intent": "provider_services", "messages": [], "slot_attempts": {}}
    return await DeliveryManagementAgent().answer_side_question(
        {**base, **state}, [], result=result, slot_name="benefits_response", extracted_value="no"
    )


async def test_a_turn_with_no_question_generates_nothing(generation):
    assert await _ask(_answered_with(None, benefits_response="no")) == ""
    assert generation == [], "a plain answer must not cost a generation call"


async def test_a_question_is_answered(generation):
    assert await _ask(_answered_with(QUESTION, benefits_response="no")) == ANSWER
    assert generation[0]["followup_query"] == QUESTION


def test_the_answer_never_ends_in_a_question(generation):
    """The handler's own next message — or the next agent's opener — is the one
    question of the turn, so the generated sentence must not add another."""
    import inspect

    body = inspect.getsource(SlotManagerMixin.answer_side_question)
    assert "will_append_ask=True" in body


async def test_it_asks_for_the_responder_guard_not_a_park(generation):
    await _ask(_answered_with(QUESTION, benefits_response="no"))
    assert generation[0]["guard"] == "FOLLOWUP_RESPOND"


async def test_the_rest_of_the_call_is_available_to_answer_from(generation):
    """A question about a later step ("will I get a text?") is answerable from
    Coming up:, the same as on the slot-pipeline path."""
    await _ask(_answered_with(QUESTION, benefits_response="no"))
    assert generation[0]["coming_up"], "no call stages passed to the responder"


# ── joining it to what the handler was already saying ────────────────────────


def test_join_puts_the_answer_first():
    assert SlotManagerMixin.join_side_answer(ANSWER, "Anything else?") == f"{ANSWER} Anything else?"


def test_join_with_no_answer_is_the_message_alone():
    assert SlotManagerMixin.join_side_answer("", "Anything else?") == "Anything else?"
    assert SlotManagerMixin.join_side_answer("   ", "Anything else?") == "Anything else?"


def test_prefix_puts_the_answer_in_front_of_a_spoken_result():
    result = {"messages": {"role": "assistant", "content": "Anything else?"}}
    assert SlotManagerMixin.prefix_side_answer(result, ANSWER)["messages"]["content"] == (
        f"{ANSWER} Anything else?"
    )


def test_prefix_gives_a_silent_handoff_something_to_say():
    """signal_complete(message="") speaks nothing — without this the answer is
    generated and then thrown away, which is the original bug with extra steps."""
    result = {"last_agent_signal": {"status": "complete"}}
    assert SlotManagerMixin.prefix_side_answer(result, ANSWER)["messages"]["content"] == ANSWER


def test_prefix_with_no_answer_leaves_the_result_untouched():
    result = {"messages": {"role": "assistant", "content": "Anything else?"}}
    assert SlotManagerMixin.prefix_side_answer(dict(result), "") == result


# ── the reported call, through the real agent ────────────────────────────────


@pytest.fixture
def delivery(monkeypatch):
    from agent.agents.delivery_management import agent as dm

    async def _no_guards(*_args, **_kwargs):
        return None

    monkeypatch.setattr(dm.DeliveryManagementAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())
    return dm


def _benefits_offer_state(last_user: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "…would you like the benefits for office visits?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "member_id": "M310188",
        "delivery_method": "fax",
        "fax": "2315553211",
        "awaiting_slot": "benefits_response",
        "provider_type": "Pediatrician",
        "zip_code": "16783",
        "zip_code_used": "16783",
        "provider_list_sent": True,
        "benefits_offer_made": True,
    }


async def test_the_reported_call_now_answers_the_id_card_question(delivery, monkeypatch, generation):
    async def _extract(*_args, **_kwargs):
        return _answered_with(QUESTION, benefits_response="no")

    monkeypatch.setattr(delivery, "extract_delivery_management_decision", _extract)

    result = await delivery.DeliveryManagementAgent().run(
        _benefits_offer_state("No. But I lost my credit ID card. Can you help me with the new one?")
    )

    assert ANSWER in result["messages"]["content"]


async def test_a_plain_no_still_hands_off_silently(delivery, monkeypatch, generation):
    """No question asked — the next agent's opener is the whole turn, as before."""

    async def _extract(*_args, **_kwargs):
        return _answered_with(None, benefits_response="no")

    monkeypatch.setattr(delivery, "extract_delivery_management_decision", _extract)

    result = await delivery.DeliveryManagementAgent().run(_benefits_offer_state("No thanks."))

    assert "messages" not in result
    assert generation == []


# ── every hand-written yes/no resolution honours it ──────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "delivery_management/agent.py",  # benefits_response, fax_confirmed, email_confirmed
        "benefits/agent.py",  # care_coach_response
        "records_coordination/agent.py",  # upload_consent, personal_guide_consent
        "notification_setup/agent.py",  # timeline_question, phone_confirmed, email_confirmed
        "provider_search/agent.py",  # zip_confirmed
    ],
)
def test_every_agent_that_resolves_a_yes_no_by_hand_answers_side_questions(path):
    """These bypass _collect_slot, so nothing else will do it for them."""
    body = Path(f"src/agent/agents/{path}").read_text()
    assert "normalize_yes_no(" in body, f"{path} no longer resolves a yes/no — update this test"
    assert "answer_side_question(" in body, f"{path} drops followup_query on its yes/no path"
