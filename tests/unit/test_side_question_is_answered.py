"""A question asked alongside a slot answer is answered, whoever collects it.

    AI      …would you like the benefits for office visits with your Pediatrician?
    Caller  No. But I lost my ID card. Can you help me with the new one?
    AI      By the way, you are eligible for a free health and wellness coach…

The extraction was right — answered_with_followup, disposition "answer",
followup_query set. _handle_benefits_response read extracted["benefits_response"],
took the "no" and returned; event_type and followup_query were never looked at.

Nothing downstream caught it: the guard layer only acts at
guard_confidence >= 0.7 and such a turn carries 0.0, and the
repeated-ignored-request escalation hangs off the OFFTOPIC_AGENT branch, so
the caller asked twice and was ignored twice.

It was never one agent's bug. _collect_slot was the only implementation of
answer-plus-question, so every slot collected by a hand-written handler
depended on that handler remembering — and a handler that forgot failed
silently, on every turn the caller asked. Wiring each site in turn fixed the
sites that had been noticed and left the rest, which is how the yes/no
handlers got fixed while the value handlers kept dropping it.

So remembering is no longer asked of handlers. The guard layer every
slot-collecting agent already runs records the question; any handler that
answers it consumes it on the way through _generate_slot_retry_response; and
BaseAgent.execute — which every agent node in the graph goes through — answers
whatever is left. A new slot cannot drop a question, because keeping it costs
the handler nothing.
"""

from __future__ import annotations

import pytest

from agent.core.agent import BaseAgent
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


# ── the net: a handler that never looks at the question ──────────────────────


class _Forgetful(BaseAgent):
    """The failure shape: read extracted[slot], branch on the value, return.

    This is every hand-written collector in the codebase, and the point of the
    net is that this agent needs no knowledge of side questions to keep one.
    """

    AGENT_NAME = "notification_setup_agent"

    def __init__(self, result):
        super().__init__()
        self._result = result

    async def run(self, state):
        if interrupt := await self.run_conversation_guards(state, user_text="x", result=self._result):
            return interrupt
        return self.ask_member(state, "Got it — SMS it is.")


def _state(**over) -> dict:
    return {
        "messages": [{"role": "user", "content": QUESTION}],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "notification_method",
        "call_intent": "claim_services",
        **over,
    }


async def test_a_handler_that_ignores_the_question_still_answers_it(generation):
    result = await _Forgetful(_answered_with(QUESTION, notification_method="sms")).execute(_state())
    assert result["messages"]["content"] == f"{ANSWER} Got it — SMS it is."


async def test_a_turn_with_no_question_is_untouched_and_costs_nothing(generation):
    result = await _Forgetful(_answered_with(None, notification_method="sms")).execute(_state())
    assert result["messages"]["content"] == "Got it — SMS it is."
    assert generation == [], "a plain answer must not reach the generation LLM"


async def test_the_answer_goes_in_front_of_what_the_turn_says(generation):
    """The caller hears their question addressed before being moved along."""
    content = (await _Forgetful(_answered_with(QUESTION, notification_method="sms")).execute(_state()))[
        "messages"
    ]["content"]
    assert content.index(ANSWER) < content.index("Got it")


async def test_a_guard_that_takes_the_turn_drops_the_question(generation):
    """A transfer or an abuse escalation owns the whole response — it must not
    carry an aside about ID cards."""
    transfer = _answered_with(QUESTION, notification_method="sms")
    transfer.guard = "TRANSFER_REQUEST"
    transfer.guard_confidence = 0.95

    result = await _Forgetful(transfer).execute(_state())

    assert "messages" not in result  # an escalation speaks via escalation_pre_message
    assert ANSWER not in result.get("escalation_pre_message", "")
    assert generation == []


async def test_an_escalation_raised_inside_the_turn_gets_no_aside(generation):
    """signal_escalate carries no "messages" key by design — attaching an
    answer would invent one, and a caller being transferred does not need an
    aside about ID cards on the way."""

    class _Escalating(_Forgetful):
        async def run(self, state):
            await self.run_conversation_guards(state, user_text="x", result=self._result)
            return self.signal_escalate(state, "Transferring you now.", reason="exhausted")

    result = await _Escalating(_answered_with(QUESTION, notification_method="sms")).execute(_state())

    assert "messages" not in result
    assert generation == []


async def test_the_question_does_not_leak_into_the_next_turn(generation):
    """Consumed once, whether or not it was answered."""
    agent = _Forgetful(_answered_with(QUESTION, notification_method="sms"))
    await agent.execute(_state())
    assert agent.consume_side_question() == {}


# ── no double answer where a handler already answers ─────────────────────────


def test_answering_through_the_shared_generator_consumes_the_question():
    """Every path that addresses a side question — the slot pipeline, intake,
    the name read-back, the records re-asks — passes a followup_query through
    _generate_slot_retry_response. That is the one place consumption hooks in,
    so no handler has to remember to claim what it answered."""
    import inspect

    body = inspect.getsource(SlotManagerMixin._generate_slot_retry_response)
    assert 'sc.get("followup_query")' in body
    assert "self.consume_side_question()" in body


async def test_the_reported_call_answers_the_id_card_question_once(generation, monkeypatch):
    """The real delivery agent, real guard layer, the transcript as it happened."""
    from agent.agents.delivery_management import agent as dm

    async def _extract(*_args, **_kwargs):
        return _answered_with(QUESTION, benefits_response="no")

    monkeypatch.setattr(dm, "extract_delivery_management_decision", _extract)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())

    state = _state(
        messages=[
            {"role": "assistant", "content": "…would you like the benefits for office visits?"},
            {"role": "user", "content": f"No. {QUESTION}"},
        ],
        awaiting_slot="benefits_response",
        call_intent="provider_services",
        delivery_method="fax",
        fax="2315553211",
        provider_type="Pediatrician",
        zip_code="16783",
        zip_code_used="16783",
        provider_list_sent=True,
        benefits_offer_made=True,
    )
    result = await dm.DeliveryManagementAgent.from_state(state).execute(state)

    assert result["messages"]["content"].count(ANSWER) == 1
    assert len(generation) == 1, "the question was answered more than once"


# ── the recording point every slot-collecting agent shares ───────────────────


def test_the_guard_entry_records_the_question_for_every_agent():
    """Recording lives in run_conversation_guards because that is the one call
    every slot-collecting agent makes with the extraction result in hand."""
    import inspect

    from agent.core.guards import ConversationGuardsMixin

    assert "self.note_side_question(result)" in inspect.getsource(
        ConversationGuardsMixin._run_conversation_guards
    )
    assert "self.discard_side_question()" in inspect.getsource(
        ConversationGuardsMixin.run_conversation_guards
    )


@pytest.mark.parametrize(
    "path",
    [
        "delivery_management/agent.py",
        "benefits/agent.py",
        "records_coordination/agent.py",
        "notification_setup/agent.py",
        "provider_search/agent.py",
        "claim_adjustment/agent.py",
        "verification/agent.py",
    ],
)
def test_every_slot_collecting_agent_runs_the_guard_layer_with_the_result(path):
    """The recording hook rides on this call. An agent that extracts without it
    would collect slots outside the net — nothing else would notice."""
    from pathlib import Path

    body = Path(f"src/agent/agents/{path}").read_text()
    assert "run_conversation_guards(" in body, f"{path} extracts without running the guard layer"
    assert "result=result" in body, f"{path} runs guards without handing over the extraction result"


# ── attaching the answer ─────────────────────────────────────────────────────


def test_join_puts_the_answer_first():
    assert SlotManagerMixin.join_side_answer(ANSWER, "Anything else?") == f"{ANSWER} Anything else?"


def test_join_with_no_answer_is_the_message_alone():
    assert SlotManagerMixin.join_side_answer("", "Anything else?") == "Anything else?"


def test_prefix_gives_a_silent_handoff_something_to_say():
    """signal_complete(message="") speaks nothing — without this the answer is
    generated and then thrown away, which is the original bug with extra steps."""
    result = {"last_agent_signal": {"status": "complete"}}
    assert SlotManagerMixin.prefix_side_answer(result, ANSWER)["messages"]["content"] == ANSWER


def test_prefix_with_no_answer_leaves_the_result_untouched():
    result = {"messages": {"role": "assistant", "content": "Anything else?"}}
    assert SlotManagerMixin.prefix_side_answer(dict(result), "") == result
