"""A call that is only ever deflected reaches a representative.

    Caller  I need to add my son to the plan.
    AI      That's not something I can help with on this call. Back to where
            we were — could I get your date of birth?
    Caller  How do I get my son covered?
    AI      That's not something I can help with on this call. …
    Caller  Who do I talk to about adding a dependent?
    AI      That's not something I can help with on this call. …

Each of those turns declines and waits, so each is a loop with no exit of its
own. Two things were meant to stop it and neither could:

  * INTERRUPTION had no counter at all.
  * The off-topic counter is keyed on _normalize_request_key(user_text) — the
    caller's exact phrasing — so rephrasing the same request opens a fresh
    budget every turn and the escalation never fires.

MAX_DEFLECTED_TURNS is one counter for the whole call, shared by every guard
that deflects, and it is checked BEFORE the response is generated: there is no
point spending a generation call on a decline the caller will not be given.
"""

from __future__ import annotations

import pytest

from agent.core.agent import BaseAgent
from agent.core.constants import MAX_DEFLECTED_TURNS
from agent.core.signal import AgentStatus
from agent.llm.schema import GuardType, WorkerResult

DECLINE = "That's not something I can help with on this call."


class _Deflected(BaseAgent):
    AGENT_NAME = "verification_agent"
    GUARD = "OFFTOPIC_AGENT"

    def __init__(self) -> None:
        super().__init__()
        self._guard = self.GUARD

    @classmethod
    def for_guard(cls, guard: str, state: dict) -> "_Deflected":
        """Built the way the graph builds every agent: slot state restored from
        LangGraph state. The deflection counter lives there, so an agent
        constructed bare would start every turn with a fresh budget."""
        agent = cls.from_state(state)
        agent._guard = guard
        return agent

    async def run(self, state):
        result = WorkerResult(guard=GuardType(self._guard), guard_confidence=0.95)
        if interrupt := await self.run_conversation_guards(
            state, user_text=state["messages"][-1]["content"], result=result
        ):
            return interrupt
        return self.ask_member(state, "Could I get your date of birth?")


def _state(said: str, slot_attempts: dict | None = None) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Could I get your date of birth?"},
            {"role": "user", "content": said},
        ],
        "slot_attempts": slot_attempts or {},
        "app_run_id": "test-run",
        "awaiting_slot": "dob",
        "call_intent": "provider_services",
        "member_status_verify": True,
    }


@pytest.fixture(autouse=True)
def generation(monkeypatch):
    """Stub LLM 2. Returns a bare decline so sanitizing leaves it standing."""
    calls: list[dict] = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return DECLINE

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    return calls


def _escalated(result: dict) -> bool:
    status = (result.get("last_agent_signal") or {}).get("status")
    return str(getattr(status, "value", status) or "") == AgentStatus.ESCALATE.value


# Each turn rephrases the same request, which is what defeated the
# per-phrasing counter.
_REPHRASINGS = [
    "I need to add my son to the plan.",
    "How do I get my son covered?",
    "Who do I talk to about adding a dependent?",
    "Can someone put my son on my policy?",
    "What about coverage for my kid?",
    "Is there a way to add a child?",
]


@pytest.mark.parametrize("guard", ["OFFTOPIC_AGENT", "INTERRUPTION"])
async def test_a_rephrased_request_cannot_loop_forever(guard):
    slot_attempts: dict = {}
    for turn, said in enumerate(_REPHRASINGS, start=1):
        turn_state = _state(said, slot_attempts)
        result = await _Deflected.for_guard(guard, turn_state).execute(turn_state)
        slot_attempts = result.get("slot_attempts") or slot_attempts
        if _escalated(result):
            assert turn <= MAX_DEFLECTED_TURNS, "the budget must not be exceeded"
            return
    pytest.fail(f"{guard} deflected {len(_REPHRASINGS)} rephrasings without ever escalating")


@pytest.mark.parametrize("guard", ["OFFTOPIC_AGENT", "INTERRUPTION"])
async def test_the_first_deflection_still_answers_the_caller(guard):
    """The budget is a backstop, not a hair trigger: a caller who goes off
    topic once is answered and brought back, exactly as before."""
    turn_state = _state(_REPHRASINGS[0])
    result = await _Deflected.for_guard(guard, turn_state).execute(turn_state)

    assert not _escalated(result)
    assert DECLINE in result["messages"]["content"]


async def test_the_budget_is_checked_before_the_generation_call(generation):
    """A decline the caller will not be given must not be generated."""
    slot_attempts = {"deflected_turns": {"attempt_count": MAX_DEFLECTED_TURNS - 1, "confirmed": False}}

    turn_state = _state(_REPHRASINGS[0], slot_attempts)
    result = await _Deflected.for_guard("OFFTOPIC_AGENT", turn_state).execute(turn_state)

    assert _escalated(result)
    assert generation == []


async def test_the_budget_sits_above_the_slot_retry_limit():
    """So the per-slot and per-guard limits still fire first where they apply."""
    from agent.core.constants import MAX_SLOT_ATTEMPTS

    assert MAX_DEFLECTED_TURNS > MAX_SLOT_ATTEMPTS
