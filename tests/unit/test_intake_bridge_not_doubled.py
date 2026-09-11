"""One acknowledgement, one ask — the intake bridge said both twice.

    AI    Thank you for calling Sagility Health… How can I help today?
    User  I want to check the claim status. Can you help me with that today?
    AI    I can certainly help you check your claim status. Could I get your
          first name? I can definitely help with that. To get started, could I
          get your first name?

One message, two complete acknowledgement-and-ask pairs.

The caller asked something alongside their intent, so the turn went through
the generation LLM, which acknowledged AND asked for the first name. Python
then appended a bridge from INTENT_BRIDGE_MSGS — which also acknowledges and
also asks, because it is written for the clean path where it is the whole turn.

Option A says the generation LLM never asks for a slot and Python appends the
ask, and sanitize_generated enforces it whenever the caller passes
will_append_ask. _collect_slot passes it. Intake, the one other place that
appends, did not — so the model's ask survived next to the appended one.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from agent.agents.intake.constants import INTENT_BRIDGE_ASKS, INTENT_BRIDGE_MSG, INTENT_BRIDGE_MSGS
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

GENERATED = "I can certainly help you check your claim status. Could I get your first name?"


@pytest.fixture
def intake(monkeypatch):
    from agent.agents.intake import agent as ik

    async def _generate(**_kwargs):
        return GENERATED

    async def _extract(*_args, **_kwargs):
        return WorkerResult(
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            followup_disposition=FollowupDisposition.ANSWER,
            followup_query="Can you help me with that today?",
            extracted={"intent": "claim_services"},
        )

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    monkeypatch.setattr(ik, "extract_intake_intent", _extract)
    monkeypatch.setattr(ik, "get_extraction_llm", lambda: object())
    return ik


def _state() -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Thank you for calling Sagility Health. How can I help?"},
            {"role": "user", "content": "I want to check the claim status. Can you help me with that today?"},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "intent",
    }


async def _turn(intake, seed: int = 0) -> str:
    random.seed(seed)
    state = _state()
    result = await intake.IntakeAgent.from_state(state).execute(state)
    return result["messages"]["content"]


# ── the reported turn ────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(6))
async def test_the_first_name_is_asked_once(intake, seed):
    """Whichever line the pool draws."""
    assert (await _turn(intake, seed)).lower().count("first name") == 1


async def test_the_models_own_ask_is_stripped(intake):
    assert "Could I get your first name? I" not in await _turn(intake)


@pytest.mark.parametrize("seed", range(6))
async def test_the_turn_acknowledges_once(intake, seed):
    """The generated acknowledgement answers what the caller actually said;
    the bridge's canned one on top of it reads as a stutter."""
    out = await _turn(intake, seed)
    assert "I can certainly help you check your claim status." in out
    for canned in ("I can definitely help with that", "Of course — happy to help", "Sure thing"):
        assert canned not in out


async def test_the_turn_still_ends_by_asking(intake):
    assert (await _turn(intake)).rstrip().endswith("?")


# ── the two pools ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("line", INTENT_BRIDGE_ASKS)
def test_the_bare_pool_is_only_an_ask(line):
    """No acknowledgement — something else has already done that."""
    assert INTENT_BRIDGE_MSG in line
    assert line.count("?") == 1
    for canned in ("help with that", "happy to help", "Absolutely", "Sure thing"):
        assert canned not in line


@pytest.mark.parametrize("line", INTENT_BRIDGE_MSGS)
def test_the_full_pool_still_acknowledges(line):
    """It is still the whole turn on the clean path, where nothing precedes it."""
    assert INTENT_BRIDGE_MSG in line


def test_the_clean_path_still_uses_the_full_bridge():
    body = Path("src/agent/agents/intake/agent.py").read_text()
    assert body.count("random.choice(INTENT_BRIDGE_MSGS)") == 2  # same-member + new-intent entries
    assert body.count("random.choice(INTENT_BRIDGE_ASKS)") == 1  # the answered-with-followup turn


# ── the invariant that was missed ────────────────────────────────────────────


def test_appending_an_ask_means_declaring_it():
    """sanitize_generated only strips the model's competing ask when the caller
    says an ask is coming. Intake is the one agent that appends one."""
    import inspect

    from agent.agents.intake.agent import IntakeAgent

    body = inspect.getsource(IntakeAgent.run)
    assert "will_append_ask=True" in body
    assert 'next_slot_label="first name"' in body


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
def test_no_other_agent_appends_to_a_generated_message_unguarded(path):
    """Every other site hands the generated sentence to ask_member whole. If one
    starts appending, it needs will_append_ask or this bug comes back."""
    body = Path(f"src/agent/agents/{path}").read_text()
    for marker in ("msg.rstrip() + ", "retry_msg.rstrip() + ", "followup_msg.rstrip() + "):
        if marker in body:
            assert "will_append_ask=True" in body, f"{path} appends after generating without declaring it"
