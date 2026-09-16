"""A caller asking out of the reference-number fallback is heard.

    AI      What is the reference number for this adjustment request?
    Caller  no in this call
    AI      No problem. I can check another way. Do you have the claim number?
    Caller  exit
    →       guard TRANSFER_REQUEST, guard_confidence 0.95
    AI      I wasn't able to capture that claim number. Could you repeat it?
    Caller  end the call
    AI      Sure — what date was the service, and what was the billed amount?

The extractor flagged the transfer request, twice, and nothing read it. The
caller was asked for the claim number again and then moved on to the next
fallback stage.

run() calls run_conversation_guards for the reference_number phase.
_collect_claim_number_fallback and _collect_dos_billed_fallback extracted,
reconciled, honoured WAIT and the pivots — and then went straight to the
value. So for the whole of the fallback path no guard was read at all: a
transfer request, abuse, and a self-harm signal were dropped in the same
place.

Two orderings matter and are covered below.

A wait must still be a wait. The soft guards defer to a caller asking for a
moment (guards.py), and these stages honour the model's own WAIT label
because detect_wait_request misses a wait carried with meta-commentary —
which a soft guard reaching the turn first would have taken before they
could. That deferral now reads the label as well as the words.

A wait wrapped around a request to leave is a request to leave. guards.py
settles that precedence, but these collectors answer a wait before the
extraction the guards read, to spare an LLM call on the commonest turn there
is — so that early check defers on the transfer phrasings a keyword pass can
settle.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.claim_adjustment import agent as claim_module
from agent.llm.schema import EventType, GuardType, WorkerResult

CLAIM_NUMBER_ASK = "No problem. I can check another way. Do you have the claim number?"
DOS_BILLED_ASK = "Sure — what date was the service, and what was the billed amount?"

# (stage, awaiting_slot, the agent's question, the collector)
CLAIM_NUMBER_STAGE = ("claim_number_ask", "fallback_claim_number", CLAIM_NUMBER_ASK)
DOS_BILLED_STAGE = ("dos_billed_ask", "fallback_dos_billed", DOS_BILLED_ASK)

BOTH_STAGES = [
    pytest.param(CLAIM_NUMBER_STAGE, id="claim_number"),
    pytest.param(DOS_BILLED_STAGE, id="dos_billed"),
]

_COLLECTOR = {
    "claim_number_ask": "_collect_claim_number_fallback",
    "dos_billed_ask": "_collect_dos_billed_fallback",
}


async def _turn(stage_spec, said: str, decision: WorkerResult, record: dict | None = None) -> dict:
    stage, awaiting, asked = stage_spec
    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "call_intent": "claim_services",
        "first_name": "James",
        "last_name": "Wilson",
        "member_id": "M310188",
        "phone_confirmed": True,
        "awaiting_slot": awaiting,
        "ref_no_fallback_stage": stage,
        "messages": [
            {"role": "assistant", "content": asked},
            {"role": "user", "content": said},
        ],
    }

    async def _extract(*_a, **_k):
        return decision

    async def _generate(**_kw):
        return "GENERATED"

    async def _lookup(*_a, **_k):
        return (record, None)

    agent = claim_module.ClaimAdjustmentAgent.from_state(state)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(claim_module, "extract_claim_adjustment_decision", _extract))
        stack.enter_context(patch.object(claim_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        stack.enter_context(patch.object(claim_module, "lookup_adjustment_by_claim_number", _lookup))
        found, interrupt = await getattr(agent, _COLLECTOR[stage])(state, state["messages"], said, asked)
    return {"record": found, "interrupt": interrupt, "awaiting": awaiting, "stage": stage}


def _escalated(out: dict) -> bool:
    r = out["interrupt"] or {}
    return (r.get("last_agent_signal") or {}).get("status") == "escalate"


def _transfers(out: dict) -> list:
    r = out["interrupt"] or {}
    return [
        e
        for e in (r.get("metadata_events") or [])
        if (e.get("data") or {}).get("eventName") == "AgentCallTransfer"
    ]


def _attempts(out: dict) -> int:
    r = out["interrupt"] or {}
    return ((r.get("slot_attempts") or {}).get(out["awaiting"]) or {}).get("attempt_count", 0)


def _guard(kind, *, confidence: float = 0.95, event=EventType.NONE) -> WorkerResult:
    return WorkerResult(event_type=event, guard=kind, guard_confidence=confidence)


# ── the hard guards take the turn ────────────────────────────────────────────


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
@pytest.mark.parametrize(
    "said, guard",
    [
        pytest.param("exit", GuardType.TRANSFER_REQUEST, id="the-reported-turn"),
        pytest.param("can I talk to a person", GuardType.TRANSFER_REQUEST, id="asks-for-a-person"),
        pytest.param("you people are useless", GuardType.ABUSE, id="abuse"),
        pytest.param("I want to hurt myself", GuardType.SELF_HARM, id="self-harm"),
    ],
)
async def test_a_hard_guard_escalates_from_either_stage(stage_spec, said, guard):
    out = await _turn(stage_spec, said, _guard(guard))

    assert _escalated(out), f"{guard} was dropped in the {out['stage']} stage"
    assert len(_transfers(out)) == 1, "the transfer must be reported"


async def test_the_reported_call_escalates_on_the_first_ask():
    """ "exit" is not in the transfer keyword list — nothing but the guard was
    ever going to catch it."""
    from agent.utils import detect_transfer_request

    assert not detect_transfer_request({"messages": [{"role": "user", "content": "exit"}]})

    out = await _turn(CLAIM_NUMBER_STAGE, "exit", _guard(GuardType.TRANSFER_REQUEST))

    assert _escalated(out)
    assert out["interrupt"]["next_node"] == "escalation_agent"


# ── the soft guards keep the stage and cost an attempt ───────────────────────


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
@pytest.mark.parametrize(
    "guard",
    [GuardType.OFFTOPIC_GLOBAL, GuardType.INTERRUPTION],
)
async def test_a_soft_guard_stays_on_the_stage(stage_spec, guard):
    """Off topic and interrupting are not requests to leave — they re-ask. The
    attempt is what stops a caller holding the stage open indefinitely."""
    out = await _turn(stage_spec, "what's the weather like", _guard(guard))

    assert not _escalated(out)
    assert out["interrupt"]["ref_no_fallback_stage"] == out["stage"]
    assert out["interrupt"]["awaiting_slot"] == out["awaiting"]
    assert _attempts(out) == 1, "a soft guard turn is still a non-answer"


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
async def test_a_guard_below_the_confidence_floor_is_ignored(stage_spec):
    """run_conversation_guards reads nothing under 0.7; the turn falls through
    to the stage's own handling."""
    out = await _turn(stage_spec, "exit", _guard(GuardType.TRANSFER_REQUEST, confidence=0.4))

    assert not _escalated(out)
    assert not _transfers(out)


# ── waits ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
async def test_a_wait_is_still_a_wait(stage_spec):
    out = await _turn(stage_spec, "hold on a second", _guard(GuardType.NONE, confidence=0.0))

    assert not _escalated(out)
    assert out["interrupt"]["ref_no_fallback_stage"] == out["stage"]
    assert _attempts(out) == 0, "waiting is not a failed attempt"


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
async def test_a_soft_guard_over_a_wait_label_defers_to_the_wait(stage_spec):
    """detect_wait_request misses a wait carried with meta-commentary, which is
    why these stages honour the model's WAIT label. A soft guard must not take
    the turn before they can."""
    out = await _turn(
        stage_spec,
        "hold on, I need to look this up",
        _guard(GuardType.INTERRUPTION, event=EventType.WAIT),
    )

    assert not _escalated(out)
    assert _attempts(out) == 0
    assert out["interrupt"]["ref_no_fallback_stage"] == out["stage"]


@pytest.mark.parametrize("stage_spec", BOTH_STAGES)
async def test_a_wait_wrapped_around_a_transfer_is_a_transfer(stage_spec):
    """guards.py: "hold on, get me a human" is a transfer. The early keyword
    wait check runs ahead of the extraction the guards read, so it defers."""
    out = await _turn(
        stage_spec,
        "hold on, get a human",
        _guard(GuardType.TRANSFER_REQUEST, event=EventType.WAIT),
    )

    assert _escalated(out)
    assert len(_transfers(out)) == 1


# ── nothing else changed ─────────────────────────────────────────────────────


async def test_a_claim_number_still_collects():
    """The guard call must not take the turn away from a caller answering."""
    out = await _turn(
        CLAIM_NUMBER_STAGE,
        "882301",
        WorkerResult(event_type=EventType.ANSWERED, extracted={"claim_number": "882301"}),
        record={"reference_number": "42695817"},
    )

    assert out["interrupt"] is None, "the answer was intercepted"
    assert out["record"] == {"reference_number": "42695817"}
