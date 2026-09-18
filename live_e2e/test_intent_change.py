"""
test_intent_change.py — Live E2E: mid-call intent switch re-verification.

Covers the Phase 3/4 behavior added for mid-call intent switches:

  * A fresh intake intent raised during follow-up FULLY resets the call and
    re-routes through verification — reset_for_new_intent() + the follow_up
    reroute + verification_routing + the pending_intent dispatch in
    VerificationAgent._signal_verified().
  * A same-intent action (re-supplying a reference number inside the claim
    flow) does NOT reset identity / verification.
  * An end-call follow-up ("no thanks") does NOT reset and does NOT re-verify.

LIVE: nothing is mocked — every LLM / Salesforce call is real. Requires the
AZURE_OPENAI_* + SF_* env vars and the Salesforce fixtures from preflight.py
(Emily Carter M907503, James Wilson M310188, adjustment 42695817). Marked
`live`, so default `pytest` runs skip it.

The scenario harness (Expected / TurnExpectation) can only assert the FINAL
state plus per-turn AI text; these tests need to inspect state MID-call (right
after the pivot, before re-verification completes), so they drive the compiled
graph directly with a tiny local driver that mirrors harness._drive mechanics.

NAMING NOTE: the staged intent value is the intent TAG "claim_services" /
"provider_services" (the call_intent / detected_intent vocabulary), not a node
name. The phase brief wrote pending_intent == "claim_adjustment"; the real,
code-accurate value is "claim_services" (mapped to the claim_adjustment_agent
node by verification's _PENDING_INTENT_NODE). Asserted as "claim_services".
"""

from __future__ import annotations

import uuid

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import extract_last_ai_message
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight

pytestmark = pytest.mark.live

_TERMINAL = ("END",)


# ──────────────────────────────────────────────────────────────────────────────
# Scripted turns (reuse the same fixtures the scenarios.py flows use)
# ──────────────────────────────────────────────────────────────────────────────

# Emily Carter M907503 — provider flow through to the follow-up "anything else?"
_EMILY_PROVIDER_TO_FOLLOW_UP = [
    "I need to find a primary care physician in my area.",  # intent
    "emily",
    "carter",
    "yes correct",  # name readback confirmed
    "m nine zero seven five zero three",  # member id
    "April twelfth nineteen eighty eight",  # dob
    "I'm calling for myself",  # relationship
    "Primary Care Physician",  # provider type
    "yes that's correct",  # zip on file confirmed
    "send it to my fax",  # delivery method
    "yes that's correct",  # fax on file confirmed
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach → follow-up "anything else?"
]

# Emily Carter M907503 — claim-flow re-verification turns (slot order:
# first/last name → readback → member id → dob → phone confirmation).
_EMILY_CLAIM_REVERIFY = [
    "emily",
    "carter",
    "yes correct",  # name readback confirmed
    "m nine zero seven five zero three",  # member id
    "April twelfth nineteen eighty eight",  # dob
    "yes that's correct",  # phone confirmation
]

# James Wilson M310188 — claim flow through to the reference-number ask.
_JAMES_CLAIM_VERIFY = [
    "I adjusted the claim and I want to follow up",  # intent
    "james",
    "wilson",
    "yes correct",  # name readback confirmed
    "m three one zero one eight eight",  # member id
    "Thirtieth of July, nineteen seventy seven",  # dob
    "yes correct",  # phone confirmation
]


# ──────────────────────────────────────────────────────────────────────────────
# Minimal live driver — mirrors harness._drive but exposes intermediate state
# ──────────────────────────────────────────────────────────────────────────────


def _new_graph():
    from langgraph.checkpoint.memory import MemorySaver

    from agent.app_graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["live_e2e", "intent_change"],
    }
    return graph, config


async def _start(graph, config) -> dict:
    return await graph.ainvoke({}, config=config)


async def _send(graph, config, user: str) -> dict:
    """Resume with one user utterance, then auto-advance through any internal
    (non-interrupt, non-terminal) super-steps until the graph pauses for input
    or reaches END. Returns the state at that pause/END."""
    from langgraph.graph import END
    from langgraph.types import Command

    state = await graph.ainvoke(Command(resume=user), config=config)
    guard = 0
    while not state.get("is_interrupt") and state.get("next_node") not in (END, "END") and guard < 12:
        state = await graph.ainvoke(Command(resume=""), config=config)
        guard += 1
    return state


async def _send_all(graph, config, turns: list[str]) -> dict:
    state: dict = {}
    for turn in turns:
        state = await _send(graph, config, turn)
    return state


def _is_terminal(state: dict) -> bool:
    from langgraph.graph import END

    return state.get("next_node") in (END, "END")


# ──────────────────────────────────────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def snapshot():
    """Preflight once per module; yield the Salesforce contact snapshot."""
    try:
        snap = await run_preflight(warm=True)
    except PreflightError as exc:
        pytest.fail(f"Preflight failed:\n{exc}", pytrace=False)
    yield snap
    await restore_contacts(snap)


# ──────────────────────────────────────────────────────────────────────────────
# Test A — intent change during follow-up forces re-verification
# ──────────────────────────────────────────────────────────────────────────────


async def test_intent_change_in_followup_reverifies(snapshot):
    """A new intake intent raised in follow-up resets the call and re-verifies,
    then lands in the new intent's domain agent (claim_adjustment)."""
    graph, config = _new_graph()

    await _start(graph, config)

    # 1. Drive Emily fully through the provider flow into the follow-up phase.
    state = await _send_all(graph, config, _EMILY_PROVIDER_TO_FOLLOW_UP)
    assert state.get("member_status_verify") is True, "Emily should be verified after the provider flow"
    assert state.get("call_intent") == "provider_services"
    assert (state.get("first_name") or "").lower() == "emily"

    # 2. Pivot to a brand-new claim request during follow-up.
    state = await _send(graph, config, "Actually, can you check a claim reprocessing for me?")

    # ── Mid-call assertions: the call was reset and re-routed to verification ──
    assert not state.get("member_status_verify"), "verification flag must be cleared on the pivot"
    assert state.get("first_name") is None, "identity must be cleared on the pivot"
    assert state.get("pending_intent") == "claim_services", (
        f"pending_intent should stage the new intent, got {state.get('pending_intent')!r}"
    )
    # Routed back into the verification node (asking for the first name again).
    assert state.get("is_interrupt") is True
    assert state.get("next_node") == "verification_agent", (
        f"next node should be the verification node, got {state.get('next_node')!r}"
    )
    ai = extract_last_ai_message(state.get("messages", []))
    assert "first name" in ai.lower(), f"agent should re-ask for the first name, got: {ai!r}"

    # The pivot turn must deliver the deterministic first-name bridge intake uses
    # (every bridge message ends with INTENT_BRIDGE_MSG), and the one-shot flag
    # must be cleared so the following turn extracts normally.
    from agent.agents.intake.constants import INTENT_BRIDGE_MSG

    assert INTENT_BRIDGE_MSG in ai, f"pivot turn should emit the first-name bridge, got: {ai!r}"
    assert not state.get("reverify_bridge_pending"), (
        "bridge one-shot flag must be cleared after the pivot turn"
    )
    assert state.get("awaiting_slot") == "first_name", (
        f"bridge should set awaiting_slot=first_name, got {state.get('awaiting_slot')!r}"
    )

    # 3. Complete re-verification (claims slot order ends with phone confirmation).
    state = await _send_all(graph, config, _EMILY_CLAIM_REVERIFY)

    # ── Landed in claim_adjustment under the new intent; pending consumed ──
    assert state.get("member_status_verify") is True, "member should be re-verified"
    assert state.get("call_intent") == "claim_services"
    assert not state.get("pending_intent"), "pending_intent must be consumed after dispatch"
    awaiting = state.get("awaiting_slot")
    ai = extract_last_ai_message(state.get("messages", []))
    assert awaiting == "reference_number" or "reference number" in ai.lower(), (
        f"claim_adjustment should now be asking for the reference number "
        f"(awaiting_slot={awaiting!r}, ai={ai!r})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test B — same-intent action inside the claim flow does NOT reset
# ──────────────────────────────────────────────────────────────────────────────


async def test_same_intent_followup_does_not_reset(snapshot):
    """Supplying a reference number inside the claim flow is a same-intent action;
    it must not reset identity or verification."""
    graph, config = _new_graph()

    await _start(graph, config)

    # Verify James and reach the reference-number ask.
    state = await _send_all(graph, config, _JAMES_CLAIM_VERIFY)
    assert state.get("member_status_verify") is True
    assert (state.get("first_name") or "").lower() == "james"

    # Supply the (valid) reference number — a same-intent step, not a pivot.
    state = await _send(graph, config, "42695817")

    # Identity + verification must be unchanged; no reset, no pending_intent.
    assert state.get("member_status_verify") is True, "verification must remain True"
    assert (state.get("first_name") or "").lower() == "james", "identity must be unchanged"
    assert state.get("call_intent") == "claim_services", "intent must be unchanged"
    assert not state.get("pending_intent"), "no intent switch → pending_intent must stay empty"


# ──────────────────────────────────────────────────────────────────────────────
# Test C — end-call follow-up does NOT reset and does NOT re-verify
# ──────────────────────────────────────────────────────────────────────────────


async def test_end_call_followup_does_not_reset(snapshot):
    """A closing follow-up answer ("no thanks") ends the call without resetting
    identity and without routing back through verification."""
    graph, config = _new_graph()

    await _start(graph, config)

    # Drive Emily into the follow-up phase, then close the call.
    await _send_all(graph, config, _EMILY_PROVIDER_TO_FOLLOW_UP)
    state = await _send(graph, config, "no thanks")

    # The call closed cleanly; identity preserved; never re-verified.
    assert state.get("member_status_verify") is True, "verification flag must not be cleared on close"
    assert (state.get("first_name") or "").lower() == "emily", "identity must be preserved on close"
    assert not state.get("pending_intent"), "no intent switch → pending_intent must stay empty"
    assert state.get("next_node") != "verification_agent", "must not route back to verification"
    assert _is_terminal(state) or state.get("active_agent") in ("closure_agent", "follow_up_agent"), (
        f"expected the call to be closing, got next_node={state.get('next_node')!r}, "
        f"active_agent={state.get('active_agent')!r}"
    )
