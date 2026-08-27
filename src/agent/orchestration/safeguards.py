"""
safeguards.py — Deterministic orchestration overrides (core agents only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.core.constants import MAX_ROUTER_LOOPS
from agent.core.signal import AgentSignal, AgentStatus
from agent.state import State

if TYPE_CHECKING:
    from agent.orchestration.orchestration import OrchestratorDecision

REGISTERED_AGENTS = {
    "verification_agent",
    "escalation_agent",
    "closure_agent",
    "provider_search_agent",
    "delivery_management_agent",
    "benefits_agent",
    "care_wellness_agent",
    "follow_up_agent",
    "claim_adjustment_agent",
    "records_coordination_agent",
    "notification_setup_agent",
}

FORBIDDEN_AGENTS = {"intake_agent"}


def apply_safeguards(
    decision: "OrchestratorDecision",
    state: State,
) -> "OrchestratorDecision":
    # 0. Router loop cap
    if state.get("router_loop_count", 0) >= MAX_ROUTER_LOOPS:
        decision.next_agent = "escalation_agent"
        decision.reasoning = f"Safeguard: router loop cap of {MAX_ROUTER_LOOPS} reached → escalation_agent"
        return decision

    signal = AgentSignal.from_state_dict(state.get("last_agent_signal", {}))

    # 1. Forbidden agents → verification
    if decision.next_agent in FORBIDDEN_AGENTS:
        decision.next_agent = "verification_agent"
        decision.reasoning = "Safeguard: forbidden route → verification_agent"

    # 2. Unknown agent → closure
    elif decision.next_agent not in REGISTERED_AGENTS:
        bad_agent = decision.next_agent
        decision.next_agent = "closure_agent"
        decision.reasoning = f"Safeguard: unknown agent '{bad_agent}' → closure_agent"

    # 3. Cannot close during active escalation
    if signal.status in {AgentStatus.ESCALATE, AgentStatus.BLOCKED}:
        if decision.next_agent != "escalation_agent":
            decision.next_agent = "escalation_agent"
            decision.reasoning = "Safeguard: escalation in progress → escalation_agent"

    # 4. Must verify before any domain agent
    if (
        not state.get("member_status_verify", False)
        and state.get("active_agent") != "verification_agent"
        and decision.next_agent not in {"verification_agent", "escalation_agent"}
    ):
        decision.next_agent = "verification_agent"
        decision.reasoning = "Safeguard: member not verified → verification_agent"

    # 5. Cannot close if there are unresolved intents
    if (
        decision.next_agent == "closure_agent"
        and state.get("intent_queue")
        and len(state["intent_queue"]) > 0
    ):
        decision.next_agent = "escalation_agent"
        decision.reasoning = (
            f"Safeguard: unresolved intent queue {state['intent_queue']} — "
            "no domain agent available → escalation_agent"
        )

    # 6. Strip oversized message_overrides
    if decision.message_override and len(decision.message_override) > 500:
        decision.message_override = None

    return decision
