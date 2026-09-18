"""
fast_path.py — Deterministic routing (core agents only).

Phase 6 note: this module only picks the next agent NODE from the last agent
signal — it never classifies or answers a member turn directly. The routed
agent's extraction + _collect_slot own every turn, so the WAIT and follow-up
disposition branches (core/slot_manager.py) are never bypassed here. If a
future fast path ever answers a turn without running extraction, it must run
detect_wait_request / detect_cannot_provide (agent.utils) first.
"""

from __future__ import annotations

from agent.core.signal import AgentSignal, AgentStatus
from agent.state import State, normalize_cross_agent_request


def get_fast_path_route(state: State) -> str | None:  # noqa: C901
    signal = AgentSignal.from_state_dict(state.get("last_agent_signal", {}))
    active_agent = state.get("active_agent")
    member_verified = state.get("member_status_verify", False)

    # Hard escalation
    if signal.status in {AgentStatus.ESCALATE, AgentStatus.BLOCKED}:
        return "escalation_agent"

    # Verification gate
    if not member_verified and active_agent != "verification_agent":
        return "verification_agent"

    # Routed cross-agent request finished → return to the requesting agent
    # (Phase 4 slot updates; Phase 6 redo/replay). The owner (e.g.
    # provider_search after a ZIP update, delivery_management after a
    # re-dispatch) signals COMPLETE with pending_cross_agent_request still
    # set; hand control back to return_to_agent. The orchestrator clears the
    # request and arms slot_update_resume when it takes this route.
    pending_request = normalize_cross_agent_request(state)
    if (
        member_verified
        and signal.status == AgentStatus.COMPLETE
        and pending_request.get("return_to_agent")
        and active_agent != pending_request.get("return_to_agent")
    ):
        return pending_request["return_to_agent"]

    # delivery_management complete → benefits (always)
    if (
        member_verified
        and active_agent == "delivery_management_agent"
        and signal.status == AgentStatus.COMPLETE
    ):
        return "benefits_agent"

    # benefits complete → care_wellness (always)
    if member_verified and active_agent == "benefits_agent" and signal.status == AgentStatus.COMPLETE:
        return "care_wellness_agent"

    # care_wellness complete → follow_up (always)
    if member_verified and active_agent == "care_wellness_agent" and signal.status == AgentStatus.COMPLETE:
        return "follow_up_agent"

    # Follow-up agent complete + closure requested → closure_agent
    if (
        member_verified
        and active_agent == "follow_up_agent"
        and signal.status == AgentStatus.COMPLETE
        and signal.closure_requested
    ):
        return "closure_agent"

    # Routing fix: follow_up_agent fast-path
    # follow_up_agent complete without closure → stay in follow_up_agent.
    # This should not normally occur (ask_member bypasses orchestrator), but
    # if signal_complete(closure_requested=False) fires for any reason,
    # routing to the LLM orchestrator risks an intake-style re-routing decision.
    if (
        member_verified
        and active_agent == "follow_up_agent"
        and signal.status == AgentStatus.COMPLETE
        and not signal.closure_requested
        and not signal.new_intent_detected
    ):
        return "follow_up_agent"

    # New intent detected by follow_up_agent — bypass LLM orchestrator entirely.
    # Read from the signal, not top-level state: _handle_new_intent resets the
    # state field via NEW_INTENT_CLEAR_FIELDS but carries the detected intent on
    # the signal. The signal is naturally consumed when the next node overwrites
    # last_agent_signal, so the domain agent does not re-trigger this shortcut on
    # its own COMPLETE (no re-route loop).
    if member_verified and signal.new_intent_detected and signal.status == AgentStatus.COMPLETE:
        intent = signal.new_intent_detected
        if intent == "provider_services":
            return "provider_search_agent"
        if intent == "claim_services":
            return "claim_adjustment_agent"
        # Unknown intent — let LLM orchestrator handle
        return None

    # General closure signal — never re-route if already inside closure_agent
    if signal.status == AgentStatus.COMPLETE and signal.closure_requested:
        if active_agent != "closure_agent":
            return "closure_agent"
        return None  # closure_agent set next_node=END; let conditional_routing handle it

    # After verification — route to the correct domain agent
    if member_verified and active_agent == "verification_agent":
        intent = state.get("call_intent", "")
        if intent == "provider_services":
            return "provider_search_agent"
        if intent == "claim_services":
            return "claim_adjustment_agent"
        return "closure_agent"

    # claim_adjustment complete → records_coordination (if records needed and not yet branched)
    if member_verified and active_agent == "claim_adjustment_agent" and signal.status == AgentStatus.COMPLETE:
        if state.get("records_required") and not state.get("records_branch_taken"):
            return "records_coordination_agent"
        if not state.get("notification_channel") or state.get("notification_channel") == "not_set":
            return "notification_setup_agent"
        return "follow_up_agent"

    # records_coordination complete → notification_setup
    if (
        member_verified
        and active_agent == "records_coordination_agent"
        and signal.status == AgentStatus.COMPLETE
    ):
        return "notification_setup_agent"

    # notification_setup complete → follow_up
    if (
        member_verified
        and active_agent == "notification_setup_agent"
        and signal.status == AgentStatus.COMPLETE
    ):
        return "follow_up_agent"

    return None
