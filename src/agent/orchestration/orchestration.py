"""
orchestration.py — LLM-based routing (core agents only).
Agents in scope: intake · verification · closure · escalation
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from agent.llm.config import get_routing_llm
from agent.logger import get_logger
from agent.orchestration.fast_path import get_fast_path_route
from agent.orchestration.safeguards import apply_safeguards
from agent.state import State, normalize_cross_agent_request
from agent.utils import _last_user_msg, build_system_prompt

logger = get_logger(__name__)


class AgentNode(str, Enum):
    INTAKE = "intake_agent"
    VERIFICATION = "verification_agent"
    ESCALATION = "escalation_agent"
    CLOSURE = "closure_agent"
    PROVIDER_SEARCH = "provider_search_agent"
    DELIVERY_MANAGEMENT = "delivery_management_agent"
    BENEFITS = "benefits_agent"
    CARE_WELLNESS = "care_wellness_agent"
    FOLLOW_UP = "follow_up_agent"
    CLAIM_ADJUSTMENT = "claim_adjustment_agent"
    RECORDS_COORDINATION = "records_coordination_agent"
    NOTIFICATION_SETUP = "notification_setup_agent"


ALL_AGENT_NODES: list[str] = [n.value for n in AgentNode]


class OrchestratorDecision(BaseModel):
    next_agent: str
    reasoning: str
    message_override: Optional[str] = None


def _build_routing_input(state: State, utterance: str) -> dict:
    last_signal_obj = state.get("last_agent_signal") or {}
    flags: dict = {
        "intent_queue": state.get("intent_queue") or [],
        "closure_requested": last_signal_obj.get("closure_requested", False),
        "router_loop": state.get("router_loop_count", 0),
    }
    if state.get("escalation_reason"):
        flags["escalation_reason"] = state["escalation_reason"]
    if state.get("new_intent_detected"):
        flags["new_intent_detected"] = state["new_intent_detected"]
    return {
        "active_agent": state.get("active_agent") or "none",
        "call_intent": state.get("call_intent") or "unknown",
        "member_verified": state.get("member_status_verify", False),
        "last_signal": last_signal_obj.get("status", "none"),
        "previous_agents": state.get("previous_agents") or [],
        "flags": flags,
        "utterance": utterance,
    }


class Orchestrator:
    def __init__(self) -> None:
        self.llm = get_routing_llm()

    async def run(self, state: State) -> dict:
        fast_route = get_fast_path_route(state)
        last_signal = state.get("last_agent_signal") or {}
        # Routing fix: follow_up_agent fast-path — debug logging
        logger.info(
            "orchestrator routing decision",
            extra={
                "active_agent": state.get("active_agent"),
                "closure_requested": last_signal.get("closure_requested"),
                "fast_route": fast_route,
                "call_intent": state.get("call_intent"),
            },
        )
        if fast_route:
            logger.info("orchestrator: fast-path", extra={"route": fast_route})
            updates = {
                "next_node": fast_route,
                "is_interrupt": False,
                "orchestrator_reasoning": f"fast-path → {fast_route}",
            }
            # Routed cross-agent request return hop (Phase 4 updates, Phase 6
            # redo/replay): consume the pending marker, restore the
            # requester's awaiting slot, and arm the one-shot resume flag the
            # requester acknowledges the completed request with. The resume
            # flag is only armed when a slot is actually being restored —
            # requests raised from non-collecting agents (return_awaiting "")
            # must not leave a stale resume marker behind.
            pending_request = normalize_cross_agent_request(state)
            if pending_request.get("return_to_agent") == fast_route:
                updates["pending_cross_agent_request"] = {}
                updates["pending_slot_update"] = {}  # legacy key
                updates["slot_update_resume"] = bool(pending_request.get("return_awaiting"))
                updates["awaiting_slot"] = pending_request.get("return_awaiting", "")
            return updates

        utterance = _last_user_msg(list(state.get("messages") or []))

        messages = [
            {"role": "system", "content": build_system_prompt("generation/orchestrator.md")},
            {"role": "user", "content": json.dumps(_build_routing_input(state, utterance))},
        ]

        decision = await self.llm.with_structured_output(OrchestratorDecision).ainvoke(messages)

        decision = apply_safeguards(decision=decision, state=state)

        logger.info(
            "orchestrator: LLM decision",
            extra={"next_agent": decision.next_agent, "reasoning": decision.reasoning},
        )

        updates = {
            "next_node": decision.next_agent,
            "is_interrupt": False,
            "orchestrator_reasoning": decision.reasoning,
        }

        if decision.message_override and decision.message_override.strip():
            updates["messages"] = {
                "role": "assistant",
                "content": decision.message_override.strip(),
            }

        return updates


_orchestrator: Optional[Orchestrator] = None


async def orchestrator(state: State) -> dict:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    loop_count = state.get("router_loop_count", 0) + 1
    state_with_count: dict = {**state, "router_loop_count": loop_count}
    updates = await _orchestrator.run(state_with_count)
    updates["router_loop_count"] = loop_count
    return updates
