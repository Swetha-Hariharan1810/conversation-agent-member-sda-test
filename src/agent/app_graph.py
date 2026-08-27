"""
app_graph.py — Trimmed LangGraph workflow (core agents only).
Agents: intake · verification · closure · escalation · orchestrator
"""

import asyncio as _asyncio

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.agents.benefits.agent import benefits_agent
from agent.agents.care_wellness.agent import care_wellness_agent
from agent.agents.claim_adjustment.agent import claim_adjustment_agent
from agent.agents.closure.agent import closure_agent
from agent.agents.delivery_management.agent import delivery_management_agent
from agent.agents.escalation.agent import escalation_agent
from agent.agents.follow_up.agent import follow_up_agent
from agent.agents.intake.agent import intake_agent
from agent.agents.notification_setup.agent import notification_setup_agent
from agent.agents.provider_search.agent import provider_search_agent
from agent.agents.records_coordination.agent import records_coordination_agent
from agent.agents.verification.agent import verification_agent
from agent.logger import get_logger
from agent.orchestration.orchestration import (
    ALL_AGENT_NODES,
    AgentNode,
    orchestrator,
)
from agent.state import State
from agent.utils import clean_asr_input

logger = get_logger(__name__)


# ── Intake routing — deterministic, bypasses Orchestrator ────────────────────
def intake_routing(state: State) -> str:
    if state.get("is_interrupt"):
        return "human_node"
    if state.get("next_node") in ("END", END):
        return END
    if state.get("next_node") == AgentNode.ESCALATION.value:
        return AgentNode.ESCALATION.value
    # Same-member confirmed path: intake signals complete → orchestrator so the
    # fast-path can route to the correct domain agent with member_verified=True.
    if state.get("next_node") == "orchestrator":
        return "orchestrator"
    return AgentNode.VERIFICATION.value


# ── Shared conditional router ────────────────────────────────────────────────
def conditional_routing(state: State) -> str:
    next_node = state.get("next_node", "")
    is_interrupt = state.get("is_interrupt", False)
    logger.info("conditional_routing", extra={"next_node": next_node, "is_interrupt": is_interrupt})
    if is_interrupt:
        return "human_node"
    if state.get("next_node") in ("END", END):
        return END
    known = ALL_AGENT_NODES + ["orchestrator", "human_node"]
    if next_node in known:
        return next_node
    logger.warning("conditional_routing: unknown node — fallback to intake", extra={"next_node": next_node})
    return AgentNode.INTAKE.value


# ── Verification routing — dispatches mid-call intent switches ───────────────
def verification_routing(state: State) -> str:
    """Conditional edge out of the verification node.

    On a mid-call intent switch, verification's _signal_verified sets next_node
    directly to the new intent's domain node (provider_search_agent /
    claim_adjustment_agent) and clears pending_intent, so this edge forwards that
    next_node. First-ever verification keeps next_node="orchestrator" (the
    fast-path then routes by call_intent). Interrupts pause at human_node;
    escalation and END are handled as usual.
    """
    next_node = state.get("next_node", "")
    is_interrupt = state.get("is_interrupt", False)
    logger.info("verification_routing", extra={"next_node": next_node, "is_interrupt": is_interrupt})
    if is_interrupt:
        return "human_node"
    if next_node in ("END", END):
        return END
    known = ALL_AGENT_NODES + ["orchestrator", "human_node"]
    if next_node in known:
        return next_node
    logger.warning("verification_routing: unknown node — fallback to intake", extra={"next_node": next_node})
    return AgentNode.INTAKE.value


# ── Orchestrator routing ──────────────────────────────────────────────────────
def orchestrator_routing(state: State) -> str:
    next_node = state.get("next_node", "")
    if state.get("is_interrupt"):
        return "human_node"
    if state.get("next_node") in ("END", END):
        return END
    if next_node in ALL_AGENT_NODES:
        return next_node
    logger.warning("orchestrator_routing: unknown node — fallback", extra={"next_node": next_node})
    return AgentNode.INTAKE.value


# ── Human interrupt node ──────────────────────────────────────────────────────
def human_node(state: State) -> Command:
    interrupt_message = ""
    try:
        messages = state.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            interrupt_message = (
                last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
            )
        elif isinstance(messages, dict):
            interrupt_message = messages.get("content", "")
        else:
            interrupt_message = getattr(messages, "content", "")
    except Exception:
        logger.exception("human_node: failed reading interrupt message")
    value = interrupt(interrupt_message)
    value = clean_asr_input(str(value) if value is not None else "")
    next_node = state.get("next_node", AgentNode.INTAKE.value)
    logger.info(f"human_node: collected → {next_node}", extra={"len": len(value)})
    if next_node in ("END", END):
        next_node = END
    logger.info(f"human_node: collected → {next_node}", extra={"len": len(value)})
    return Command(
        goto=next_node,
        update={"is_interrupt": False, "messages": [{"role": "user", "content": value}]},
    )


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("human_node", human_node)
    g.add_node(AgentNode.INTAKE.value, intake_agent)
    g.add_node(AgentNode.VERIFICATION.value, verification_agent)
    g.add_node(AgentNode.ESCALATION.value, escalation_agent)
    g.add_node(AgentNode.CLOSURE.value, closure_agent)
    g.add_node(AgentNode.PROVIDER_SEARCH.value, provider_search_agent)
    g.add_node(AgentNode.DELIVERY_MANAGEMENT.value, delivery_management_agent)
    g.add_node(AgentNode.BENEFITS.value, benefits_agent)
    g.add_node(AgentNode.CARE_WELLNESS.value, care_wellness_agent)
    g.add_node(AgentNode.FOLLOW_UP.value, follow_up_agent)
    g.add_node(AgentNode.CLAIM_ADJUSTMENT.value, claim_adjustment_agent)
    g.add_node(AgentNode.RECORDS_COORDINATION.value, records_coordination_agent)
    g.add_node(AgentNode.NOTIFICATION_SETUP.value, notification_setup_agent)
    g.add_conditional_edges(AgentNode.INTAKE.value, intake_routing)
    # Verification has a dedicated router so a mid-call intent switch can be
    # dispatched straight to the new intent's domain node (see verification_routing
    # and VerificationAgent._signal_verified / pending_intent).
    g.add_conditional_edges(AgentNode.VERIFICATION.value, verification_routing)
    for node in [
        AgentNode.ESCALATION.value,
        AgentNode.CLOSURE.value,
        AgentNode.PROVIDER_SEARCH.value,
        AgentNode.DELIVERY_MANAGEMENT.value,
        AgentNode.BENEFITS.value,
        AgentNode.CARE_WELLNESS.value,
        AgentNode.FOLLOW_UP.value,
        AgentNode.CLAIM_ADJUSTMENT.value,
        AgentNode.RECORDS_COORDINATION.value,
        AgentNode.NOTIFICATION_SETUP.value,
    ]:
        g.add_conditional_edges(node, conditional_routing)
    g.add_conditional_edges("orchestrator", orchestrator_routing)
    g.set_entry_point(AgentNode.INTAKE.value)
    # langgraph dev provides its own checkpointer — only attach when explicitly given
    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


graph = build_graph()


async def warm_llm_connections() -> None:
    """
    Establish HTTP keep-alive to Azure OpenAI at process startup.
    Fires minimal concurrent calls on extraction, follow-up, and generation
    LLMs so the first real conversational turn does not pay TCP + TLS cost.
    Also pre-populates the lru_cache for build_extraction_prompt_core and
    read_prompt as a side-effect of building the warm-up messages.

    Call this once from your ASGI lifespan or startup handler:
        await warm_llm_connections()

    Do NOT call this from app_graph.py itself — it must be called from
    the application entry point that owns the event loop.

    Failures are silently swallowed — this is best-effort only.
    The system operates correctly if this function is never called.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm.config import get_extraction_llm, get_follow_up_llm, get_generation_llm
    from agent.llm.extractor import build_worker_input
    from agent.llm.schema import FollowUpResult, WorkerResult
    from agent.utils import build_extraction_prompt_core, build_generation_prompt

    _warmup_messages = build_worker_input(
        system_prompt=build_extraction_prompt_core("extraction/benefits.md"),
        awaiting_slot="care_coach_response",
        last_agent_message="Would you like Care Coach details?",
        last_user_message="yes",
        attempt=0,
    )

    async def _ping_extraction(llm):
        try:
            await llm.with_structured_output(WorkerResult).ainvoke(_warmup_messages)
        except Exception:
            pass

    async def _ping_follow_up(llm):
        try:
            await llm.with_structured_output(FollowUpResult).ainvoke(_warmup_messages)
        except Exception:
            pass

    async def _ping_generation(llm):
        try:
            system_prompt_text = build_generation_prompt()
            await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt_text),
                    HumanMessage(content="Collecting: first_name\nTone: first ask\nEvent: RETRY"),
                ]
            )
        except Exception:
            pass

    async def _ping_salesforce():
        """
        Fire a minimal SOQL query to establish TCP + TLS to Salesforce.
        Uses a lazy import to avoid circular import risk at module load time.
        Swallows all exceptions — this is best-effort only.
        """
        try:
            from agent.storage.db import _get_sf

            sf = _get_sf()
            await sf.query("SELECT Id FROM M_Member__c LIMIT 1")
            logger.info("warm_llm_connections: Salesforce connection warmed")
        except Exception:
            logger.warning("warm_llm_connections: Salesforce ping failed — will connect on first call")

    extraction_llm = get_extraction_llm()
    follow_up_llm = get_follow_up_llm()
    generation_llm = get_generation_llm()

    await _asyncio.gather(
        _ping_extraction(extraction_llm),
        _ping_follow_up(follow_up_llm),
        _ping_generation(generation_llm),
        _ping_salesforce(),
    )
