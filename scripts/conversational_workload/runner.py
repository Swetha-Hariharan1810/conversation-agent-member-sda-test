"""
runner.py — Conversational evaluation runner for the PCP flow.

EVAL ARCHITECTURE (must stay true):
  ground_truth  = static transcript line for that AI message
                  (authoritative — what a correct user says here)
  user_response = LLM-generated response from the simulator
                  (independent — reacts to the live AI message)
  judge         = LLM that scores how well user_response satisfies ground_truth

These three must remain completely independent.  The simulator must never
read the transcript; the ground truth must never read the simulator output.

SCORING:
  Fast-path (no LLM judge call):
    - Normalised exact match → 1.0
    - One string contains the other → 0.9
  LLM judge (fired when fast-path misses):
    - Four-dimension rubric: intent, constraint, completeness, naturalness
    - Fires whenever the agent deviates from the expected script
    - A high LLM-judge call rate in pcp_happy_path indicates agent drift

TURN COUNTER TIMING:
  turn_counters tracks how many times (scenario, slot) has been visited.
  It is incremented AFTER scoring + graph advance so that the NEXT
  visit to the same slot sees the updated count.  Both simulator and
  ground_truth_builder read the same count before the increment.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, List, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

from scripts.conversational_workload.ground_truth_builder import (
    build_dynamic_ground_truth,
)
from scripts.conversational_workload.intent_classifier import classify_ai_slot
from scripts.conversational_workload.judge import judge_turn
from scripts.conversational_workload.models import (
    ClaimAdjustmentEntity,
    ConversationReport,
    PCPInquiryEntity,
    TurnEvaluation,
)
from scripts.conversational_workload.report_builder import save_report
from scripts.conversational_workload.user_simulator import (
    simulate_user_response_async,
)
from scripts.conversational_workload.utils import (
    extract_last_ai_message,
    new_conversation_id,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 30


def _normalise(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _fast_score(user_text: str, ground_truth: str):
    """
    Return (score, method) without LLM if possible, else (None, None).
    """
    if not ground_truth or not ground_truth.strip():
        return 1.0, "no_ground_truth"

    u = _normalise(user_text)
    g = _normalise(ground_truth)

    if u == g:
        return 1.0, "exact_match"

    # Substring match handles minor phrasing differences like
    # "emily" vs "my name is emily"
    if len(u) > 2 and len(g) > 2 and (u in g or g in u):
        return 0.9, "substring_match"

    return None, None


async def run_evaluation_async(
    entity_data: dict,
    flow: str = "pcp",
    scenario_tag: str = "pcp_happy_path",
) -> ConversationReport:
    entity = ClaimAdjustmentEntity(**entity_data) if flow == "claim" else PCPInquiryEntity(**entity_data)

    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["evaluation", flow, scenario_tag],
        "metadata": {"flow": flow, "scenario_tag": scenario_tag},
    }

    state = await graph.ainvoke({}, config=config)

    conversation_id = new_conversation_id()
    turns: List[TurnEvaluation] = []
    turn_counters: Dict[Tuple[str, str], int] = {}

    for _ in range(MAX_TURNS):
        if state.get("next_node") == END:
            break

        if state.get("is_interrupt"):
            ai_msg = extract_last_ai_message(state.get("messages", []))
            if not ai_msg:
                state = await graph.ainvoke(Command(resume=""), config=config)
                continue

            slot = classify_ai_slot(ai_msg, flow)

            # ── Ground truth: from static transcript (authoritative) ──────────
            # Uses the CURRENT turn_counters (before increment).
            ground_truth = build_dynamic_ground_truth(
                ai_msg,
                entity,
                flow,
                scenario_tag=scenario_tag,
                turn_counters=turn_counters,
            )

            # ── Simulator: LLM generates response independently ───────────────
            # Also uses CURRENT turn_counters.
            # MUST NOT read ground_truth or the transcript.
            user_text = await simulate_user_response_async(
                ai_msg,
                entity,
                flow,
                scenario_tag=scenario_tag,
                turn_counters=turn_counters,
            )

            # ── Scoring ───────────────────────────────────────────────────────
            score, method = _fast_score(user_text, ground_truth)
            if score is not None:
                scores = {
                    "intent_score": score,
                    "constraint_score": score,
                    "completeness_score": score,
                    "naturalness_score": score,
                    "overall": score,
                    "verdict": "PASS" if score >= 0.8 else "FAIL",
                    "judge_method": method,
                }
            else:
                # Genuine mismatch — call LLM judge
                judge_result = judge_turn(user_text, ground_truth)
                scores = {
                    **judge_result.model_dump(),
                    "judge_method": "llm",
                }

            turns.append(
                TurnEvaluation(
                    ai_prompt=ai_msg,
                    user_response=user_text,
                    ground_truth=ground_truth,
                    slot=slot,
                    scenario=scenario_tag,
                    scores=scores,
                )
            )

            # ── Advance the graph with the simulator's response ───────────────
            state = await graph.ainvoke(Command(resume=user_text), config=config)

            # ── Increment AFTER scoring and graph advance ─────────────────────
            # The NEXT time this (scenario, slot) appears, the count
            # correctly reflects this completed visit.
            key = (scenario_tag, slot)
            turn_counters[key] = turn_counters.get(key, 0) + 1

        else:
            state = await graph.ainvoke(Command(resume=""), config=config)

    final_score = round(sum(t.scores.get("overall", 0.0) for t in turns) / len(turns), 2) if turns else 0.0

    report = ConversationReport(
        conversation_id=conversation_id,
        flow=flow,
        scenario_tag=scenario_tag,
        completed=state.get("next_node") == END,
        turns=turns,
        final_score=final_score,
    )

    save_report(report)
    return report


def run_evaluation(
    entity_data: dict,
    flow: str = "pcp",
    scenario_tag: str = "pcp_happy_path",
) -> ConversationReport:
    return asyncio.run(run_evaluation_async(entity_data, flow, scenario_tag))
