"""llm.py — LLM extraction for BenefitsAgent.

Extracted slot values live in result.extracted:
  care_coach_response  — "yes" | "no"
"""

from __future__ import annotations

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger

logger = get_logger(__name__)


async def extract_benefits_decision(
    llm,
    system_prompt: str,
    *,
    awaiting_slot: str,
    last_agent_message: str,
    last_user_message: str,
    confirmed_slots: dict | None = None,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract the member's response to the Care Coach offer.
    Falls back to an empty WorkerResult on any exception.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot=awaiting_slot,
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=confirmed_slots,
        pending_slots=pending_slots,
        attempt=attempt,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        return result
    except Exception:
        logger.exception("extract_benefits_decision: LLM extraction failed")
        return WorkerResult()
