"""
llm.py — LLM extraction for DeliveryManagementAgent.

Public API:
    extract_delivery_management_decision(llm, system_prompt, *, awaiting_slot,
                                         last_agent_message, last_user_message,
                                         confirmed_slots, attempt, recent_messages)
        → WorkerResult

Extracted slot values live in result.extracted:
  delivery_method, fax, email, contact_confirmed, benefits_response
"""

from __future__ import annotations

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger

logger = get_logger(__name__)


async def extract_delivery_management_decision(
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
    Run one LLM call to extract delivery management slots from the latest user utterance.

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

    # ADD HERE — after build_worker_input, before the try block
    # import json
    # logger.debug(
    #     "delivery_management LLM raw payload: %s",
    #     json.dumps(messages, ensure_ascii=False)
    # )

    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        # ADD THIS
        # logger.debug(
        #     "delivery_management LLM result: extracted=%r event_type=%r guard=%r",
        #     result.extracted,
        #     result.event_type,
        #     result.guard,
        # )

        return result
    except Exception:
        logger.exception("extract_delivery_management_decision: LLM extraction failed")
        return WorkerResult()
