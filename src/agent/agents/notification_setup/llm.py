"""llm.py — LLM extraction for NotificationSetupAgent.

Extracted slot values:
  notification_method  — "sms" | "email"
  contact_confirmed    — "yes" | "no"
  phone                — 10-digit string (updated phone if on-file declined)
  email                — valid email string (updated email if on-file declined)
"""

from __future__ import annotations

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger

logger = get_logger(__name__)


async def extract_notification_decision(
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
    """Extract notification setup slots. Falls back to empty WorkerResult on exception."""
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
        logger.exception("extract_notification_decision: LLM extraction failed")
        return WorkerResult()
