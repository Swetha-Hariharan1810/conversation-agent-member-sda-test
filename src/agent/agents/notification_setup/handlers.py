"""handlers.py — Notification Setup workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_SAVE_FAIL = [
    "I'm sorry, I wasn't able to save your notification preference. "
    "Let me connect you with a representative who can help.",
    "I wasn't able to update your notification settings. Connecting you with a specialist.",
]


async def save_timeline_notification_preference(
    agent, state: State, method: str, destination: str
) -> dict | None:
    """
    Write claim timeline/progress notification preference to Salesforce.
    Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import set_claim_timeline_notification

    member_id = state.get("member_id", "")
    reference_number = state.get("reference_number", "")

    if not member_id or not reference_number:
        logger.warning(
            "save_timeline_notification_preference: missing member_id or reference_number — skipping write"
        )
        return None

    try:
        success = await set_claim_timeline_notification.ainvoke(
            {
                "member_id": member_id,
                "reference_number": reference_number,
                "method": method,
                "destination": destination,
            }
        )
        if not success:
            return agent.signal_escalate(
                state, pick(_MSG_SAVE_FAIL), reason="timeline_notification_preference_save_failed"
            )
        return None
    except Exception:
        logger.exception("save_timeline_notification_preference: tool call failed")
        return agent.signal_escalate(
            state, pick(_MSG_SAVE_FAIL), reason="timeline_notification_preference_save_error"
        )


async def save_notification_preference(agent, state: State, method: str, destination: str) -> dict | None:
    """
    Write notification preference to Salesforce.
    Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import set_claim_notification

    member_id = state.get("member_id", "")
    reference_number = state.get("reference_number", "")

    if not member_id or not reference_number:
        logger.warning("save_notification_preference: missing member_id or reference_number — skipping write")
        return None

    try:
        success = await set_claim_notification.ainvoke(
            {
                "member_id": member_id,
                "reference_number": reference_number,
                "method": method,
                "destination": destination,
            }
        )
        if not success:
            return agent.signal_escalate(
                state, pick(_MSG_SAVE_FAIL), reason="notification_preference_save_failed"
            )
        return None
    except Exception:
        logger.exception("save_notification_preference: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_SAVE_FAIL), reason="notification_preference_save_error")
