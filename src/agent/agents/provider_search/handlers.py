"""
handlers.py — Provider search workflow handlers.
"""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_ZIP_UPDATE_FAIL = [
    "I'm sorry, I was unable to update your ZIP code. Let me connect you with a representative.",
    "I wasn't able to save your ZIP code update. Connecting you with a specialist.",
]


async def update_zip_in_salesforce(agent, state: State, new_zip: str) -> dict | None:
    """
    Write the updated ZIP to Salesforce.
    Returns an escalation interrupt dict if the update fails, else None.
    """
    # Lazy import keeps handlers.py free of agent → storage → agent circular risk
    from agent.storage.tools import update_zip_code

    member_id = state.get("member_id", "")
    if not member_id:
        logger.warning("update_zip_in_salesforce: no member_id in state — skipping write")
        return None

    try:
        success = await update_zip_code.ainvoke({"member_id": member_id, "zip_code": new_zip})
        if not success:
            return agent.signal_escalate(state, pick(_MSG_ZIP_UPDATE_FAIL), reason="zip_update_failed")
        return None
    except Exception:
        logger.exception("update_zip_in_salesforce: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_ZIP_UPDATE_FAIL), reason="zip_update_error")


async def confirm_zip_from_state(state: State) -> str:
    """Return the ZIP code from state (already populated by verification_agent)."""
    return (state.get("zip_code") or "").strip()
