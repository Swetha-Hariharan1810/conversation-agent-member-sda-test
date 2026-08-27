"""
handlers.py — Delivery management workflow handlers.
"""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_CONTACT_UPDATE_FAIL = [
    "I'm sorry, I was unable to update your contact information. Let me connect you with a representative.",
    "I wasn't able to save your contact update. Connecting you with a specialist.",
]

_MSG_DISPATCH_FAIL = [
    "I'm sorry, I was unable to send the provider list. Let me connect you with a representative.",
    "I wasn't able to dispatch the list. Connecting you with a specialist.",
]


async def update_fax_in_salesforce(agent, state: State, new_fax: str) -> dict | None:
    """Update fax in Salesforce. Returns escalation interrupt on failure, else None."""
    from agent.storage.tools import update_member_contact

    member_id = state.get("member_id", "")
    if not member_id:
        logger.warning("update_fax_in_salesforce: no member_id in state — skipping write")
        return None

    try:
        success = await update_member_contact.ainvoke({"member_id": member_id, "fax": new_fax})
        if not success:
            return agent.signal_escalate(state, pick(_MSG_CONTACT_UPDATE_FAIL), reason="fax_update_failed")
        return None
    except Exception:
        logger.exception("update_fax_in_salesforce: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_CONTACT_UPDATE_FAIL), reason="fax_update_error")


async def update_email_in_salesforce(agent, state: State, new_email: str) -> dict | None:
    """Update email in Salesforce. Returns escalation interrupt on failure, else None."""
    from agent.storage.tools import update_member_contact

    member_id = state.get("member_id", "")
    if not member_id:
        logger.warning("update_email_in_salesforce: no member_id in state — skipping write")
        return None

    try:
        success = await update_member_contact.ainvoke({"member_id": member_id, "email": new_email})
        if not success:
            return agent.signal_escalate(state, pick(_MSG_CONTACT_UPDATE_FAIL), reason="email_update_failed")
        return None
    except Exception:
        logger.exception("update_email_in_salesforce: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_CONTACT_UPDATE_FAIL), reason="email_update_error")


async def dispatch_provider_list(agent, state: State, method: str, destination: str) -> dict | None:
    """
    Call dispatch_provider_list storage tool.
    Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import dispatch_provider_list as _dispatch_tool

    member_id = state.get("member_id", "")
    provider_type = state.get("provider_type", "")
    zip_code = (state.get("zip_code_used") or state.get("zip_code") or "").strip()

    try:
        success = await _dispatch_tool.ainvoke(
            {
                "member_id": member_id,
                "provider_type": provider_type,
                "zip_code": zip_code,
                "delivery_method": method,
                "delivery_address": destination,
            }
        )
        if not success:
            return agent.signal_escalate(state, pick(_MSG_DISPATCH_FAIL), reason="dispatch_failed")
        return None
    except Exception:
        logger.exception("dispatch_provider_list: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_DISPATCH_FAIL), reason="dispatch_error")
