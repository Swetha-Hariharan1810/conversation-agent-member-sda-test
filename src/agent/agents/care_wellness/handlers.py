"""handlers.py — Care & Wellness workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)


def _resolve_delivery_contact(state: State) -> tuple[str, str]:
    """
    Return (method, contact) from already-confirmed session context.
    Priority: fax first (used in provider search flow), then email.
    Returns ("", "") if nothing is on file.
    """
    fax = (state.get("fax") or "").strip()
    email = (state.get("email") or "").strip()
    delivery_method = (state.get("delivery_method") or "").strip().lower()

    if delivery_method == "fax" and fax:
        return "fax", fax
    if delivery_method == "email" and email:
        return "email", email
    # Fall back to whatever is available
    if fax:
        return "fax", fax
    if email:
        return "email", email
    return "", ""


async def dispatch_care_coach(
    agent,
    state: State,
    method: str,
    contact: str,
) -> dict | None:
    """
    Dispatch Care Coach details. Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import dispatch_care_coach_details as _tool

    member_id = state.get("member_id", "")
    if not member_id:
        logger.warning("dispatch_care_coach: no member_id in state — skipping write")
        return None

    try:
        success = await _tool.ainvoke(
            {
                "member_id": member_id,
                "delivery_method": method,
                "delivery_address": contact,
            }
        )
        if not success:
            from agent.agents.care_wellness.constants import MSG_DISPATCH_FAIL

            return agent.signal_escalate(state, pick(MSG_DISPATCH_FAIL), reason="care_coach_dispatch_failed")
        return None
    except Exception:
        logger.exception("dispatch_care_coach: tool call failed")
        from agent.agents.care_wellness.constants import MSG_DISPATCH_FAIL

        return agent.signal_escalate(state, pick(MSG_DISPATCH_FAIL), reason="care_coach_dispatch_error")
