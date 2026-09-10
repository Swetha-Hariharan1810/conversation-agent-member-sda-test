"""handlers.py — Records Coordination workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_SEND_LINK_FAIL = [
    "I'm sorry, I wasn't able to generate the upload link. "
    "Let me connect you with a representative who can help.",
    "I wasn't able to send the upload link. Connecting you with a specialist.",
]

_MSG_GUIDE_FAIL = [
    "I'm sorry, I wasn't able to schedule the Personal Guide outreach. "
    "Let me connect you with a representative who can help.",
    "I wasn't able to trigger the Personal Guide workflow. Connecting you with a specialist.",
]


async def dispatch_upload_link(agent, state: State, email: str) -> dict | None:
    """
    Generate and send the secure medical records upload link.
    Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import send_claim_upload_link

    member_id = state.get("member_id", "")
    reference_number = state.get("reference_number", "")

    try:
        success = await send_claim_upload_link.ainvoke(
            {
                "member_id": member_id,
                "reference_number": reference_number,
                "email": email,
            }
        )
        if not success:
            return agent.signal_escalate(
                state, pick(_MSG_SEND_LINK_FAIL), reason="upload_link_dispatch_failed"
            )
        return None
    except Exception:
        logger.exception("dispatch_upload_link: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_SEND_LINK_FAIL), reason="upload_link_dispatch_error")


async def dispatch_personal_guide(agent, state: State) -> dict | None:
    """
    Trigger Personal Guide workflow to contact the provider for records.
    Returns escalation interrupt on failure, else None.
    """
    from agent.storage.tools import trigger_claim_personal_guide

    member_id = state.get("member_id", "")
    reference_number = state.get("reference_number", "")

    try:
        success = await trigger_claim_personal_guide.ainvoke(
            {
                "member_id": member_id,
                "reference_number": reference_number,
            }
        )
        if not success:
            return agent.signal_escalate(
                state, pick(_MSG_GUIDE_FAIL), reason="personal_guide_dispatch_failed"
            )
        return None
    except Exception:
        logger.exception("dispatch_personal_guide: tool call failed")
        return agent.signal_escalate(state, pick(_MSG_GUIDE_FAIL), reason="personal_guide_dispatch_error")
