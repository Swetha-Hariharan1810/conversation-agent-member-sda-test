"""handlers.py — Records Coordination workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.slots.options import match_option
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


# ── Reading the records option out of the caller's words ─────────────────────
#     AI      …we'll need a complete copy of the medical records for this
#             adjustment. Are you able to provide those?
#     Caller  Can I ask my doctor to send it over?
#     AI      That's a great question — yes, your doctor can send the records
#             over to us.
#
# The caller chose an option. records_coordination has the branch for it — an
# acknowledgement plus the upload-link offer, both static — and the branch was
# not taken: extraction returned no upload_method, so the turn burned a retry
# attempt and answered with generated prose instead. Nothing moved.
#
# records_coordination.md lists "my doctor will send it" and "the provider can
# send it" under doctor_direct, so the classification was not unspecified. The
# caller phrased it as asking permission, which is how people pick an option,
# and the question mark sent the turn down the side-question path.
#
# upload_method is the only branch-selecting slot with no normalizer at all, so
# the extraction model was the single point of failure. This is the deterministic
# reading, consulted when extraction produced nothing.

# The patterns that read these three options out of the caller's words moved to
# agent.slots.options.UPLOAD_METHOD, which is also what renders the accepted
# answers into the extraction prompt. One table, so the list the model is shown
# and the list this screen can read cannot drift apart.
#
# "decline" is still deliberately unmatchable. Inferring one escalates the call,
# which is the worst outcome to reach on a guess — a caller who has not refused
# is handed to a representative they did not ask for. That stays with the model.


def screen_upload_method(utterance: str) -> str:
    """The records option named in ``utterance``, or "".

    Returns one of the upload_method values the agent branches on. Only the
    three positive branches are read; see the note above on "decline".
    """
    value = match_option("upload_method", utterance)
    if value:
        logger.info(
            "screen_upload_method: read the records option from the caller's words",
            extra={"value": value, "utterance": (utterance or "").strip()[:60]},
        )
    return value
