"""handlers.py — Benefits workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_FETCH_FAIL = [
    "I'm sorry, I wasn't able to retrieve your plan details right now. Let me connect you with a specialist.",
    "I'm having trouble pulling up your benefits. Connecting you with a representative.",
]


async def fetch_benefits(agent, state: State) -> tuple[dict | None, dict | None]:
    """
    Fetch member benefit plan from Salesforce.

    Returns (benefits_record, interrupt_or_none).
    If interrupt is not None, the caller must return it immediately.
    """
    from agent.storage.queries.benefits import get_member_benefits

    member_id = state.get("member_id", "")
    if not member_id:
        logger.warning("fetch_benefits: no member_id in state")
        return None, agent.signal_escalate(state, pick(_MSG_FETCH_FAIL), reason="no_member_id_for_benefits")

    # If already fetched and stored in state (e.g. prefetched during verification),
    # return from state directly — no Salesforce call needed.
    if state.get("individual_deductible"):
        logger.info(
            "fetch_benefits: cache hit — benefits already in state from verification prefetch, "
            "skipping Salesforce call"
        )
        return {
            "individual_deductible": state.get("individual_deductible", ""),
            "family_deductible": state.get("family_deductible", ""),
            "coinsurance_percent": state.get("coinsurance_percent", ""),
            "individual_oop_max": state.get("individual_oop_max", ""),
            "family_oop_max": state.get("family_oop_max", ""),
        }, None

    try:
        record = await get_member_benefits(member_id)
        if not record:
            return None, agent.signal_escalate(
                state, pick(_MSG_FETCH_FAIL), reason="benefits_record_not_found"
            )
        return record, None
    except Exception:
        logger.exception("fetch_benefits: Salesforce call failed")
        return None, agent.signal_escalate(state, pick(_MSG_FETCH_FAIL), reason="benefits_fetch_error")
