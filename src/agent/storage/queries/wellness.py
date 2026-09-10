"""wellness.py — async care coach and wellness queries."""

from agent.storage.queries.members import normalize_member_id


async def dispatch_care_coach_details(
    member_id: str,
    delivery_method: str,
    delivery_address: str,
) -> bool:
    """
    Record a Care Coach detail dispatch in Salesforce.
    Returns True on success.
    """
    from agent.storage.queries.communication import create_provider_outreach

    try:
        await create_provider_outreach(
            normalize_member_id(member_id),
            information_request_type="care_coach",
            method=delivery_method,
            destination=delivery_address,
            status="sent",
        )
        return True
    except Exception:
        return False
