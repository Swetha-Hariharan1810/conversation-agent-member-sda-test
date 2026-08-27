"""providers.py — async provider list dispatch queries."""

import logging

from agent.storage.queries.communication import set_provider_request_delivery
from agent.storage.queries.members import normalize_member_id

logger = logging.getLogger(__name__)


async def send_provider_list(
    member_id: str,
    provider_type: str,
    zip_code: str,
    delivery_method: str,
    delivery_address: str,
) -> bool:
    """
    Record a provider list dispatch request in Salesforce.
    Returns True on success.
    """
    try:
        await set_provider_request_delivery(
            normalize_member_id(member_id),
            provider_type=provider_type,
            method=delivery_method,
            destination=delivery_address,
            update_status="sent",
        )
        return True
    except Exception:
        logger.exception(
            "send_provider_list: failed to write M_Provider_Update__c for member=%s provider_type=%s",
            member_id,
            provider_type,
        )
        return False
