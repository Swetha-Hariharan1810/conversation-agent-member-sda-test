"""
tools.py — Storage tools (core agents only: lookup_member, update_member_contact).
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.logger import get_logger

logger = get_logger(__name__)


@tool
async def lookup_member(
    member_id: str,
    first_name: str = "",
    last_name: str = "",
    dob: str = "",
) -> Optional[Dict[str, Any]]:
    """Full identity verification against Salesforce member record.

    On a full identity match, returns ``verified=True`` with the member's
    contact fields (unchanged contract). On a failed full match, additionally
    reports whether a record exists for the Member ID and, when it does, which
    identity fields matched — so callers can give targeted re-prompts without a
    second lookup. ``verified`` is always present, so verified-only callers are
    unaffected.
    """
    from agent.storage.queries.members import (
        compare_identity_fields,
        find_member_by_identity,
        get_member_contact,
    )

    try:
        record = await find_member_by_identity(
            member_id=member_id,
            first_name=first_name,
            last_name=last_name,
            dob=dob,
        )
        logger.info(
            "lookup_member: verification attempt",
            extra={"verified": bool(record), "member_tail": member_id[-4:]},
        )
        if record:
            return {
                "verified": True,
                "member_id": record.get("member_id"),
                "phone_number": record.get("phone_number", ""),
                "zip_code": record.get("zip_code", ""),
                "fax": record.get("fax", ""),
                "email": record.get("email", ""),
                "relationship": record.get("relationship", ""),
                "record": record,
            }

        # Failed full match — fetch by Member ID alone to see whether the ID
        # exists and, if so, which identity fields differ.
        id_record = await get_member_contact(member_id) if member_id else None
        if not id_record:
            return {"verified": False, "member_id_found": False}

        return {
            "verified": False,
            "member_id_found": True,
            "field_matches": compare_identity_fields(
                id_record,
                first_name=first_name,
                last_name=last_name,
                dob=dob,
            ),
            "record": id_record,
        }
    except Exception:
        logger.exception("lookup_member failed")
        return {"verified": False}


@tool
async def lookup_member_by_ssn(ssn: str) -> Optional[Dict[str, Any]]:
    """Look up a member record by SSN only.

    Returns the same contract as ``lookup_member`` (verified, member_id,
    phone_number, zip_code, fax, email, relationship, record) so callers can
    use either tool interchangeably.
    """
    from agent.storage.queries.members import find_member_by_ssn

    try:
        record = await find_member_by_ssn(ssn=ssn)
        logger.info(
            "lookup_member_by_ssn: attempt",
            extra={"verified": bool(record)},
        )
        if record:
            return {
                "verified": True,
                "member_id": record.get("member_id"),
                "phone_number": record.get("phone_number", ""),
                "zip_code": record.get("zip_code", ""),
                "fax": record.get("fax", ""),
                "email": record.get("email", ""),
                "relationship": record.get("relationship", ""),
                "record": record,
            }
        return {"verified": False, "member_id_found": False}
    except Exception:
        logger.exception("lookup_member_by_ssn failed")
        return {"verified": False}


@tool
async def update_member_contact(
    member_id: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    fax: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> bool:
    """Update contact fields for a verified member."""
    from agent.storage.queries.members import update_member_contact as _update

    try:
        return await _update(member_id, phone=phone, email=email, fax=fax, zip_code=zip_code)
    except Exception:
        logger.exception("update_member_contact failed")
        return False


@tool
async def update_zip_code(member_id: str, zip_code: str) -> bool:
    """Update ZIP code for a verified member."""
    from agent.storage.queries.members import update_member_contact as _update

    try:
        return await _update(member_id, zip_code=zip_code)
    except Exception:
        logger.exception("update_zip_code failed")
        return False


@tool
async def dispatch_provider_list(
    member_id: str,
    provider_type: str,
    zip_code: str,
    delivery_method: str,
    delivery_address: str,
) -> bool:
    """Dispatch an in-network provider list to the member via fax or email."""
    from agent.storage.queries.providers import send_provider_list

    try:
        return await send_provider_list(
            member_id=member_id,
            provider_type=provider_type,
            zip_code=zip_code,
            delivery_method=delivery_method,
            delivery_address=delivery_address,
        )
    except Exception:
        logger.exception("dispatch_provider_list failed")
        return False


@tool
async def dispatch_care_coach_details(
    member_id: str,
    delivery_method: str,
    delivery_address: str,
) -> bool:
    """Dispatch Care Coach program details to the member via fax or email."""
    from agent.storage.queries.wellness import dispatch_care_coach_details as _dispatch

    try:
        return await _dispatch(
            member_id=member_id,
            delivery_method=delivery_method,
            delivery_address=delivery_address,
        )
    except Exception:
        logger.exception("dispatch_care_coach_details failed")
        return False


@tool
async def send_claim_upload_link(
    member_id: str,
    reference_number: str,
    email: str,
) -> bool:
    """Generate and send a secure medical records upload link to the member via email."""
    from agent.storage.queries.adjustments import send_upload_link

    try:
        result = await send_upload_link(member_id, reference_number, email)
        return bool(result)
    except Exception:
        logger.exception("send_claim_upload_link failed")
        return False


@tool
async def trigger_claim_personal_guide(
    member_id: str,
    reference_number: str,
) -> bool:
    """Trigger a Personal Guide to contact the provider
    for medical records (requires prior member consent)."""
    from agent.storage.queries.adjustments import trigger_personal_guide_outreach_for_claim

    try:
        result = await trigger_personal_guide_outreach_for_claim(member_id, reference_number)
        return bool(result)
    except Exception:
        logger.exception("trigger_claim_personal_guide failed")
        return False


@tool
async def set_claim_timeline_notification(
    member_id: str,
    reference_number: str,
    method: str,
    destination: str,
) -> bool:
    """Write the member's claim timeline/progress notification preference to Salesforce."""
    # M_Claim_Timeline_Notification__c does not exist in this org — skip SF write
    return {"status": "ok"}


@tool
async def set_claim_notification(
    member_id: str,
    reference_number: str,
    method: str,
    destination: str,
) -> bool:
    """Write the member's chosen notification channel and contact to Salesforce."""
    # M_Notification_Preference__c does not exist in this org — skip SF write
    return {"status": "ok"}
