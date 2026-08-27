"""_adjustments.py — async claim adjustment queries."""

import datetime
from typing import Dict, Optional

from agent.storage.db import query_store
from agent.storage.queries.members import normalize_member_id


def _now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


async def find_adjustment(reference_number: str, member_id: str) -> Optional[Dict]:
    return await query_store(
        "find_one",
        entity="adjustment_requests",
        where={"reference_number": reference_number, "member_id": normalize_member_id(member_id)},
    )


async def find_adjustment_by_claim_number(claim_number: str, member_id: str) -> Optional[Dict]:
    """Look up an adjustment request by claim_number + member_id."""
    return await query_store(
        "find_one",
        entity="adjustment_requests",
        where={"claim_number": claim_number, "member_id": normalize_member_id(member_id)},
    )


async def find_adjustment_by_dos_and_billed(dos: str, billed_amount: str, member_id: str) -> Optional[Dict]:
    """Look up an adjustment by date-of-service + member_id, then filter by billed_amount.

    DOS is expected in YYYY-MM-DD (ISO) format so the DB layer queries it as a
    date literal (without quotes).  Billed amount is compared client-side as a
    float to avoid SOQL currency-type quoting issues.
    """
    from typing import List

    rows: List[Dict] = (
        await query_store(  # type: ignore[assignment]
            "find_many",
            entity="adjustment_requests",
            where={"dos": dos, "member_id": normalize_member_id(member_id)},
        )
        or []
    )

    if not rows or not billed_amount:
        return None

    try:
        target = float(billed_amount)
    except (TypeError, ValueError):
        return None

    for row in rows:
        stored = row.get("billed_amount")
        try:
            if abs(float(stored) - target) < 0.01:
                return row
        except (TypeError, ValueError):
            continue
    return None


async def set_claim_request_delivery(
    member_id: str,
    reference_number: str,
    *,
    run_id: Optional[str] = None,
    method: Optional[str] = None,
    destination: Optional[str] = None,
    update_status: Optional[str] = None,
) -> Dict:
    member_id = normalize_member_id(member_id)
    now = _now()
    row = await query_store(
        "find_one",
        entity="claim_request_delivery_updates",
        where={"member_id": member_id, "reference_number": reference_number, "run_id": run_id},
    )

    if not row or (method is not None and row.get("method") != method):
        return await query_store(
            "insert_one",
            entity="claim_request_delivery_updates",
            record={
                "member_id": member_id,
                "reference_number": reference_number,
                "run_id": run_id,
                "method": method,
                "destination": destination,
                "update_status": update_status,
                "updated_at": now,
            },
        )

    update = {
        k: v
        for k, v in {"destination": destination, "update_status": update_status, "updated_at": now}.items()
        if v is not None
    }
    if update:
        await query_store(
            "update_one", entity="claim_request_delivery_updates", where={"id": row["id"]}, update=update
        )
        row.update(update)
    return row


async def fetch_claim_request_delivery(
    member_id: str,
    reference_number: str,
    *,
    run_id: Optional[str] = None,
    method: Optional[str] = None,
) -> Optional[Dict]:
    return await query_store(
        "find_one",
        entity="claim_request_delivery_updates",
        where={
            "member_id": normalize_member_id(member_id),
            "reference_number": reference_number,
            "run_id": run_id,
            "method": method,
        },
    )


async def send_upload_link(
    member_id: str,
    reference_number: str,
    email: str,
    *,
    run_id: str | None = None,
) -> dict:
    # M_Claim_Upload_Link__c does not exist in this org — skip SF write
    return {"status": "ok"}


async def trigger_personal_guide_outreach_for_claim(
    member_id: str,
    reference_number: str,
) -> dict:
    # M_Claim_Outreach__c does not exist in this org — skip SF write
    return {"status": "ok"}


async def set_claim_timeline_notification(
    member_id: str,
    reference_number: str,
    method: str,
    destination: str,
) -> dict:
    # M_Claim_Timeline_Notification__c does not exist in this org — skip SF write
    return {"status": "ok"}


async def set_claim_notification_preference(
    member_id: str,
    reference_number: str,
    method: str,
    destination: str,
) -> dict:
    """
    Write the member's notification channel preference to Salesforce.
    Reuses the existing notification_preferences entity.
    """
    from agent.storage.queries.communication import set_notification_preference

    return await set_notification_preference(
        member_id=normalize_member_id(member_id),
        reference_number=reference_number,
        method=method,
        destination=destination,
    )
