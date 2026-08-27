"""_communication.py — async provider delivery, outreach, notification queries."""

import datetime
from typing import Dict, List, Optional

from agent.storage.db import query_store
from agent.storage.queries.members import normalize_member_id


def _now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


async def list_provider_request_deliveries_by_run(run_id: str) -> List[Dict]:
    return (
        await query_store("find_many", entity="provider_request_delivery_updates", where={"run_id": run_id})
        or []
    )


async def set_provider_request_delivery(
    member_id: str,
    *,
    run_id: Optional[str] = None,
    provider_type: Optional[str] = None,
    method: Optional[str] = None,
    destination: Optional[str] = None,
    update_status: Optional[str] = None,
) -> Dict:
    # Idempotency contract: run_id is unique per conversation turn.
    # Duplicate inserts are prevented by the caller generating a fresh
    # run_id on each invocation. The prior find_one + conditional insert/update
    # pattern cost two serial SF HTTP calls per dispatch — replaced here with
    # a single insert_one.
    member_id = normalize_member_id(member_id)
    return await query_store(
        "insert_one",
        entity="provider_request_delivery_updates",
        record={
            "member_id": member_id,
            "run_id": run_id,
            "provider_type": provider_type,
            "method": method,
            "destination": destination,
            "update_status": update_status,
            "updated_at": _now(),
        },
    )


async def create_provider_outreach(
    member_id: str,
    information_request_type: str,
    *,
    method: Optional[str] = None,
    destination: Optional[str] = None,
    status: Optional[str] = "initiated",
) -> Dict:
    return await query_store(
        "insert_one",
        entity="provider_outreach",
        record={
            "member_id": normalize_member_id(member_id),
            "information_request_type": information_request_type,
            "method": method,
            "destination": destination,
            "status": status,
            "is_active": True,
        },
    )


async def set_notification_preference(
    member_id: str,
    reference_number: str,
    method: str,
    destination: str,
) -> Dict:
    member_id = normalize_member_id(member_id)
    existing = await query_store(
        "find_one",
        entity="notification_preferences",
        where={"member_id": member_id, "reference_number": reference_number},
    )
    if existing:
        await query_store(
            "update_one",
            entity="notification_preferences",
            where={"id": existing["id"]},
            update={"method": method, "destination": destination, "is_active": True},
        )
        return existing
    return await query_store(
        "insert_one",
        entity="notification_preferences",
        record={
            "member_id": member_id,
            "reference_number": reference_number,
            "method": method,
            "destination": destination,
            "is_active": True,
        },
    )
