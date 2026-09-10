"""
db.py

Async Salesforce query engine.
All functions are async — await them directly.

Parallelization:
  Since query_store is now async, multiple independent reads can run
  concurrently using asyncio.gather():

    member, benefits = await asyncio.gather(
        query_store("find_one", entity="members", where={"member_id": mid}),
        query_store("find_one", entity="benefit_plans", where={"member_id": mid}),
    )

  Each SF HTTP call runs in parallel — total wait = max(call1, call2),
  not call1 + call2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from langsmith import traceable

from agent.storage.client import SalesforceClient

# ── Entity → Salesforce sObject mapping ──────────────────────────────────────

_SF_ENTITIES: Dict[str, Dict[str, Any]] = {
    "members": {
        "sobject": "M_Member__c",
        "field_map": {
            "member_id": "Member_ID__c",
            "account_id": "Account_ID__c",
            "first_name": "First_Name__c",
            "last_name": "Last_Name__c",
            "dob": "Date_of_Birth__c",
            "relationship": "Relationship__c",
            "zip_code": "Zip_Code__c",
            "phone_number": "Phone_Number__c",
            "fax": "Fax__c",
            "email": "Email__c",
            "ssn": "SSN__c",
        },
        # Fields that are native Salesforce Date type — queried without quotes.
        # Text fields that happen to store ISO-format values must NOT be listed here.
        "date_fields": {"dob"},
    },
    "benefit_plans": {
        "sobject": "M_Benefit_Plan__c",
        "field_map": {
            "member_id": "Member_ID__c",
            "individual_deductible": "Individual_Deductible__c",
            "family_deductible": "Family_Deductible__c",
            "coinsurance_percent": "Coinsurance_Percent__c",
            "individual_oop_max": "Individual_OOP_Max__c",
            "family_oop_max": "Family_OOP_Max__c",
        },
    },
    "adjustment_requests": {
        "sobject": "M_Adjustment_Request__c",
        "field_map": {
            "name": "Name",
            "reference_number": "Reference_Number__c",
            "member_id": "Member_Id__c",
            "claim_number": "Claim_Number__c",
            "claim_status": "Claim_Status__c",
            "claim_update_date": "Claim_Update_Date__c",
            "claim_created_at": "Claim_Created_At__c",
            "claim_updated_at": "Claim_Updated_At__c",
            "dos": "DOS__c",
            "billed_amount": "Billed_Amount__c",
        },
        # DOS__c is a Text field in this org — it must be quoted in SOQL
        # even though its values happen to look like ISO dates.
        "date_fields": set(),
    },
    "claim_request_delivery_updates": {
        "sobject": "M_Claim_Update__c",
        "field_map": {
            "id": "ID__c",
            "run_id": "Run_ID__c",
            "reference_number": "Reference_Number__c",
            "member_id": "Member_ID__c",
            "method": "Method__c",
            "destination": "Destination__c",
            "update_status": "Update_Status__c",
            "updated_at": "Updated_At__c",
        },
    },
    "provider_request_delivery_updates": {
        "sobject": "M_Provider_Update__c",
        "field_map": {
            "id": "ID__c",
            "run_id": "Run_ID__c",
            "member_id": "Member_ID__c",
            "provider_type": "Provider_Type__c",
            "method": "Method__c",
            "destination": "Destination__c",
            "update_status": "Update_Status__c",
            "updated_at": "Updated_At__c",
        },
    },
    "provider_outreach": {
        "sobject": "M_Provider_Outreach__c",
        "field_map": {
            "id": "ID__c",
            "member_id": "Member_ID__c",
            "information_request_type": "Information_Request_Type__c",
            "method": "Method__c",
            "destination": "Destination__c",
            "status": "Status__c",
            "is_active": "Is_Active__c",
        },
    },
    "claim_upload_links": {
        "sobject": "M_Claim_Upload_Link__c",
        "field_map": {
            "id": "ID__c",
            "run_id": "Run_ID__c",
            "member_id": "Member_ID__c",
            "reference_number": "Reference_Number__c",
            "upload_url": "Upload_URL__c",
            "email_sent_to": "Email_Sent_To__c",
            "status": "Status__c",
            "created_at": "Created_At__c",
        },
    },
    "claim_outreach": {
        "sobject": "M_Claim_Outreach__c",
        "field_map": {
            "id": "ID__c",
            "member_id": "Member_ID__c",
            "reference_number": "Reference_Number__c",
            "outreach_type": "Outreach_Type__c",
            "status": "Status__c",
            "is_active": "Is_Active__c",
            "created_at": "Created_At__c",
        },
    },
    "notification_preferences": {
        "sobject": "M_Notification_Preference__c",
        "field_map": {
            "id": "ID__c",
            "member_id": "Member_ID__c",
            "reference_number": "Reference_Number__c",
            "method": "Method__c",
            "destination": "Destination__c",
            "is_active": "Is_Active__c",
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sf_escape(val: Any) -> str:
    return str(val).replace("\\", "\\\\").replace("'", "\\'")


def _is_iso_date(val: Any) -> bool:
    return isinstance(val, str) and len(val) == 10 and val[4] == "-" and val[7] == "-"


def _normalize_sf_value(val: Any) -> Any:
    if isinstance(val, str) and "T" in val:
        return val[:10]
    return val


def _sf_select_list(field_map: Dict[str, str]) -> List[str]:
    fields = {"Id"}
    fields.update(field_map.values())
    return sorted(fields)


def _sf_to_app(record: Dict[str, Any], field_map: Dict[str, str]) -> Dict[str, Any]:
    return {k: record.get(v) for k, v in field_map.items()}


# ── Singleton client ──────────────────────────────────────────────────────────

_SF_CLIENT: Optional[SalesforceClient] = None


def _get_sf() -> SalesforceClient:
    global _SF_CLIENT
    if _SF_CLIENT is None:
        _SF_CLIENT = SalesforceClient()
    return _SF_CLIENT


def reset_salesforce_client() -> None:
    global _SF_CLIENT
    _SF_CLIENT = None


# ── Errors ────────────────────────────────────────────────────────────────────


class DBError(Exception):
    pass


# ── Core async query engine ───────────────────────────────────────────────────


async def _sf_query_store(
    action: str,
    *,
    entity: str,
    where: Optional[Dict[str, Any]] = None,
    update: Optional[Dict[str, Any]] = None,
    record: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]], int, None]:
    if entity not in _SF_ENTITIES:
        raise DBError(f"Unknown entity '{entity}'")

    cfg = _SF_ENTITIES[entity]
    sobject = cfg["sobject"]
    fmap = cfg["field_map"]

    # Build WHERE clause
    date_fields: set = cfg.get("date_fields", set())
    conds: List[str] = []
    if where:
        for k, v in where.items():
            if v is None:
                continue
            sf_field = fmap.get(k)
            if not sf_field:
                raise DBError(f"Unknown filter field '{k}'")
            # Only omit quotes for fields explicitly declared as Salesforce Date
            # types.  Text fields that happen to store ISO-formatted values must
            # be quoted; otherwise SOQL rejects the bare date literal.
            if _is_iso_date(v) and k in date_fields:
                conds.append(f"{sf_field} = {v}")
            else:
                conds.append(f"{sf_field} = '{_sf_escape(v)}'")

    where_sql = f" WHERE {' AND '.join(conds)}" if conds else ""
    limit_sql = f" LIMIT {limit}" if limit else ""
    soql = f"SELECT {', '.join(_sf_select_list(fmap))} FROM {sobject}{where_sql}{limit_sql}"

    sf = _get_sf()

    # READ
    if action in ("find_one", "find_many"):
        res = await sf.query(soql)
        rows = [_sf_to_app(r, fmap) for r in res.get("records", [])]
        return rows[0] if action == "find_one" and rows else (None if action == "find_one" else rows)

    # INSERT
    if action == "insert_one":
        payload = {
            fmap[k]: _normalize_sf_value(v) for k, v in (record or {}).items() if k in fmap and v is not None
        }
        return await sf.create(sobject=sobject, payload=payload)

    # UPDATE
    if action == "update_one":
        res = await sf.query(soql)
        records = res.get("records", [])
        if not records:
            return 0

        payload = {
            fmap[k]: _normalize_sf_value(v) for k, v in (update or {}).items() if k in fmap and v is not None
        }
        if payload:
            await sf.update(sobject=sobject, record_id=records[0]["Id"], payload=payload)
        return 1

    raise DBError(f"Unsupported action '{action}'")


@traceable(name="query_store")
async def query_store(
    action: str,
    *,
    entity: str,
    where: Optional[Dict[str, Any]] = None,
    update: Optional[Dict[str, Any]] = None,
    record: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]], int, None]:
    """
    Async Salesforce query engine.

    Use asyncio.gather() for parallel independent reads:
        member, benefits = await asyncio.gather(
            query_store("find_one", entity="members", where={"member_id": mid}),
            query_store("find_one", entity="benefit_plans", where={"member_id": mid}),
        )
    """
    return await _sf_query_store(
        action, entity=entity, where=where, update=update, record=record, limit=limit
    )
