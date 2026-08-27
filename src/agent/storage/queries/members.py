"""_members.py — async member queries."""

from typing import Dict, Optional

from agent.logger import get_logger
from agent.slots.normalizers import normalize_dob as _normalize_dob_slot
from agent.slots.normalizers import normalize_name as _normalize_name_slot
from agent.storage.db import query_store

logger = get_logger(__name__)


def normalize_member_id(member_id: Optional[str]) -> Optional[str]:
    return member_id.strip().upper() if member_id else None


def normalize_name(value: str) -> str:
    return _normalize_name_slot(value)


async def find_member_by_identity(
    *, member_id: str, first_name: str, last_name: str, dob: str, ssn: Optional[str] = None
) -> Optional[Dict]:
    where = {
        "first_name": normalize_name(first_name),
        "last_name": normalize_name(last_name),
        "dob": dob_to_db_format(dob),
    }
    if member_id:
        where["member_id"] = normalize_member_id(member_id)
    if ssn:
        where["ssn"] = ssn.strip()
    return await query_store("find_one", entity="members", where=where)


async def find_member_by_ssn(ssn: str) -> Optional[Dict]:
    return await query_store("find_one", entity="members", where={"ssn": ssn.strip()})


async def get_member_contact(member_id: str) -> Optional[Dict]:
    return await query_store(
        "find_one", entity="members", where={"member_id": normalize_member_id(member_id)}
    )


async def update_member_contact(
    member_id: str,
    *,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    fax: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> bool:
    payload = {
        k: v
        for k, v in {"phone_number": phone, "email": email, "fax": fax, "zip_code": zip_code}.items()
        if v is not None
    }
    if not payload:
        return False
    updated = await query_store(
        "update_one", entity="members", where={"member_id": normalize_member_id(member_id)}, update=payload
    )
    return bool(updated)


def dob_to_db_format(dob: str) -> str:
    """Convert MM/DD/YYYY (normalizer output) to YYYY-MM-DD (DB format)."""
    if not dob:
        return ""
    if len(dob) == 10 and dob[4] == "-":
        return dob  # already YYYY-MM-DD
    try:
        # Lazy import: avoids circular dependency
        from datetime import datetime

        return datetime.strptime(dob, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return dob


def _dob_key(value: Optional[str]) -> str:
    """Canonical DOB key for comparison: normalize then convert to DB (YYYY-MM-DD) form.

    Mirrors the identity pipeline + storage path, so spoken/MM-DD-YYYY/ISO inputs
    all collapse to the same key (e.g. "7/13/1977" and "1977-07-13" → "1977-07-13").
    """
    return dob_to_db_format(_normalize_dob_slot(value))


def compare_identity_fields(
    record: Optional[Dict],
    *,
    first_name: str = "",
    last_name: str = "",
    dob: str = "",
) -> Dict[str, bool]:
    """Compare caller-provided identity fields against a fetched member record.

    Returns a per-field match map for ``first_name``, ``last_name`` and ``dob``.
    Both sides are normalized with the same normalizers the identity pipeline
    uses — names via ``normalize_name``, dob via ``normalize_dob`` +
    ``dob_to_db_format`` — so equivalent values ("JAMES" vs "James",
    "7/13/1977" vs "1977-07-13") are never flagged as mismatches.

    An empty caller-provided field is treated as "not yet provided" rather than
    a mismatch, so it reports ``True``. This is a pure helper: it performs no
    I/O and does not mutate state.
    """
    record = record or {}

    def _name_match(provided: str, stored: Optional[str]) -> bool:
        if not (provided and provided.strip()):
            return True  # not yet provided — not a mismatch
        return normalize_name(provided) == normalize_name(stored or "")

    def _dob_match(provided: str, stored: Optional[str]) -> bool:
        if not (provided and provided.strip()):
            return True  # not yet provided — not a mismatch
        return _dob_key(provided) == _dob_key(stored)

    return {
        "first_name": _name_match(first_name, record.get("first_name")),
        "last_name": _name_match(last_name, record.get("last_name")),
        "dob": _dob_match(dob, record.get("dob")),
    }
