"""_benefits.py — async benefits queries."""

from typing import Dict, Optional

from agent.storage.db import query_store
from agent.storage.queries.members import normalize_member_id


async def get_member_benefits(member_id: str) -> Optional[Dict]:
    return await query_store(
        "find_one", entity="benefit_plans", where={"member_id": normalize_member_id(member_id)}
    )
