"""
preflight.py — Environment + Salesforce fixture verification and teardown.

Run before any live scenario:
  1. Verify required env vars (fail fast, list what is missing).
  2. Live-verify the Salesforce fixtures the scenarios depend on, via the real
     query layer (agent.storage.queries) — no mocks. Snapshot contact fields so
     mutating scenarios can be reverted.
  3. Warm LLM + Salesforce connections via agent.app_graph.warm_llm_connections().

Teardown: restore_contacts(snapshot) re-writes the snapshotted zip/fax/email/
phone for both fixture members via update_member_contact. Call it in a
`finally` block after any mutating scenario, even on failure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("live_e2e.preflight")

REQUIRED_ENV = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "SF_CLIENT_ID",
    "SF_CLIENT_SECRET",
    "SF_REFRESH_TOKEN",
    "SF_INSTANCE_URL",
]

# ── Fixture identities (must exist in the target Salesforce org) ──────────────
EMILY = {
    "member_id": "M907503",
    "first_name": "Emily",
    "last_name": "Carter",
    "dob": "1988-04-12",
}
JAMES = {
    "member_id": "M310188",
    "first_name": "James",
    "last_name": "Wilson",
    "dob": "1977-07-30",
}
JAMES_PHONE_DIGITS = "5125556101"  # 512-555-6101
JAMES_EMAIL = "james.wilson@gmail.com"
ADJUSTMENT_REF = "42695817"

FIXTURE_INSTRUCTIONS = """
Live E2E fixtures are missing or wrong in the target Salesforce org.
Required records (create/fix them, then re-run):

  M_Member__c  Emily Carter
    Member_ID__c=M907503, First_Name__c=Emily, Last_Name__c=Carter,
    Date_of_Birth__c=1988-04-12, with non-empty Zip_Code__c, Fax__c, Email__c.

  M_Member__c  James Wilson
    Member_ID__c=M310188, First_Name__c=James, Last_Name__c=Wilson,
    Date_of_Birth__c=1977-07-30, Phone_Number__c=512-555-6101,
    Email__c=james.wilson@gmail.com.

  M_Adjustment_Request__c
    Reference_Number__c=42695817 linked to Member_Id__c=M310188.
    (records_required is not a field on this object — the application code
    defaults it to True, which these tests rely on.)

  M_Benefit_Plan__c
    Member_ID__c=M907503 with deductible / coinsurance / OOP fields populated.

Do NOT skip these silently — the scenario assertions depend on this exact data.
"""


class PreflightError(RuntimeError):
    """Raised when env or Salesforce fixtures are missing. Abort the run."""


@dataclass
class MemberSnapshot:
    member_id: str
    zip_code: str = ""
    fax: str = ""
    email: str = ""
    phone_number: str = ""


@dataclass
class FixtureSnapshot:
    members: dict[str, MemberSnapshot] = field(default_factory=dict)

    def get(self, member_id: str) -> MemberSnapshot:
        return self.members[member_id]


def _digits(value: str | None) -> str:
    return "".join(c for c in (value or "") if c.isdigit())


def check_env() -> list[str]:
    """Return the list of missing required env vars (also consults the repo
    .env loader used by the app, so a populated .env counts)."""
    # Importing Config triggers the same dotenv loading the app uses.
    from agent.llm.config import Config  # noqa: F401

    return [name for name in REQUIRED_ENV if not os.getenv(name, "")]


async def verify_fixtures() -> FixtureSnapshot:
    """Verify Salesforce fixtures with REAL queries and snapshot contact fields."""
    from agent.storage.queries.adjustments import find_adjustment
    from agent.storage.queries.benefits import get_member_benefits
    from agent.storage.queries.members import find_member_by_identity

    problems: list[str] = []
    snapshot = FixtureSnapshot()

    emily = await find_member_by_identity(**EMILY)
    if not emily:
        problems.append("Member Emily Carter / M907503 / DOB 1988-04-12 not found.")
    else:
        for fld in ("zip_code", "fax", "email"):
            if not (emily.get(fld) or "").strip():
                problems.append(f"Emily Carter (M907503) has no {fld} on file.")
        snapshot.members["M907503"] = MemberSnapshot(
            member_id="M907503",
            zip_code=emily.get("zip_code") or "",
            fax=emily.get("fax") or "",
            email=emily.get("email") or "",
            phone_number=emily.get("phone_number") or "",
        )

    james = await find_member_by_identity(**JAMES)
    if not james:
        problems.append("Member James Wilson / M310188 / DOB 1977-07-30 not found.")
    else:
        if _digits(james.get("phone_number")) != JAMES_PHONE_DIGITS:
            problems.append(
                f"James Wilson phone on file is {james.get('phone_number')!r}, expected 512-555-6101."
            )
        if (james.get("email") or "").strip().lower() != JAMES_EMAIL:
            problems.append(f"James Wilson email on file is {james.get('email')!r}, expected {JAMES_EMAIL}.")
        snapshot.members["M310188"] = MemberSnapshot(
            member_id="M310188",
            zip_code=james.get("zip_code") or "",
            fax=james.get("fax") or "",
            email=james.get("email") or "",
            phone_number=james.get("phone_number") or "",
        )

    adjustment = await find_adjustment(ADJUSTMENT_REF, "M310188")
    if not adjustment:
        problems.append(f"Adjustment request {ADJUSTMENT_REF} linked to M310188 not found.")
    # records_required is not a real field on M_Adjustment_Request__c; the
    # claim agent defaults it to True, which the records scenarios depend on.

    benefits = await get_member_benefits("M907503")
    if not benefits:
        problems.append("Benefit plan record (M_Benefit_Plan__c) for M907503 not found.")

    if problems:
        raise PreflightError(
            "Salesforce fixture verification failed:\n  - "
            + "\n  - ".join(problems)
            + "\n"
            + FIXTURE_INSTRUCTIONS
        )

    logger.info(
        "preflight: fixtures verified — Emily(zip=%s, fax=%s, email=%s) James(phone=%s)",
        snapshot.members["M907503"].zip_code,
        snapshot.members["M907503"].fax,
        snapshot.members["M907503"].email,
        snapshot.members["M310188"].phone_number,
    )
    return snapshot


async def restore_contacts(snapshot: FixtureSnapshot) -> None:
    """Restore snapshotted contact fields for every fixture member.

    Safe to call repeatedly; writes the original values back via the real
    update_member_contact query. Call from a `finally` block after any
    scenario that mutates Salesforce contact fields, even if it failed.
    """
    from agent.storage.queries.members import update_member_contact

    for member in snapshot.members.values():
        try:
            await update_member_contact(
                member.member_id,
                phone=member.phone_number or None,
                email=member.email or None,
                fax=member.fax or None,
                zip_code=member.zip_code or None,
            )
            logger.info("preflight: restored contact fields for %s", member.member_id)
        except Exception:
            logger.exception(
                "preflight: FAILED to restore contact fields for %s — fix manually: %r",
                member.member_id,
                member,
            )
            raise


async def run_preflight(warm: bool = True) -> FixtureSnapshot:
    """Full preflight: env check → fixture verification → connection warm-up."""
    missing = check_env()
    if missing:
        raise PreflightError(
            "Missing required environment variables:\n  - "
            + "\n  - ".join(missing)
            + "\nSet them (or populate .env at the repo root) and re-run. "
            "These tests hit LIVE Azure OpenAI and Salesforce — no mocks."
        )

    snapshot = await verify_fixtures()

    if warm:
        from agent.app_graph import warm_llm_connections

        await warm_llm_connections()
        logger.info("preflight: LLM + Salesforce connections warmed")

    return snapshot
