"""
llm.py — Single LLM call for FollowUpAgent.

One call does everything: guard classification, intent routing,
and answer generation from session context.

Previously two calls (extract_follow_up_decision + generate_follow_up_answer)
are now collapsed into one. The session snapshot is injected into the user
content block so the model classifies intent AND writes the answer in a
single forward pass. The answer field on WorkerResult carries it back.

Spoken-form requirement: every email address and website written into the
SESSION SNAPSHOT is rendered in fully spoken words ("at" / "dot") via
speak_email/speak_url, so any answer the LLM generates from the snapshot
speaks them verbatim.
"""

from __future__ import annotations

from agent.llm.extractor import build_worker_input
from agent.llm.schema import FollowUpResult
from agent.logger import get_logger
from agent.state import State
from agent.utils import speak_email, speak_url

logger = get_logger(__name__)


def _build_session_snapshot(state: State) -> str:  # noqa: C901
    """
    Serialise all relevant session state into a compact, readable text block.

    Rules:
      - Only include fields that have a non-empty value.
      - Group related fields under a plain-text heading.
      - Use concrete dollar amounts, not variable names.
      - Keep it short — this is LLM context, not a report.
      - Emails and URLs appear in spoken-word form ("at" / "dot") so the
        generated answer reads them out verbatim.
    """
    lines: list[str] = []

    # ── Member identity ───────────────────────────────────────────────────────
    first = (state.get("first_name") or "").strip()
    last = (state.get("last_name") or "").strip()
    name = f"{first} {last}".strip()
    if name:
        lines.append(f"Member name: {name}")
    member_email = (state.get("email") or "").strip()
    if member_email:
        lines.append(f"Member email on file: {speak_email(member_email)}")
    phone_number = (state.get("phone_number") or "").strip()
    if phone_number and state.get("phone_confirmed"):
        lines.append(f"Phone number on file (confirmed this call): {phone_number}")

    # ── Benefits ──────────────────────────────────────────────────────────────
    indv_ded = state.get("individual_deductible") or ""
    fam_ded = state.get("family_deductible") or ""
    coins = state.get("coinsurance_percent") or ""
    indv_oop = state.get("individual_oop_max") or ""
    fam_oop = state.get("family_oop_max") or ""

    if any([indv_ded, fam_ded, coins, indv_oop, fam_oop]):
        lines.append("\nBenefits:")
        if indv_ded:
            lines.append(f"  Individual deductible: ${indv_ded} per calendar year")
        if fam_ded:
            lines.append(f"  Family deductible: ${fam_ded}")
        if coins:
            lines.append(f"  Coinsurance: {coins}% after deductible is met")
        if indv_oop:
            lines.append(f"  Individual out-of-pocket maximum: ${indv_oop} per year")
        if fam_oop:
            lines.append(f"  Family out-of-pocket maximum: ${fam_oop}")
        lines.append(
            "  Once the out-of-pocket maximum is reached, the plan pays "
            "100% of covered in-network services for the rest of the year."
        )
        lines.append("  Member is eligible for a free health and wellness coach.")

    # ── Provider search ───────────────────────────────────────────────────────
    provider_type = state.get("provider_type") or ""
    zip_used = state.get("zip_code_used") or state.get("zip_code") or ""
    if provider_type or zip_used:
        lines.append("\nProvider search:")
        if provider_type:
            lines.append(f"  Provider type requested: {provider_type}")
        if zip_used:
            lines.append(f"  ZIP code used: {zip_used}")
        if state.get("provider_list_sent"):
            lines.append("  In-network provider list was sent to the member this call.")

    # ── Delivery ──────────────────────────────────────────────────────────────
    method = (state.get("delivery_method") or "").strip()
    fax = (state.get("fax") or "").strip()
    email = (state.get("email") or "").strip()
    if method == "fax" and fax:
        lines.append(f"\nDelivery: sent by fax to {fax}")
    elif method == "email" and email:
        lines.append(f"\nDelivery: sent by email to {speak_email(email)}")

    # ── Care Coach ────────────────────────────────────────────────────────────
    if state.get("care_coach_details_sent"):
        contact = fax if method == "fax" else speak_email(email)
        lines.append(
            "\nCare Coach details were sent to the member"
            + (f" ({method}: {contact})" if contact else "")
            + " this call."
        )
    elif state.get("care_coach_offered"):
        lines.append("\nCare Coach offer was presented to the member this call.")

    # ── Claim Adjustment ──────────────────────────────────────────────────────────
    ref = (state.get("reference_number") or "").strip()
    claim_status = (state.get("claim_status") or "").strip()
    last_update_date = (state.get("last_update_date") or "").strip()

    lines.append(
        "\nWellness rewards / incentive points portal: "
        f"{speak_url('www.mysagilityhealth.com')} → My Wellness section. "
        "(Always say the website exactly in this spoken form.)"
    )

    if any([ref, claim_status]):
        lines.append("\nClaim adjustment:")
        if ref:
            lines.append(f"  Reference number: {ref}")
        if claim_status:
            lines.append(f"  Status: {claim_status}")
        if last_update_date:
            lines.append(f"  Last update date: {last_update_date}")
        records_branch = (state.get("records_branch_taken") or "").strip()
        if records_branch:
            lines.append(f"  Records coordination: {records_branch}")
            if records_branch == "declined_personal_guide":
                lines.append(
                    "  Records coordination was declined by the member — "
                    "this call's claim assistance is complete."
                )
        if state.get("upload_link_sent"):
            email_dest = (state.get("email") or "").strip()
            lines.append(
                "  Upload link was sent to member"
                + (f" ({speak_email(email_dest)})" if email_dest else "")
                + " this call."
            )
        if state.get("personal_guide_outreach_requested"):
            lines.append("  Personal Guide will contact the provider within 24 hours.")
        notif_channel = (state.get("notification_channel") or "").strip()
        notif_contact = (state.get("claim_notification_contact") or "").strip()
        if notif_channel and notif_channel != "not_set":
            spoken_contact = speak_email(notif_contact) if "@" in notif_contact else notif_contact
            lines.append(
                f"  Notifications: {notif_channel}" + (f" to {spoken_contact}" if spoken_contact else "")
            )
        lines.append("  Resolution timeline: 5 to 10 business days from receipt of required information.")

    return "\n".join(lines)


async def extract_follow_up_decision(
    llm,
    system_prompt: str,
    *,
    last_agent_message: str,
    last_user_message: str,
    pending_slots: list[str] | None = None,
    recent_messages: list | None = None,
    state: State | None = None,
) -> FollowUpResult:
    """
    Single LLM call: classifies guard + intent AND generates answer if needed.

    Session snapshot is injected into the user content so the model can answer
    questions directly without a second generation call.

    The answer field on WorkerResult carries the generated response back.
    Falls back to an empty WorkerResult on any exception.
    """
    session_snapshot = _build_session_snapshot(state) if state else ""

    messages = build_worker_input(
        system_prompt,
        awaiting_slot="",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        pending_slots=pending_slots,
        recent_messages=recent_messages,
    )

    # Inject session snapshot into the user message content so the model
    # has all the information it needs to generate an answer in one pass.
    if session_snapshot and messages:
        messages[-1]["content"] = f"SESSION SNAPSHOT:\n{session_snapshot}\n\n" + messages[-1]["content"]

    try:
        result: FollowUpResult = await llm.with_structured_output(FollowUpResult).ainvoke(messages)
        return result
    except Exception:
        logger.exception("extract_follow_up_decision: LLM call failed")
        return FollowUpResult()
