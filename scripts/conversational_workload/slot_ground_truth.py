"""
Maps slot labels to expected user responses for the PCP and claim-adjustment
evaluation flows.

Scenario-specific overrides handle clarification re-asks and mid-conversation
corrections. turn_counters tracks how many times each (scenario_tag, slot)
pair has been visited within a single run so the override can return a
different response on the second visit.
"""

from __future__ import annotations


def _pcp_map(entity) -> dict:
    provider = getattr(entity, "provider_type", None) or "Primary Care Physician"
    fax = getattr(entity, "fax_number", None) or "6175554199"
    zip_code = getattr(entity, "zip_code", None) or "12139"

    return {
        "intent_selection": lambda: "Can I get a list of in-network providers, please?",
        "first_name": lambda: entity.first_name,
        "last_name": lambda: entity.last_name,
        "member_id": lambda: entity.member_id,
        "dob": lambda: entity.date_of_birth,
        "subscriber_confirm": lambda: "I'm calling for myself.",
        "provider_type": lambda: f"I'm looking for a {provider}.",
        "zip_confirm": lambda: "Yes, that's correct.",
        "zip_update": lambda: zip_code,
        "fax_or_email": lambda: "Please send it to my fax.",
        "fax_confirm": lambda: "Yes, that fax number is correct.",
        "fax_update": lambda: fax,
        "benefits_offer": lambda: "Yes please.",
        "benefits_explanation": lambda: "Yes please.",
        "coach_offer": lambda: "Yes, that sounds interesting.",
        "follow_up": lambda: "Can you summarize the PCP benefits?",
        "closing": lambda: "No, that was very helpful. Thank you.",
        "name_confirmed": lambda: "yes that's correct",
        "clarification": lambda: f"Sorry about that — {entity.member_id}.",
        "correction_ack": lambda: "Yes, that's right now. Thank you.",
        "other": lambda: "Okay.",
    }


def _claim_map(entity) -> dict:
    # email = getattr(entity, "email", None) or ""

    return {
        "intent_selection": lambda: "I adjusted a claim and I want to follow up",
        "first_name": lambda: entity.first_name,
        "last_name": lambda: entity.last_name,
        "member_id": lambda: entity.member_id,
        "dob": lambda: entity.date_of_birth,
        "phone_confirmed": lambda: "yes correct",
        "reference_number": lambda: entity.reference_number,
        "records_method": lambda: "Can I ask my doctor to send it over?",
        "upload_consent": lambda: "Yes, please",
        "email_confirmed_upload": lambda: "Yes, that's correct",
        "personal_guide_consent": lambda: "Perfect. Please do that",
        "notification_method": lambda: "You can send me to my phone",
        "timeline_question": lambda: "Okay, how long will it take to finalize the request?",
        "n2_notification_method": lambda: "email them to me",
        "upload_link_sent": lambda: "Perfect. Please do that",
        "guide_scheduled": lambda: "You can send me to my phone",
        "follow_up": lambda: "Yes, can you tell me where I can see how many rewards"
        " I earned from my annual check up last week?",
        "closing": lambda: "No, that's it for me. Thanks!",
        "name_confirmed": lambda: "yes that's correct",
        "clarification": lambda: f"Sorry about that — {entity.member_id}.",
        "correction_ack": lambda: "Yes, that's right now. Thank you.",
        "other": lambda: "Okay.",
    }


def ground_truth_for_slot(
    slot: str,
    entity,
    flow: str = "pcp",
    scenario_tag: str = "",
    turn_counters: dict | None = None,
) -> str:
    if turn_counters is None:
        turn_counters = {}

    visit = turn_counters.get((scenario_tag, slot), 0)

    # ── claim_adjustment_no_proceed ───────────────────────────────────────
    if scenario_tag == "claim_adjustment_no_proceed":
        if slot == "upload_consent" and visit == 0:
            return "no thanks"
        if slot == "personal_guide_consent" and visit == 0:
            return "no i dont want to proceed"

    # ── claim_adjustment_happy_path ───────────────────────────────────────
    if scenario_tag == "claim_adjustment_happy_path":
        if slot == "n2_notification_method":
            return "email them to me"
        if slot == "follow_up":
            return (
                "Yes, can you tell me where I can see how many rewards"
                " I earned from my annual check up last week?"
            )

    # ── Claim flow default ─────────────────────────────────────────────────
    if flow == "claim" or scenario_tag.startswith("claim_"):
        claim_map = _claim_map(entity)
        handler = claim_map.get(slot, lambda: "Okay.")
        return handler()

    # ── pcp_clarification_zip ──────────────────────────────────────────────
    if scenario_tag == "pcp_clarification_zip":
        if slot == "zip_confirm":
            return "Hmm, let me check." if visit == 0 else "Yes, that's correct."

    # ── pcp_correction_first_name ──────────────────────────────────────────
    # if scenario_tag == "pcp_correction_first_name":
    #     if slot == "last_name" and visit == 0:
    #         return "Actually my first name is Emilia, E-M-I-L-I-A. Last name is Carter."
    #     if slot == "correction_ack" and visit == 0:
    #         return entity.member_id

    # ── pcp_correction_member_id ───────────────────────────────────────────
    if scenario_tag == "pcp_correction_member_id":
        if slot == "member_id" and visit == 0:
            return "m nine oh seven five oh two"
        if slot == "member_id" and visit == 1:
            return "Sorry, that's wrong. It's m nine zero seven five zero three."
        if slot == "correction_ack" and visit == 0:
            return f"April twelfth {entity.date_of_birth[:4]}"

    # ── pcp_clarification_fax ──────────────────────────────────────────────
    if scenario_tag == "pcp_clarification_fax":
        if slot == "fax_confirm":
            return (
                "I'm not sure if that's the right one."
                if visit == 0
                else "Yes, that is the correct fax number."
            )

    # ── Default ────────────────────────────────────────────────────────────
    pcp_map = _pcp_map(entity)
    handler = pcp_map.get(slot, lambda: "Okay.")
    return handler()
