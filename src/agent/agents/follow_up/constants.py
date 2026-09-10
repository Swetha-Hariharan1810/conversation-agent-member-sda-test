"""constants.py — Configuration constants for FollowUpAgent."""

from __future__ import annotations

AGENT_NAME = "follow_up_agent"

# ── New-intake-intent restart ─────────────────────────────────────────────────
# Top-level, routable intake intents. When one of these is newly detected
# mid-follow-up, the call is fully reset and re-routed through verification.
# Mirrors the routable service flows in intake.models.IntentTag.
INTAKE_INTENTS: frozenset[str] = frozenset({"provider_services", "claim_services"})

# Subset of INTAKE_INTENTS that must be handed back to the *intake* agent rather
# than straight to verification, so intake re-applies its front-door screening
# (e.g. the unsupported-provider-type gate that escalates before identity is ever
# collected). Intents NOT listed here stay on the direct-to-verification path.
#
# claim_services is included so that mid-call claim requests from follow_up also
# route through intake, allowing the same-member disambiguation check to fire
# (intake can offer to reuse the previously verified member context rather than
# forcing a full re-verification).
INTAKE_RESCREEN_INTENTS: frozenset[str] = frozenset({"provider_services", "claim_services"})

# Appeal / grievance keyword gate. Appeals and grievances are out_of_scope topics
# (see extraction/intake.md) that intake routes to the appeals/grievance team. The
# follow-up classifier has NO tag for them, and its new_intent branch only fires on
# a cross-intent switch (provider ↔ claim) — so an appeal raised mid-call surfaces
# as a plain `question`. This keyword set lets follow_up detect them directly and
# reroute back through intake for out_of_scope screening, independent of the LLM
# classification. Matched as whole words (word boundaries) to avoid false positives.
APPEAL_GRIEVANCE_KEYWORDS: frozenset[str] = frozenset(
    {"appeal", "appeals", "appealing", "appealed", "grievance", "grievances", "denial", "denials"}
)

# Per-intent "prior flow already completed" flags. Used so a same-intent
# request (e.g. a second, distinct claim) qualifies as a fresh intake while a
# same-intent clarification does not.
FLOW_COMPLETE_FLAGS: dict[str, str] = {
    "claim_services": "claim_flow_complete",
    "provider_services": "provider_list_sent",
}

# ── Escalation thresholds ─────────────────────────────────────────────────────
# UPDATE_REQUEST: immediate escalation every time — no threshold, no counting.
# MSG_CANNOT_ANSWER: escalate after this many consecutive cannot-answer turns.
MAX_CANNOT_ANSWER_BEFORE_ESCALATE: int = 3

# ── Log labels ────────────────────────────────────────────────────────────────
LOG_ENTERED = "follow_up_agent: entered"
LOG_ANSWERED = "follow_up_agent: answered from session context"
LOG_CLOSURE = "follow_up_agent: closure signal detected"
LOG_CANNOT_ANSWER = "follow_up_agent: LLM could not answer from context"
LOG_NEW_INTENT = "follow_up_agent: new intent detected — restarting flow"

# State fields that must be cleared when a new intent is detected mid-call.
# Identity and verification fields are intentionally excluded.
NEW_INTENT_CLEAR_FIELDS: list[str] = [
    # Call intent
    "call_intent",
    # Slot tracking
    "awaiting_slot",
    "slot_attempts",
    "correction_return_to",
    "ambiguous_counts",
    "wait_count",
    "parked_followups",
    "pending_cross_agent_request",
    "pending_slot_update",  # legacy key — cleared for old checkpoints
    "slot_update_resume",
    # Provider search
    "provider_type",
    "zip_code_used",
    "zip_code_updated",
    "provider_list_sent",
    "delivery_timestamp",
    "fax_confirmed",
    "fax_update_requested",
    "email_confirmed",
    "email_update_requested",
    "delivery_method",
    "benefits_offer_made",
    # Benefits & Wellness
    "individual_deductible",
    "family_deductible",
    "coinsurance_percent",
    "individual_oop_max",
    "family_oop_max",
    "benefits_explained",
    "care_coach_offer_made",
    "care_coach_offered",
    "care_coach_details_sent",
    "rewards_portal_shared",
    "care_coach_nooffer_sent",
    # Follow-up agent counters
    "follow_up_turn_count",
    "follow_up_last_question",
    "follow_up_cannot_answer_count",
    # Claim adjustment
    "reference_number",
    "claim_status",
    "last_update_date",
    "records_required",
    "ref_no_fallback_stage",
    "fallback_claim_number",
    "fallback_dos",
    "fallback_billed_amount",
    "records_branch_taken",
    "upload_link_sent",
    "personal_guide_outreach_requested",
    "notification_channel",
    "claim_notification_contact",
    "claim_timeline_notification_channel",
    "claim_timeline_notification_contact",
    "claim_flow_complete",
    # Delivery management
    "pending_zip_code",
    "pending_fax",
    "pending_email",
    "pending_phone",
    # Orchestration
    "new_intent_detected",
    "intent_queue",
    "proactive_offer_available",
    "closure_requested",
    "router_loop_count",
    "orchestrator_reasoning",
    # NOTE: last_agent_signal is deliberately NOT cleared here. _build() applies
    # context_updates AFTER writing result["last_agent_signal"], so listing it
    # would clobber the COMPLETE signal that carries new_intent_detected — which
    # the fast-path needs to route to the new domain agent.
    "previous_agents",
]

# ── Closure keywords ──────────────────────────────────────────────────────────
CLOSURE_KEYWORDS: frozenset[str] = frozenset(
    {
        "no",
        "nope",
        "nah",
        "no thank",
        "no thanks",
        "that's all",
        "thats all",
        "that's it",
        "thats it",
        "nothing else",
        "nothing more",
        "all good",
        "all set",
        "i'm good",
        "im good",
        "i'm done",
        "im done",
        "done",
        "good",
        "ok",
        "okay",
        "great",
        "perfect",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "that was helpful",
        "that helped",
        "very helpful",
        "have a good",
        "have a great",
    }
)

# Bare affirmations — intercepted before the LLM call.
BARE_AFFIRMATIONS: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "okay",
        "ok",
        "please",
        "yes please",
        "yes thank you",
        "sure please",
    }
)

# ── Message pools ─────────────────────────────────────────────────────────────

MSG_FOLLOW_UP_ASK: list[str] = [
    "Aside from this, is there anything else I can help you with today?",
    "Is there anything else I can help you with today?",
    "Is there anything else from our call today I can help you with?",
]

MSG_NUDGE: list[str] = [
    "Is there a specific question I can help you with, or are you all set for today?",
    "Did you have a specific question, or is there anything else I can help you with?",
    "Is there something specific from our call I can clarify, or are you good to go?",
]

MSG_CONTINUATION: list[str] = [
    "Is there anything else from our call today I can help with?",
    "Do you have any other questions about what we covered?",
    "Is there anything else I can clarify for you?",
]

MSG_CANNOT_ANSWER: list[str] = [
    "I'm sorry, I don't have that information from our call today.",
    "That isn't something we covered during this call, so I don't have those details available.",
    "I don't have that information available from what we discussed today.",
]

# Delivered on every UPDATE_REQUEST turn, immediately before signal_escalate().
# A single fixed string — no variation, no pool.
MSG_UPDATE_REQUEST_ESCALATE: str = (
    "I'm sorry, I'm not able to make changes during this part of the call. "
    "However, I will transfer you to a representative for further assistance."
)

# Max total follow-up turns (safety net).
MAX_FOLLOW_UP_TURNS: int = 10
