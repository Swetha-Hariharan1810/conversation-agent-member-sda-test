"""constants.py — Configuration constants for BenefitsAgent."""

from __future__ import annotations

AGENT_NAME = "benefits_agent"

LOG_ENTERED = "benefits_agent: entered"
LOG_BENEFITS_FETCHED = "benefits_agent: benefits fetched from Salesforce"
LOG_BENEFITS_EXPLAINED = "benefits_agent: benefits explained to member"

# ── Benefits explanation template ─────────────────────────────────────────────
# Placeholders: {individual_deductible}, {family_deductible},
#               {coinsurance_percent}, {individual_oop_max}, {family_oop_max}
BENEFITS_EXPLANATION_TEMPLATE = (
    "Your individual deductible is ${individual_deductible} per calendar year. "
    "Once it's met, benefits apply at the plan's standard cost-share.\n\n"
    "Your plan also has a family deductible of ${family_deductible}, "
    "with {coinsurance_percent}% coinsurance for covered services until the "
    "share limit is met.\n\n"
    "Deductible amounts count toward your plan's out-of-pocket maximum.\n\n"
    "Your individual out-of-pocket maximum is ${individual_oop_max} per year, "
    "and the family out-of-pocket maximum is ${family_oop_max}.\n\n"
    "Once these limits are reached, your plan pays 100% of covered in-network "
    "services for the remainder of the year."
)

# ── Care Coach proactive offer ────────────────────────────────────────────────
CARE_COACH_OFFER_TEMPLATES = [
    "By the way, you are eligible for a free health and wellness coach who can "
    "help you understand your medications and doctor's orders. Would you like us "
    "to send you details about our Care Coach Guides?",
    "I also want to let you know you have access to a free personal health coach. "
    "They can help with medications and doctor's instructions. Want me to send "
    "you the details?",
    "One more thing — you're eligible for a complimentary health and wellness "
    "coach. Would you like me to send you information on how to get started?",
]


BENEFITS_NOEXPLANATION_TEMPLATES = [
    "By the way, you are eligible to have a free health and wellness coach who "
    "can help you with understanding your medications and doctor's orders. Our "
    "coaching can also guide you in reducing any health risks you may have. "
    "Do you want us to send the details of our Care Coach Guides?",
]


# ── Escalation / fallback ─────────────────────────────────────────────────────
MSG_BENEFITS_FETCH_FAIL = [
    "I'm sorry, I wasn't able to retrieve your plan details right now. Let me connect you with a specialist.",
    "I'm having trouble pulling up your benefits. Let me transfer you to a representative who can help.",
]

MSG_CARE_COACH_OFFER_EXHAUST = [
    "No problem. Is there anything else I can help you with?",
]
