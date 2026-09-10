"""constants.py — Configuration for IntakeAgent."""

from __future__ import annotations

MAX_CLARIFICATION_ATTEMPTS = 2

# Regulatory: contractual opening statement — must not be LLM-generated
# GREETING = (
#     "Thank you for calling Sagility Health. "
#     "For quality assurance purposes, your call may be monitored or recorded. "
#     "At the end of your call today, there will be a short survey to provide "
#     "feedback regarding the representative you spoke with. "
#     "Your participation is greatly appreciated. "
#     "Please tell me how I can assist you today."
# )
GREETING = (
    "Thank you for calling Sagility Health. "
    "This call may be monitored or recorded for quality. "
    "How can I help today?"
)

INTENT_SLOT = "intent"

UNCLEAR_INTENT_REASON = "Intent could not be classified after maximum clarification attempts"
OFFTOPIC_REASON = "Repeated unsupported or off-topic requests"
EXPLICIT_ESCALATION_REASON = "Caller requested human assistance"

LOG_INTAKE_GREETING = "IntakeAgent: delivering greeting"
LOG_INTENT_CLASSIFIED = "IntakeAgent: intent classified"
LOG_INTENT_UNCLEAR = "IntakeAgent: intent unclear"
LOG_OFFTOPIC = "IntakeAgent: off-topic request"
LOG_MAX_RETRY = "IntakeAgent: max clarification reached"
LOG_CLASSIFICATION_FAILURE = "IntakeAgent: classification failure"
LOG_SAME_MEMBER_CHECK = "IntakeAgent: same-member disambiguation triggered"
LOG_SAME_MEMBER_CONFIRMED = "IntakeAgent: same member confirmed — restoring verification context"
LOG_DIFFERENT_MEMBER = "IntakeAgent: different member indicated — routing to verification"
LOG_SAME_MEMBER_AMBIGUOUS = "IntakeAgent: same-member response ambiguous — asking clarification"

INTENT_BRIDGE_MSGS = [
    "I can definitely help with that. To get started, could I get your first name?",
    "Of course — happy to help. Can I start with your first name?",
    "Absolutely. Let me pull that up for you — first, what's your first name?",
    "Sure thing. Could I start with your first name?",
]
# Backward-compatible sentinel: common substring present in every pool message.
INTENT_BRIDGE_MSG = "your first name?"

# ── Same-member disambiguation (follow-up re-entry) ───────────────────────────
# Delivered when intake detects that the previous agent was follow_up, the new
# intent is PCP or Claims, and there is a verified member context on record.
SAME_MEMBER_CHECK_QUESTION = (
    "Is this request for the same member we've been discussing, or is this for a different member?"
)
SAME_MEMBER_CLARIFICATION_MSGS = [
    "Just to confirm — is this request for the same member as before, or for someone new?",
    "Could you clarify — is this for the member we already have on file, or a different person?",
]

# Escalation handoff messages — delivered at moment of escalation
UNCLEAR_ESCALATION = "Not a problem — let me connect you with a representative who can assist you further."
OFFTOPIC_ESCALATION = "Let me connect you with a representative who may better assist you."

UNCLEAR_FIRST_ATTEMPT_MSGS = [
    "I'd be happy to help. Could you tell me a little more about what you need today?",
    "Of course. What can I help you with?",
    "Sure thing. What brings you in today?",
    "Happy to help — what would you like assistance with?",
]

# Out-of-scope handling
LOG_OUT_OF_SCOPE = "IntakeAgent: out-of-scope intent detected"

OUT_OF_SCOPE_REASON = "Caller intent is outside covered workflows"

# Keyword → (human-readable team label, phone number)
# Checked against the caller's last utterance using substring match.
# Order matters — more specific phrases must come before shorter ones.
# e.g. "insurance card" before "card", "prior approval" before "prior"
OUT_OF_SCOPE_KEYWORD_ROUTING: list[tuple[str, str, str]] = [
    # appeals — must come before all other entries to prevent partial shadowing
    ("appeal", "our appeals team", "1-800-555-0105"),
    # billing
    ("insurance card", "our member services team", "1-800-555-0102"),
    ("member card", "our member services team", "1-800-555-0102"),
    ("id card", "our member services team", "1-800-555-0102"),
    ("prior approval", "our authorizations team", "1-800-555-0103"),
    ("prior auth", "our authorizations team", "1-800-555-0103"),
    ("billing", "our billing team", "1-800-555-0101"),
    ("invoice", "our billing team", "1-423-872-2404"),
    ("payment", "our billing team", "1-423-872-24041"),
    ("pay my", "our billing team", "1-800-555-0101"),
    ("my bill", "our billing team", "1-800-555-0101"),
    ("referral", "our authorizations team", "1-800-555-0103"),
    ("authorization", "our authorizations team", "1-800-555-0103"),
    ("prescription", "our pharmacy benefits team", "1-800-555-0104"),
    ("pharmacy", "our pharmacy benefits team", "1-800-555-0104"),
    ("medication", "our pharmacy benefits team", "1-800-555-0104"),
    ("drug", "our pharmacy benefits team", "1-800-555-0104"),
    ("enrol", "our member services team", "1-800-555-0102"),
    ("enroll", "our member services team", "1-800-555-0102"),
    ("cancel", "our member services team", "1-800-555-0102"),
    ("coverage", "our member services team", "1-800-555-0102"),
]

# Fallback when no keyword matches
OUT_OF_SCOPE_FALLBACK_TEAM = "our support team"
OUT_OF_SCOPE_FALLBACK_NUMBER = "1-800-555-0102"

# Message templates. Placeholders:
#   {topic_description} — what the caller said, in their words
#   {team}              — human-readable team name from routing table
#   {number}            — phone number from routing table
OUT_OF_SCOPE_MSG_TEMPLATES = [
    (
        "I understand you're calling about {topic_description}. "
        "That's handled by {team}, Their number is {number}, — I'll connect you now."
    ),
    (
        "Got it — {topic_description} is something {team} can help you with."
        "Their number is {number},"
        "Let me transfer you to them now. "
    ),
    (
        "Thanks for letting me know. For {topic_description}, "
        "you'll want to speak with {team}. "
        "Their number is {number}, I will connect you with them now"
    ),
]

# ── Unsupported provider type — escalation at intake ─────────────────────────
# Fires when the caller names a medical specialty outside the five supported types.
# Mirrors MSG_PROVIDER_TYPE_UNSUPPORTED in provider_search/constants.py so the
# member hears a consistent message regardless of where the check fires.
# Placeholder: {provider_type} — filled with what the caller said.

PROVIDER_TYPE_UNSUPPORTED_REASON = "provider_type_unsupported_at_intake"
LOG_PROVIDER_TYPE_UNSUPPORTED = "IntakeAgent: unsupported provider type at intake — escalating"

PROVIDER_TYPE_UNSUPPORTED_ESCALATION = [
    (
        "I can see you're looking for a {provider_type}. Unfortunately, "
        "our in-network search currently covers Primary Care Physicians, "
        "Pediatricians, Cardiologists, Dermatologists, and Orthopedic "
        "Specialists. Let me connect you with a representative who can "
        "help you find the right provider."
    ),
    (
        "For {provider_type} searches I'll need to connect you with one "
        "of our representatives — our self-service search covers Primary "
        "Care Physicians, Pediatricians, Cardiologists, Dermatologists, "
        "and Orthopedic Specialists at this time."
    ),
]
