"""constants.py — Configuration constants for ClaimAdjustmentAgent."""

from __future__ import annotations

AGENT_NAME = "claim_adjustment_agent"

LOG_ENTERED = "claim_adjustment_agent: entered"
LOG_REF_COLLECTED = "claim_adjustment_agent: reference_number collected"
LOG_STATUS_REPORTED = "claim_adjustment_agent: claim status reported to member"

# ── Reference number collection bridge ────────────────────────────────────────
REFERENCE_NUMBER_BRIDGE_MSGS = [
    "May I have the reference number of the adjustment request?",
    "Could you provide the reference number for your adjustment?",
    "I'll need your adjustment request reference number — go ahead whenever you're ready.",
    "What is the reference number for this adjustment request?",
]

# ── Status report templates ────────────────────────────────────────────────────
# Placeholders: {status}, {last_update_date}
STATUS_REPORT_TEMPLATES = [
    (
        "Thank you. I found the request. "
        "The claim is still {status}. "
        "The last update on record was {last_update_date}."
    ),
    (
        "I found the adjustment request. "
        "Current status: {status}. "
        "Our last recorded update was on {last_update_date}."
    ),
    (
        "Got it. The request is currently {status}. "
        "The most recent update on file is dated {last_update_date}."
    ),
]

# ── Records needed message ─────────────────────────────────────────────────────
MSG_RECORDS_NEEDED = [
    "We also need a copy of the complete medical records for this request. Can you send it over?",
    "To move forward, we'll need a complete copy of the medical records for this adjustment. "
    "Are you able to provide those?",
    "Our team also requires a complete set of medical records for this request. "
    "Could you arrange to have those sent over?",
]

# ── Timeline explanation ───────────────────────────────────────────────────────
TIMELINE_MSG = [
    ("It will take 5 to 10 business days from the date we receive the required additional information."),
    (
        "Once we have the required documents on file, the adjustment typically "
        "resolves within 5 to 10 business days."
    ),
    (
        "After we receive the complete medical records, "
        "you can expect a resolution within 5 to 10 business days."
    ),
]

# ── Escalation messages ────────────────────────────────────────────────────────
MSG_REF_NOT_FOUND = [
    "I wasn't able to find an adjustment request with that reference number. "
    "Let me connect you with a representative who can help.",
    "I couldn't locate an adjustment request matching that reference number. "
    "I'm going to connect you with a specialist who can assist.",
    "That reference number didn't match any open adjustment request on file. "
    "Let me transfer you to a representative for further assistance.",
]

MSG_REF_EXHAUST = [
    "I wasn't able to capture a valid reference number after a few attempts. "
    "Let me connect you with a representative who can assist.",
    "I wasn't able to get a valid reference number. Connecting you with a specialist now.",
]

# ── Reference not found — retry prompt (one attempt before escalating) ────────
MSG_REF_NOT_FOUND_RETRY = [
    "I wasn't able to find a request with that reference number. "
    "Could you double-check and provide it again?",
    "That reference number didn't match any open request on file. "
    "Could you verify the number and try once more?",
    "I couldn't locate a request with that number. Please check the reference number and provide it again.",
]

# ── Reference number fallback: ask for claim number ──────────────────────────
MSG_REF_FALLBACK_CLAIM_NUMBER_ASK = [
    "No problem. I can check another way. Do you have the claim number?",
    "That's okay. I can look it up differently — do you happen to have the claim number?",
    "No worries. I can try another approach. Do you have the claim number available?",
]

# Used when the member confirms they HAVE the claim number (bare affirmative) — no extraction failure.
MSG_REF_FALLBACK_CLAIM_NUMBER_HAVE_IT = [
    "Perfect — could you go ahead and share that claim number?",
    "Great! Please go ahead and provide the claim number.",
    "Thank you — could you give me the claim number now?",
]

MSG_REF_FALLBACK_CLAIM_NUMBER_RETRY = [
    "I wasn't able to capture that claim number. Could you repeat it for me?",
    "I didn't quite catch the claim number — could you say it once more?",
    "Could you provide the claim number once more? I want to make sure I have it right.",
]

# ── Reference number fallback: ask for date of service + billed amount ────────
MSG_REF_FALLBACK_DOS_BILLED_ASK = [
    "That's okay. What was the date of service and the billed amount?",
    "No problem. Could you give me the date of service and the billed amount?",
    "Sure — what date was the service, and what was the billed amount?",
]

# Used when the member confirms they HAVE the DOS+billed info (bare affirmative).
MSG_REF_FALLBACK_DOS_BILLED_HAVE_IT = [
    "Perfect — please go ahead with the date of service and the billed amount.",
    "Great! Could you share the date of service and the billed amount?",
    "Thank you — please provide the date of service and the billed amount.",
]

MSG_REF_FALLBACK_DOS_BILLED_RETRY = [
    "I need both the date of service and the billed amount to locate the claim. Could you provide those?",
    "Thank you — could you give me both the date of service and the billed amount?",
    "Could you give me the date of service and the billed amount? I need both to find your record.",
]

# ── Fallback lookup not found ─────────────────────────────────────────────────
MSG_REF_FALLBACK_NOT_FOUND = [
    "I wasn't able to find an adjustment request with those details. "
    "Let me connect you with a representative who can help.",
    "I couldn't locate a matching adjustment request with that information. "
    "I'm going to connect you with a specialist.",
]
