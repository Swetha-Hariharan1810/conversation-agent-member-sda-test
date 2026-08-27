"""
Configuration constants for VerificationAgent.

IMPORTANT:
This module centralizes all fixed values:
- retry limits
- slot ordering
- error messages
- logging event names

Do NOT place:
- orchestration logic
- prompt strings
- LLM schemas
inside this file.
"""

from __future__ import annotations

# =========================================================
# Retry Limits
# =========================================================

MAX_LOOKUP_ATTEMPTS = 2

# Maximum rejection cycles before escalating the name confirmation loop.
# One cycle = one readback delivered → member rejects → (correction collected or not).
# Successful confirmation never increments this counter.
MAX_NAME_CONFIRM_ATTEMPTS = 2

# =========================================================
# Agent Identity
# =========================================================

VERIFICATION_AGENT_NAME = "verification_agent"

# =========================================================
# Slot Ordering
# =========================================================

IDENTITY_SLOT_ORDER = [
    "first_name",
    "last_name",
    "member_id",
    "dob",
]

# =========================================================
# Completion Message Template
# =========================================================

VERIFIED_MSG_TEMPLATES = [
    "Thank you, {first_name}. I've verified your account.",
    "Got it, {first_name} — your account is verified.",
    "Perfect, {first_name}. I've confirmed your identity.",
    "All set, {first_name}. Your account is verified.",
]

# =========================================================
# Logging Events
# =========================================================

LOG_ENTERED = "verification_agent: entered"
LOG_VERIFIED = "VerificationAgent: fully verified — signalling complete"
LOG_LOOKUP_FAIL = "VerificationAgent: SF lookup failed"
LOG_SLOT_CORRECTED = "VerificationAgent: slot corrected"
LOG_LLM_EXTRACT_FAIL = "VerificationAgent: LLM extraction failed — using empty decision"
LOG_INVALID_MEMBER_ID = "VerificationAgent: invalid member ID blocked"
LOG_INVALID_DOB = "VerificationAgent: invalid DOB blocked"

# =========================================================
# Name readback / confirmation messages
# =========================================================
# Placeholder: {spelled} — full name spelled with hyphens, e.g. "E-M-I-L-Y  C-A-R-T-E-R"

NAME_READBACK_TEMPLATES = [
    "Thank you. Just to confirm — is your name {spelled}, correct?",
    "Got it. So that's {spelled} — is that right?",
    "Thank you. Just to confirm {spelled}. Is that correct?",
]

# Used after bare "no" — agent asks for the correct name before re-reading back.
NAME_CORRECTION_PROMPTS = [
    "No problem — could you give me the correct first and last name?",
    "Got it — what is the correct name on the account?",
    "Sure, what is the correct name?",
]

# Escalation message when MAX_NAME_CONFIRM_ATTEMPTS is reached.
MSG_NAME_CONFIRM_EXHAUST = [
    "I wasn't able to confirm the name on the account after a few tries. "
    "Let me connect you with a representative who can assist.",
    "I wasn't able to verify the name after several attempts. Connecting you with a specialist now.",
]

# =========================================================
# Partial re-ask messages (targeted identity correction)
# =========================================================
# Delivered when the Member ID was found but one identity field didn't match.
# Disclosing style (Phase 0 decision): name the single mismatched field so the
# caller knows exactly what to restate. Used by lookup_and_verify's partial
# re-ask path; matched fields and the Member ID are preserved.

MSG_REASK_DOB = [
    "Thank you. Everything matched except the date of birth — "
    "could you tell me your date of birth once more?",
    "I found your account, but the date of birth didn't quite match. What is your date of birth?",
    "Almost there — the only detail that didn't match was the date of birth. "
    "Could you confirm your date of birth for me again?",
]

MSG_REASK_LAST_NAME = [
    "Thank you. Everything matched except the last name — could you give me your last name once more?",
    "I found your account, but the last name didn't quite match. Could you confirm your last name for me?",
    "Almost there — the only detail that didn't match was the last name. "
    "Could you spell your last name for me again?",
]

MSG_REASK_FIRST_NAME = [
    "Thank you. Everything matched except the first name — could you give me your first name once more?",
    "I found your account, but the first name didn't quite match. Could you confirm your first name for me?",
    "Almost there — the only detail that didn't match was the first name. "
    "Could you tell me your first name again?",
]

# Non-disclosing fallback: used when more than one field mismatched, so we don't
# enumerate every wrong detail back to the caller.
MSG_REASK_GENERIC = [
    "Thank you. A couple of details didn't quite match — "
    "could you confirm your name and date of birth for me again?",
    "I found your account, but some of the details didn't match. "
    "Let's recheck them — could you give me your name and date of birth once more?",
    "We're almost there — a couple of details didn't line up. "
    "Could you confirm your name and date of birth again?",
]

# ── New log labels ────────────────────────────────────────────────────────────
LOG_PARTIAL_REASK = "VerificationAgent: partial re-ask — clearing only mismatched slots"
LOG_NAME_READBACK = "VerificationAgent: name readback delivered"
LOG_NAME_CONFIRMED = "VerificationAgent: name confirmed by member"
LOG_NAME_CORRECTED = "VerificationAgent: name corrected by member"
LOG_NAME_CONFIRM_EXHAUST = "VerificationAgent: name confirmation exhausted — escalating"

# =========================================================
# SSN Fallback
# =========================================================

MSG_SSN_ASK = "No problem. I can check another way. Do you have the SSN?"

MSG_SSN_COLLECT = [
    "Perfect — please go ahead and provide your SSN.",
    "Great — could you share your Social Security number?",
    "Thank you — please provide your SSN",
]

MSG_SSN_DOB = "Thank you. Could you also provide your date of birth?"

MSG_SSN_INVALID = (
    "I'm having a little trouble with that — could you provide your SSN in the format XXX-XX-XXXX?"
)

MSG_SSN_RETRY_EXHAUSTED = (
    "I'm having trouble with that SSN. Let me connect you with a representative who can help."
)

MSG_SSN_EITHER = (
    "I need either your Member ID or your SSN to continue. "
    "If you have either one available, please provide it."
)

MSG_SSN_ESCALATE = (
    "I'm unable to verify your account without either your Member ID or your SSN. "
    "I'll connect you with a representative who can assist you further."
)

MSG_SSN_SUCCESS = "Thank you, I found your account. Are you the subscriber or dependent?"

MAX_SSN_ATTEMPTS = 3

# Member ID denial phrases (deterministic detection)
MEMBER_ID_DENIAL_PHRASES = (
    "don't have",
    "dont have",
    "i don't have it",
    "i dont have it",
    "don't know",
    "dont know",
    "i don't know",
    "cant find",
    "can't find",
    "i can't find",
    "no member id",
    "don't have my member",
    "dont have my member",
    "not sure",
    "have no idea",
    "lost it",
)

# Phrases indicating the user definitely has no SSN to provide (escalate)
NO_SSN_AVAILABLE_PHRASES = (
    "don't have it",
    "dont have it",
    "don't have my ssn",
    "dont have my ssn",
    "neither",
    "don't know it",
    "dont know it",
    "can't access",
    "cant access",
    "lost it",
    "i can't",
    "i cant",
    "i don't",
    "i dont",
    "no ssn",
)

YES_PHRASES = frozenset({"yes", "yeah", "yep", "sure", "okay", "ok", "yup"})
NO_PHRASES = frozenset({"no", "nope", "nah"})
