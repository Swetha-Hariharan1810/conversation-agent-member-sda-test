"""constants.py — Configuration constants for CareWellnessAgent."""

from __future__ import annotations

AGENT_NAME = "care_wellness_agent"

LOG_ENTERED = "care_wellness_agent: entered"
LOG_DETAILS_SENT = "care_wellness_agent: care coach details dispatched"
LOG_PORTAL_SHARED = "care_wellness_agent: rewards portal link shared"

WELLNESS_PORTAL_URL = "www.mysagilityhealth.com"

# ── Care Coach intro message ──────────────────────────────────────────────────
CARE_COACH_INTRO_TEMPLATES = [
    (
        "Great! You can also set up an appointment for a consultation at the link "
        "I will include in the details.\n\n"
        "I'll send it to the same {method} {contact} you provided. "
        "Please expect to receive it within 30 minutes.\n"
        "Is there anything else I can help you with?"
    ),
    (
        "Wonderful! I'll include a consultation appointment link with the details.\n\n"
        "Sending it to your {method} at {contact} — you should receive it within 30 minutes.\n"
        "Is there anything else I can help you with?"
    ),
]

CARE_COACH_NOOFFER_TEMPLATES = [
    "No problem. Is there anything else I can help you with?",
]

# ── Rewards portal message ─────────────────────────────────────────────────────
REWARDS_PORTAL_TEMPLATES = [
    (
        "You can track your wellness incentives and reward points on our member portal "
        "at {url}, under the My Wellness section."
    ),
    (
        "For wellness rewards and incentive tracking, visit {url} and look for the "
        "My Wellness section in your member account."
    ),
]

# ── No delivery contact fallback ──────────────────────────────────────────────
MSG_NO_CONTACT = [
    "I'm sorry, I don't have a delivery address on file. "
    "Let me connect you with a representative who can help.",
]

MSG_DISPATCH_FAIL = [
    "I'm sorry, I was unable to send the Care Coach details. Let me connect you with a representative.",
    "I wasn't able to dispatch the Care Coach information. Connecting you with a specialist now.",
]
