"""
constants.py — Configuration constants for DeliveryManagementAgent.
"""

from __future__ import annotations

# ── Agent identity ────────────────────────────────────────────────────────────

AGENT_NAME = "delivery_management_agent"

MAX_CONTACT_CHANGE_CYCLES = 3

# ── Log labels ────────────────────────────────────────────────────────────────

LOG_ENTERED = "delivery_management_agent: entered"
LOG_METHOD_COLLECTED = "delivery_management_agent: delivery_method collected"
LOG_CONTACT_CONFIRMED = "delivery_management_agent: contact confirmed"
LOG_CONTACT_UPDATED = "delivery_management_agent: contact updated"
LOG_LIST_DISPATCHED = "delivery_management_agent: provider list dispatched"

# ── Delivery window ───────────────────────────────────────────────────────────

DELIVERY_WINDOW_MSG = [
    "Great! Expect to receive the list within 30 minutes.",
    "Perfect — you should receive it within 30 minutes.",
    "The list will be sent within 30 minutes.",
]

# Used when the member updated their ZIP code during this call — the
# confirmation explicitly includes the new ZIP so the member hears it
# was applied to the search. Placeholder: {zip_code}
DELIVERY_WINDOW_MSG_ZIP_UPDATED = [
    "Great! Expect to receive the list of in-network providers for your "
    "current ZIP code {zip_code} within 30 minutes.",
    "Perfect — you should receive the list of in-network providers for "
    "your updated ZIP code {zip_code} within 30 minutes.",
]

# ── Benefits offer ────────────────────────────────────────────────────────────

BENEFITS_OFFER_TEMPLATES = [
    (
        "Since you're calling about a list of in-network providers, "
        "would you like to also get the benefits for office visits with your {provider_type}?"
    ),
    ("Would you also like me to go over the benefits for office visits with your {provider_type}?"),
    ("I can also provide your benefits information for {provider_type} visits — would that be helpful?"),
]

# ── Delivery method prompt ────────────────────────────────────────────────────

FAX_CONFIRM_TEMPLATES = [
    "How do you want us to send your request — via fax or email?",
    "Would you prefer to receive the provider list by fax or email?",
]

# ── Contact readback templates ────────────────────────────────────────────────

FAX_READBACK_TEMPLATES = [
    "Definitely. The fax number we have on file is {fax}. Is this correct?",
    "I'll send it to {fax} — is that the right fax number?",
]

EMAIL_READBACK_TEMPLATES = [
    "I'll send it to {email}. Is that the right email address?",
    "The email we have on file is {email}. Is that correct?",
]

# ── Contact update prompts ────────────────────────────────────────────────────

FAX_UPDATE_PROMPTS = [
    "No problem — what is the correct fax number?",
    "Got it — could I get the updated fax number?",
    "Sure — what fax number should we use?",
]

EMAIL_UPDATE_PROMPTS = [
    "No problem — what is the correct email address?",
    "Got it — could I get the updated email address?",
    "Sure — what email address should we use?",
]

# ── Escalation messages ───────────────────────────────────────────────────────

MSG_CONTACT_EXHAUST = [
    "I'm having trouble confirming your contact details. Let me connect you with a representative.",
    "I wasn't able to confirm your delivery details. Connecting you with a specialist now.",
]

MSG_PHONE_DELIVERY_NOT_SUPPORTED = [
    "I understand — unfortunately, I can only send the provider list by fax or email, not over the phone. "
    "Which would work better for you — fax or email?",
    "I hear you, but phone delivery isn't available for the provider list. "
    "I can send it by fax or email — which would you prefer?",
    "I'm sorry, the provider list can only be delivered electronically, by fax or email. "
    "Which would you like?",
]

MSG_DISPATCH_FAIL = [
    "I'm sorry, I was unable to send the provider list. Let me connect you with a representative.",
    "I wasn't able to dispatch the list. Connecting you with a specialist.",
]
# Static slot collection order (all awaiting_slot phases, in flow sequence) —
# used to build the "Pending:" extraction context line.
DELIVERY_SLOT_ORDER = [
    "delivery_method",
    "fax_confirmed",
    "fax",
    "email_confirmed",
    "email",
    "benefits_response",
]
