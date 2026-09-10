"""
constants.py — Configuration constants for ProviderSearchAgent.

Contains ONLY:
  - Agent name string
  - Retry limits
  - Log labels
  - Static message pools

No logic, no imports from other agent modules.
"""

from __future__ import annotations

# ── Agent identity ────────────────────────────────────────────────────────────

AGENT_NAME = "provider_search_agent"

# ── Retry limits ──────────────────────────────────────────────────────────────

MAX_SLOT_ATTEMPTS = 3

MAX_CONTACT_CHANGE_CYCLES = 3

# ── Log labels ────────────────────────────────────────────────────────────────

LOG_ENTERED = "provider_search_agent: entered"
LOG_PROVIDER_TYPE = "provider_search_agent: provider_type collected"
LOG_ZIP_CONFIRMED = "provider_search_agent: zip_code confirmed"
LOG_ZIP_UPDATED = "provider_search_agent: zip_code updated"

# ── ZIP confirmation templates ────────────────────────────────────────────────

ZIP_CONFIRM_TEMPLATES = [
    "Thank you. Your ZIP code is {zip_code}, correct?",
    "I have your ZIP code as {zip_code}. Is that right?",
    "Just to confirm — your ZIP code is {zip_code}?",
]

ZIP_UPDATE_PROMPT = "No problem — what is your current 5-digit ZIP code?"

# ── Provider search bridge: first question asked on fresh entry from verification ──
# Plain strings — no interpolation. Used by the first-entry fast path in
# agent.py so no LLM call is needed on the initial turn.
PROVIDER_SEARCH_BRIDGE_MSGS = [
    "What type of provider are you looking for?",
    "What kind of doctor do you need?",
    "What type of care can I help you find today?",
]

# ── Delivery bridge (interrupt — pauses graph for user's fax/email answer) ───
# Plain strings — no {provider_type} interpolation.
# _signal_done uses ask_member (is_interrupt=True) so the graph pauses here
# and delivery_management_agent receives the user's answer as last_user,
# extracting the delivery method in one LLM call with no double-ask.
DELIVERY_BRIDGE_TEMPLATES = [
    "Alright — I have a list of in-network providers ready for you. "
    "Would you like that sent via fax or email?",
    "Great — I have your in-network provider list ready. Would you prefer to receive it by fax or email?",
    "I have a list of in-network providers in your area ready to send. Shall I deliver it by fax or email?",
    "Your in-network provider list is ready. Would you like it sent by fax or email?",
]

# ── Escalation messages ───────────────────────────────────────────────────────

MSG_NOT_VERIFIED = [
    "I'm sorry, I need to verify your account first. Let me connect you with a representative.",
    "I wasn't able to confirm your identity. Connecting you with a specialist now.",
]

MSG_ZIP_EXHAUST = [
    "I'm having trouble confirming your ZIP code. Let me connect you with a representative.",
    "I wasn't able to confirm your ZIP code. Connecting you with a specialist now.",
]

MSG_PROVIDER_TYPE_UNSUPPORTED = [
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
# Static slot collection order — used to build the "Pending:" extraction
# context line (agent.llm.extractor.build_worker_input pending_slots kwarg).
# delivery_method is owned by delivery_management_agent but is listed here
# so the extraction LLM parks delivery/presentation side-questions instead
# of declining them — they will be answered in the very next agent stage.
PROVIDER_SEARCH_SLOT_ORDER = [
    "provider_type",
    "zip_code",
    "delivery_method",
]
