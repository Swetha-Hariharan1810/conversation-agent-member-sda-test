"""constants.py — Configuration constants for RecordsCoordinationAgent."""

from __future__ import annotations

AGENT_NAME = "records_coordination_agent"

LOG_ENTERED = "records_coordination_agent: entered"
LOG_UPLOAD_LINK_SENT = "records_coordination_agent: upload link sent"
LOG_GUIDE_TRIGGERED = "records_coordination_agent: personal guide outreach triggered"
LOG_DOCTOR_DIRECT = "records_coordination_agent: doctor-direct branch acknowledged"

MAX_CONTACT_CHANGE_CYCLES = 3

# ── Upload link offer ──────────────────────────────────────────────────────────
MSG_UPLOAD_OFFER = [
    "I can also send you a link where you can upload the records directly. Would you like me to do that?",
    "Thank You. I can also generate a secure upload link and send it to your email. "
    "Would you like me to do that?",
    "Sure. I can also send a secure link to your email so you can upload the records yourself. "
    "Would you like me to send that over?",
]

# ── Email readback for upload link ─────────────────────────────────────────────
# Placeholder: {email}
EMAIL_READBACK_FOR_UPLOAD = [
    "The email address we have on file is {email}. Is this correct or has it been changed?",
    "I'll send the link to {email}. Is that still the right address?",
    "I have {email} on file. Is that the correct email to send the upload link to?",
]

# ── Upload confirmation message ────────────────────────────────────────────────
MSG_UPLOAD_SENT = [
    "Great! Expect to receive the link within 30 minutes.",
    "The upload link is on its way — expect it within 30 minutes.",
    "Done! You should receive the upload link within the next 30 minutes.",
]

# ── Doctor-direct acknowledgement ─────────────────────────────────────────────
MSG_DOCTOR_DIRECT_ACK = [
    "Of course — that works as well.",
    "Sure, that's fine.",
]

# ── Personal Guide offer ───────────────────────────────────────────────────────
MSG_PERSONAL_GUIDE_OFFER = [
    "I can have one of our Personal Guides conduct the outreach to your doctor "
    "to request a complete copy of the medical records for this request. "
    "Would you like us to proceed?",
    "Our Personal Guides can reach out directly to your provider to request the records. "
    "Would you like us to proceed with that?",
    "I can also have one of our Personal Guides contact your doctor's office on your behalf. "
    "Would you like us to proceed with that?",
]

# ── Personal Guide scheduled ───────────────────────────────────────────────────
MSG_GUIDE_SCHEDULED = [
    "Our Personal Guide will call the provider in the next 24 hours.",
    "A Personal Guide will reach out to your doctor within 24 hours.",
    "Our team will contact the provider within 24 hours to request the records.",
]

# ── Notification bridge (after Guide scheduled) ────────────────────────────────
MSG_NOTIFICATION_BRIDGE = [
    "We can also keep you posted on the status of the provider outreach. "
    "I can send the status updates to your email or phone. How do you want to be notified?",
    "I can send you updates as we progress with the outreach. Would you prefer those by email or phone?",
]

# ── Decline / escalation ───────────────────────────────────────────────────────
MSG_DECLINE_ESCALATE = [
    "I'm not able to complete this step right now. "
    "I can connect you to a representative to finish this request.",
    "No problem. I'll connect you with a representative who can help you through this step.",
]

# ── Contact update prompts ─────────────────────────────────────────────────────
MSG_EMAIL_UPDATE_PROMPT = [
    "No problem — what is the correct email address?",
    "Got it — could I get the updated email address?",
    "Sure — what email address should we use?",
]
# Static slot collection order — used to build the "Pending:" extraction
# context line.
RECORDS_SLOT_ORDER = [
    "upload_method",
    "upload_consent",
    "personal_guide_consent",
]
