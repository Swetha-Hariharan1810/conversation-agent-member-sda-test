"""
response_builder.py — Context-aware conversational response generation.

Single source of truth for all slot-prompt variation.

Public API used by agents and slot infrastructure:
  build_initial_prompt(slot_type)               → str
  build_transition_prompt(slot_type, context)   → str
  build_retry_prompt(slot_type, attempt, ...)   → str

All selection is pure Python — no LLM calls, no I/O, zero latency.
"""

from __future__ import annotations

import random

from agent.conversation.context import (
    ConversationContext,
)
from agent.slots.types import SlotType

__all__ = [
    "build_initial_prompt",
    "build_transition_prompt",
    "build_retry_prompt",
]

# ---------------------------------------------------------------------------
# Transition templates: moving from the previous confirmed slot to this one
# ---------------------------------------------------------------------------

_TRANSITION_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.LAST_NAME: [
        "And your last name?",
        "Could you please provide your last name?",
        "Could I get your last name?",
        "May I have your last name?",
    ],
    SlotType.MEMBER_ID: [
        "Thank you{name_part}. May I have your Member ID?",
        "Could you provide your Member ID number?",
        "And your Member ID — whenever you're ready.",
        "Perfect. Could I get your Member ID?",
    ],
    SlotType.DOB: [
        "Thank you. And your date of birth?",
        "What's the date of birth on the account?",
        "Almost there{name_part} — and your date of birth?",
        "Could I get your date of birth?",
        "And the date of birth, including the year?",
    ],
    SlotType.RELATIONSHIP: [
        "Thank you. Are you the subscriber, or are you calling for a dependent?",
        "Could you confirm — are you the subscriber or a dependent?",
        "And are you the primary subscriber?",
    ],
    SlotType.PHONE_NUMBER: [
        "Thank you{name_part}. What is the best number to reach you?",
        "Could you provide your phone number?",
        "And the phone number on the account?",
    ],
    SlotType.ZIP_CODE: [
        "Could you confirm your ZIP code?",
        "And your five-digit ZIP code?",
        "What ZIP code are we working with?",
        "And the ZIP code?",
    ],
    SlotType.EMAIL: [
        "And what email address should we use?",
        "Could you give me your email address?",
        "What email address should I put down?",
    ],
    SlotType.FAX: [
        "And the fax number you'd like us to use?",
        "Could you provide your fax number?",
        "What fax number should we send that to?",
    ],
    SlotType.PROVIDER_TYPE: [
        "Thank you{name_part}. What type of provider are you looking for?",
        "What kind of doctor or specialist do you need?",
        "And what type of provider are you searching for?",
        "What type of care are you looking for today?",
    ],
    SlotType.CLAIM_NUMBER: [
        "Thank you{name_part}. May I have the reference number for the adjustment?",
        "Could you provide the reference number from your adjustment?",
        "And the adjustment reference number?",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "We can keep you posted on the status of the provider outreach. "
        "Would you prefer updates by SMS or email?",
        "To keep you in the loop, would you like notifications by SMS or email?",
        "How would you like to receive status updates — SMS or email?",
    ],
    SlotType.DELIVERY_METHOD: [
        "How would you like us to send this — fax or email?",
        "Would you prefer fax or email for that?",
        "And for delivery — fax or email works best for you?",
    ],
}

_DEFAULT_TRANSITION = [
    "Thank you{name_part}. Could you provide {slot_label}?",
    "Could you provide {slot_label}?",
    "Thank you. Could you provide {slot_label}?",
    "And {slot_label}?",
]

# ---------------------------------------------------------------------------
# First-ask templates
# ---------------------------------------------------------------------------

_INITIAL_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.FIRST_NAME: [
        "Can I get your first name, please?",
        "Could you start with your first name?",
        "To get started, what's your first name?",
        "Please go ahead with your first name.",
    ],
    SlotType.LAST_NAME: [
        "What's the last name on the account?",
        "Could I get your last name?",
        "And the last name?",
        "What last name should I look under?",
    ],
    SlotType.MEMBER_ID: [
        "May I ask for your Member ID, please?",
        "Could you provide your Member ID?",
        "I'll need your Member ID — go ahead whenever you're ready.",
        "Please share your Member ID.",
    ],
    SlotType.DOB: [
        "To validate your account, could I get your date of birth?",
        "Could you provide your date of birth, including the year?",
        "May I have your date of birth?",
        "I'll need your date of birth to confirm the account.",
    ],
    SlotType.PROVIDER_TYPE: [
        "What type of provider are you looking for?",
        "What kind of doctor or specialist do you need?",
        "What type of care are you looking for today?",
        "Are you looking for a primary care physician, a specialist, or another type of provider?",
    ],
    SlotType.FAX: [
        "What is the correct fax number?",
        "Could I get the updated fax number?",
        "What fax number should we use?",
    ],
    SlotType.EMAIL: [
        "What is the correct email address?",
        "Could I get the updated email address?",
        "What email address should we use?",
    ],
    SlotType.DELIVERY_METHOD: [
        "Would you prefer to receive that by fax or email?",
        "How would you like us to send that — by fax or email?",
        "Should I send that via fax or email?",
    ],
    SlotType.REFERENCE_NUMBER: [
        "May I have the reference number of the adjustment request?",
        "Could you provide the reference number for your adjustment?",
        "I'll need the reference number from your adjustment request — go ahead whenever you're ready.",
        "What is the reference number for the adjustment?",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "How would you like to receive notifications — by SMS or email?",
        "Would you prefer status updates by SMS or email?",
        "I can send you notifications by SMS or email — which do you prefer?",
    ],
}

# ---------------------------------------------------------------------------
# Retry templates: the caller did not give a usable value — re-ask the SAME
# slot. Two tiers by attempt: a gentle first retry, then a hinted retry that
# spells out the expected shape. Deterministic Python — no LLM call, so a
# plain retry can never drift onto a different slot.
# ---------------------------------------------------------------------------

_RETRY_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.FIRST_NAME: [
        "Sorry, I didn't catch that — could you say your first name again?",
        "I want to make sure I get this right — what's your first name?",
        "Could you repeat your first name for me?",
    ],
    SlotType.LAST_NAME: [
        "Sorry, I didn't catch that — could you say your last name again?",
        "I want to make sure I get this right — what's your last name?",
        "Could you repeat your last name for me?",
    ],
    SlotType.FULL_NAME: [
        "Sorry, I didn't catch that — could you say your full name again?",
        "Could you repeat your full name for me?",
    ],
    SlotType.MEMBER_ID: [
        "Sorry, I didn't catch that — could you repeat your Member ID?",
        "Could you say your Member ID once more for me?",
    ],
    SlotType.DOB: [
        "Sorry, I didn't catch that — could you repeat your date of birth?",
        "Could you say your date of birth once more for me?",
    ],
    SlotType.ZIP_CODE: [
        "Sorry, I didn't catch that — could you repeat your ZIP code?",
        "Could you say your ZIP code once more?",
    ],
    SlotType.PHONE_NUMBER: [
        "Sorry, I didn't catch that — could you repeat the phone number?",
        "Could you say that phone number once more?",
    ],
    SlotType.EMAIL: [
        "Sorry, I didn't catch that — could you repeat the email address?",
        "Could you say that email address once more?",
    ],
    SlotType.FAX: [
        "Sorry, I didn't catch that — could you repeat the fax number?",
        "Could you say that fax number once more?",
    ],
    SlotType.RELATIONSHIP: [
        "Sorry, I didn't catch that — are you the subscriber, or calling for a dependent?",
        "Just to confirm — are you the subscriber or a dependent?",
    ],
    SlotType.PROVIDER_TYPE: [
        "Sorry, I didn't catch that — what type of provider are you looking for?",
        "Could you tell me again what type of provider you need?",
    ],
    SlotType.DELIVERY_METHOD: [
        "Sorry, I didn't catch that — would you like that by fax or email?",
        "Just to confirm — fax or email?",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "Sorry, I didn't catch that — would you prefer updates by SMS or email?",
        "Just to confirm — SMS or email?",
    ],
    SlotType.REFERENCE_NUMBER: [
        "Sorry, I didn't catch that — could you repeat the reference number?",
        "Could you say that reference number once more?",
    ],
    SlotType.CLAIM_NUMBER: [
        "Sorry, I didn't catch that — could you repeat the claim number?",
        "Could you say that claim number once more?",
    ],
}

# Second-tier retry: same slot, but now say what the value should look like.
_RETRY_HINTED_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.FIRST_NAME: [
        "I still don't have your first name — could you say it slowly, or spell it out for me?",
    ],
    SlotType.LAST_NAME: [
        "I still don't have your last name — could you say it slowly, or spell it out for me?",
    ],
    SlotType.FULL_NAME: [
        "I still don't have your name — could you say it slowly, or spell it out for me?",
    ],
    SlotType.MEMBER_ID: [
        "I still don't have your Member ID — it starts with an M followed by six digits. "
        "Could you read it out one character at a time?",
    ],
    SlotType.DOB: [
        "I still don't have your date of birth — could you give me the month, day, and year?",
    ],
    SlotType.ZIP_CODE: [
        "I still don't have your ZIP code — could you read me the five digits?",
    ],
    SlotType.PHONE_NUMBER: [
        "I still don't have the phone number — could you read me the ten digits?",
    ],
    SlotType.EMAIL: [
        "I still don't have the email address — could you spell it out for me?",
    ],
    SlotType.FAX: [
        "I still don't have the fax number — could you read me the ten digits?",
    ],
    SlotType.RELATIONSHIP: [
        "I still need to know whether you're the subscriber on the plan, or calling for a dependent.",
    ],
    SlotType.DELIVERY_METHOD: [
        "I still need to know how to send this — please say fax or email.",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "I still need to know how to reach you with updates — please say SMS or email.",
    ],
    SlotType.REFERENCE_NUMBER: [
        "I still don't have the reference number — it's eight digits. "
        "Could you read it out one digit at a time?",
    ],
}

_DEFAULT_RETRY = [
    "Sorry, I didn't catch that — could you repeat your {slot_label}?",
    "Could you say your {slot_label} once more for me?",
]

_DEFAULT_RETRY_HINTED = [
    "I still don't have your {slot_label} — could you say it slowly for me?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _name_part(context: ConversationContext) -> str:
    if context.should_use_name and context.caller_first_name:
        return f", {context.caller_first_name}"
    return ""


def _slot_label(slot_type: SlotType) -> str:
    return slot_type.value.replace("_", " ")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def build_initial_prompt(slot_type: SlotType) -> str:
    """First ask for a slot at the start of a pipeline."""
    pool = _INITIAL_TEMPLATES.get(slot_type)
    if pool:
        return random.choice(pool)
    return f"Could you provide your {_slot_label(slot_type)}?"


def build_transition_prompt(
    slot_type: SlotType,
    context: ConversationContext,
) -> str:
    """
    Prompt that acknowledges the just-confirmed previous slot and asks for the
    next one. Produces natural flow rather than isolated form-filling.
    """
    np = _name_part(context)
    pool = _TRANSITION_TEMPLATES.get(slot_type, _DEFAULT_TRANSITION)
    template = random.choice(pool)
    return template.format(
        name_part=np,
        slot_label=_slot_label(slot_type),
    )


def build_retry_prompt(
    slot_type: SlotType | None,
    *,
    attempt: int = 1,
    slot_label: str = "",
) -> str:
    """Re-ask for the SAME slot after a non-answer — static, no LLM call.

    ``attempt`` is the slot's attempt count *after* the failure was recorded.
    Attempt 1 gets a gentle "didn't catch that"; attempt 2 and beyond get the
    hinted variant that names the expected shape of the value.

    ``slot_label`` is the spoken label used when the slot has no SlotType (or
    no template) — e.g. "your Member ID". It is only consulted for the generic
    templates.
    """
    label = (slot_label or (slot_type.value.replace("_", " ") if slot_type else "that")).strip()
    hinted = attempt >= 2
    pool: list[str] | None = None
    if slot_type is not None:
        pool = (_RETRY_HINTED_TEMPLATES if hinted else _RETRY_TEMPLATES).get(slot_type)
        if pool is None and hinted:
            # No hinted variant for this slot — the gentle pool still re-asks
            # the right slot, which matters more than the escalation in tone.
            pool = _RETRY_TEMPLATES.get(slot_type)
    if pool is None:
        pool = _DEFAULT_RETRY_HINTED if hinted else _DEFAULT_RETRY
    return random.choice(pool).format(slot_label=label)
