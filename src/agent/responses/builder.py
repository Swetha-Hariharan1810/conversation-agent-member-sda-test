"""
response_builder.py — Context-aware conversational response generation.

Single source of truth for all slot-prompt variation.

Public API used by agents and slot infrastructure:
  build_initial_prompt(slot_type)               → str
  build_transition_prompt(slot_type, context)   → str

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
