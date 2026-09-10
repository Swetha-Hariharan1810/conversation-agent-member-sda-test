"""
pipelines.py — Slot pipelines for the Verification Agent.
slot_type added to every SlotConfig to enable contextual_response_builder.
"""

from __future__ import annotations

from agent.slots.normalizers import (
    normalize_caller_role,
    normalize_dob,
    normalize_member_id,
    normalize_name,
    normalize_yes_no,
)
from agent.slots.pipeline import SlotConfig, SlotPipeline
from agent.slots.types import SlotType
from agent.slots.validators import (
    validate_dob,
    validate_member_id,
    validate_name,
    validate_relationship,
    validate_yes_no,
)


def build_identity_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="first_name",
                slot_type=SlotType.FIRST_NAME,
                normalizer=normalize_name,
                validator=validate_name,
                prompt="",
            ),
            SlotConfig(
                name="last_name",
                slot_type=SlotType.LAST_NAME,
                normalizer=normalize_name,
                validator=validate_name,
                prompt="",
            ),
            SlotConfig(
                name="member_id",
                slot_type=SlotType.MEMBER_ID,
                normalizer=normalize_member_id,
                validator=validate_member_id,
                prompt="",
            ),
            SlotConfig(
                name="dob",
                slot_type=SlotType.DOB,
                normalizer=normalize_dob,
                validator=validate_dob,
                prompt="",
            ),
        ],
    )


def build_claims_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="phone_confirmed",
                slot_type=None,
                normalizer=normalize_yes_no,
                validator=validate_yes_no,
                prompt="Thank you. Could you confirm your phone number on file?",
            ),
        ],
    )


def build_provider_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="relationship",
                slot_type=None,
                normalizer=normalize_caller_role,
                validator=validate_relationship,
                prompt="Thank you, I found your account. Are you the subscriber or dependent?",
            ),
        ],
    )
