"""
pipelines.py — Slot pipelines for DeliveryManagementAgent.
"""

from __future__ import annotations

from agent.agents.delivery_management.constants import (
    EMAIL_UPDATE_PROMPTS,
    FAX_UPDATE_PROMPTS,
)
from agent.slots.normalizers import (
    normalize_delivery_method,
    normalize_email,
    normalize_fax_number,
)
from agent.slots.pipeline import SlotConfig, SlotPipeline
from agent.slots.types import SlotType
from agent.slots.validators import (
    validate_delivery_method,
    validate_email,
    validate_fax_number,
)
from agent.utils import pick


def build_delivery_method_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="delivery_method",
                slot_type=SlotType.DELIVERY_METHOD,
                normalizer=normalize_delivery_method,
                validator=validate_delivery_method,
                prompt="",
            ),
        ],
    )


def build_fax_pipeline(agent) -> SlotPipeline:
    """Pipeline for collecting a new fax number after the member declines the existing one."""
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="fax",
                slot_type=SlotType.FAX,
                normalizer=normalize_fax_number,
                validator=validate_fax_number,
                prompt=pick(FAX_UPDATE_PROMPTS),
            ),
        ],
    )


def build_email_pipeline(agent) -> SlotPipeline:
    """Pipeline for collecting a new email after the member declines the existing one."""
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="email",
                slot_type=SlotType.EMAIL,
                normalizer=normalize_email,
                validator=validate_email,
                prompt=pick(EMAIL_UPDATE_PROMPTS),
            ),
        ],
    )
