"""
pipelines.py — Slot pipelines for ProviderSearchAgent.
"""

from __future__ import annotations

from agent.agents.provider_search.constants import ZIP_UPDATE_PROMPT
from agent.slots.normalizers import normalize_provider_type, normalize_zip_code
from agent.slots.pipeline import SlotConfig, SlotPipeline
from agent.slots.types import SlotType
from agent.slots.validators import validate_provider_type, validate_zip_code


def build_provider_type_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="provider_type",
                slot_type=SlotType.PROVIDER_TYPE,
                normalizer=normalize_provider_type,
                validator=validate_provider_type,
                prompt="",
            ),
        ],
    )


def build_zip_confirmation_pipeline(agent) -> SlotPipeline:
    """Pipeline for collecting a new ZIP code after the member declines the existing one.
    Prompt is dynamic — built at runtime from ZIP on file when needed."""
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="zip_code",
                slot_type=SlotType.ZIP_CODE,
                normalizer=normalize_zip_code,
                validator=validate_zip_code,
                prompt=ZIP_UPDATE_PROMPT,
            ),
        ],
    )
