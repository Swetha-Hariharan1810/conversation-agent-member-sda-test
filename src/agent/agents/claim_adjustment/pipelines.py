# Dead code — no longer used by ClaimAdjustmentAgent.
# Reference number is now collected directly in agent.run().
# Retained for reference; safe to delete.
"""pipelines.py — Slot pipelines for ClaimAdjustmentAgent."""

from __future__ import annotations

from agent.slots.normalizers import normalize_reference_number
from agent.slots.pipeline import SlotConfig, SlotPipeline
from agent.slots.types import SlotType
from agent.slots.validators import validate_reference_number


def build_reference_number_pipeline(agent) -> SlotPipeline:
    return SlotPipeline(
        agent,
        [
            SlotConfig(
                name="reference_number",
                slot_type=SlotType.REFERENCE_NUMBER,
                normalizer=normalize_reference_number,
                validator=validate_reference_number,
                prompt="",  # initial prompt driven by SlotType via build_initial_prompt
            ),
        ],
    )
