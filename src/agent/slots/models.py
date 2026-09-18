"""
slot_models.py — Slot validation and definition schemas.

SlotValidationResult is returned by every validator function.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SlotValidationResult(BaseModel):
    """Returned by every validator. `.valid` is the key field."""

    valid: bool
    normalized_value: Optional[str] = None
    error_reason: Optional[str] = None
