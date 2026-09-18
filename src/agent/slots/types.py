"""
Reusable slot type definitions.

IMPORTANT:
Slot types provide standardized semantic
categories across conversational agents.

Used by:
- validators
- extractors
- prompts
- slot registry
- analytics
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "SlotType",
    "SlotValues",
    "IDENTITY_SLOT_TYPES",
    "CONTACT_SLOT_TYPES",
    "CLAIM_SLOT_TYPES",
    "PROVIDER_SLOT_TYPES",
    "CONFIRMATION_REQUIRED_SLOT_TYPES",
]

# Type alias for the collected-slots dict passed through every slot pipeline.
# Keys are slot names (e.g. "first_name"), values are normalized string values.
SlotValues = dict[str, str]

# =========================================================
# Core Slot Types
# =========================================================


class SlotType(str, Enum):
    """
    Standardized conversational slot types.
    """

    # -----------------------------------------------------
    # Identity
    # -----------------------------------------------------

    FIRST_NAME = "first_name"

    LAST_NAME = "last_name"

    FULL_NAME = "full_name"

    MEMBER_ID = "member_id"

    DOB = "dob"

    ZIP_CODE = "zip_code"

    PHONE_NUMBER = "phone_number"

    EMAIL = "email"

    RELATIONSHIP = "relationship"

    CALLER_ROLE = "caller_role"

    # -----------------------------------------------------
    # Contact / Delivery
    # -----------------------------------------------------

    DELIVERY_METHOD = "delivery_method"

    FAX = "fax"

    # -----------------------------------------------------
    # Provider Services
    # -----------------------------------------------------

    PROVIDER_TYPE = "provider_type"

    # -----------------------------------------------------
    # Claim Services
    # -----------------------------------------------------

    CLAIM_NUMBER = "claim_number"

    CLAIM_STATUS = "claim_status"

    REFERENCE_NUMBER = "reference_number"

    NOTIFICATION_METHOD = "notification_method"

    # -----------------------------------------------------
    # Generic
    # -----------------------------------------------------

    FREE_TEXT = "free_text"

    YES_NO = "yes_no"


# =========================================================
# Slot Type Groups
# =========================================================

IDENTITY_SLOT_TYPES = {
    SlotType.FIRST_NAME,
    SlotType.LAST_NAME,
    SlotType.FULL_NAME,
    SlotType.MEMBER_ID,
    SlotType.DOB,
}

CONTACT_SLOT_TYPES = {
    SlotType.PHONE_NUMBER,
    SlotType.EMAIL,
    SlotType.FAX,
    SlotType.ZIP_CODE,
}

CLAIM_SLOT_TYPES = {
    SlotType.CLAIM_NUMBER,
    SlotType.CLAIM_STATUS,
}

PROVIDER_SLOT_TYPES = {
    SlotType.PROVIDER_TYPE,
}

# =========================================================
# Confirmation Recommended Types
# =========================================================

CONFIRMATION_REQUIRED_SLOT_TYPES = {
    SlotType.MEMBER_ID,
    SlotType.DOB,
    SlotType.PHONE_NUMBER,
    SlotType.EMAIL,
}
