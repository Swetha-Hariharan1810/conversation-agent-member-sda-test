"""
slot_validators.py — Reusable deterministic slot validators.

Validators verify business correctness of normalized slot values.
  - valid member ID format
  - realistic date of birth
  - 10-digit phone number

Rules:
  - Deterministic and stateless
  - Reusable across all agents
  - Must NOT normalize, mutate state, or emit signals
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional

from agent.slots.models import SlotValidationResult

__all__ = [
    "validate_name",
    "validate_member_id",
    "validate_ssn",
    "validate_dob",
    "validate_zip_code",
    "validate_phone_number",
    "validate_fax_number",
    "validate_email",
    "validate_yes_no",
    "validate_relationship",
    "validate_provider_type",
    "validate_delivery_method",
    "validate_claim_number",
    "validate_date_of_service",
    "validate_billed_amount",
    "VALIDATOR_REGISTRY",
    "get_validator",
    "validate_slot_value",
]

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _ok(value: str | None = None) -> SlotValidationResult:
    return SlotValidationResult(valid=True, normalized_value=value)


def _fail(reason: str) -> SlotValidationResult:
    return SlotValidationResult(valid=False, error_reason=reason)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def validate_name(value: str | None) -> SlotValidationResult:
    """Accept names: letters, spaces, hyphens, apostrophes. Min 2 chars."""
    if not value:
        return _fail("Name is required")
    cleaned = value.strip()
    if len(cleaned) < 2:
        return _fail("Name is too short")
    if not re.fullmatch(r"[A-Za-z\s\-']+", cleaned):
        return _fail("Invalid name format")
    return _ok(cleaned)


# ---------------------------------------------------------------------------
# Member ID
# ---------------------------------------------------------------------------


def validate_member_id(value: str | None) -> SlotValidationResult:
    """Accept M followed by exactly 6 digits (e.g. 'M907503')."""
    if not value:
        return _fail("Member ID is required")
    cleaned = value.strip()
    if not re.fullmatch(r"M[0-9]{6}", cleaned):
        return _fail("Member ID must start with M followed by exactly 6 digits")
    return _ok(cleaned)


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


def validate_ssn(value: str | None) -> SlotValidationResult:
    """Accept XXX-XX-XXXX format (normalized form)."""
    if not value:
        return _fail("SSN is required")
    if not re.fullmatch(r"\d{3}-\d{2}-\d{4}", value.strip()):
        return _fail("SSN must be in XXX-XX-XXXX format")
    return _ok(value.strip())


# ---------------------------------------------------------------------------
# Date of Birth
# ---------------------------------------------------------------------------


def validate_dob(value: str | None) -> SlotValidationResult:
    """Accept MM/DD/YYYY dates that are in the past and within 120 years.

    Rejects ambiguous digit strings that the normalizer could not resolve
    (e.g. '01021985' which could be Jan 2 or Feb 1).
    """
    if not value:
        return _fail("DOB is required")

    # Detect unresolved digit strings — normalizer returns these as-is
    # when the format is ambiguous (valid as both YYYYMMDD and DDMMYYYY)
    digits_only = re.sub(r"\D", "", value)
    if len(digits_only) == 8 and not re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        return _fail("Date of birth format is ambiguous — please confirm as month, day, and year")

    try:
        parsed = datetime.strptime(value, "%m/%d/%Y")
    except ValueError:
        return _fail("Invalid DOB format — expected MM/DD/YYYY")

    today = datetime.today()
    if parsed > today:
        return _fail("DOB cannot be in the future")
    if (today.year - parsed.year) > 120:
        return _fail("DOB appears invalid")
    return _ok(value)


# ---------------------------------------------------------------------------
# ZIP Code
# ---------------------------------------------------------------------------


def validate_zip_code(value: str | None) -> SlotValidationResult:
    """Accept exactly 5 digits."""
    if not value:
        return _fail("ZIP code required")
    if not re.fullmatch(r"\d{5}", value):
        return _fail("Invalid ZIP code — expected 5 digits")
    return _ok(value)


# ---------------------------------------------------------------------------
# Phone Number
# ---------------------------------------------------------------------------


def validate_phone_number(value: str | None) -> SlotValidationResult:
    """Accept exactly 10 digits (after stripping non-digit characters)."""
    if not value:
        return _fail("Phone number required")
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return _fail("Phone number must be 10 digits")
    return _ok(digits)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def validate_email(value: str | None) -> SlotValidationResult:
    if not value:
        return _fail("Email required")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
        return _fail("Invalid email address")
    return _ok(value)


# ---------------------------------------------------------------------------
# Yes / No
# ---------------------------------------------------------------------------


def validate_yes_no(value: str | None) -> SlotValidationResult:
    if not value:
        return _fail("Response required")
    if value.lower().strip() not in {"yes", "no"}:
        return _fail("Expected yes or no")
    return _ok(value.lower().strip())


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

_VALID_RELATIONSHIPS = {"plan_holder", "subscriber", "dependent"}


def validate_relationship(value: str | None) -> SlotValidationResult:
    """Accept 'plan_holder' | 'subscriber' | 'dependent'."""
    if not value:
        return _fail("Relationship is required")
    if value.strip().lower() in _VALID_RELATIONSHIPS:
        return _ok(value.strip().lower())
    return _fail("Unrecognized relationship value")


# ---------------------------------------------------------------------------
# Provider Type
# ---------------------------------------------------------------------------

_KNOWN_PROVIDER_TYPES = {
    "Primary Care Physician",
    "Pediatrician",
    "Cardiologist",
    "Dermatologist",
    "Orthopedic Specialist",
}


def validate_provider_type(value: str | None) -> SlotValidationResult:
    if not value or not value.strip():
        return _fail("Provider type is required")
    if value.strip() not in _KNOWN_PROVIDER_TYPES:
        return _fail(f"Unrecognized provider type: {value.strip()!r}")
    return _ok(value.strip())


# ---------------------------------------------------------------------------
# Delivery Method
# ---------------------------------------------------------------------------


def validate_delivery_method(value: str | None) -> SlotValidationResult:
    """Accept 'fax' or 'email' only."""
    if not value:
        return _fail("Delivery method required")
    if value.strip().lower() not in {"fax", "email"}:
        return _fail("Expected fax or email")
    return _ok(value.strip().lower())


# ---------------------------------------------------------------------------
# Fax Number (same rules as phone number)
# ---------------------------------------------------------------------------


def validate_fax_number(value: str | None) -> SlotValidationResult:
    """Accept exactly 10 digits."""
    return validate_phone_number(value)


def validate_reference_number(value: str | None) -> SlotValidationResult:
    """Accept exactly 8-digit reference numbers (digits only after normalization)."""
    if not value:
        return _fail("Reference number required")
    if not re.fullmatch(r"\d{8}", value):
        return _fail("Reference number must be exactly 8 digits")
    return _ok(value)


def validate_notification_method(value: str | None) -> SlotValidationResult:
    """Accept 'sms' or 'email' only."""
    if not value:
        return _fail("Notification method required")
    if value.strip().lower() not in {"sms", "email"}:
        return _fail("Expected sms or email")
    return _ok(value.strip().lower())


def validate_claim_number(value: str | None) -> SlotValidationResult:
    """Accept numeric claim numbers (at least 4 digits, e.g. '882301')."""
    if not value:
        return _fail("Claim number required")
    cleaned = value.strip()
    if not re.fullmatch(r"\d+", cleaned):
        return _fail("Claim number must be digits only")
    if len(cleaned) < 4:
        return _fail("Claim number too short")
    return _ok(cleaned)


def validate_date_of_service(value: str | None) -> SlotValidationResult:
    """Accept ISO date YYYY-MM-DD for date of service."""
    if not value:
        return _fail("Date of service required")
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return _ok(value)
    except ValueError:
        return _fail("Invalid date of service — expected YYYY-MM-DD after normalization")


def validate_billed_amount(value: str | None) -> SlotValidationResult:
    """Accept positive decimal billed amounts (e.g. '1240.00')."""
    if not value:
        return _fail("Billed amount required")
    try:
        amount = float(value)
        if amount <= 0:
            return _fail("Billed amount must be positive")
        return _ok(value)
    except (TypeError, ValueError):
        return _fail("Invalid billed amount")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VALIDATOR_REGISTRY: dict[str, Callable[[Optional[str]], SlotValidationResult]] = {
    "validate_name": validate_name,
    "validate_member_id": validate_member_id,
    "validate_ssn": validate_ssn,
    "validate_dob": validate_dob,
    "validate_zip_code": validate_zip_code,
    "validate_phone_number": validate_phone_number,
    "validate_fax_number": validate_fax_number,
    "validate_reference_number": validate_reference_number,
    "validate_claim_number": validate_claim_number,
    "validate_date_of_service": validate_date_of_service,
    "validate_billed_amount": validate_billed_amount,
    "validate_notification_method": validate_notification_method,
    "validate_email": validate_email,
    "validate_yes_no": validate_yes_no,
    "validate_relationship": validate_relationship,
    "validate_provider_type": validate_provider_type,
    "validate_delivery_method": validate_delivery_method,
}


def get_validator(name: str) -> Callable[[Optional[str]], SlotValidationResult]:
    if name not in VALIDATOR_REGISTRY:
        raise ValueError(f"Unknown validator: {name}")
    return VALIDATOR_REGISTRY[name]


def validate_slot_value(validator_name: str | None, value: str | None) -> SlotValidationResult:
    if not validator_name:
        return _ok(value)
    return get_validator(validator_name)(value)
