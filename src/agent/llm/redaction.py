"""
redaction.py — single source of truth for masking slot values in LLM-2 payloads.

Masking is slot-NAME based (MASKED_SLOTS), with a value-shape regex kept as a
belt-and-braces second pass: even if a sensitive value arrives under an
unexpected slot name, anything member_id-shaped (M123456) or dob-shaped
(MM/DD/YYYY) is still rendered as "on file".

Imported by both llm.response_generator and core.slot_manager — keep this
module dependency-free so it never participates in the core ↔ llm cycle.
"""

from __future__ import annotations

import re

# Slots whose raw values must never be spoken back or handed to the
# generation LLM. Rendered as "on file" wherever a confirmed value appears.
# email and fax are included: at verification stage they are prefetched from
# Salesforce but should never be recited back (the member didn't just state
# them). They are only recitable at delivery stage after the member confirms
# them — but delivery builds its own Confirmed context independently.
MASKED_SLOTS: frozenset[str] = frozenset({"member_id", "dob", "email", "fax"})

_RAW_SENSITIVE_VALUE_RE = re.compile(r"^\s*(?:[Mm]\d{6}|\d{2}/\d{2}/\d{4})\s*$")

# Pseudo-slots that are internal counters/flags (yes/no confirmations, loop
# counters, update-detour budgets), never caller-facing collected values.
_PSEUDO_SLOT_RE = re.compile(r"(?:_confirmed|_cycles)$|^update_")


def _is_reportable_slot(name: str) -> bool:
    """True when ``name`` denotes a real collected value worth reporting on
    the Confirmed: line — filters out counter/flag pseudo-slots such as
    ``name_confirmed``, ``phone_confirmed``, ``*_cycles`` and ``update_*``."""
    return bool(name) and not _PSEUDO_SLOT_RE.search(name)


def mask_confirmed(values: dict | None) -> dict[str, str]:
    """Render confirmed slot values safe for the LLM-2 payload.

    Masked slots (and anything that merely LOOKS like a masked value) become
    "on file"; everything else is stringified as-is.
    """
    masked: dict[str, str] = {}
    for name, value in (values or {}).items():
        v = str(value)
        masked[name] = "on file" if name in MASKED_SLOTS or _RAW_SENSITIVE_VALUE_RE.match(v) else v
    return masked
