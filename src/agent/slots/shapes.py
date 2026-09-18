"""
shapes.py — how long a value is, and what to do when it arrives in pieces.

A slot value is either present or absent. Four digits of an eight-digit
reference number counts as present, so it enters the confirmation path:

    AI      May I have the reference number of the adjustment request?
    Caller  It is four two six nine.
    AI      And that reference number is four two six nine?
    Caller  five eight one seven.
    AI      And that reference number is four two six nine five eight one seven?
    Caller  Yes. That is correct.

``validate_reference_number`` requires exactly eight digits and the caller gave
four. Instead of asking for the rest, the system read the fragment back for
confirmation — then read the whole number back a second time once the rest
arrived. Two confirmation round-trips for one number, on a voice call.

The shape is already written down, in English, in the retry templates: "it's
eight digits", "it starts with an M followed by six digits". This module makes
it data, so a value can be measured against it:

  * :func:`assess`   — COMPLETE, PARTIAL or NONE for what the caller just said.
  * :func:`merge_digits` — the running digit run, when they give it in pieces.
  * :func:`remaining`— how many digits are still missing, for the ask.

Completeness is computed here, in Python, and never asked of the extraction
model. The model perceives "four two six nine"; whether that is a whole
reference number is a fact about the slot, not about the utterance, and asking
a perception model to rule on policy is what produced most of the flags this
schema is carrying. ``WorkerResult.completeness`` is annotated
``SkipJsonSchema`` for exactly that reason — the field exists, the model is
never shown it.

Only fixed-length digit runs are declared. ``claim_number`` is "at least four
digits", which has no partial state — a short one is simply a different number.
``dob`` is a date, and half a date is a different problem with a different fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from agent.slots.normalizers import (
    normalize_fax_number,
    normalize_member_id,
    normalize_phone_number,
    normalize_reference_number,
    normalize_ssn,
    normalize_zip_code,
    spoken_digits,
)

__all__ = [
    "Completeness",
    "SlotShape",
    "SLOT_SHAPES",
    "get_shape",
    "has_shape",
    "assess",
    "remaining",
    "merge_digits",
    "canonical_value",
    "describe_remainder",
]


class Completeness(str, Enum):
    """What the caller has given for a slot, measured against its shape."""

    UNKNOWN = "unknown"  # no shape declared, or nothing to measure
    COMPLETE = "complete"  # every digit the shape calls for
    PARTIAL = "partial"  # a real start, but short — keep it and ask for the rest
    NONE = "none"  # nothing usable: no digits, or more than the shape allows


@dataclass(frozen=True)
class SlotShape:
    """The declared shape of one slot's value.

    ``digits`` is how many a complete value carries. ``prefix`` is a fixed
    leading literal the canonical form always has ("M" on a member ID) — it is
    added back rather than required from the caller, because "nine zero seven
    five zero three" is how people read their member ID out loud.

    ``normalizer`` turns a complete digit run into the value the validator
    accepts: nine digits become "555-12-3456" for an SSN, six become "M907503"
    for a member ID.
    """

    slot: str
    digits: int
    label: str
    spoken: str  # "eight digits" — how the shape is said out loud
    normalizer: Callable[[Optional[str]], str]
    prefix: str = ""


SLOT_SHAPES: dict[str, SlotShape] = {
    s.slot: s
    for s in (
        SlotShape("reference_number", 8, "reference number", "eight digits", normalize_reference_number),
        SlotShape("zip_code", 5, "ZIP code", "five digits", normalize_zip_code),
        SlotShape("ssn", 9, "Social Security number", "nine digits", normalize_ssn),
        SlotShape("phone_number", 10, "phone number", "ten digits", normalize_phone_number),
        SlotShape("fax", 10, "fax number", "ten digits", normalize_fax_number),
        SlotShape("member_id", 6, "Member ID", "an M followed by six digits", normalize_member_id, "M"),
    )
}


def get_shape(slot: str | None) -> Optional[SlotShape]:
    if not slot:
        return None
    return SLOT_SHAPES.get(slot.strip())


def has_shape(slot: str | None) -> bool:
    return get_shape(slot) is not None


def read_digits(slot: str | None, value: str | None) -> str:
    """The digit run in ``value``, or "" when the slot has no declared shape.

    Reads the digits directly rather than going through the slot's normalizer,
    because several of those only emit a canonical form: normalize_ssn returns
    "" for anything that is not exactly nine digits, so a caller who has read
    out five has nothing left to measure.
    """
    if get_shape(slot) is None:
        return ""
    return spoken_digits(value)


def assess(slot: str | None, value: str | None) -> Completeness:
    """Measure what the caller gave against the slot's declared shape.

    More digits than the shape allows is NONE, not PARTIAL. Eleven digits for a
    ten-digit fax is a misheard number or a run-on, and treating it as a start
    to build on would append onto a value that is already wrong.
    """
    shape = get_shape(slot)
    if shape is None:
        return Completeness.UNKNOWN
    digits = spoken_digits(value)
    if not digits:
        return Completeness.NONE
    if len(digits) == shape.digits:
        return Completeness.COMPLETE
    if len(digits) < shape.digits:
        return Completeness.PARTIAL
    return Completeness.NONE


def remaining(slot: str | None, value: str | None) -> int:
    """How many digits are still missing. 0 when complete, or no shape."""
    shape = get_shape(slot)
    if shape is None:
        return 0
    return max(0, shape.digits - len(spoken_digits(value)))


def merge_digits(slot: str | None, held: str | None, incoming: str | None) -> str:
    """The digit run after the caller's next piece arrives.

    Four cases, and the order matters:

      * ``incoming`` already starts with everything held — the extraction model
        merged them itself, which it does whenever the earlier fragment is
        still in the history window it was given. Taking both would produce
        "42694269581 7".
      * held + incoming reaches the shape exactly — the ordinary second half.
      * held + incoming is still short — a third piece is coming.
      * neither fits — the caller started the number over ("no, sorry, it's
        seven one..."), so the new piece replaces what was held.
    """
    shape = get_shape(slot)
    if shape is None:
        return ""
    old = spoken_digits(held)
    new = spoken_digits(incoming)
    if not new:
        return old
    if not old:
        return new
    if new.startswith(old):
        return new
    joined = old + new
    if len(joined) <= shape.digits:
        return joined
    return new


def canonical_value(slot: str | None, digits: str | None) -> str:
    """The stored form of a complete digit run, or "" when it is not complete.

    Goes back through the slot's own normalizer so the value that reaches the
    validator is built exactly the way an ordinary single-turn answer is.
    """
    shape = get_shape(slot)
    if shape is None:
        return ""
    run = spoken_digits(digits)
    if len(run) != shape.digits:
        return ""
    return shape.normalizer(shape.prefix + run)


def describe_remainder(slot: str | None, digits: str | None) -> str:
    """How many digits are left, as the word a person would say: "four", "one".

    A count, not a phrase, so the caller-facing sentence decides how to say it
    ("the last four digits", "just the last digit"). Returns "" when nothing is
    missing or the slot has no shape, so a caller is never asked for the
    remainder of a value that is already whole.
    """
    if get_shape(slot) is None:
        return ""
    left = remaining(slot, digits)
    if left <= 0:
        return ""
    return _NUMBER_WORDS.get(left, str(left))


_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
