"""Tests for declared slot shapes and the partial-value handling they enable.

Transcript 2 is the acceptance criterion:

    AI      May I have the reference number of the adjustment request?
    Caller  It is four two six nine.
    AI      And that reference number is four two six nine?
    Caller  five eight one seven.
    AI      And that reference number is four two six nine five eight one seven?
    Caller  Yes. That is correct.

Four of eight digits was treated as a whole value, so it entered the
confirmation path — and a caller says yes to "is that four two six nine?".
"""

from __future__ import annotations

import pytest

from agent.core.request_detection import reconcile_worker_result
from agent.llm.schema import WorkerResult
from agent.responses.builder import build_remainder_prompt
from agent.slots import validators
from agent.slots.shapes import (
    SLOT_SHAPES,
    Completeness,
    assess,
    canonical_value,
    describe_remainder,
    get_shape,
    has_shape,
    merge_digits,
    remaining,
)

_VALIDATOR_FOR = {
    "reference_number": validators.validate_reference_number,
    "zip_code": validators.validate_zip_code,
    "ssn": validators.validate_ssn,
    "phone_number": validators.validate_phone_number,
    "fax": validators.validate_fax_number,
    "member_id": validators.validate_member_id,
}


# ── The declared shape agrees with the validator ─────────────────────────────


@pytest.mark.parametrize("slot", sorted(SLOT_SHAPES))
def test_a_complete_run_passes_the_slots_validator(slot):
    """The shape is only a source of truth if the value it builds is storable.

    A shape that declares nine digits for a slot whose validator wants ten
    would have the flow collect a full value and then reject it, which is worse
    than the bug being fixed.
    """
    shape = SLOT_SHAPES[slot]
    digits = "1234567890"[: shape.digits] if shape.digits <= 10 else "1" * shape.digits
    value = canonical_value(slot, digits)
    assert value, f"{slot}: a complete run produced no canonical value"
    assert _VALIDATOR_FOR[slot](value).valid, f"{slot}: {value!r} fails its own validator"


def test_registry_is_keyed_by_its_own_slot_names():
    for key, shape in SLOT_SHAPES.items():
        assert key == shape.slot


def test_slots_without_a_declared_shape_are_left_alone():
    """claim_number is "at least four digits" and dob is a date. Neither has a
    partial state, and inventing one would break values that are already fine."""
    for slot in ("claim_number", "dob", "first_name", "email", ""):
        assert not has_shape(slot)
        assert get_shape(slot) is None
        assert assess(slot, "four two six nine") is Completeness.UNKNOWN
        assert remaining(slot, "4269") == 0
        assert canonical_value(slot, "4269") == ""


# ── Measuring a value ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "slot,spoken,expected",
    [
        ("reference_number", "four two six nine", Completeness.PARTIAL),
        ("reference_number", "four two six nine five eight one seven", Completeness.COMPLETE),
        ("reference_number", "42695817", Completeness.COMPLETE),
        ("reference_number", "", Completeness.NONE),
        ("reference_number", "somewhere in the letter", Completeness.NONE),
        # More digits than the shape allows is NONE, not PARTIAL: appending to a
        # value that is already wrong makes it wronger.
        ("reference_number", "one two three four five six seven eight nine", Completeness.NONE),
        ("zip_code", "one two three", Completeness.PARTIAL),
        ("zip_code", "one two three one nine", Completeness.COMPLETE),
        ("member_id", "m nine zero seven", Completeness.PARTIAL),
        ("member_id", "m nine zero seven five zero three", Completeness.COMPLETE),
        # normalize_ssn only ever emits nine digits, so a partial SSN has no
        # canonical form — the shape reads the digits directly for this reason.
        ("ssn", "five five five one two", Completeness.PARTIAL),
        ("ssn", "five five five one two three four five six", Completeness.COMPLETE),
        ("phone_number", "five one two five five five", Completeness.PARTIAL),
        ("fax", "six one seven one two three four one nine nine", Completeness.COMPLETE),
    ],
)
def test_assess(slot, spoken, expected):
    assert assess(slot, spoken) is expected


def test_remaining_and_its_spoken_form():
    assert remaining("reference_number", "4269") == 4
    assert describe_remainder("reference_number", "4269") == "four"
    assert remaining("reference_number", "4269581") == 1
    assert describe_remainder("reference_number", "4269581") == "one"
    # Nothing missing — a caller must never be asked for the rest of a whole value.
    assert remaining("reference_number", "42695817") == 0
    assert describe_remainder("reference_number", "42695817") == ""


def test_canonical_value_rebuilds_the_stored_form():
    assert canonical_value("reference_number", "42695817") == "42695817"
    assert canonical_value("member_id", "907503") == "M907503"
    assert canonical_value("ssn", "555123456") == "555-12-3456"
    assert canonical_value("zip_code", "12319") == "12319"
    # Short runs have no canonical form: a partial is never a value.
    assert canonical_value("reference_number", "4269") == ""


# ── The fragment merge ───────────────────────────────────────────────────────


def test_the_second_half_completes_the_first():
    assert merge_digits("reference_number", "4269", "five eight one seven") == "42695817"


def test_a_model_that_merged_it_itself_is_not_doubled():
    """The extraction model merges from history whenever the earlier fragment is
    still in the window it was given. Taking both would produce 426942695817."""
    assert merge_digits("reference_number", "4269", "four two six nine five eight one seven") == "42695817"


def test_a_restart_replaces_what_was_held():
    """ "No, sorry, it's seven one three two five five one two." The joined run
    would overrun the shape, so the new reading wins outright."""
    assert merge_digits("reference_number", "4269", "71325512") == "71325512"


def test_a_third_piece_keeps_accumulating():
    assert merge_digits("reference_number", "42", "six nine") == "4269"
    assert merge_digits("reference_number", "4269", "five eight") == "426958"


def test_merge_handles_the_empty_cases():
    assert merge_digits("reference_number", "", "four two six nine") == "4269"
    assert merge_digits("reference_number", "4269", "") == "4269"
    assert merge_digits("reference_number", "", "") == ""
    assert merge_digits("dob", "4269", "5817") == ""


# ── What the caller hears ────────────────────────────────────────────────────


def test_the_remainder_ask_names_what_we_have_and_what_is_missing():
    msg = build_remainder_prompt("reference_number", "4269")
    assert "four two six nine" in msg
    assert "four digits" in msg


def test_one_digit_left_gets_its_own_sentence():
    """ "the last one digits" is not a sentence."""
    msg = build_remainder_prompt("reference_number", "4269581")
    assert "one digits" not in msg
    assert "last digit" in msg


def test_the_remainder_ask_is_never_a_yes_no_question_about_a_fragment():
    """The bug is that "And that reference number is four two six nine?" gets a
    yes. Every remainder line has to ask for digits, not for agreement."""
    for digits in ("4", "42", "426", "4269", "42695", "426958", "4269581"):
        msg = build_remainder_prompt("reference_number", digits)
        assert msg
        assert "is that" not in msg.lower()
        assert "correct" not in msg.lower()


def test_the_remainder_ask_is_empty_when_there_is_nothing_to_ask_for():
    assert build_remainder_prompt("reference_number", "42695817") == ""
    assert build_remainder_prompt("reference_number", "") == ""
    assert build_remainder_prompt("first_name", "42") == ""


# ── completeness on the extraction result ────────────────────────────────────


def test_completeness_is_not_in_the_schema_the_model_is_shown():
    """The model perceives "four two six nine"; whether that is a whole
    reference number is a fact about the slot. Asking a perception model to
    rule on policy is what produced most of the flags this schema carries."""
    assert "completeness" not in WorkerResult.model_json_schema()["properties"]
    assert "completeness" in WorkerResult.model_fields


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("four two six nine", Completeness.PARTIAL),
        ("four two six nine five eight one seven", Completeness.COMPLETE),
        ("somewhere in the letter", Completeness.NONE),
    ],
)
def test_reconcile_measures_the_awaited_value(spoken, expected):
    result = reconcile_worker_result(
        WorkerResult(extracted={"reference_number": spoken}),
        f"It is {spoken}.",
        awaiting_slot="reference_number",
    )
    assert result.completeness is expected


def test_a_correction_is_measured_too():
    """ "No, it's four two six nine" is as short as the first answer was."""
    result = reconcile_worker_result(
        WorkerResult(corrections={"reference_number": "four two six nine"}),
        "No, it's four two six nine.",
        awaiting_slot="reference_number",
    )
    assert result.completeness is Completeness.PARTIAL


def test_a_slot_with_no_shape_is_left_unknown():
    result = reconcile_worker_result(
        WorkerResult(extracted={"first_name": "Monique"}),
        "Monique.",
        awaiting_slot="first_name",
    )
    assert result.completeness is Completeness.UNKNOWN


def test_no_value_leaves_completeness_alone():
    result = reconcile_worker_result(
        WorkerResult(),
        "sorry, what was that?",
        awaiting_slot="reference_number",
    )
    assert result.completeness is Completeness.UNKNOWN
