"""A number dictated with "triple five" in it must survive normalization.

    AI     Got it — could I get the updated fax number?
    Caller Two three one, triple five, three two one one.
    AI     Could you say that fax number once more?

The utterance is a complete ten-digit fax number. Two independent holes ate it.

The prompt defined fax normalization as a strict word-to-digit map — "Map each
spoken word to EXACTLY ONE digit", valid words zero through nine, "return
ambiguous if not exactly 10 single-digit words". "triple" is not one of those
words, so mapping what was left gave eight digits, and the rule said ambiguous.
The extraction LLM was doing as it was told. (It sometimes got the number out
anyway — on a retry turn, with the same utterance twice in the history, it
overrode the rule. Same input, different answer: the rule as written made every
success a disobedience.)

_convert_spoken_digits had the same hole and could not rescue it: a flat
word-by-word lookup, so "triple" passed through unmapped and was then erased by
the digit filter. It also dropped any digit word a transcript had punctuated —
"one," is not a key — which cost digits on numbers with no repetition word in
them at all, and on member IDs did worse: an unmapped word survives the
alphanumeric filter as LETTERS, so "m nine, zero, three" normalized to
"MNINEZERO3".
"""

from __future__ import annotations

import re

import pytest

from agent.slots.normalizers import (
    _convert_spoken_digits,
    normalize_claim_number,
    normalize_fax_number,
    normalize_member_id,
    normalize_phone_number,
    normalize_reference_number,
    normalize_ssn,
    normalize_zip_code,
)
from agent.slots.validators import validate_fax_number

THE_CALL = "Two three one, triple five, three two one one."


# ── the reported call ────────────────────────────────────────────────────────


def test_the_fax_number_from_the_call_normalizes_and_validates():
    assert normalize_fax_number(THE_CALL) == "2315553211"
    assert validate_fax_number(normalize_fax_number(THE_CALL)).valid


def test_the_same_number_without_the_repetition_word_is_unchanged():
    plain = "Two three one, five five five, three two one one."
    assert normalize_fax_number(plain) == normalize_fax_number(THE_CALL)


# ── repetition words ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spoken, digits",
    [
        ("double five", "55"),
        ("triple five", "555"),
        ("treble five", "555"),  # British, and a common ASR reading of "triple"
        ("quadruple five", "5555"),
        ("double oh", "00"),
        ("double zero", "00"),
        ("triple nine nine", "9999"),  # the repeat ends at the word it applies to
        ("triple 5", "555"),  # already numeric
    ],
)
def test_a_repetition_word_repeats_the_digit_after_it(spoken, digits):
    assert re.sub(r"\D", "", _convert_spoken_digits(spoken)) == digits


def test_a_repetition_word_with_no_digit_after_it_is_left_alone():
    """It then falls through the word map and is erased with the other
    non-digits — never swallowing a real digit on its way out."""
    assert re.sub(r"\D", "", _convert_spoken_digits("one two triple")) == "12"
    assert re.sub(r"\D", "", _convert_spoken_digits("one triple banana two")) == "12"


def test_a_repetition_word_does_not_repeat_a_multi_digit_token():
    """ "triple 55" is not 555555 — only a single digit repeats."""
    assert re.sub(r"\D", "", _convert_spoken_digits("triple 55")) == "55"


# ── transcript punctuation ───────────────────────────────────────────────────


def test_punctuated_digit_words_are_still_digits():
    """This one cost digits with no repetition word anywhere in the number."""
    assert normalize_fax_number("Two three one, five five five, three two one one.") == "2315553211"


def test_punctuation_alone_used_to_shorten_the_number():
    spoken = "One six, seven eight three."
    assert normalize_zip_code(spoken) == "16783"


def test_separators_inside_a_written_number_still_work():
    assert normalize_fax_number("231-555-3211") == "2315553211"
    assert normalize_phone_number("(415) 555-3211") == "4155553211"


# ── every slot that takes a dictated number ──────────────────────────────────


@pytest.mark.parametrize(
    "normalizer, spoken, expected",
    [
        (normalize_fax_number, "two three one triple five three two one one", "2315553211"),
        (normalize_phone_number, "four one five triple five three two one one", "4155553211"),
        (normalize_zip_code, "one six double seven three", "16773"),
        (normalize_ssn, "five two seven, triple four, three eight two", "527-44-4382"),
        (normalize_reference_number, "one two, four nine, triple one five", "12491115"),
        (normalize_claim_number, "double eight two three zero one", "882301"),
        (normalize_member_id, "em nine zero triple seven five", "M907775"),
    ],
)
def test_each_spoken_number_slot_handles_repetition(normalizer, spoken, expected):
    assert normalizer(spoken) == expected


# ── member ID, where an unmapped word became letters ─────────────────────────


def test_a_punctuated_member_id_does_not_spell_its_digits_out():
    assert normalize_member_id("m nine, zero, seven, five, oh, three") == "M907503"


def test_a_trailing_full_stop_does_not_spell_the_last_digit_out():
    assert normalize_member_id("m nine zero seven five oh three.") == "M907503"


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("m nine zero seven five oh three", "M907503"),
        ("M907503", "M907503"),
        ("n907503", "M907503"),  # N → M prefix correction still applies
    ],
)
def test_member_ids_that_already_worked_still_work(spoken, expected):
    assert normalize_member_id(spoken) == expected


# ── the prompts agree with the normalizer ────────────────────────────────────


def _prompt(path: str) -> str:
    from pathlib import Path

    return " ".join(Path(f"src/agent/prompts/{path}").read_text().lower().split())


def test_the_rule_is_stated_once_where_every_extraction_prompt_reads_it():
    """global_extraction.md is the only file all three prompt tiers include."""
    body = _prompt("system/global_extraction.md")
    assert "triple five" in body
    assert '"double", "triple", "treble", "quadruple"' in body


def test_the_fax_rule_no_longer_counts_words_instead_of_digits():
    """ "Return ambiguous if not exactly 10 single-digit words" is what made the
    reported call ambiguous: "triple five" is one word and three digits."""
    body = _prompt("extraction/delivery_management.md")
    assert "not exactly 10 single-digit words" not in body
    assert "count digits, not words" in body
    # The enumeration of which words ARE single digits is a different line and
    # stays — it is the map, not the counting rule.
    assert "valid single-digit words: zero" in body


def test_the_ssn_digit_map_expands_repetition_before_counting():
    assert "expand repetition words before counting" in _prompt("extraction/ssn_fallback.md")


@pytest.mark.parametrize(
    "path, example, expected",
    [
        ("system/global_extraction.md", "two three one, triple five, three two one one", "2315553211"),
        ("system/global_extraction.md", "one six double seven three", "16773"),
        ("extraction/delivery_management.md", "four one five, triple oh, seven seven two one", "4150007721"),
        ("extraction/ssn_fallback.md", "five two seven, triple four, three eight two", "527444382"),
    ],
)
def test_every_worked_example_in_the_prompts_is_arithmetically_right(path, example, expected):
    """A prompt that teaches the rule with a wrong answer is worse than silence."""
    from pathlib import Path

    body = Path(f"src/agent/prompts/{path}").read_text()
    assert example in " ".join(body.lower().split()), f"{example!r} is no longer in {path}"
    assert expected in body, f"{path} shows a different result for {example!r}"
    assert re.sub(r"\D", "", _convert_spoken_digits(example)) == expected
