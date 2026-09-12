"""A value the normalizer can read is never re-asked.

    AI      …and your date of birth?
    Caller  April twelvee nineteen eighty-eight
    AI      Sorry, I didn't catch that — could you repeat your date of birth?

Nothing was wrong with normalization. normalize_dob reads that utterance to
04/12/1988 — ASR typo and all — and has done all along. It was never called.

The normalizer only ever runs on what the extraction LLM hands over, so when
the model declines to extract, the parser that could have read the value sits
unused and the caller is asked again. The extraction header instructs the model
to return nothing when speech "sounds garbled", and "twelvee" looks garbled to
something that cannot try parsing it. The one component that can, does not get
the chance.

So on a turn where extraction produced nothing, the slot's own normalizer and
validator are pointed at the caller's raw words before an attempt is counted.
The validator is what makes that safe: a date either parses or it does not.

That safety is why it is restricted to format-gated slot types. normalize_name
accepts anything, so an unrestricted version would confirm "I don't have it" as
a last name — the test at the bottom of this file is what pins that down.
"""

from __future__ import annotations

import pytest

from agent.core.agent import BaseAgent
from agent.core.slot_manager import SlotManagerMixin, _InternalSlotConfig
from agent.slots import normalizers as N
from agent.slots import validators as V
from agent.slots.types import SlotType


class _Agent(BaseAgent):
    AGENT_NAME = "verification_agent"

    async def run(self, state):  # pragma: no cover - never called
        raise AssertionError


def _messages(said: str) -> list[dict]:
    return [
        {"role": "assistant", "content": "And your date of birth?"},
        {"role": "user", "content": said},
    ]


_CONFIGS = {
    "dob": _InternalSlotConfig("dob", "", N.normalize_dob, V.validate_dob, SlotType.DOB),
    "member_id": _InternalSlotConfig(
        "member_id", "", N.normalize_member_id, V.validate_member_id, SlotType.MEMBER_ID
    ),
    "zip_code": _InternalSlotConfig(
        "zip_code", "", N.normalize_zip_code, V.validate_zip_code, SlotType.ZIP_CODE
    ),
    "last_name": _InternalSlotConfig("last_name", "", N.normalize_name, V.validate_name, SlotType.LAST_NAME),
}


def _salvage(slot: str, said: str) -> str:
    return _Agent().salvage_slot_value(_CONFIGS[slot], _messages(said))


# ── the value is read from what the caller said ──────────────────────────────


@pytest.mark.parametrize(
    "said, expected",
    [
        ("April twelvee nineteen eighty-eight", "04/12/1988"),  # the reported turn
        ("April twelve nineteen eighty-eight", "04/12/1988"),
        ("April twelfth nineteen eighty eight", "04/12/1988"),
        ("twelfth of April nineteen eighty-eight", "04/12/1988"),
        ("November 5 1992", "11/05/1992"),
    ],
)
def test_a_spoken_date_is_read(said, expected):
    assert _salvage("dob", said) == expected


@pytest.mark.parametrize(
    "said, expected",
    [
        # The normalizers expect the bare value extraction would have handed
        # them, and a caller saying it behind a lead-in is the ordinary case.
        ("it's 04/12/1988", "04/12/1988"),
        ("I think it's April 12 1988", "04/12/1988"),
        ("my date of birth is April twelve nineteen eighty-eight", "04/12/1988"),
    ],
)
def test_a_value_behind_a_lead_in_is_read(said, expected):
    assert _salvage("dob", said) == expected


@pytest.mark.parametrize(
    "slot, said, expected",
    [
        ("member_id", "it's M451982", "M451982"),
        ("member_id", "my member id is m four five one nine eight two", "M451982"),
        ("zip_code", "it's 90210", "90210"),
    ],
)
def test_the_other_format_gated_slots_too(slot, said, expected):
    assert _salvage(slot, said) == expected


# ── and nothing else is ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "said",
    [
        "hold on, let me check",  # a wait has no value to salvage
        "I don't have it",  # cannot-provide keeps its own path
        "uh",
        "hmm sorry",
        "can you repeat that?",
        "I want to speak to a human",
        "nineteen eighty-eight",  # a year is not a date
        "",
    ],
)
def test_a_turn_with_no_value_salvages_nothing(said):
    assert _salvage("dob", said) == ""


@pytest.mark.parametrize(
    "said",
    [
        "I don't have it",
        "the weather is terrible today",
        "uh",
        "my last name is Customer",  # a real value, and still not salvaged
    ],
)
def test_a_name_is_never_salvaged(said):
    """The reason the slot list exists. normalize_name accepts anything, so an
    unrestricted salvage would confirm "I don't have it" as a last name. Names
    stay with extraction, which can tell a name from a sentence."""
    assert _salvage("last_name", said) == ""


def test_the_slot_list_only_holds_format_gated_types():
    """Each entry must reject prose through its validator — that is what makes
    reading the caller's raw words safe. A type added here without a real format
    check reopens the name hole."""
    prose = [
        "I don't have it",
        "the weather is terrible today",
        "yes that's right",
        "hold on, let me check",
        "uh",
        "I want to speak to a human",
    ]
    import inspect

    for slot_type in SlotManagerMixin._SALVAGEABLE_SLOT_TYPES:
        normalizer = getattr(N, f"normalize_{slot_type}", None) or getattr(
            N, f"normalize_{slot_type}_number", None
        )
        validator = getattr(V, f"validate_{slot_type}", None) or getattr(
            V, f"validate_{slot_type}_number", None
        )
        assert normalizer and validator, f"{slot_type} has no normalizer/validator pair"
        assert inspect.isfunction(normalizer)
        for said in prose:
            value = normalizer(said)
            accepted = bool(value) and (validator(value).valid)
            assert not accepted, f"{slot_type} accepts prose {said!r} as {value!r} — not format-gated"


def test_names_are_deliberately_absent_from_the_list():
    assert "first_name" not in SlotManagerMixin._SALVAGEABLE_SLOT_TYPES
    assert "last_name" not in SlotManagerMixin._SALVAGEABLE_SLOT_TYPES
    assert "full_name" not in SlotManagerMixin._SALVAGEABLE_SLOT_TYPES
    assert "free_text" not in SlotManagerMixin._SALVAGEABLE_SLOT_TYPES
