"""Denial and pivot detection is semantic, not a phrase list.

The phrase lists could never finish enumerating English — "I do not have a
member ID" was missed because only the contracted form was listed, and the
caller went to a representative instead of being offered the SSN. The
extraction model now decides (WorkerResult.cannot_provide / fallback_pivot)
and the regex survives only as a backstop for a missed call or an extraction
that threw.
"""

from __future__ import annotations

import pytest

from agent.core.request_detection import reconcile_worker_result
from agent.llm.schema import WorkerResult


def _reconciled(text: str, **fields) -> WorkerResult:
    return reconcile_worker_result(WorkerResult(**fields), text)


# ── The model is the primary source ──────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "that's in my wallet at home",
        "my husband handles all of that",
        "I've no idea what that even is",
        "I don't think they ever sent me one",
    ],
)
def test_model_flag_is_honoured_where_no_pattern_matches(utterance):
    """These are exactly the phrasings a substring list will never contain."""
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is False, "precondition: regex does not match"
    assert _reconciled(utterance, cannot_provide=True).cannot_provide is True


# ── The regex is the backstop ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "I do not have a member ID",
        "I don't have the claim number",
        "I never received a card",
        "I lost it",
        "I can't find it anywhere",
    ],
)
def test_regex_fills_in_when_the_model_misses_the_denial(utterance):
    """An extraction that threw returns an empty WorkerResult — still routes."""
    assert _reconciled(utterance).cannot_provide is True


def test_regex_never_clears_a_flag_the_model_set():
    assert _reconciled("mmhm", cannot_provide=True).cannot_provide is True


# ── Precedence: an answer or a pivot outranks a denial ───────────────────────


def test_a_value_in_the_same_utterance_is_not_a_denial():
    result = _reconciled(
        "I don't have my card but the member id is M907503",
        extracted={"member_id": "M907503"},
    )
    assert result.cannot_provide is False


def test_a_named_alternative_is_a_pivot_not_a_denial():
    result = _reconciled("I don't have the member ID, can I use my social?", fallback_pivot="ssn")
    assert result.cannot_provide is False
    assert result.fallback_pivot == "ssn"


@pytest.mark.parametrize("utterance", ["M907503", "hold on, let me look", "yes", "sure", "um"])
def test_ordinary_turns_are_not_denials(utterance):
    assert _reconciled(utterance).cannot_provide is False


def test_reconcile_tolerates_a_non_worker_result_shim():
    class _Frozen:
        __slots__ = ("extracted",)

        def __init__(self):
            self.extracted = None

    # Must not raise even though cannot_provide cannot be assigned.
    reconcile_worker_result(_Frozen(), "I don't have it")
