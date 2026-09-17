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


# ── A denial names what is missing ───────────────────────────────────────────
#
# Every call site in _collect_slot escalates on this flag with no retry, so a
# pattern that also fits an ordinary sentence hangs up on a caller who was
# cooperating. The three that did:
#
#   "I lost my ..."        — accepted any noun, so "I lost my job" was a denial
#   "I never received ..." — same, so "I never received the list you faxed"
#                            (the reason for the call) was a denial
#   "left it" / "not with me" — no first-person anchor at all, despite the
#                            docstring promising one
#
# and the bare "I don't know", which is filler at least as often as refusal.


@pytest.mark.parametrize(
    "utterance",
    [
        # A life event, not a missing identifier.
        "I lost my job so I'm on COBRA now",
        "I lost my husband last year",
        "I lost my train of thought",
        # The reason for the call, not an inability to answer this question.
        "I never received the provider list you faxed last month",
        # Somebody other than the caller.
        "she left it with the doctor",
        "we left it at that",
        # The caller describing a form they filled in.
        "I left it blank on the form",
    ],
)
def test_an_ordinary_sentence_is_not_a_denial(utterance):
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "I lost my card",
        "I lost my insurance card",
        "I lost my credit ID card",
        "I lost the paper that had the reference number",
        "I misplaced my card",
        "I never received a card",
        "I never received one",
        "I never received my member id",
        "I never received a reference number for this",
        "I left it at home",
        "I left it with my doctor",
        "I don't carry it with me",
    ],
)
def test_a_missing_identifier_is_still_a_denial(utterance):
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is True


# ── "I don't know" denies only when it is the whole turn ─────────────────────


@pytest.mark.parametrize(
    "utterance",
    ["I don't know", "I don't know.", "Sorry, I don't know", "I don't know, sorry", "I don't know right now"],
)
def test_a_bare_dont_know_is_a_denial(utterance):
    """Politeness and hedges are not "something else"."""
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "I don't know, is it the one ending in 5309?",
        "I don't know if this helps but my ID is M907503",
    ],
)
def test_a_dont_know_that_carries_an_answer_is_not_a_denial(utterance):
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is False


@pytest.mark.parametrize(
    "utterance",
    ["I don't know the claim number, but I have my reference number", "I don't know where my card is"],
)
def test_a_qualified_dont_know_denies_wherever_it_sits(utterance):
    from agent.utils import detect_cannot_provide

    assert detect_cannot_provide(utterance) is True


def test_a_caller_hedging_while_they_look_is_waiting_not_refusing():
    """cannot-provide outranks wait, so a loose denial pattern silently ate the
    wait path: "hold on, I don't know, let me find my card" escalated instead
    of acknowledging the hold."""
    from agent.utils import detect_cannot_provide, detect_wait_request

    utterance = "hold on, I don't know, let me find my card"
    assert detect_cannot_provide(utterance) is False
    assert detect_wait_request(utterance) is True
