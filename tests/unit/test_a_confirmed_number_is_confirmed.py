"""A caller who confirms the number on file is not asked for it again.

    AI      Thank you. Is your phone number 512-555-6101?
    Caller  yep, that's the right number
    AI      Sorry, I didn't catch that — is 512-555-6101 still the best
            number to reach you?

The caller confirmed, in the plainest words there are, and the turn re-asked.
The reply is build_retry_prompt for phone_confirmed, which is reached only
after the slot records a failed attempt.

Three separate ways a confirmation the model made never reached the slot, all
of them landing on that one sentence:

  - The model wrote the verdict in words. `phone_confirmed` normalised
    "yep, that's the right number" to "yes" (the "yep," opener), but not
    "that's right", "that's correct" or "that's the right number" — the value
    passed through normalize_yes_no unchanged and failed validate_yes_no,
    which counts as a failed attempt. Those are the two commonest
    confirmations in the flow and they cost the caller a retry each.
  - The model filed it under `phone_confirmation`. That is the name in
    verification/llm.py's own docstring, and redirect_off_topic already
    treats both names as this slot; the pipeline read only `phone_confirmed`.
  - The model placed nothing. This slot's raw-utterance fallback was removed
    on the ground that the model has the context and the contract by this
    turn. It mostly does; when it does not, nothing catches it.

What is deliberately NOT read from the caller's words is a decline. A "no"
here ends the call, the phone on file is human_only so there is no
replacement to collect, and the ways of refusing do not make a list that
finishes. An affirmation is the opposite case — a closed set (see
core.confirmation) — so it is read, and only it.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.agents.verification.handlers import collect_post_lookup, screen_phone_affirmation
from agent.agents.verification.pipelines import build_claims_pipeline, build_provider_pipeline
from agent.llm.schema import EventType, WorkerResult
from agent.slots.normalizers import normalize_yes_no

PHONE = "512-555-6101"
REPORTED = "yep, that's the right number"


# ── the normalizer: the model's verdict, written out ─────────────────────────


@pytest.mark.parametrize(
    "said",
    [
        "that's right",
        "thats right",
        "that's correct",
        "that's the right number",
        "that's the correct number",
        "that's my number",
        "that's my cell",
        "that's it",
        "that's the one",
        "still correct",
        "still the same",
        "it is",
        "sounds good",
        "looks right",
        "all good",
        "perfect",
        "confirmed",
        "that's right, that's my cell",
    ],
)
def test_a_confirmation_written_out_is_canonicalised(said):
    assert normalize_yes_no(said) == "yes", f"{said!r} reached the validator as-is"


@pytest.mark.parametrize(
    "said",
    [
        "that's not right",
        "that's not correct",
        "that's not the right number",
        "that's not my number",
        "that isn't right",
        "that's the wrong number",
        "that's my old number",
        "that's outdated",
        "that's a different number",
        "that's changed",
        "no, that's right",
    ],
)
def test_a_negation_is_never_canonicalised_to_yes(said):
    """ "that's right" and "that's not right" differ by one word and mean
    opposite things, so a negation disqualifies the whole set."""
    assert normalize_yes_no(said) != "yes"


def test_the_decline_side_is_left_to_the_model():
    """Declines are not enumerated here. "that's wrong" stays unmapped — it
    re-asks, which is what this slot does with a turn it cannot place, rather
    than ending the call on a phrase match."""
    assert normalize_yes_no("that's wrong") != "no"


# ── the screen: an affirmation in the caller's words ─────────────────────────


@pytest.mark.parametrize(
    "said",
    [
        REPORTED,
        "yes",
        "yes correct",
        "yep",
        "yeah that's the one",
        "correct",
        "that's right",
        "that's my cell, yes",
        "sure, that's it",
        "absolutely",
        "still correct",
    ],
)
def test_an_affirmation_is_read_from_the_callers_words(said):
    assert screen_phone_affirmation(said) == "yes"


@pytest.mark.parametrize(
    "said",
    [
        # Refusals — never read here, whatever the wording.
        "no",
        "no, that's my old number",
        "that's not my number",
        "I changed it last month",
        # An affirmative opener with something that argues with it.
        "yes, but I changed it recently",
        "yeah, that's my old one",
        "yes, can you update it to my cell?",
        # Not an answer at all.
        "is that the number ending 6101?",
        "what number do you have?",
        "hold on, let me check",
        "",
    ],
)
def test_the_screen_reads_nothing_else(said):
    """It returns "yes" or nothing. A decline read from words ends a call the
    caller never asked to end."""
    assert screen_phone_affirmation(said) == ""


def test_the_screen_never_returns_no():
    for said in ("no", "nope", "that's wrong", "not mine", "no thanks", "wrong number"):
        assert screen_phone_affirmation(said) != "no"


# ── end to end, through collect_post_lookup ──────────────────────────────────


async def _turn(said: str, decision: WorkerResult) -> dict | None:
    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "call_intent": "claim_services",
        "member_status_verify": True,
        "first_name": "James",
        "last_name": "Wilson",
        "member_id": "M310188",
        "dob": "1977-07-30",
        "phone_number": PHONE,
        "awaiting_slot": "phone_confirmed",
        "messages": [
            {"role": "assistant", "content": f"Thank you. Is your phone number {PHONE}?"},
            {"role": "user", "content": said},
        ],
    }

    async def _generate(**_kw):
        return "GENERATED"

    agent = VerificationAgent.from_state(state)
    collected: dict = {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        interrupt = await collect_post_lookup(
            agent,
            state,
            state["messages"],
            collected,
            "claim_services",
            {"phone_number": PHONE},
            decision,
            build_claims_pipeline(agent),
            build_provider_pipeline(agent),
        )
    return {"interrupt": interrupt, "collected": collected}


@pytest.mark.parametrize(
    "decision",
    [
        pytest.param(WorkerResult(extracted={"phone_confirmed": "yes"}), id="canonical"),
        pytest.param(WorkerResult(extracted={"phone_confirmed": "that's right"}), id="written-out"),
        pytest.param(WorkerResult(extracted={"phone_confirmation": "yes"}), id="the-other-field-name"),
        pytest.param(WorkerResult(), id="placed-nowhere"),
    ],
)
async def test_the_reported_turn_confirms_the_number(decision):
    out = await _turn(REPORTED, decision)

    assert out["interrupt"] is None, "the caller was asked again"
    assert out["collected"]["phone_confirmed"] is True
    assert out["collected"]["phone_update_requested"] is False


@pytest.mark.parametrize(
    "said, decision",
    [
        pytest.param(
            "I'm not sure, let me check the other phone",
            WorkerResult(event_type=EventType.AMBIGUOUS),
            id="does-not-know",
        ),
        pytest.param(
            "yeah I think so",
            WorkerResult(event_type=EventType.AMBIGUOUS),
            id="labelled-unsure-despite-the-opener",
        ),
        pytest.param("yes, but I changed it recently", WorkerResult(), id="affirms-then-contradicts"),
        pytest.param("no, that's my old number", WorkerResult(), id="declines-in-words-only"),
        pytest.param(REPORTED, WorkerResult(extracted={"phone_confirmed": PHONE}), id="the-number-itself"),
    ],
)
async def test_what_must_still_be_asked_again(said, decision):
    """A label of AMBIGUOUS or WAIT is honoured over the words — the screen is
    for a turn the model placed nowhere, not one it placed as unsure. And a
    refusal is never read from words: the re-ask is the caller's chance to be
    heard, where a wrong "no" would have ended the call."""
    out = await _turn(said, decision)

    assert out["interrupt"] is not None, f"{said!r} was read as a confirmation"
    assert out["interrupt"].get("next_node") != "END"
    assert not out["collected"].get("phone_confirmed")


async def test_a_decline_the_model_placed_still_ends_the_call():
    """The screen adds nothing to this path and must not take it away."""
    out = await _turn("no, that's my old number", WorkerResult(extracted={"phone_confirmed": "no"}))

    assert out["interrupt"]["next_node"] == "END"
    assert out["interrupt"]["call_end_detail"] == "phone_not_confirmed"
