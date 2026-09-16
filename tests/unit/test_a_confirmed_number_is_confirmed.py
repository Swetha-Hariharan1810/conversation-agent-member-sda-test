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

The decline is read too, and did not used to be, for a reason that has since
changed: it ended the call where it stood, so a wrong read cost the caller
their call. A decline escalates now, which is where the re-ask was taking
them anyway once the attempts ran out. Explicit refusals only — a garbled
turn, a hold or a side question has taken no position and still re-asks — and
never a mixed turn: normalize_yes_no("no, that's right") is "no", and that
caller confirmed.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.agents.verification.handlers import collect_post_lookup, screen_phone_confirmation
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
    assert screen_phone_confirmation(said) == "yes"


@pytest.mark.parametrize(
    "said",
    [
        "no",
        "nope",
        "no, that's not my number",
        "that's not my number",
        "that's not right",
        "that's the wrong number",
        "that's my old number",
        "that's a different number",
        "I changed it last month",
        "I don't use that anymore",
        "that number's been disconnected",
    ],
)
def test_an_explicit_refusal_is_read_from_the_callers_words(said):
    assert screen_phone_confirmation(said) == "no"


@pytest.mark.parametrize(
    "said",
    [
        # Affirms and refuses in one breath — read as neither. "no, that's
        # right" is the one normalize_yes_no gets wrong, and that caller
        # confirmed.
        "no, that's right",
        "yes, but I changed it recently",
        "yeah, that's my old one",
        "yes, can you update it to my cell?",
        # No position on the number at all.
        "is that the number ending 6101?",
        "what number do you have?",
        "hold on, let me check",
        "let me grab my card",
        "umm",
        "",
    ],
)
def test_a_turn_with_no_position_reads_as_none(said):
    """Not "anything that is not a yes is a no" — that would escalate a caller
    whose turn was garbled or who only asked a question."""
    assert screen_phone_confirmation(said) == ""


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
        pytest.param("no, that's right", WorkerResult(), id="affirms-and-refuses-at-once"),
        pytest.param(REPORTED, WorkerResult(extracted={"phone_confirmed": PHONE}), id="the-number-itself"),
    ],
)
async def test_what_must_still_be_asked_again(said, decision):
    """A label of AMBIGUOUS or WAIT is honoured over the words — the screen is
    for a turn the model placed nowhere, not one it placed as unsure."""
    out = await _turn(said, decision)

    assert out["interrupt"] is not None, f"{said!r} was read as a confirmation"
    assert out["interrupt"].get("next_node") != "END"
    assert not out["collected"].get("phone_confirmed")


# ── a decline escalates ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "said, decision",
    [
        pytest.param("no, that's not my number", WorkerResult(), id="the-reported-turn"),
        pytest.param(
            "no, that's not my number",
            WorkerResult(extracted={"phone_confirmed": "no"}),
            id="canonical",
        ),
        pytest.param(
            "that's not my number",
            WorkerResult(extracted={"phone_confirmation": "no"}),
            id="the-other-field-name",
        ),
    ],
)
async def test_a_declined_number_escalates(said, decision):
    """Identity cannot be verified without the number and the field is
    human_only, so the answer goes to someone who can act on it. This used to
    route to END while telling the caller they were being transferred — no
    AgentCallTransfer, no reference number, and the call reported itself
    finished."""
    out = await _turn(said, decision)
    result = out["interrupt"]

    assert result["next_node"] == "escalation_agent"
    assert result["last_agent_signal"]["status"] == "escalate"
    assert result["phone_update_requested"] is True
    assert "unable to verify" in result["escalation_pre_message"]

    transfers = [
        e
        for e in (result.get("metadata_events") or [])
        if (e.get("data") or {}).get("eventName") == "AgentCallTransfer"
    ]
    assert len(transfers) == 1, "the transfer the caller is promised must be reported"
    assert transfers[0]["data"]["detail"] == "phone_not_confirmed"


def test_the_pre_message_leaves_the_sign_off_to_the_escalation_agent():
    """escalation_agent appends the reference number and the goodbye, so the
    pre-message must not carry one of its own."""
    from agent.agents.verification.handlers import MSG_PHONE_NOT_CONFIRMED

    assert "unable to verify" in MSG_PHONE_NOT_CONFIRMED
    assert "representative" in MSG_PHONE_NOT_CONFIRMED
    assert "Have a great day" not in MSG_PHONE_NOT_CONFIRMED
    assert "Thank you for calling" not in MSG_PHONE_NOT_CONFIRMED
