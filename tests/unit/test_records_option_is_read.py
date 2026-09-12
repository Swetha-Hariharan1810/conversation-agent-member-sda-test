"""A caller who names a records option gets that option's branch.

    AI      …we'll need a complete copy of the medical records for this
            adjustment. Are you able to provide those?
    Caller  Can I ask my doctor to send it over?
    AI      That's a great question — yes, your doctor can send the records
            over to us.

The caller chose. records_coordination has the branch — a static
acknowledgement plus the upload-link offer, and awaiting_slot moves to
upload_consent — and it was not taken: extraction reported no upload_method, so
the turn burned a retry attempt and generated prose confirming what it had just
declined to act on. Nothing moved.

records_coordination.md lists "my doctor will send it" and "the provider can
send it" under doctor_direct, so the classification was never unspecified. The
caller asked permission, which is how people pick an option, and the question
mark sent the turn down the side-question path.

upload_method was the only branch-selecting slot in the codebase with no
normalizer at all, which made the extraction model a single point of failure for
a three-way choice. screen_upload_method is the deterministic reading, consulted
when extraction produced nothing.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.records_coordination import agent as records_module
from agent.agents.records_coordination.handlers import screen_upload_method
from agent.llm.schema import WorkerResult

REPORTED = "Can I ask my doctor to send it over?"


# ── the screen ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "said",
    [
        REPORTED,
        "my doctor will send it",
        "I'll have my doctor's office send them",
        "the provider can send it",
        "the office will handle it",
        "can my physician fax them over",
        "could the clinic forward them",
    ],
)
def test_the_doctor_sending_is_doctor_direct(said):
    assert screen_upload_method(said) == "doctor_direct"


@pytest.mark.parametrize(
    "said",
    [
        "I can upload it myself",
        "I'll do it online",
        "sure, send me the link",
        "can I upload them",
        "I'll scan them and send them in",
    ],
)
def test_the_caller_sending_is_member_upload(said):
    assert screen_upload_method(said) == "member_upload"


@pytest.mark.parametrize(
    "said",
    [
        "can you contact them for me",
        "could you reach out to my doctor",
        "can you get them from the provider",  # names the provider AND asks us
        "please do that on my behalf",
    ],
)
def test_us_doing_the_chasing_is_personal_guide(said):
    """Order matters: "can you get them from the provider" names the provider,
    so personal_guide has to be tested before doctor_direct."""
    assert screen_upload_method(said) == "personal_guide"


@pytest.mark.parametrize(
    "said",
    [
        # Real clarifications — these must still clarify, not pick a branch.
        "what records do you need exactly?",
        "which records?",
        "how long does this take?",
        "why do you need those?",
        # Not an option either way.
        "hold on, let me check",
        "I don't have them",
        "I'm not sure",
        "",
    ],
)
def test_anything_that_is_not_a_choice_reads_as_no_choice(said):
    assert screen_upload_method(said) == ""


@pytest.mark.parametrize("said", ["no", "no thanks", "I don't want to proceed", "not right now"])
def test_a_decline_is_never_screened(said):
    """Inferring a decline escalates the call — the worst outcome to reach on a
    guess, since a caller who has not refused is handed to a representative they
    did not ask for. That stays with the model."""
    assert screen_upload_method(said) == ""


# ── end to end ───────────────────────────────────────────────────────────────


async def _turn(said: str, extracted: dict | None = None) -> dict:
    async def _extract(*_a, **_k):
        return WorkerResult(extracted=extracted or {})

    async def _generate(**_kw):
        return "GENERATED"

    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "call_intent": "claim_services",
        "reference_number": "12345678",
        "claim_status": "Review",
        "records_required": True,
        "email": "james.wilson@gmail.com",
        "first_name": "James",
        "last_name": "Wilson",
        "member_id": "M451982",
        "awaiting_slot": "upload_method",
        "messages": [
            {"role": "assistant", "content": "Are you able to provide those?"},
            {"role": "user", "content": said},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(records_module, "extract_records_decision", _extract))
        stack.enter_context(patch.object(records_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        return await records_module.RecordsCoordinationAgent.from_state(state).execute(state)


def _retries(result: dict) -> int:
    return (result.get("slot_attempts") or {}).get("upload_method", {}).get("attempt_count", 0)


async def test_the_reported_turn_takes_the_doctor_direct_branch():
    result = await _turn(REPORTED)

    assert result["awaiting_slot"] == "upload_consent", "the branch was not taken"
    assert _retries(result) == 0, "choosing an option is not a failed attempt"
    spoken = (result.get("messages") or {}).get("content") or ""
    assert "upload" in spoken.lower(), "the branch's own upload-link offer should follow the ack"


@pytest.mark.parametrize(
    "said, expected_awaiting",
    [
        ("my doctor will send it", "upload_consent"),
        ("I can upload it myself", "upload_consent"),
        ("can you contact them for me", "personal_guide_consent"),
    ],
)
async def test_each_option_reaches_its_own_branch(said, expected_awaiting):
    result = await _turn(said)

    assert result["awaiting_slot"] == expected_awaiting
    assert _retries(result) == 0


@pytest.mark.parametrize("said", ["what records do you need exactly?", "how long does this take?"])
async def test_a_clarification_still_clarifies(said):
    """The screen must not take the turn away from a caller who asked a real
    question about the options."""
    result = await _turn(said)

    assert result["awaiting_slot"] == "upload_method"
    assert _retries(result) == 1


async def test_extraction_still_wins_when_it_reports_a_value():
    """The screen fills a gap; it never overrides the model."""
    result = await _turn("can you contact them for me", {"upload_method": "member_upload"})

    assert result["awaiting_slot"] == "upload_consent"


# ── two pools, one opener ────────────────────────────────────────────────────


def test_joining_two_messages_does_not_open_twice():
    """The doctor_direct branch speaks an acknowledgement and then an offer, and
    both pools carry their own opener because each is usually a whole turn:

        "Sure, that's fine."  +  "Sure. I can also send a secure link…"

    One in six combinations stuttered.
    """
    import itertools

    from agent.agents.records_coordination.constants import MSG_DOCTOR_DIRECT_ACK, MSG_UPLOAD_OFFER
    from agent.utils import join_turn

    for ack, offer in itertools.product(MSG_DOCTOR_DIRECT_ACK, MSG_UPLOAD_OFFER):
        combined = join_turn(ack, offer)
        tail = combined.split("\n\n", 1)[1]
        assert not tail.lower().startswith(("sure", "thank you", "of course", "okay")), combined
        assert "I can also" in tail, "the offer itself must survive the trim"


def test_join_turn_handles_the_empty_halves():
    from agent.utils import join_turn

    assert join_turn("", "Sure. I can help.") == "Sure. I can help."
    assert join_turn("Of course.", "") == "Of course."
    assert join_turn("", "") == ""


def test_join_turn_keeps_a_second_half_that_is_only_an_opener():
    """Trimming everything would leave the turn saying less than it did."""
    from agent.utils import join_turn

    assert join_turn("Got it.", "Thank you.").endswith("Thank you.")
