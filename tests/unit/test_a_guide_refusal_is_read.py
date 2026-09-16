"""A caller who refuses Personal Guide outreach is not asked again.

    AI      I can have one of our Personal Guides contact your doctor's office
            on your behalf. Would you like us to proceed with that?
    Caller  no i dont want to proceed
    AI      Just to confirm — should we have a Personal Guide reach out to your
            provider for those records?

The caller refused, in the plainest words there are, and the turn spent the
refusal on a retry. The reply is the static re-ask from
responses.builder._RETRY_BY_SLOT_NAME["personal_guide_consent"], reached
whenever the phase reads neither "yes" nor "no" — and on the third pass that
path escalates, handing the caller a representative they never asked for.

Two things put it there, and this file covers both.

The refusal was filed under the neighbouring field. records_coordination.md
listed "I don't want to proceed" under upload_method's decline AND "no I don't
want to proceed" under personal_guide_consent's no, so the caller's sentence was
the documented example for two different fields. The phase read one key and the
model had filled the other. The prompt no longer overlaps, and the phase now
reads the option slot as a position on the offer that was actually made —
"decline" is a no, "personal_guide" is a yes.

And nothing caught the miss. The email_confirmed phase above it treats a turn
that takes no position on its question as a decline (core.confirmation); the
consent phase went straight to the retry, so any turn the model could not place
became an ambiguity. It runs the same backstop now, with the same asymmetry
behind it — except for "yes", which stays with the model, because a Personal
Guide calling a provider is not an outcome to reach on a guess.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.records_coordination import agent as records_module
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

REPORTED = "no i dont want to proceed"

GUIDE_OFFER = (
    "I can have one of our Personal Guides contact your doctor's office "
    "on your behalf. Would you like us to proceed with that?"
)


async def _turn(said: str, result: WorkerResult | None = None) -> dict:
    async def _extract(*_a, **_k):
        return result if result is not None else WorkerResult()

    async def _generate(**_kw):
        return "GENERATED"

    async def _dispatch(*_a, **_k):
        return None  # the Salesforce workflow succeeded

    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "call_intent": "claim_services",
        "reference_number": "42695817",
        "claim_status": "Review",
        "records_required": True,
        "email": "james.wilson@gmail.com",
        "first_name": "James",
        "last_name": "Wilson",
        "member_id": "M451982",
        "awaiting_slot": "personal_guide_consent",
        "messages": [
            {"role": "user", "content": "no thanks"},
            {"role": "assistant", "content": GUIDE_OFFER},
            {"role": "user", "content": said},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(records_module, "extract_records_decision", _extract))
        stack.enter_context(patch.object(records_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        stack.enter_context(patch.object(records_module, "dispatch_personal_guide", _dispatch))
        return await records_module.RecordsCoordinationAgent.from_state(state).execute(state)


def _retries(result: dict) -> int:
    return (result.get("slot_attempts") or {}).get("personal_guide_consent", {}).get("attempt_count", 0)


def _declined(result: dict) -> bool:
    return result.get("records_branch_taken") == "declined_personal_guide"


# ── the reported turn ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "extracted",
    [
        pytest.param({"personal_guide_consent": "no"}, id="on-contract"),
        pytest.param({"upload_method": "decline"}, id="filed-as-the-option"),
        pytest.param({"upload_consent": "no"}, id="filed-as-the-earlier-offer"),
        pytest.param({}, id="placed-nowhere"),
    ],
)
async def test_the_reported_refusal_closes_the_branch(extracted):
    """Wherever the model files the refusal, the caller is not asked again."""
    result = await _turn(REPORTED, WorkerResult(extracted=extracted))

    assert _declined(result), "the refusal did not reach the decline branch"
    assert result["next_node"] == "follow_up_agent"
    assert result["awaiting_slot"] == ""
    assert not result.get("escalate"), "declining an offer is not an escalation"
    assert _retries(result) == 0, "refusing is not a failed attempt"


@pytest.mark.parametrize(
    "said",
    [
        "no",
        "no thanks",
        "no thank you",
        "not right now",
        "maybe some other time",
        "that's not needed",
        "I'll handle it myself",
        "no, my doctor's office will send them directly",
    ],
)
async def test_a_refusal_needs_no_phrasing_recognised(said):
    """The ways of declining an offer do not make a list that finishes, so none
    of these has to be matched — a turn taking no position on the only question
    on the table is read as a no."""
    result = await _turn(said)

    assert _declined(result), f"{said!r} did not close the branch"
    assert _retries(result) == 0


# ── what must still be asked again ───────────────────────────────────────────


@pytest.mark.parametrize(
    "said, result",
    [
        pytest.param("I'm not sure", WorkerResult(event_type=EventType.AMBIGUOUS), id="does-not-know"),
        pytest.param("let me think", WorkerResult(event_type=EventType.WAIT), id="holding"),
        pytest.param("hold on a second", WorkerResult(), id="holding-unlabelled"),
        pytest.param(
            "what would they ask my doctor for?",
            WorkerResult(
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                followup_disposition=FollowupDisposition.ANSWER,
                followup_query="what would they ask my doctor for?",
            ),
            id="side-question",
        ),
    ],
)
async def test_a_turn_that_takes_no_position_still_asks(said, result):
    """A caller who does not know, is not ready, or asked something else has not
    refused — reading a decline into those turns is the mirror of the bug."""
    out = await _turn(said, result)

    assert not _declined(out), f"{said!r} was read as a refusal"
    assert out["awaiting_slot"] == "personal_guide_consent"


# ── consent stays with the model ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "extracted",
    [
        pytest.param({"personal_guide_consent": "yes"}, id="on-contract"),
        pytest.param({"upload_method": "personal_guide"}, id="filed-as-the-option"),
    ],
)
async def test_consent_reaches_the_guide_branch(extracted):
    """The option slot is read both ways — and this is why it is read before the
    decline backstop: a yes filed under upload_method would otherwise reach the
    backstop and be read as a no."""
    result = await _turn("yes please, go ahead", WorkerResult(extracted=extracted))

    assert result["records_branch_taken"] == "personal_guide"
    assert result["personal_guide_outreach_requested"] is True


async def test_the_prompt_no_longer_files_a_refusal_in_two_places():
    """The root cause: one sentence, two documented fields. Both the phase's
    backstop and the prompt fix are needed — the backstop keeps the call moving
    when the model misfiles a refusal, and the prompt stops it misfiling."""
    from agent.utils import build_extraction_prompt_extraction

    prompt = build_extraction_prompt_extraction("extraction/records_coordination.md")

    guide_field = prompt.index('personal_guide_consent  "yes" | "no"')
    decline_option = prompt.index("    decline — ")

    assert "I don't want to proceed" in prompt[decline_option:guide_field], (
        "upload_method's decline should still name the phrasing it does own"
    )
    assert "only while upload_method is the awaiting slot" in prompt[decline_option:guide_field], (
        "upload_method's decline must be scoped to its own turn"
    )
