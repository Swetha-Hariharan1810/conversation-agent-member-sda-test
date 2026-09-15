"""A decline that also asks something is still a decline.

    ai    The fax number we have on file is 4155553211. Is this correct?
    human Yeah. That's kind of an old fax number. I'll give you a new number
          if you can do that?
    ai    No worries at all, I can update that for you. I'll send it to
          4155553211 — is that the right fax number?
    human Four one five five five five three two one one.

The caller rejected the number and offered a replacement in one breath. They
were read the same number back, agreed to it out of confusion, and the list
went to the fax they had just called old.

The extractor heard the decline — fax_confirmed "no" — and the tail
("I'll give you a new number if you can do that?") was reported as a side
question. is_not_an_answer returned True on the side question alone, without
ever looking at extracted{}, so delivery_management took its "uncertain,
holding, or raising something else" branch: re-read pending_fax or fax_on_file
and burn a retry attempt. The decline branch below it — commented "Anything
else the caller says ... declines the number on file. Ask for the current
one" — was unreachable on this turn.

owned_slots was already the answer and guarded only the update_target test at
the bottom of the function, so the same intent put as a QUESTION rather than
an update walked past it.

This is the failure the module docstring names: "Mistaking a decline for a
non-answer re-asks a question the caller already answered — the failure this
rule exists to stop."

Note the perverse trigger: _recover_missed_followup needs a captured value
before it will recover a question, so with NOTHING extracted the same
utterance always behaved correctly. Only a turn the extractor got right could
fail — see test_extraction_failure_was_hiding_this.
"""

from __future__ import annotations

import pytest

from agent.core.confirmation import is_not_an_answer
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

SAID = "Yeah. That's kind of an old fax number. I'll give you a new number if you can do that?"
# A side question with no change-intent word in it: the caller took no position
# on the number, in the extractor's labels OR in their own words.
NEUTRAL = "Will I get this by email?"
OWNED = ("fax", "fax_confirmed")


def _result(**kwargs) -> WorkerResult:
    base = {
        "event_type": EventType.ANSWERED_WITH_FOLLOWUP,
        "followup_disposition": FollowupDisposition.ANSWER,
        "followup_query": "I'll give you a new number if you can do that?",
    }
    return WorkerResult(**{**base, **kwargs})


# ── the transcript ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("verdict", ["no", "yes"])
def test_a_position_on_the_value_beats_the_side_question(verdict):
    """Either way it is an answer to the read-back — the branch that reads it
    decides what to do, and it must be allowed to run."""
    assert is_not_an_answer(_result(extracted={"fax_confirmed": verdict}), SAID, owned_slots=OWNED) is False


def test_a_replacement_value_is_a_position_too():
    assert is_not_an_answer(_result(extracted={"fax": "4155559999"}), SAID, owned_slots=OWNED) is False


def test_the_side_question_alone_is_still_not_an_answer():
    """Nothing extracted for an owned slot and nothing in the words, so the
    caller took no position and the read-back is still owed one."""
    assert is_not_an_answer(_result(extracted={}), NEUTRAL, owned_slots=OWNED) is True


def test_the_words_carry_the_position_when_the_extractor_reports_none():
    """The reported shape: no fax_confirmed, no update_target, the whole turn
    labelled a side question. Both outs the prompt gives the extractor went
    unused, so the caller's own words are the last thing left."""
    assert is_not_an_answer(_result(extracted={}), SAID, owned_slots=OWNED) is False


@pytest.mark.parametrize(
    "said",
    [
        "That's my old one.",
        "That number has changed.",
        "Can you use a different fax?",
        "That's not right anymore.",
        "We switched providers last spring.",
        "I'll give you a new number if you can do that.",
    ],
)
def test_change_intent_in_the_callers_words_is_a_position(said):
    assert is_not_an_answer(_result(extracted={}), said, owned_slots=OWNED) is False


# ── what must not change ─────────────────────────────────────────────────────


@pytest.mark.parametrize("event", [EventType.AMBIGUOUS, EventType.WAIT])
def test_uncertainty_and_waiting_still_win(event):
    """ "I'm not sure" and "hold on" are not positions, whatever else the
    extractor filled in — these stay ahead of the new rule."""
    result = _result(event_type=event, extracted={"fax_confirmed": "yes"})
    assert is_not_an_answer(result, "I'm not sure, hold on", owned_slots=OWNED) is True


def test_a_wait_read_from_the_words_still_wins():
    result = _result(event_type=EventType.ANSWERED, extracted={"fax_confirmed": "yes"})
    assert is_not_an_answer(result, "hold on, let me dig out the letter", owned_slots=OWNED) is True


def test_an_update_aimed_elsewhere_still_routes():
    """A value for an owned slot plus a request for a DIFFERENT one: the value
    answers the read-back, so this turn is an answer. The routing of the other
    request is the caller's business, not this function's."""
    result = _result(extracted={"fax_confirmed": "no"}, update_target="zip_code")
    assert is_not_an_answer(result, SAID, owned_slots=OWNED) is False

    # ... but with no position taken, it is still a detour.
    assert is_not_an_answer(_result(extracted={}, update_target="zip_code"), SAID, owned_slots=OWNED) is True


def test_a_value_for_an_unowned_slot_is_not_a_position():
    """Only the slots this read-back is about count."""
    result = _result(extracted={"zip_code": "78701"})
    assert is_not_an_answer(result, NEUTRAL, owned_slots=OWNED) is True


def test_an_empty_value_is_not_a_position():
    result = _result(extracted={"fax_confirmed": "", "fax": "   "})
    assert is_not_an_answer(result, NEUTRAL, owned_slots=OWNED) is True


def test_no_owned_slots_is_unchanged():
    """Callers that pass no owned_slots keep the old behaviour exactly — the
    words are never consulted without them."""
    assert is_not_an_answer(_result(extracted={"fax_confirmed": "no"}), SAID) is True
    assert is_not_an_answer(_result(extracted={}), SAID) is True


def test_a_request_for_another_slot_still_routes():
    """The change-intent words must not swallow a request aimed elsewhere:
    "Actually my last name is wrong" during a fax read-back is last_name's,
    and carries "wrong". The extractor placed a target, so it decides."""
    result = _result(extracted={}, update_target="last_name")
    assert is_not_an_answer(result, "Actually my last name is wrong.", owned_slots=OWNED) is True


def test_a_missing_result_is_still_not_an_answer():
    assert is_not_an_answer(None, SAID, owned_slots=OWNED) is True
    assert is_not_an_answer(_result(extracted={"fax_confirmed": "no"}), "", owned_slots=OWNED) is True


# ── through the agent ────────────────────────────────────────────────────────

ON_FILE_FAX = "4155553211"
ON_FILE_EMAIL = "james.wilson@gmail.com"


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


async def _delivery_turn(said: str, extracted: dict, awaiting: str) -> dict:
    import contextlib
    from unittest.mock import patch

    from agent.agents.delivery_management import agent as dm
    from agent.core.request_detection import reconcile_worker_result

    dispatched: dict = {}

    async def _extract(*_a, **_k):
        return reconcile_worker_result(
            WorkerResult(event_type=EventType.ANSWERED, extracted=dict(extracted)), said
        )

    async def _gen(**_k):
        return "No worries at all, I can update that for you."

    async def _save(*_a, **_k):
        return None

    async def _dispatch(_agent, _state, *a, **k):
        dispatched["sent"] = True
        return None

    async def _noguard(*_a, **_k):
        return None

    state = {
        "app_run_id": "r",
        "slot_attempts": {},
        "member_id": "M451982",
        "fax": ON_FILE_FAX,
        "email": ON_FILE_EMAIL,
        "delivery_method": "fax" if awaiting.startswith("fax") else "email",
        "awaiting_slot": awaiting,
        "provider_type": "Primary Care Physician",
        "zip_code": "16783",
        "zip_code_used": "16783",
        "messages": [
            {
                "role": "assistant",
                "content": f"The number we have on file is {ON_FILE_FAX}. Is this correct?",
            },
            {"role": "user", "content": said},
        ],
    }
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(dm, "extract_delivery_management_decision", _extract))
        st.enter_context(patch.object(dm, "get_extraction_llm", lambda: object()))
        st.enter_context(patch.object(dm, "update_fax_in_salesforce", _save))
        st.enter_context(patch.object(dm, "update_email_in_salesforce", _save))
        st.enter_context(patch.object(dm, "dispatch_provider_list", _dispatch))
        st.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _gen))
        st.enter_context(patch.object(dm.DeliveryManagementAgent, "run_conversation_guards", _noguard))
        out = await dm.DeliveryManagementAgent.from_state(state).run(state)
    out["_dispatched"] = bool(dispatched)
    return out


async def test_the_reported_turn_asks_for_the_new_fax():
    result = await _delivery_turn(SAID, {"fax_confirmed": "no"}, "fax_confirmed")

    assert result["awaiting_slot"] == "fax", "the caller is asked for the replacement"
    assert ON_FILE_FAX not in _text(result).replace("-", ""), "the rejected number is not read back again"
    assert not result["_dispatched"], "nothing is sent to a number the caller called old"


async def test_the_same_shape_on_the_email_read_back():
    """is_not_an_answer is shared by six read-backs — fax, email twice, phone
    and zip. The fix is in the shared rule, not in one branch."""
    said = "Yeah, but that's my old address. Can I give you a different one?"
    result = await _delivery_turn(said, {"email_confirmed": "no"}, "email_confirmed")

    assert result["awaiting_slot"] == "email"
    assert ON_FILE_EMAIL not in _text(result)
    assert not result["_dispatched"]


async def test_extraction_failure_was_hiding_this():
    """With nothing extracted the recovery never runs, so followup_query stays
    empty and the decline default was always reached. Only a turn the extractor
    got RIGHT could fail — which is why this never showed up as a bad
    extraction."""
    result = await _delivery_turn(SAID, {}, "fax_confirmed")

    assert result["awaiting_slot"] == "fax"
    assert not result["_dispatched"]
