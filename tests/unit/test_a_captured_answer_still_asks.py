"""A pipeline that speaks must not leave the turn without its question.

    ai    Would you prefer to receive it by fax or email?
    human Fax. By the way, how do I get a replacement card?
    ai    You can request one from the member portal.      ← no question
    human okay
    ai    I'll send it to 415-555-3299 — is that the right fax number?

The caller answered the slot AND asked something. The follow-up was answered,
the answer was captured — and the fax read-back that should have ridden along
in the same breath came a whole turn later, after the caller said "okay" to a
turn that asked them nothing.

Not a state bug. next_slot in _handle_answered_followup is chosen from
``pending_slots``, which holds only the slots of the pipeline currently
running, and build_delivery_method_pipeline has exactly one:

    DELIVERY_SLOT_ORDER  delivery_method, fax_confirmed, fax, email_confirmed,
                         email, benefits_response     ← the conversation
    pipeline.order       delivery_method              ← what next_slot can see

fax_confirmed is not a pipeline slot at all; it is asked by
_ask_contact_confirmation, on the far side of the ``return interrupt`` that
this turn takes. So remaining is empty, no static ask is appended, and
awaiting_slot is set to "". The next turn recovered only through the
"delivery_method known but awaiting_slot not matched — handles unexpected
re-entry" fallback at the bottom of run().

Every single-slot pipeline has this shape, so the fix is not to teach the
pipeline about slots it does not own: awaits_nothing recognises the turn, and
carry_unasked_turn hands its speech to the agent's own next ask.
"""

from __future__ import annotations

import pytest

from agent.agents.delivery_management import agent as dm
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

ON_FILE_FAX = "4155553299"
ON_FILE_EMAIL = "james.wilson@gmail.com"
ANSWER = "You can request a replacement card from the member portal."
SAID = "Fax. By the way, how do I get a replacement card?"


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


def _state(last_user: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Would you prefer to receive it by fax or email?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_id": "M310188",
        "fax": ON_FILE_FAX,
        "email": ON_FILE_EMAIL,
        "delivery_method": "",
        "awaiting_slot": "delivery_method",
        "provider_type": "Primary Care Physician",
        "zip_code": "58797",
        "zip_code_used": "58797",
    }


@pytest.fixture
def agent(monkeypatch):
    async def _no_guards(*_args, **_kwargs):
        return None

    async def _save(*_args, **_kwargs):
        return None

    monkeypatch.setattr(DeliveryManagementAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(dm, "update_fax_in_salesforce", _save)
    monkeypatch.setattr(dm, "update_email_in_salesforce", _save)
    monkeypatch.setattr(dm, "dispatch_provider_list", _save)
    return DeliveryManagementAgent()


@pytest.fixture
def generation(monkeypatch):
    """Stub LLM 2 — the follow-up answer the pipeline speaks."""
    calls: list[dict] = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return ANSWER

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    return calls


def _with_extraction(monkeypatch, result: WorkerResult):
    async def _extract(*_args, **_kwargs):
        return result

    monkeypatch.setattr(dm, "extract_delivery_management_decision", _extract)


def _answered_with_followup() -> WorkerResult:
    return WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.ANSWER,
        followup_query="how do I get a replacement card",
        extracted={"delivery_method": "fax"},
    )


# ── the transcript ───────────────────────────────────────────────────────────


async def test_the_answer_and_the_next_ask_arrive_in_one_turn(agent, monkeypatch, generation):
    _with_extraction(monkeypatch, _answered_with_followup())

    result = await agent.run(_state(SAID))

    spoken = _text(result)
    assert ANSWER in spoken, "the follow-up answer is still spoken"
    assert ON_FILE_FAX in spoken.replace("-", ""), "and the fax read-back rides along"
    assert spoken.index(ANSWER) < spoken.replace("-", "").index(ON_FILE_FAX), "answer first, then the ask"
    assert spoken.rstrip().endswith("?"), "the turn must leave the caller something to answer"


async def test_the_turn_awaits_the_slot_it_just_asked_for(agent, monkeypatch, generation):
    """awaiting_slot "" is what sent the next turn through the unexpected
    re-entry fallback."""
    _with_extraction(monkeypatch, _answered_with_followup())

    result = await agent.run(_state(SAID))

    assert result["awaiting_slot"] == "fax_confirmed"
    assert result["delivery_method"] == "fax"


async def test_the_answer_is_not_left_pending(agent, monkeypatch, generation):
    """Carried through pending_side_answer, it must be drained by the ask it
    rode in on — not held over to be spoken again on a later turn."""
    _with_extraction(monkeypatch, _answered_with_followup())

    result = await agent.run(_state(SAID))

    assert not (result.get("pending_side_answer") or "")
    assert _text(result).count(ANSWER) == 1


async def test_email_takes_the_same_path(agent, monkeypatch, generation):
    _with_extraction(
        monkeypatch,
        WorkerResult(
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            followup_disposition=FollowupDisposition.ANSWER,
            followup_query="how do I get a replacement card",
            extracted={"delivery_method": "email"},
        ),
    )

    result = await agent.run(_state("Email. By the way, how do I get a replacement card?"))

    assert result["awaiting_slot"] == "email_confirmed"
    assert ANSWER in _text(result)
    assert _text(result).rstrip().endswith("?")


# ── what must not change ─────────────────────────────────────────────────────


async def test_a_clean_answer_is_untouched(agent, monkeypatch, generation):
    """No follow-up, so nothing is carried and the ask stands on its own."""
    _with_extraction(monkeypatch, WorkerResult(extracted={"delivery_method": "fax"}))

    result = await agent.run(_state("Fax, please."))

    assert result["awaiting_slot"] == "fax_confirmed"
    assert ON_FILE_FAX in _text(result).replace("-", "")
    assert ANSWER not in _text(result)
    assert generation == [], "a clean answer costs no generation call"
