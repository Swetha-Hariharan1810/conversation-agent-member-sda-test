"""Every read-back confirmation honours a decline without naming the wording.

The ZIP confirmation was fixed first, after a caller was asked twice for a ZIP
they had already given. The same shape ran in four more places — the fax and
email read-backs in delivery_management, the phone and email read-backs in
notification_setup, and the email read-back in records_coordination. Each had
to RECOGNISE a decline before it would ask for the new value, and each carried
its own list to do it with: an extraction prompt full of example phrasings, a
normalize_yes_no pass over the caller's raw words, and in delivery_management a
_STALE_CONTACT_RE of stale-value patterns.

A list of ways to say "that's not my number any more" is never finished. So
none of them recognises declines now. They recognise an affirmation and they
recognise a contact value — a closed set and a shape — and everything else a
caller says in answer to "is this right?" is a decline by default.

core.confirmation.is_not_an_answer holds the one exception: turns that took no
position at all, read off the extraction model's classification rather than the
caller's words.
"""

from __future__ import annotations

import pytest

from agent.core.confirmation import is_not_an_answer
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

# Phrasings that appear in no prompt list and no regex, paired with the empty
# extraction a model returns when it does not recognise them either.
UNLISTED_DECLINES = [
    "That's my old one.",
    "My daughter handles my mail now.",
    "Nah, that hasn't been right since the divorce.",
    "I closed that account years ago.",
    "I would like to change.",
]

# "We relocated last spring" is deliberately NOT in that list. detect_request
# reads it as a ZIP update — which it is — and the Phase-7 reroute hands the
# turn to provider_search, the ZIP's owner, before the confirmation branch sees
# it. That routing runs first by design and this change does not touch it.
ROUTES_TO_THE_ZIP_OWNER = "We relocated last spring."


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


# ── the shared rule ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("utterance", UNLISTED_DECLINES)
def test_an_unrecognised_answer_is_not_a_non_answer(utterance):
    """The model placed nothing — that is a decline, not a turn to re-ask."""
    assert is_not_an_answer(WorkerResult(), utterance, owned_slots=("fax", "fax_confirmed")) is False


@pytest.mark.parametrize(
    "extraction",
    [
        WorkerResult(event_type=EventType.AMBIGUOUS),
        WorkerResult(event_type=EventType.WAIT),
        WorkerResult(
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            followup_disposition=FollowupDisposition.PARK,
            followup_query="will this cost anything?",
        ),
        WorkerResult(event_type=EventType.CORRECTED, update_target="last_name"),
    ],
)
def test_a_turn_that_took_no_position_re_asks(extraction):
    assert is_not_an_answer(extraction, "something", owned_slots=("fax", "fax_confirmed")) is True


def test_an_update_aimed_at_the_slot_itself_is_a_decline():
    """ "change my fax" is declining the fax, not a request to route elsewhere."""
    decision = WorkerResult(event_type=EventType.CORRECTED, update_target="fax")
    assert is_not_an_answer(decision, "change my fax", owned_slots=("fax", "fax_confirmed")) is False


def test_nothing_heard_re_asks():
    assert is_not_an_answer(WorkerResult(), "   ", owned_slots=("fax",)) is True
    assert is_not_an_answer(None, "anything", owned_slots=("fax",)) is True


# ── delivery_management: fax and email read-backs ────────────────────────────


@pytest.fixture
def delivery(monkeypatch):
    from agent.agents.delivery_management import agent as dm

    async def _no_guards(*_args, **_kwargs):
        return None

    async def _ok(*_args, **_kwargs):
        return None

    monkeypatch.setattr(dm.DeliveryManagementAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(dm, "update_fax_in_salesforce", _ok)
    monkeypatch.setattr(dm, "update_email_in_salesforce", _ok)
    return dm


def _delivery_state(method: str, last_user: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "I'll send it to 2155553299 — is that right?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_id": "M310188",
        "delivery_method": method,
        "fax": "2155553299",
        "email": "james.wilson@gmail.com",
        "awaiting_slot": f"{method}_confirmed",
        "provider_type": "Primary Care Physician",
        "zip_code": "58797",
        "zip_code_used": "58797",
    }


@pytest.mark.parametrize("method, next_slot", [("fax", "fax"), ("email", "email")])
@pytest.mark.parametrize("utterance", UNLISTED_DECLINES)
async def test_delivery_asks_for_the_new_contact(delivery, monkeypatch, method, next_slot, utterance):
    async def _extract(*_args, **_kwargs):
        return WorkerResult()

    monkeypatch.setattr(delivery, "extract_delivery_management_decision", _extract)

    agent = delivery.DeliveryManagementAgent()
    result = await agent.run(_delivery_state(method, utterance))

    assert result["awaiting_slot"] == next_slot
    assert result["slot_attempts"].get(f"{method}_confirmed") is None  # no retry burned


@pytest.mark.parametrize("method", ["fax", "email"])
async def test_delivery_re_asks_when_the_caller_is_unsure(delivery, monkeypatch, method):
    async def _extract(*_args, **_kwargs):
        return WorkerResult(event_type=EventType.AMBIGUOUS)

    monkeypatch.setattr(delivery, "extract_delivery_management_decision", _extract)

    agent = delivery.DeliveryManagementAgent()
    result = await agent.run(_delivery_state(method, "I'm not sure."))

    assert result["awaiting_slot"] == f"{method}_confirmed"


# ── records_coordination: email read-back ────────────────────────────────────


@pytest.fixture
def records(monkeypatch):
    from agent.agents.records_coordination import agent as rc

    async def _no_guards(*_args, **_kwargs):
        return None

    monkeypatch.setattr(rc.RecordsCoordinationAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(rc, "get_extraction_llm", lambda: object())
    return rc


def _records_state(last_user: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "I have james.wilson@gmail.com — is that right?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_id": "M310188",
        "email": "james.wilson@gmail.com",
        "awaiting_slot": "email_confirmed",
        "records_required": True,
    }


@pytest.mark.parametrize("utterance", UNLISTED_DECLINES)
async def test_records_asks_for_the_new_email(records, monkeypatch, utterance):
    async def _extract(*_args, **_kwargs):
        return WorkerResult()

    monkeypatch.setattr(records, "extract_records_decision", _extract)

    agent = records.RecordsCoordinationAgent()
    result = await agent.run(_records_state(utterance))

    assert result["awaiting_slot"] == "email"
    assert result["slot_attempts"].get("email_confirmed") is None


async def test_records_re_asks_when_the_caller_is_unsure(records, monkeypatch):
    async def _extract(*_args, **_kwargs):
        return WorkerResult(event_type=EventType.AMBIGUOUS)

    monkeypatch.setattr(records, "extract_records_decision", _extract)

    agent = records.RecordsCoordinationAgent()
    result = await agent.run(_records_state("I'm not sure."))

    assert result["awaiting_slot"] == "email_confirmed"


async def test_a_relocation_still_routes_to_the_zip_owner(delivery, monkeypatch):
    """Unchanged: "we relocated" is a ZIP update, and ZIP has an owner.

    The decline default must not swallow a turn that belongs to another agent —
    the Phase-7 reroute runs first and still wins.
    """

    async def _extract(*_args, **_kwargs):
        return WorkerResult()

    monkeypatch.setattr(delivery, "extract_delivery_management_decision", _extract)

    agent = delivery.DeliveryManagementAgent()
    result = await agent.run(_delivery_state("fax", ROUTES_TO_THE_ZIP_OWNER))

    assert result["awaiting_slot"] != "fax"  # not treated as a fax decline
