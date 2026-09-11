"""A contact given in the same breath as the channel is the caller's answer.

    ai    Would you prefer to receive it by fax or email?
    human Please send the list by text.
    ai    That's not a delivery option we have for provider lists, but I can
          send it by fax or email. Which would you prefer?
    human Please send the list by fax. Use four one five five five five three
          two one one.
    ai    Definitely. The fax number we have on file is 4155553299. Is this
          correct?                                    ← the caller said ...3211
    human Yes.

The caller named the channel and gave the number together. Only the channel
was taken; the number was dropped and the one on file read back in its place,
so the "yes" confirmed a destination the caller had never given and the
provider list was bound for the wrong fax with their apparent agreement.

_maybe_switch_method already did the right thing with a contact carried in a
channel SWITCH ("just email it to jane at example dot com"). The path that
collects the channel the first time did not, and that is the path this
transcript took.
"""

from __future__ import annotations

import pytest

from agent.agents.delivery_management import agent as dm
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import WorkerResult
from agent.slots.normalizers import normalize_delivery_method

ON_FILE_FAX = "4155553299"
GIVEN_FAX = "4155553211"
ON_FILE_EMAIL = "james.wilson@gmail.com"
GIVEN_EMAIL = "jim@example.com"


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


def _state(last_user: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "That's not a delivery option we have for provider lists, "
                "but I can send it by fax or email. Which would you prefer?",
            },
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
    saved: dict = {}

    async def _no_guards(*_args, **_kwargs):
        return None

    async def _save(_agent, _state, value):
        saved["value"] = value
        return None

    monkeypatch.setattr(DeliveryManagementAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(dm, "update_fax_in_salesforce", _save)
    monkeypatch.setattr(dm, "update_email_in_salesforce", _save)
    monkeypatch.setattr(dm, "dispatch_provider_list", _save)

    instance = DeliveryManagementAgent()
    instance.saved = saved
    return instance


def _with_extraction(monkeypatch, result: WorkerResult):
    async def _extract(*_args, **_kwargs):
        return result

    monkeypatch.setattr(dm, "extract_delivery_management_decision", _extract)


# ── the transcript ───────────────────────────────────────────────────────────


async def test_the_fax_the_caller_gave_is_the_one_read_back(agent, monkeypatch):
    _with_extraction(
        monkeypatch,
        WorkerResult(extracted={"delivery_method": "fax", "fax": GIVEN_FAX}),
    )

    result = await agent.run(
        _state("Please send the list by fax. Use four one five five five five three two one one.")
    )

    assert result["awaiting_slot"] == "fax_confirmed"
    assert result["pending_fax"] == GIVEN_FAX
    assert "415-555-3211" in _text(result)
    assert ON_FILE_FAX not in _text(result).replace("-", "")
    # Nothing is written until the caller confirms the read-back.
    assert result["fax"] == ON_FILE_FAX
    assert agent.saved == {}


async def test_an_email_given_with_the_channel_is_read_back_too(agent, monkeypatch):
    _with_extraction(
        monkeypatch,
        WorkerResult(extracted={"delivery_method": "email", "email": GIVEN_EMAIL}),
    )

    result = await agent.run(_state("Email it to jim at example dot com."))

    assert result["awaiting_slot"] == "email_confirmed"
    assert result["pending_email"] == GIVEN_EMAIL
    assert ON_FILE_EMAIL not in _text(result)


# ── unchanged when no contact rides along ────────────────────────────────────


async def test_the_channel_alone_still_confirms_what_is_on_file(agent, monkeypatch):
    _with_extraction(monkeypatch, WorkerResult(extracted={"delivery_method": "fax"}))

    result = await agent.run(_state("Fax, please."))

    assert result["awaiting_slot"] == "fax_confirmed"
    assert ON_FILE_FAX in _text(result).replace("-", "")
    assert not result.get("pending_fax")


async def test_a_contact_that_does_not_validate_falls_back_to_the_one_on_file(agent, monkeypatch):
    """A half-heard number must not be read back as if it were given."""
    _with_extraction(
        monkeypatch,
        WorkerResult(extracted={"delivery_method": "fax", "fax": "415555"}),
    )

    result = await agent.run(_state("Fax it to four one five five five five."))

    assert result["awaiting_slot"] == "fax_confirmed"
    assert ON_FILE_FAX in _text(result).replace("-", "")
    assert not result.get("pending_fax")


async def test_a_contact_for_the_other_channel_is_not_used(agent, monkeypatch):
    """An email in the extraction must not be read back as a fax, or vice versa."""
    _with_extraction(
        monkeypatch,
        WorkerResult(extracted={"delivery_method": "fax", "email": GIVEN_EMAIL}),
    )

    result = await agent.run(_state("Fax, please."))

    assert result["awaiting_slot"] == "fax_confirmed"
    assert GIVEN_EMAIL not in _text(result)
    assert ON_FILE_FAX in _text(result).replace("-", "")


# ── "text" is not a delivery channel, and is not quietly turned into one ─────


@pytest.mark.parametrize("utterance", ["text", "by text", "Please send the list by text.", "text me"])
def test_text_is_not_a_delivery_method(utterance):
    """It must never normalise to fax: a provider list faxed to a mobile is
    PHI sent to a destination the caller did not choose."""
    assert normalize_delivery_method(utterance) == ""
