"""Naming the channel and giving the contact together happens everywhere.

The provider-list flow dropped the fax a caller gave alongside "by fax" and
read the one on file back instead, so the caller's "yes" confirmed a
destination they had never given. The same sentence shape reaches three more
places, and each dropped the contact the same way:

  - notification_setup, claim notifications: "text me at 415-555-3211"
  - notification_setup, progress updates (N2): the same — and this path SAVES
    AND COMPLETES in one turn with no read-back at all, so a wrong number is
    committed with nothing in the sentence for the caller to catch
  - records_coordination, the upload link: "yes, send it to jim at example dot
    com"

core.confirmation.carried_contact is the one reader of a contact given with
the channel, for fax, email and SMS alike.
"""

from __future__ import annotations

import pytest

from agent.core.confirmation import carried_contact
from agent.llm.schema import WorkerResult

GIVEN_PHONE = "4155553211"
ON_FILE_PHONE = "4155553299"
GIVEN_EMAIL = "jim@example.com"
ON_FILE_EMAIL = "james.wilson@gmail.com"


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


# ── the shared reader ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "channel, extracted, expected",
    [
        ("fax", {"fax": "4155553211"}, "4155553211"),
        ("email", {"email": "jim@example.com"}, "jim@example.com"),
        ("sms", {"phone": "4155553211"}, "4155553211"),
        ("phone", {"phone": "4155553211"}, "4155553211"),
        # The other channel's value is never borrowed.
        ("fax", {"email": "jim@example.com"}, ""),
        ("sms", {"email": "jim@example.com"}, ""),
        # A half-heard value is not one the caller can be held to.
        ("fax", {"fax": "415555"}, ""),
        ("email", {"email": "jim@"}, ""),
        ("sms", {"phone": "415"}, ""),
        # Nothing given.
        ("fax", {}, ""),
        ("email", {}, ""),
        # An unknown channel asks for nothing.
        ("carrier pigeon", {"fax": "4155553211"}, ""),
    ],
)
def test_carried_contact(channel, extracted, expected):
    assert carried_contact(WorkerResult(extracted=extracted), channel) == expected


def test_carried_contact_survives_a_missing_result():
    assert carried_contact(None, "fax") == ""
    assert carried_contact(WorkerResult(), "fax") == ""


# ── notification_setup: claim notifications ──────────────────────────────────


@pytest.fixture
def notify(monkeypatch):
    from agent.agents.notification_setup import agent as ns

    saved: dict = {}

    async def _no_guards(*_args, **_kwargs):
        return None

    async def _save(_agent_or_self, *args, **kwargs):
        saved["args"] = args
        return None

    monkeypatch.setattr(ns.NotificationSetupAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(ns, "get_extraction_llm", lambda: object())
    ns._saved = saved
    return ns


def _notify_state(awaiting: str, last_user: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "How would you like to be notified — text or email?"},
            {"role": "user", "content": last_user},
        ],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_id": "M310188",
        "phone_number": ON_FILE_PHONE,
        "email": ON_FILE_EMAIL,
        "awaiting_slot": awaiting,
        "reference_number": "ADJ-1",
    }


def _with_notify_extraction(monkeypatch, ns, result: WorkerResult):
    async def _extract(*_args, **_kwargs):
        return result

    monkeypatch.setattr(ns, "extract_notification_decision", _extract)


async def test_a_number_given_with_text_is_read_back_not_the_one_on_file(notify, monkeypatch):
    _with_notify_extraction(
        monkeypatch,
        notify,
        WorkerResult(extracted={"notification_method": "sms", "phone": GIVEN_PHONE}),
    )

    agent = notify.NotificationSetupAgent()
    result = await agent.run(_notify_state("notification_method", "Text me at 415-555-3211."))

    assert result["awaiting_slot"] == "phone_confirmed"
    assert result["pending_phone"] == GIVEN_PHONE
    assert "415-555-3211" in _text(result)
    assert ON_FILE_PHONE not in _text(result).replace("-", "")


async def test_an_email_given_with_the_channel_is_read_back(notify, monkeypatch):
    _with_notify_extraction(
        monkeypatch,
        notify,
        WorkerResult(extracted={"notification_method": "email", "email": GIVEN_EMAIL}),
    )

    agent = notify.NotificationSetupAgent()
    result = await agent.run(_notify_state("notification_method", "Email it to jim at example dot com."))

    assert result["awaiting_slot"] == "email_confirmed"
    assert result["pending_email"] == GIVEN_EMAIL
    assert ON_FILE_EMAIL not in _text(result)


async def test_the_channel_alone_still_confirms_the_number_on_file(notify, monkeypatch):
    _with_notify_extraction(monkeypatch, notify, WorkerResult(extracted={"notification_method": "sms"}))

    agent = notify.NotificationSetupAgent()
    result = await agent.run(_notify_state("notification_method", "Text, please."))

    assert result["awaiting_slot"] == "phone_confirmed"
    assert ON_FILE_PHONE in _text(result).replace("-", "")
    assert not result.get("pending_phone")


# ── notification_setup: progress updates (N2), which saves in one turn ───────


async def test_n2_saves_the_number_the_caller_gave(notify, monkeypatch):
    """No read-back on this path, so the saved value must be the right one."""
    saved: dict = {}

    async def _n2_save(self, state, method, contact, msg):
        saved.update({"method": method, "contact": contact, "msg": msg})
        return {"messages": {"role": "assistant", "content": msg}}

    monkeypatch.setattr(notify.NotificationSetupAgent, "_n2_save_and_complete", _n2_save, raising=False)
    _with_notify_extraction(
        monkeypatch,
        notify,
        WorkerResult(extracted={"notification_method": "sms", "phone": GIVEN_PHONE}),
    )

    agent = notify.NotificationSetupAgent()
    await agent.run(_notify_state("n2_notification_method", "Text me at 415-555-3211."))

    assert saved["contact"] == GIVEN_PHONE
    assert "415-555-3211" in saved["msg"]
    # It must not be described as the number already on record.
    assert "on record" not in saved["msg"].lower()
    assert "same" not in saved["msg"].lower()


async def test_n2_still_uses_the_number_on_file_when_none_is_given(notify, monkeypatch):
    saved: dict = {}

    async def _n2_save(self, state, method, contact, msg):
        saved.update({"contact": contact, "msg": msg})
        return {"messages": {"role": "assistant", "content": msg}}

    monkeypatch.setattr(notify.NotificationSetupAgent, "_n2_save_and_complete", _n2_save, raising=False)
    _with_notify_extraction(monkeypatch, notify, WorkerResult(extracted={"notification_method": "sms"}))

    agent = notify.NotificationSetupAgent()
    await agent.run(_notify_state("n2_notification_method", "By text."))

    assert saved["contact"] == ON_FILE_PHONE


# ── records_coordination: the upload link ────────────────────────────────────


@pytest.fixture
def records(monkeypatch):
    from agent.agents.records_coordination import agent as rc

    async def _no_guards(*_args, **_kwargs):
        return None

    monkeypatch.setattr(rc.RecordsCoordinationAgent, "run_conversation_guards", _no_guards, raising=False)
    monkeypatch.setattr(rc, "get_extraction_llm", lambda: object())
    return rc


async def test_records_reads_back_the_email_given_with_the_consent(records, monkeypatch):
    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"upload_consent": "yes", "email": GIVEN_EMAIL})

    monkeypatch.setattr(records, "extract_records_decision", _extract)

    agent = records.RecordsCoordinationAgent()
    result = await agent.run(
        {
            "messages": [
                {"role": "assistant", "content": "Shall I send you a secure upload link?"},
                {"role": "user", "content": "Yes, send it to jim at example dot com."},
            ],
            "app_run_id": "test-run",
            "slot_attempts": {},
            "member_id": "M310188",
            "email": ON_FILE_EMAIL,
            "awaiting_slot": "upload_consent",
            "records_required": True,
        }
    )

    assert result["awaiting_slot"] == "email_confirmed"
    assert result["pending_email"] == GIVEN_EMAIL
    assert ON_FILE_EMAIL not in _text(result)
