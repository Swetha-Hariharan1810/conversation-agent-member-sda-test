"""A caller who declines the value on file and gives the new one is heard once.

    AI      The email address we have on file is james.wilson@gmail.com.
            Is this correct or has it been changed?
    Caller  no, use james.one@example.com
    AI      No problem — what is the correct email address?
    Caller  no, actually use james.two@example.com
    AI      No problem — what is the correct email address?

The caller handed over the new address and was asked for it again. Then again.

The extraction contract says a replacement contact and a yes/no on the
read-back are mutually exclusive, and the model breaks it both ways, so the
confirmation branches compensate. Three of the four compensated in the wrong
direction — clearing the replacement whenever a "no" arrived with it, on the
assumption it was an echo of the "Confirmed:" context line:

    if contact_conf == "no":
        new_email_raw = ""

records_coordination is the clearest case: that line sat three lines above a
branch commented "Inline replacement: member declined AND provided new email in
same utterance", which it made unreachable. delivery_management had already
been fixed to compare the value first; the fix was never carried to the other
three.

Telling an echo from a replacement needs no model — an echo is the value we
just said, a replacement is a different one. core.confirmation.is_read_back_echo
compares them, against what the read-back actually put to the caller (the
pending value when there is one, since a second read-back reads that back).
"""

from __future__ import annotations

import contextlib
import pathlib
import re
from unittest.mock import patch

import pytest

from agent.core.confirmation import is_read_back_echo
from agent.llm.schema import WorkerResult
from agent.slots.normalizers import normalize_email

ON_FILE = "james.wilson@gmail.com"
REPLACEMENT = "james.one@example.com"


# ── the echo test ────────────────────────────────────────────────────────────


def test_a_different_value_is_a_replacement():
    assert is_read_back_echo(REPLACEMENT, ON_FILE, normalize_email) is False


def test_the_value_we_read_back_is_an_echo():
    assert is_read_back_echo(ON_FILE, ON_FILE, normalize_email) is True
    assert is_read_back_echo("JAMES.WILSON@GMAIL.COM", ON_FILE, normalize_email) is True


def test_nothing_to_keep_is_an_echo():
    assert is_read_back_echo("", ON_FILE, normalize_email) is True
    assert is_read_back_echo(None, ON_FILE, normalize_email) is True


def test_the_pending_value_is_what_a_second_read_back_says():
    """After "your email is james.one@example.com, correct?", an echo is
    james.one — not the address still sitting in state as on-file."""
    assert is_read_back_echo(REPLACEMENT, REPLACEMENT, normalize_email) is True
    assert is_read_back_echo("james.two@example.com", REPLACEMENT, normalize_email) is False


# ── end to end, at every confirmation branch ─────────────────────────────────


async def _turn(module, cls, extractor, state_extra, awaiting, said, extracted):
    async def _extract(*_a, **_k):
        return WorkerResult(extracted=extracted)

    async def _generate(**_kw):
        return "GENERATED"

    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "first_name": "James",
        "last_name": "Wilson",
        "member_id": "M451982",
        "email": ON_FILE,
        "phone_number": "5558675309",
        "fax": "2315553211",
        "awaiting_slot": awaiting,
        "messages": [
            {"role": "assistant", "content": "Is this correct or has it been changed?"},
            {"role": "user", "content": said},
        ],
        **state_extra,
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, extractor, _extract))
        stack.enter_context(patch.object(module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        return await getattr(module, cls).from_state(state).execute(state)


def _notification_setup():
    from agent.agents.notification_setup import agent as module

    return (
        module,
        "NotificationSetupAgent",
        "extract_notification_decision",
        {
            "call_intent": "claim_services",
            "records_branch_taken": "member_upload",
            "reference_number": "12345678",
            "notification_channel": "email",
        },
    )


def _records_coordination():
    from agent.agents.records_coordination import agent as module

    return (
        module,
        "RecordsCoordinationAgent",
        "extract_records_decision",
        {
            "call_intent": "claim_services",
            "reference_number": "12345678",
            "claim_status": "open",
            "records_required": True,
        },
    )


def _delivery_management():
    from agent.agents.delivery_management import agent as module

    return (
        module,
        "DeliveryManagementAgent",
        "extract_delivery_management_decision",
        {
            "call_intent": "provider_services",
            "delivery_method": "email",
            "provider_type": "Pediatrician",
            "zip_code": "16783",
            "zip_code_used": "16783",
        },
    )


_EMAIL_SITES = [
    pytest.param(_notification_setup, id="notification_setup"),
    pytest.param(_records_coordination, id="records_coordination"),
    pytest.param(_delivery_management, id="delivery_management"),
]


@pytest.mark.parametrize("site", _EMAIL_SITES)
async def test_a_decline_with_a_new_email_takes_the_new_email(site):
    module, cls, extractor, extra = site()

    result = await _turn(
        module,
        cls,
        extractor,
        extra,
        "email_confirmed",
        f"no, use {REPLACEMENT}",
        {"contact_confirmed": "no", "email_confirmed": "no", "email": REPLACEMENT},
    )

    spoken = (result.get("messages") or {}).get("content") or ""
    assert result.get("pending_email") == REPLACEMENT, "the address the caller gave was thrown away"
    assert "what is the correct email" not in spoken.lower(), "asked again for what they just said"


@pytest.mark.parametrize("site", _EMAIL_SITES)
async def test_a_decline_echoing_the_read_back_is_still_a_decline(site):
    """The case the discard was written for: the model returns "no" alongside
    the address we just read out. That must stay a plain decline."""
    module, cls, extractor, extra = site()

    result = await _turn(
        module,
        cls,
        extractor,
        extra,
        "email_confirmed",
        "no, that's not right",
        {"contact_confirmed": "no", "email_confirmed": "no", "email": ON_FILE},
    )

    assert result.get("awaiting_slot") == "email"
    assert not result.get("pending_email")


@pytest.mark.parametrize("site", _EMAIL_SITES)
async def test_a_plain_decline_still_asks(site):
    module, cls, extractor, extra = site()

    result = await _turn(
        module,
        cls,
        extractor,
        extra,
        "email_confirmed",
        "no, it's changed",
        {"contact_confirmed": "no", "email_confirmed": "no"},
    )

    assert result.get("awaiting_slot") == "email"


async def test_the_phone_branch_too():
    """notification_setup's SMS path carried the same line."""
    module, cls, extractor, extra = _notification_setup()

    result = await _turn(
        module,
        cls,
        extractor,
        {**extra, "notification_channel": "sms"},
        "phone_confirmed",
        "no, use 555-111-2222",
        {"contact_confirmed": "no", "phone": "5551112222"},
    )

    assert result.get("pending_phone") == "5551112222"


async def test_the_fax_branch_too():
    module, cls, extractor, extra = _delivery_management()

    result = await _turn(
        module,
        cls,
        extractor,
        {**extra, "delivery_method": "fax"},
        "fax_confirmed",
        "no, use 555-111-2222",
        {"fax_confirmed": "no", "fax": "5551112222"},
    )

    assert result.get("pending_fax") == "5551112222"


# ── the structural guarantee ─────────────────────────────────────────────────


def test_no_confirmation_branch_clears_a_replacement_unconditionally():
    """The bug in one line, in three files. Any branch that clears the
    replacement on a bare "no" throws away a value the caller gave."""
    import agent as agent_pkg

    root = pathlib.Path(agent_pkg.__file__).parent / "agents"
    offenders: list[str] = []
    pattern = re.compile(
        r'if\s+\w*conf\w*\s*==\s*"no":\s*\n\s+new_\w+_raw\s*=\s*""',
        re.MULTILINE,
    )
    for path in sorted(root.glob("*/agent.py")):
        for match in pattern.finditer(path.read_text()):
            offenders.append(f"{path.parent.name}: {match.group(0).split(chr(10))[0].strip()}")

    assert not offenders, (
        "these discard the caller's replacement on a bare decline — gate the "
        f"clear on core.confirmation.is_read_back_echo: {offenders}"
    )


def test_the_echo_test_has_one_implementation():
    """Five call sites, one definition. delivery_management had its own inline
    comparison and the other three had none, which is how they drifted."""
    import agent as agent_pkg

    root = pathlib.Path(agent_pkg.__file__).parent / "agents"
    sites = sum(path.read_text().count("is_read_back_echo(") for path in sorted(root.glob("*/agent.py")))
    assert sites == 5, f"expected the five confirmation branches to share the helper, found {sites}"
