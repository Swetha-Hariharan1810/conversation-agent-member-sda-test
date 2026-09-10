"""Static retry fast path — a plain re-ask must not cost a generation-LLM call.

Covers the two halves of the fix:
  1. needs_freeform_response() routes plain retries to build_retry_prompt.
  2. sanitize_generated() strips a foreign-slot ask when the generation LLM
     does run, so a retry can never drift onto a different slot.
"""

from __future__ import annotations

import logging

import pytest

from agent.conversation.context import ConversationContext
from agent.core.slot_manager import SlotManagerMixin
from agent.llm import response_generator
from agent.llm.response_generator import needs_freeform_response, sanitize_generated
from agent.llm.schema import WorkerResult
from agent.responses.builder import build_retry_prompt, has_static_retry
from agent.slots.types import SlotType

# ── needs_freeform_response ──────────────────────────────────────────────────


def test_plain_non_answer_skips_the_generation_llm():
    assert needs_freeform_response(guard="RETRY", decision=WorkerResult()) is False
    assert needs_freeform_response(guard="CLARIFY", decision=WorkerResult()) is False


def test_flagged_turn_uses_the_generation_llm():
    flagged = WorkerResult(needs_freeform_response=True)
    assert needs_freeform_response(guard="RETRY", decision=flagged) is True


@pytest.mark.parametrize(
    "guard",
    ["CORRECTION", "CORRECTION_ACK", "INTERRUPTION", "OFFTOPIC_AGENT", "FOLLOWUP_PARK", "FOLLOWUP_RESPOND"],
)
def test_content_bearing_guards_always_generate(guard):
    assert needs_freeform_response(guard=guard, decision=WorkerResult()) is True


def test_missing_decision_preserves_legacy_behaviour():
    assert needs_freeform_response(guard="RETRY", decision=None) is True


@pytest.mark.parametrize(
    "decision, kwargs",
    [
        (WorkerResult(corrections={"last_name": "Smith"}), {}),
        (WorkerResult(update_target="email"), {}),
        (WorkerResult(followup_query="what is a member id?"), {}),
        (WorkerResult(), {"followup_query": "can you repeat that?"}),
        (WorkerResult(), {"extracted_value": "M907503"}),
    ],
)
def test_caller_content_overrides_a_false_flag(decision, kwargs):
    """A canned re-ask cannot carry these, whatever the model flagged."""
    assert needs_freeform_response(guard="RETRY", decision=decision, **kwargs) is True


# ── build_retry_prompt ───────────────────────────────────────────────────────


@pytest.mark.parametrize("attempt", [1, 2, 3])
@pytest.mark.parametrize(
    "slot_type, term",
    [
        (SlotType.FIRST_NAME, "first name"),
        (SlotType.LAST_NAME, "last name"),
        (SlotType.MEMBER_ID, "member id"),
        (SlotType.DOB, "date of birth"),
        (SlotType.ZIP_CODE, "zip code"),
        (SlotType.REFERENCE_NUMBER, "reference number"),
    ],
)
def test_static_retry_always_re_asks_its_own_slot(slot_type, term, attempt):
    for _ in range(25):  # templates are picked at random — check the whole pool
        msg = build_retry_prompt(slot_type, attempt=attempt).lower()
        assert term in msg


def test_second_attempt_adds_a_format_hint():
    assert "six digits" in build_retry_prompt(SlotType.MEMBER_ID, attempt=2)


def test_unknown_slot_falls_back_to_its_label():
    msg = build_retry_prompt(None, attempt=1, slot_label="upload method")
    assert "upload method" in msg


# ── sanitize_generated: cross-slot hallucination guard ───────────────────────


def test_foreign_slot_ask_is_stripped():
    """The transcript bug: re-asking last_name, the model asked for member_id."""
    out = sanitize_generated(
        "Monique Customer — and your Member ID?",
        guard="RETRY",
        confirmed_labels=("first_name",),
        collecting_slot="last_name",
        fallback_slot_label="last name",
    )
    assert "Member ID" not in out
    assert "last name" in out


def test_own_slot_re_ask_survives():
    text = "Sorry, could you say your last name once more?"
    assert (
        sanitize_generated(text, guard="RETRY", confirmed_labels=("first_name",), collecting_slot="last_name")
        == text
    )


def test_overlapping_phone_slots_are_not_stripped():
    """phone / phone_confirmed / phone_confirmation all say "phone number"."""
    text = "Is 555-867-5309 still the best phone number for you?"
    assert sanitize_generated(text, guard="RETRY", collecting_slot="phone_confirmed") == text


def test_guard_is_off_when_no_collecting_slot_given():
    text = "And your Member ID?"
    assert sanitize_generated(text, guard="FOLLOWUP_PARK") == text


# ── _generate_slot_retry_response wiring ─────────────────────────────────────


class _StubAgent(SlotManagerMixin):
    """Minimal host for the mixin — only what _generate_slot_retry_response uses."""

    AGENT_NAME = "stub"

    def __init__(self):
        self._slots = {}
        self._newly_confirmed = set()
        self.logger = logging.getLogger("stub")


async def test_retry_path_makes_no_llm_call(monkeypatch):
    """The whole point: a plain retry never reaches generate_recovery_message."""

    async def _boom(**_kwargs):
        raise AssertionError("generation LLM was called on a plain retry")

    monkeypatch.setattr(response_generator, "generate_recovery_message", _boom)

    agent = _StubAgent()
    agent.slot_fail("last_name", None, is_asr=True)
    msg = await agent._generate_slot_retry_response(
        {},
        "last_name",
        ConversationContext(confirmed_slots=["first_name"], caller_first_name="Monique"),
        [{"role": "assistant", "content": "And your last name?"}, {"role": "user", "content": "hmm"}],
        decision=WorkerResult(),
        slot_type=SlotType.LAST_NAME,
    )
    assert "last name" in msg.lower()
    assert "member id" not in msg.lower()


async def test_flagged_retry_still_reaches_the_llm(monkeypatch):
    calls = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return "No problem at all — what's your last name?"

    monkeypatch.setattr(response_generator, "generate_recovery_message", _fake)

    agent = _StubAgent()
    agent.slot_fail("last_name", None, is_asr=True)
    msg = await agent._generate_slot_retry_response(
        {},
        "last_name",
        ConversationContext(confirmed_slots=["first_name"]),
        [{"role": "user", "content": "sorry, my dog was barking"}],
        decision=WorkerResult(needs_freeform_response=True),
        slot_type=SlotType.LAST_NAME,
    )
    assert len(calls) == 1
    assert msg == "No problem at all — what's your last name?"


# ── Coverage across agents: slots with no SlotType ───────────────────────────


@pytest.mark.parametrize(
    "slot_name, term",
    [
        ("timeline_question", "timeline"),
        ("upload_consent", "upload link"),
        ("personal_guide_consent", "personal guide"),
        ("upload_method", "records"),
        ("phone_confirmed", "number"),
        ("relationship", "subscriber"),
        ("name_correction", "name"),
    ],
)
def test_slots_without_a_slot_type_have_a_spoken_re_ask(slot_name, term):
    """These slots' names are not spoken nouns — a generic template reads badly."""
    assert has_static_retry(None, slot_name)
    for _ in range(25):
        assert term in build_retry_prompt(None, slot_name=slot_name, attempt=1).lower()


def test_slot_with_no_template_is_kept_on_the_llm_path():
    assert has_static_retry(None, "some_internal_counter") is False
    assert has_static_retry(SlotType.FIRST_NAME, "first_name") is True


def test_confirmation_re_ask_names_the_value():
    msg = build_retry_prompt(None, slot_name="phone_confirmed", value="555-867-5309")
    assert "555-867-5309" in msg


def test_confirmation_without_a_value_still_reads_naturally():
    msg = build_retry_prompt(None, slot_name="phone_confirmed")
    assert "{value}" not in msg
    assert "best number" in msg


async def test_unknown_slot_falls_through_to_the_llm(monkeypatch):
    """No purpose-written template → generate rather than read out a field name."""
    calls = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return "Sorry — could you tell me that again?"

    monkeypatch.setattr(response_generator, "generate_recovery_message", _fake)

    agent = _StubAgent()
    agent.slot_fail("some_internal_counter", None, is_asr=True)
    await agent._generate_slot_retry_response(
        {},
        "some_internal_counter",
        ConversationContext(),
        [{"role": "user", "content": "huh"}],
        decision=WorkerResult(),
    )
    assert len(calls) == 1


async def test_phone_confirmation_retry_is_static_and_names_the_number(monkeypatch):
    async def _boom(**_kwargs):
        raise AssertionError("generation LLM was called on a plain confirmation retry")

    monkeypatch.setattr(response_generator, "generate_recovery_message", _boom)

    agent = _StubAgent()
    agent.slot_fail("phone_confirmed", None, is_asr=True)
    msg = await agent._generate_slot_retry_response(
        {"phone_number": "5558675309"},
        "phone_confirmed",
        ConversationContext(),
        [{"role": "user", "content": "sorry what"}],
        decision=WorkerResult(),
    )
    assert "555-867-5309" in msg


# ── Correction acks are guarded too ──────────────────────────────────────────


def test_correction_ack_may_name_the_corrected_field():
    text = "Got it — I've updated your last name to Smith. And your date of birth?"
    out = sanitize_generated(text, guard="CORRECTION", collecting_slot="dob", exempt_slots=("last_name",))
    assert out == text


def test_correction_ack_still_strips_a_foreign_slot_ask():
    out = sanitize_generated(
        "Got it — I've updated your last name. And your Member ID?",
        guard="CORRECTION",
        collecting_slot="dob",
        exempt_slots=("last_name",),
        fallback_slot_label="date of birth",
    )
    assert "Member ID" not in out
    assert "last name" in out
