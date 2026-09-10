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
from agent.responses.builder import build_retry_prompt
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
