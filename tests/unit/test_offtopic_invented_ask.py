"""An off-topic decline must not invent an ask the caller never was asked for.

    AI    All set — I've sent that same Pediatrician list to your fax at
          2315553211 as well. Perfect — you should receive it within 30
          minutes. Would you also like me to go over the benefits for office
          visits with your Pediatrician?
    User  <something this line does not handle>
    AI    That's something a representative can help with — for now, would you
          prefer SMS or email for claim status updates?

Nothing in this call ever mentioned claim status updates, and the question on
the table was the benefits offer. The turn was waiting on benefits_response —
a yes/no to that offer — and the payload said:

    Collecting: benefits response

which is not something anyone can be asked for. _SLOT_LABELS had no entry for
benefits_response (nor for nine other slots that reach awaiting_slot), so the
label fell back to the slot name with its underscore removed. Told to redirect
to it, the model reached instead for the nearest askable thing it knew about —
"Setting up notification preferences (SMS or email) for claim status updates",
off the system prompt's list of what this line does.

The slot pipeline sanitizes its own generated re-asks against exactly this
(``collecting_slot``); the guard path did not, so the invented ask went out.
"""

from __future__ import annotations

import re

import pytest

from agent.core.guards import ConversationGuardsMixin
from agent.llm.response_generator import _SLOT_LABELS, _render_payload, sanitize_generated

BENEFITS_OFFER = (
    "All set — I've sent that same Pediatrician list to your fax at 2315553211 as well. "
    "Perfect — you should receive it within 30 minutes. "
    "Would you also like me to go over the benefits for office visits with your Pediatrician?"
)
HISTORY = [
    {"role": "assistant", "content": BENEFITS_OFFER},
    {"role": "user", "content": "Can you also tell me if my appeal went through?"},
]
INVENTED = (
    "That's something a representative can help with — "
    "for now, would you prefer SMS or email for claim status updates?"
)


# ── the payload names the question that is actually pending ──────────────────


def test_the_benefits_offer_is_described_as_the_yes_no_it_is():
    payload = _render_payload(
        slot_name="benefits_response",
        attempt=0,
        guard="OFFTOPIC_AGENT",
        last_messages=HISTORY,
        user_utterance=HISTORY[-1]["content"],
    )
    assert "Collecting: benefits response\n" not in payload
    assert "whether they want to hear the benefits for office visits" in payload
    assert "yes or no" in payload


def _awaiting_slots_in_source() -> set[str]:
    """Every slot name written to state["awaiting_slot"] as a literal."""
    import pathlib

    found: set[str] = set()
    for path in pathlib.Path("src/agent").rglob("*.py"):
        found.update(re.findall(r'\["awaiting_slot"\]\s*=\s*"([a-z_]+)"', path.read_text()))
    found.update({"benefits_response", "care_coach_response", "name_confirmed", "name_correction"})
    return {s for s in found if s}


def test_every_slot_that_can_be_awaited_has_a_label():
    """A slot with no label renders as its own field name. "benefits response"
    and "email confirmed" are not questions, and the model does not ask them —
    it asks something else."""
    missing = sorted(s for s in _awaiting_slots_in_source() if s not in _SLOT_LABELS)
    assert not missing, f"slots reachable as awaiting_slot with no _SLOT_LABELS entry: {missing}"


# ── the sanitizer catches the invented ask ───────────────────────────────────


def test_the_invented_notification_ask_is_stripped_on_the_benefits_turn():
    out = sanitize_generated(
        INVENTED,
        guard="OFFTOPIC_AGENT",
        collecting_slot="benefits_response",
        fallback_text="That's not something I can help with on this call — anything else?",
    )
    assert "SMS or email" not in out
    assert "claim status" not in out


def test_the_fallback_is_used_rather_than_an_ask_for_the_slot_name():
    """_FALLBACKS["OFFTOPIC_AGENT"] ends in "could I get your {slot}?" — which
    for this slot would be "could I get your benefits response?"."""
    out = sanitize_generated(
        INVENTED,
        guard="OFFTOPIC_AGENT",
        collecting_slot="benefits_response",
        fallback_text="That's not something I can help with on this call — anything else?",
    )
    assert "benefits response" not in out
    assert out == "That's not something I can help with on this call — anything else?"


def test_the_right_redirect_survives_sanitizing():
    good = (
        "That's something a representative can help with — "
        "would you still like me to go over those office visit benefits?"
    )
    assert sanitize_generated(good, guard="OFFTOPIC_AGENT", collecting_slot="benefits_response") == good


def test_nothing_being_collected_means_every_slot_ask_is_foreign():
    """collecting_slot="" is not "check nothing" — it is "no ask is allowed"."""
    out = sanitize_generated(
        "That's something a representative can help with — could I get your email address?",
        guard="OFFTOPIC_AGENT",
        collecting_slot="",
        fallback_text="That's not something I can help with — anything else?",
    )
    assert "email address" not in out


@pytest.mark.parametrize(
    ("slot", "ask"),
    [
        ("email_confirmed", "Sorry about that — is that email address still correct?"),
        ("phone_confirmed", "Sorry about that — is that still the best phone number for you?"),
        ("zip_confirmed", "Sorry about that — is your ZIP code still the right one?"),
        ("fax_confirmed", "Sorry about that — is that the right fax number?"),
    ],
)
def test_a_confirmation_may_still_name_the_value_it_confirms(slot, ask):
    """The confirmation slots share their wording with the value slots they
    confirm — stripping the legitimate question would be the worse bug."""
    assert sanitize_generated(ask, guard="OFFTOPIC_AGENT", collecting_slot=slot) == ask


# ── the decline hands back to the question already asked ─────────────────────


def test_the_decline_falls_back_to_the_pending_question():
    out = ConversationGuardsMixin._decline_handoff({"messages": HISTORY})
    assert out.endswith(
        "Would you also like me to go over the benefits for office visits with your Pediatrician?"
    )
    assert out.startswith("That's not something I can help with on this call.")


def test_the_decline_falls_back_to_anything_else_when_no_question_is_pending():
    out = ConversationGuardsMixin._decline_handoff(
        {"messages": [{"role": "assistant", "content": "The list is on its way."}]}
    )
    assert "anything else" in out


# ── the prompt knows the yes/no case ─────────────────────────────────────────


def test_the_prompt_puts_a_yes_no_slot_back_as_a_question():
    body = " ".join(open("src/agent/prompts/generation/events/offtopic_agent.md").read().lower().split())
    assert 'a slot that begins "whether' in body
    assert "would you prefer sms or email for claim status updates?" in body  # named as WRONG


# ── …without silencing the call's own subject ────────────────────────────────
#
#     AI    I understand you'd like to check your claim status, and I can help
#           with that, but I do need your first name to get started.
#     User  How are you doing today?
#     AI    That's not something I can help with on this call. Is there
#           anything else I can help you with today?
#
# A pleasantry, mid-verification, answered with a decline and an offer to end
# the call. The generation was fine; three things above it were not, and all
# three arrived with the fix at the top of this file.


def test_the_callers_own_topic_is_not_a_foreign_slot_ask():
    """ "claim status" and "status updates" were listed as ways of asking for the
    notification channel. They are the caller's TOPIC. On a claim-services call
    every question that named the reason for calling was stripped."""
    kept = "I'm doing well, thanks — shall we carry on with your claim status?"
    assert sanitize_generated(kept, guard="OFFTOPIC_AGENT", collecting_slot="first_name") == kept


def test_the_ask_that_needed_catching_is_still_caught():
    """Narrowing the terms must not give back the hallucination they were added
    for — "sms or email" is the ask, and it is the half that matched."""
    out = sanitize_generated(
        INVENTED, guard="OFFTOPIC_AGENT", collecting_slot="benefits_response", fallback_text="FALLBACK"
    )
    assert out == "FALLBACK"


async def test_with_nothing_collected_an_invented_ask_is_still_stripped(monkeypatch):
    """The strict rule stays: the reported decline was caused by the term list,
    not by this, and loosening it here gives back the original hallucination."""
    from agent.agents.verification.agent import VerificationAgent

    async def _generated(**_kwargs):
        return "That's something a representative can help with — could I get your email address?"

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generated)

    out = await VerificationAgent()._generate_guard_response(
        {
            "messages": [{"role": "assistant", "content": "Is there anything else I can help you with?"}],
            "awaiting_slot": "",
            "slot_attempts": {},
        },
        "OFFTOPIC_AGENT",
    )
    assert "email address" not in out


async def test_a_pending_slot_is_re_asked_rather_than_offered_the_exit(monkeypatch):
    """The decline fallback ends with "anything else I can help you with?" —
    right when the call is on that question, catastrophic when it is three
    turns into taking a name."""
    from agent.agents.verification.agent import VerificationAgent

    async def _generated(**_kwargs):
        return "Would you prefer SMS or email?"  # stripped: foreign to first_name

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generated)

    out = await VerificationAgent()._generate_guard_response(
        {
            "messages": [{"role": "user", "content": "How are you doing today?"}],
            "awaiting_slot": "first_name",
            "slot_attempts": {},
        },
        "OFFTOPIC_AGENT",
    )
    assert "anything else" not in out.lower()
    assert "first name" in out.lower()


async def test_with_nothing_pending_the_open_question_is_still_handed_back(monkeypatch):
    async def _generated(**_kwargs):
        return "Would you prefer SMS or email?"

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generated)

    from agent.agents.verification.agent import VerificationAgent

    out = await VerificationAgent()._generate_guard_response(
        {
            "messages": [
                {"role": "assistant", "content": "Is there anything else I can help you with?"},
                {"role": "user", "content": "How are you doing today?"},
            ],
            "awaiting_slot": "",
            "slot_attempts": {},
        },
        "OFFTOPIC_AGENT",
    )
    assert out.endswith("Is there anything else I can help you with?")
