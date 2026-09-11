"""An off-topic question must not send the caller back to a finished step.

    AI    Sure, I will use the same email address on record, james dot wilson
          at gmail dot com.
          Is there anything else from our call today I can help you with?
    User  Can I also find an inlet for adjuster?
    AI    That's something a representative can help with — for now, could I
          get your email address?

The decline was right; the question after it was not. The caller had already
chosen email and heard it read back, and the open question was "anything
else?" — nothing was being collected.

The payload behind it said otherwise:

    Collecting: preferred channel for claim progress updates — SMS or email
    Event:      OFFTOPIC_AGENT

_generate_guard_response passes state["awaiting_slot"] straight through as the
slot to return to, and _n2_save_and_complete had saved the preference, moved
the call to "anything else?", and left awaiting_slot naming the answered slot.
The off-topic prompt then did exactly what it is told to do: redirect to
"Collecting:". Its first natural pattern is, verbatim, "That's something a
representative can help with — for now, could I get your [slot]?".

Five of the six hand-offs to follow_up left awaiting_slot behind that way.
"""

from __future__ import annotations

import pytest

from agent.llm.response_generator import _render_payload

HISTORY = [
    {"role": "assistant", "content": "Is there anything else I can help you with?"},
    {"role": "user", "content": "Can I also find an inlet for adjuster?"},
]


def _payload(slot_name: str) -> str:
    return _render_payload(
        slot_name=slot_name,
        attempt=0,
        guard="OFFTOPIC_AGENT",
        last_messages=HISTORY,
        user_utterance="Can I also find an inlet for adjuster?",
        confirmed_slots={"first_name": "James"},
    )


# ── the payload no longer invents a step ─────────────────────────────────────


def test_no_slot_is_reported_as_no_slot():
    payload = _payload("")
    assert "no slot is being collected" in payload
    assert "do not ask for or re-confirm any slot" in payload


def test_no_slot_does_not_fall_back_to_the_intent_label():
    """The old fallback said "what they need help with today" — an invented step."""
    assert "what they need help with today" not in _payload("")


def test_a_real_slot_is_still_named():
    assert "Collecting: date of birth" in _payload("dob") or "Collecting: dob" in _payload("dob")


# ── the hand-offs leave nothing behind ───────────────────────────────────────


def _hands_off_clearing_awaiting(path: str) -> list[tuple[int, str]]:
    """Every "next_node = follow_up_agent" line, with whether the slot is cleared."""
    lines = open(path).read().split("\n")
    out = []
    for i, line in enumerate(lines):
        if 'next_node"] = "follow_up_agent"' not in line:
            continue
        window = "\n".join(lines[max(0, i - 6) : i + 8])
        out.append((i + 1, window))
    return out


@pytest.mark.parametrize(
    "path",
    [
        "src/agent/agents/care_wellness/agent.py",
        "src/agent/agents/delivery_management/agent.py",
        "src/agent/agents/notification_setup/agent.py",
        "src/agent/agents/records_coordination/agent.py",
    ],
)
def test_every_follow_up_handoff_clears_awaiting_slot(path):
    """Handing to follow_up means the collection step is over — say so in state.

    A stale awaiting_slot is read by the next turn's guards as a step still in
    progress, and the caller is redirected to a question they already answered.
    """
    for line_no, window in _hands_off_clearing_awaiting(path):
        assert 'awaiting_slot"] = ""' in window, (
            f"{path}:{line_no} hands off to follow_up without clearing awaiting_slot"
        )


# ── the prompt knows both cases ──────────────────────────────────────────────


def _offtopic_prompt() -> str:
    body = open("src/agent/prompts/generation/events/offtopic_agent.md").read()
    return " ".join(body.lower().split())


def test_the_prompt_forbids_inventing_a_slot_when_none_is_being_collected():
    body = _offtopic_prompt()
    assert "nothing is being collected" in body
    assert "must not invent one" in body


def test_the_prompt_distinguishes_a_choice_slot_from_a_value_slot():
    """ "SMS or email" is a choice to offer, not an email address to ask for."""
    body = _offtopic_prompt()
    assert "would you prefer sms or email?" in body
    assert "could i get your email address?" in body  # named as the WRONG form
