"""A question about a step that is coming gets answered, not deferred.

A caller who asks "will I get this by email?" while their ZIP is being taken
used to hear "I'll get to that in a moment" and have the question queued for
the end of the call. The reason was not that the answer was unknown — the call
was three turns from asking them to choose fax or email. It was that the
generation LLM was never told what was coming. Its payload carried Confirmed:
and nothing about the road ahead, so a question whose answer lay ahead had no
answer it could give, and promising to return to it was the honest move.

The payload now carries "Coming up:", and such a question routes to
FOLLOWUP_RESPOND — answered in the sentence the caller is already getting.

Two things still park, and they are what parking is for:
  - an ACTION: an update aimed at a slot this agent cannot honour here. That
    is work to carry to its owner, not a question to answer.
  - anything with no "Coming up:" to answer from — promising to return to a
    question still beats telling the caller it cannot be answered.
"""

from __future__ import annotations

from agent.conversation.context import ConversationContext
from agent.core.slot_manager import SlotManagerMixin
from agent.llm.response_generator import _render_payload

HISTORY = [
    {"role": "assistant", "content": "And your ZIP code?"},
    {"role": "user", "content": "90210 — will I receive this by email?"},
]


def _payload(**kwargs) -> str:
    base = {
        "slot_name": "zip_code",
        "attempt": 0,
        "guard": "FOLLOWUP_RESPOND",
        "last_messages": HISTORY,
        "user_utterance": "90210 — will I receive this by email?",
        "extracted_value": "90210",
        "followup_query": "will I receive this by email?",
    }
    base.update(kwargs)
    return _render_payload(**base)


# ── the payload can say what is coming ───────────────────────────────────────


def test_coming_up_reaches_the_generation_llm():
    payload = _payload(coming_up=["delivery method"])
    assert "Coming up: delivery method" in payload


def test_several_steps_are_listed_in_order():
    payload = _payload(coming_up=["zip code", "delivery method"])
    assert "Coming up: zip code, delivery method" in payload


def test_no_coming_up_line_when_there_is_nothing_ahead():
    assert "Coming up:" not in _payload(coming_up=[])
    assert "Coming up:" not in _payload()


def test_the_followup_and_the_road_ahead_are_both_present():
    """The LLM needs the question AND the steps to answer from."""
    payload = _payload(coming_up=["delivery method"])
    assert "Followup: will I receive this by email?" in payload
    assert "Coming up: delivery method" in payload


# ── the context carries it across turns ──────────────────────────────────────


def test_with_coming_up_records_the_steps_on_the_context():
    state = {"conversation_context": {"caller_first_name": "James"}}
    updated = SlotManagerMixin.with_coming_up(state, ["zip_code", "delivery_method"])

    ctx = ConversationContext.from_state(updated)
    assert ctx.coming_up == ["zip_code", "delivery_method"]
    assert ctx.caller_first_name == "James"  # nothing else disturbed


def test_with_coming_up_does_not_mutate_the_state_it_was_given():
    state = {"conversation_context": {"caller_first_name": "James"}}
    SlotManagerMixin.with_coming_up(state, ["zip_code"])
    assert "coming_up" not in state["conversation_context"]


def test_coming_up_survives_a_serialisation_round_trip():
    ctx = ConversationContext(coming_up=["delivery_method"])
    assert ConversationContext.from_dict(ctx.to_dict()).coming_up == ["delivery_method"]


def test_a_context_from_before_this_change_still_loads():
    """Checkpoints written without the field must not break."""
    ctx = ConversationContext.from_dict({"caller_first_name": "James"})
    assert ctx.coming_up == []


def test_with_coming_up_drops_empties():
    state: dict = {}
    updated = SlotManagerMixin.with_coming_up(state, ["zip_code", "", None])
    assert ConversationContext.from_state(updated).coming_up == ["zip_code"]


# ── the prompts agree with the routing ───────────────────────────────────────


def _prompt(name: str) -> str:
    """Prompt body with line wrapping collapsed, so assertions match phrases."""
    from pathlib import Path

    body = Path(f"src/agent/prompts/generation/events/{name}.md").read_text()
    return " ".join(body.lower().split())


def test_respond_is_told_to_answer_from_coming_up_without_deferring():
    body = _prompt("followup_respond")
    assert "coming up:" in body
    assert "this is an answer, not a deferral" in body


def test_respond_is_told_not_to_state_an_outcome_that_has_not_happened():
    """ "you'll choose fax or email" is an answer; "it'll go to your email" is a guess."""
    assert "never state the outcome of a step that has not happened" in _prompt("followup_respond")


def test_park_is_narrowed_to_what_still_parks():
    assert "followup_respond" in _prompt("followup_park")
