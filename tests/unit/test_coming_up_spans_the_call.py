"""A question about any later step is answerable, not just the next few slots.

"Coming up:" is how the generation LLM answers a side question about a step the
call has not reached — "will I get a text about this?" during verification —
in the sentence the caller is already getting. What it cannot answer is
declined, now that questions no longer park.

The line used to carry only the remaining slots of the CURRENT agent's
pipeline. So it ran out at the tail of every pipeline, and it was empty for the
whole of verification and intake — which is where a caller is most likely to
ask about something further on, having just been told what the call is about.
The question was declined for no better reason than that the payload stopped at
the edge of one agent.

It now carries the rest of the call: this pipeline's remaining slots first,
then the stages still ahead of the running agent, with the conditional ones
filtered the way the router filters them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.conversation.context import ConversationContext
from agent.core.call_stages import _CLAIM_FLOW, _PROVIDER_FLOW, STAGE_LABELS, remaining_call_stages
from agent.core.slot_manager import SlotManagerMixin

PROVIDER = "provider_services"
CLAIM = "claim_services"


def _stages(agent: str, intent: str = PROVIDER, **state) -> list[str]:
    return remaining_call_stages(intent=intent, current_agent=agent, state=state)


# ── the stages ahead of where the call is ────────────────────────────────────


def test_verification_has_the_whole_provider_flow_ahead():
    assert _stages("verification_agent") == [STAGE_LABELS[a] for a in _PROVIDER_FLOW]


def test_intake_has_the_flow_ahead_too():
    """Intake is where the caller has just heard what the call is for — and
    where they ask what else is coming."""
    assert _stages("intake_agent", CLAIM) == [
        STAGE_LABELS["claim_adjustment_agent"],
        STAGE_LABELS["notification_setup_agent"],
    ]


def test_a_stage_whose_condition_is_not_known_yet_is_not_promised():
    """records_required comes off the claim record, which intake has not read.
    Unknown is not the same as happening: promising to arrange records for a
    claim that turns out not to need them is the false promise all over again."""
    assert STAGE_LABELS["records_coordination_agent"] not in _stages("intake_agent", CLAIM)


def test_a_running_agent_is_not_listed_as_still_ahead():
    ahead = _stages("provider_search_agent")
    assert STAGE_LABELS["provider_search_agent"] not in ahead
    assert STAGE_LABELS["delivery_management_agent"] in ahead


def test_the_last_stage_has_nothing_after_it():
    assert _stages("care_wellness_agent") == []


@pytest.mark.parametrize("agent", ["follow_up_agent", "closure_agent", "escalation_agent"])
def test_an_agent_outside_the_flow_promises_nothing(agent):
    """follow_up is "anything else?" — there is no step ahead to name."""
    assert _stages(agent) == []


def test_an_unknown_intent_promises_nothing():
    assert _stages("verification_agent", "") == []
    assert _stages("verification_agent", "something_else") == []


# ── a stage this call will skip is never promised ────────────────────────────


def test_records_are_only_ahead_when_the_claim_needs_them():
    without = _stages("claim_adjustment_agent", CLAIM, records_required=False)
    with_records = _stages("claim_adjustment_agent", CLAIM, records_required=True)
    assert STAGE_LABELS["records_coordination_agent"] not in without
    assert STAGE_LABELS["records_coordination_agent"] in with_records


def test_records_already_handled_are_not_ahead():
    done = _stages(
        "claim_adjustment_agent", CLAIM, records_required=True, records_branch_taken="member_upload"
    )
    assert STAGE_LABELS["records_coordination_agent"] not in done


@pytest.mark.parametrize("channel", ["sms", "email"])
def test_a_chosen_notification_channel_is_not_ahead(channel):
    assert _stages("claim_adjustment_agent", CLAIM, notification_channel=channel) == []


def test_not_set_still_counts_as_unchosen():
    assert STAGE_LABELS["notification_setup_agent"] in _stages(
        "claim_adjustment_agent", CLAIM, notification_channel="not_set"
    )


@pytest.mark.parametrize(
    "flag, agent",
    [
        ("benefits_explained", "benefits_agent"),
        ("care_coach_offered", "care_wellness_agent"),
    ],
)
def test_a_completed_stage_is_not_promised_again(flag, agent):
    assert STAGE_LABELS[agent] not in _stages("provider_search_agent", **{flag: True})


# ── the two parts, in order ──────────────────────────────────────────────────


class _Agent(SlotManagerMixin):
    AGENT_NAME = "provider_search_agent"


def _coming_up(*, ctx_slots: list, remaining: list, slot_name: str, **state) -> list[str]:
    ctx = ConversationContext(coming_up=ctx_slots)
    return _Agent().build_coming_up(state, ctx=ctx, remaining=remaining, slot_name=slot_name)


def test_this_pipelines_slots_come_before_the_rest_of_the_call():
    out = _coming_up(
        ctx_slots=["zip_code", "delivery_method"], remaining=[], slot_name="", call_intent=PROVIDER
    )
    assert out[:2] == ["zip code", "delivery method"]
    assert STAGE_LABELS["delivery_management_agent"] in out[2:]


def test_the_slot_being_collected_is_not_listed_as_coming():
    out = _coming_up(ctx_slots=["zip_code"], remaining=[], slot_name="zip_code", call_intent=PROVIDER)
    assert "zip code" not in out


def test_the_tail_of_a_pipeline_still_has_the_call_ahead_of_it():
    """This is the case that used to decline: no slots left, nothing to answer
    from, so a question about delivery or benefits got "I can't help with that"."""
    out = _coming_up(ctx_slots=[], remaining=[], slot_name="", call_intent=PROVIDER)
    assert out == [
        STAGE_LABELS["delivery_management_agent"],
        STAGE_LABELS["benefits_agent"],
        STAGE_LABELS["care_wellness_agent"],
    ]


def test_remaining_is_the_fallback_when_the_context_carries_nothing():
    out = _coming_up(ctx_slots=[], remaining=["delivery_method"], slot_name="", call_intent=PROVIDER)
    assert out[0] == "delivery method"


# ── the labels are spoken, not internal ──────────────────────────────────────


def test_every_stage_in_a_flow_has_a_label():
    missing = [a for a in (*_PROVIDER_FLOW, *_CLAIM_FLOW) if a not in STAGE_LABELS]
    assert not missing, f"stages with no spoken label: {missing}"


@pytest.mark.parametrize("label", sorted(STAGE_LABELS.values()))
def test_a_label_reads_as_a_step_not_a_node_name(label):
    assert "_agent" not in label
    assert "_" not in label
    assert label == label.lstrip()  # no stray indentation reaching the payload


def test_the_flows_match_the_agents_the_router_chains():
    """The stage order is the spoken view of fast_path's routing. If a stage is
    added to the router and not here, a caller stops being told about it."""
    router = Path("src/agent/orchestration/fast_path.py").read_text()
    for agent in (*_PROVIDER_FLOW, *_CLAIM_FLOW):
        assert re.search(rf'"{agent}"', router), f"{agent} is not a node the router routes to"


# ── the responder is told what the wider line means ──────────────────────────


def test_the_responder_answers_one_step_and_does_not_recite_the_line():
    body = " ".join(
        Path("src/agent/prompts/generation/events/followup_respond.md").read_text().lower().split()
    )
    assert "name only the step the question is about" in body
    assert "never read the line back" in body


def test_the_responder_is_told_to_match_the_distance():
    """ "In just a moment" is a promise, and it is false for a step four stages
    out."""
    body = " ".join(
        Path("src/agent/prompts/generation/events/followup_respond.md").read_text().lower().split()
    )
    assert "later in this call" in body
    assert "do not promise a step is imminent when it is not" in body
