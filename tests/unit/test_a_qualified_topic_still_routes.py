"""A topic the caller qualifies is still that topic.

    ai    Your individual deductible is $750 per calendar year. ...
          Your individual out-of-pocket maximum is $3000 per year ...
          One more thing — you're eligible for a complimentary health and
          wellness coach. Would you like me to send you information?
    human Can you summarize the BCP benefits?
    ai    Emily, I can't summarize BCP benefits, but you can find information
          about your health and wellness coach at www dot mysagilityhealth dot
          com ... I'll come back to that in just a moment. ...
    human Can you summarize the PCP benefits?
    ai    I'll come back to that in just a moment. ...
    human yes

The caller asked for a summary of the benefits that had just been read out,
twice, and never got one. Not an ASR problem: "BCP" is a mis-hear of "PCP",
but the correctly transcribed second attempt failed in exactly the same way.

capability_topic was an exact-match lookup. It knew "benefits", "benefit" and
"my_benefits", so any qualifier at all — the one word naming which benefits —
resolved to "", the capability was unroutable, and benefits_agent took its
unknown-topic branch: park as an action, promise to come back, re-ask the
Care Coach offer. The promise is never kept. follow_up routes a parked action
by slot ownership, "pcp_benefits" is in no registry, and an unknown slot is
OWNER_HUMAN — so a caller asking to hear their own deductible again ends up
transferred to a representative.

("replay", "benefits") existed the whole time and stays in-flow with no
routing at all: _replay_benefits re-states the same amounts.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.benefits import agent as benefits_module
from agent.core.slot_ownership import canonical_capability_topic, capability_topic, resolve_capability
from agent.llm.schema import RequestKind, WorkerResult

# ── the transcript ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "target",
    [
        "pcp benefits",
        "pcp_benefits",
        "PCP benefits",
        "bcp benefits",  # the mis-hear resolves the same way
        "my plan benefits",
        "benefits summary",
        "summarize the pcp benefits",
        "office_visit_benefits",
    ],
)
def test_a_qualified_benefits_request_reaches_the_replay(target):
    assert capability_topic(target) == "benefits"
    assert canonical_capability_topic("replay", target) == "benefits"
    assert resolve_capability("replay", target) is not None


@pytest.mark.parametrize(
    "target",
    [
        # What a caller says instead of the word "benefits". _replay_benefits
        # re-states all of these together, so each is vocabulary for the topic.
        "deductible",
        "my deductible",
        "family deductible",
        "coinsurance",
        "co-insurance",
        "copay",
        "out of pocket",
        "out-of-pocket maximum",
        "office visit",
        "cost share",
    ],
)
def test_the_parts_of_the_benefits_resolve_to_it(target):
    assert capability_topic(target) == "benefits"


# ── every target that resolved before resolves the same way ──────────────────


@pytest.mark.parametrize(
    "target, topic",
    [
        ("benefits", "benefits"),
        ("benefit", "benefits"),
        ("my_benefits", "benefits"),
        ("delivery", "delivery"),
        ("delivery_method", "delivery"),
        ("provider_list", "provider_list"),
        ("provider list", "provider_list"),
        ("providers", "provider_list"),
        ("list", "provider_list"),
        ("notification", "notification"),
        ("notification method", "notification"),
        ("claim", "claim_status"),
        ("claim_status", "claim_status"),
        ("claim status", "claim_status"),
        ("my claim", "claim_status"),
    ],
)
def test_the_exact_table_is_unchanged(target, topic):
    assert capability_topic(target) == topic


def test_the_longest_alias_wins():
    """ "claim status" must not lose to "claim", nor "provider list" to "list"."""
    assert capability_topic("that claim status again") == "claim_status"
    assert capability_topic("my provider list") == "provider_list"


@pytest.mark.parametrize(
    "target",
    [
        # Whole words only — the alias "list" is not inside "specialist", and
        # "copay" is not inside "copayments due".
        "specialist",
        "specialists near me",
        # Nothing here names a routable topic, and an unknown target must stay
        # unroutable: it degrades to the question path, never a hard decline.
        "car insurance",
        "my son's school form",
        "",
        "   ",
    ],
)
def test_an_unknown_target_is_still_unknown(target):
    assert capability_topic(target) == ""
    assert canonical_capability_topic("replay", target) == ""


def test_a_redo_of_benefits_is_still_not_routable():
    """Only ("replay", "benefits") exists. Resolving the topic must not invent
    a redo capability the registry does not have."""
    assert capability_topic("pcp benefits") == "benefits"
    assert canonical_capability_topic("redo", "pcp benefits") == ""
    assert resolve_capability("redo", "deductible") is None


# ── the agent no longer parks the request ────────────────────────────────────

SAID = "Can you summarize the PCP benefits?"


async def _turn(target: str, said: str = SAID) -> dict:
    """One care-coach-offer turn carrying a replay request for ``target``."""

    async def _extract(*_args, **_kwargs):
        return WorkerResult(request_kind=RequestKind.REPLAY, update_target=target)

    async def _generate(**_kwargs):
        return "Generated."

    state = {
        "app_run_id": "test-run",
        "slot_attempts": {},
        "benefits_explained": True,
        "individual_deductible": "750",
        "family_deductible": "2500",
        "coinsurance_percent": "20",
        "individual_oop_max": "3000",
        "family_oop_max": "7000",
        "first_name": "Emily",
        "member_id": "M451982",
        "awaiting_slot": "care_coach_response",
        "messages": [
            {"role": "assistant", "content": "Would you like information on the health and wellness coach?"},
            {"role": "user", "content": said},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(benefits_module, "extract_benefits_decision", _extract))
        stack.enter_context(patch.object(benefits_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        return await benefits_module.BenefitsAgent.from_state(state).execute(state)


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


@pytest.mark.parametrize("target", ["pcp benefits", "pcp_benefits", "deductible"])
async def test_the_benefits_are_summarized_instead_of_parked(target):
    result = await _turn(target)

    spoken = _text(result)
    assert "750" in spoken and "3000" in spoken, "the amounts are actually re-stated"
    assert "come back to that" not in spoken, "the request is answered now, not promised"
    assert not result.get("parked_followups"), "nothing is parked, so nothing is owed"


async def test_the_replay_stays_in_flow():
    """No routing: the Care Coach offer is still owed an answer and is re-asked
    in the same turn."""
    result = await _turn("pcp benefits")

    assert result["awaiting_slot"] == "care_coach_response"
    assert not result.get("pending_cross_agent_request")
    assert result.get("next_node") == "benefits_agent"


async def test_an_unknown_topic_still_parks():
    """The unknown-topic branch is not removed — a request this call genuinely
    cannot serve must still park rather than hard-decline."""
    result = await _turn("car insurance", said="Can you quote me for car insurance?")

    assert "come back to that" in _text(result)
    assert result.get("parked_followups")
