"""A caller who asks for a moment is told to take their time.

    AI      May I have the reference number of the adjustment request?
    Caller  hold on, let me find the letter...
    AI      Could you say that reference number once more?

    AI      May I have the reference number of the adjustment request?
    Caller  hold on, let me dig out the letter... one second
    AI      Sorry, I didn't catch that — could you repeat the reference number?

The caller was reading us their paperwork and was told we had not heard them —
and it cost one of three retry attempts each time. Two independent faults:

1. detect_wait_request vetoed the second one. After stripping the matched wait
   phrases ("hold on", "one second") it counted the leftover words, and three
   or more meant "a slot value follows, let extraction win". "let me dig out
   the letter" is six words, so the wait was thrown away. Word count is the
   wrong proxy: a reference number is eight digits and a member ID is M plus
   six — six words of English is the opposite of a value. The guard now tests
   for something value-SHAPED instead.

2. reference_number never consulted the wait path at all. _collect_slot has
   always handled waits, and its own two fallback sub-flows (claim number,
   date + amount) both check — but the primary collection is hand-written and
   did not, so the first example failed even though detection worked.

Fault 2 was not one slot's bug. Every hand-written collector had to remember,
and most did not: notification_setup and records_coordination had eight
hand-rolled retries between them with no wait check, and verification's name
correction had none either. So remembering is no longer asked of them —
SlotManagerMixin.wait_ack is the one implementation, and the last test here
fails if a new retry site appears without it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from agent.core.constants import MAX_WAIT_TURNS
from agent.utils import detect_wait_request

# ── the detector ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "hold on, let me dig out the letter... one second",  # the reported turn
        "hold on, let me find the letter...",
        "hold on, let me go grab that paperwork from the other room",
        "bear with me, I think it is in the kitchen drawer somewhere",
        "give me a sec",
        "one second",
        "hold on",
        "just a sec",
        "let me check",
    ],
)
def test_narrating_the_search_is_a_wait(utterance):
    assert detect_wait_request(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        # A value follows the wait phrase — extraction decides, not the ack.
        "hold on... okay it's M451982",
        "hold on, it's 12345678",
        "one second, the number is four five one nine eight two",
        # cannot-provide outranks wait.
        "hold on, I lost the letter",
        "I don't have it",
        # A correction outranks wait.
        "hold on, my ZIP changed",
        # No wait phrase at all.
        "M451982",
        "",
    ],
)
def test_a_value_or_a_denial_still_wins(utterance):
    assert detect_wait_request(utterance) is False


def test_the_guard_is_about_shape_not_length():
    """The regression in one line: leftover prose must not read as a value."""
    from agent.utils import _looks_like_a_value

    assert _looks_like_a_value("let me dig out the letter") is False
    assert _looks_like_a_value("it's m451982") is True
    assert _looks_like_a_value("the number is 12345678") is True
    assert _looks_like_a_value("four five one nine") is True


# ── wait_ack ─────────────────────────────────────────────────────────────────


class _Collector:
    """The mixin under test, with just enough of BaseAgent to run."""

    AGENT_NAME = "claim_adjustment_agent"

    def __init__(self):
        from agent.core.agent import BaseAgent

        class _Agent(BaseAgent):
            AGENT_NAME = "claim_adjustment_agent"

            async def run(self, state):  # pragma: no cover - never called
                raise AssertionError

        self.agent = _Agent()


def _state(said: str, wait_count: int = 0) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "May I have the reference number?"},
            {"role": "user", "content": said},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "reference_number",
        "wait_count": wait_count,
    }


def test_a_wait_costs_no_attempt():
    agent = _Collector().agent
    result = agent.wait_ack(_state("hold on, let me find the letter..."), "reference_number")

    assert result is not None
    assert result["awaiting_slot"] == "reference_number"
    assert result["wait_count"] == 1
    assert agent.get_slot("reference_number").attempt_count == 0, "waiting is not a failed attempt"
    assert "didn't catch" not in result["messages"]["content"]


def test_a_non_wait_turn_returns_nothing():
    agent = _Collector().agent
    assert agent.wait_ack(_state("uh"), "reference_number") is None
    assert agent.wait_ack(_state("12345678"), "reference_number") is None


def test_the_nudge_names_the_slot_after_enough_waits():
    agent = _Collector().agent
    result = agent.wait_ack(
        _state("still looking, give me a sec", wait_count=MAX_WAIT_TURNS - 1),
        "reference_number",
        slot_label="reference number",
    )

    assert result["wait_count"] == MAX_WAIT_TURNS
    assert "reference number" in result["messages"]["content"]


def test_cannot_provide_is_not_a_wait():
    """ "I don't have it" must reach the cannot-provide path, never an ack that
    leaves the caller holding for something they already said they lack."""
    agent = _Collector().agent
    assert agent.wait_ack(_state("hold on — actually I never got the letter"), "reference_number") is None


# ── the structural guarantee ─────────────────────────────────────────────────


def _agent_sources() -> list[pathlib.Path]:
    import agent as agent_pkg

    root = pathlib.Path(agent_pkg.__file__).parent / "agents"
    paths = sorted(root.glob("*/agent.py"))
    assert paths, "no agent modules found"
    return paths


_WAIT_MARKERS = ("wait_ack(", "detect_wait_request", "MSG_WAIT_ACK", "EventType.WAIT")


def test_every_hand_rolled_retry_checks_for_a_wait():
    """A hand-rolled retry burns an attempt and speaks a re-ask. Any such site
    that cannot recognise a wait will tell a caller who asked for a moment that
    they were not heard — which is what reference_number did.

    Slots collected through _collect_slot are covered by the mixin and do not
    appear here; this finds the ones written out by hand.
    """
    offenders: list[str] = []
    for path in _agent_sources():
        source = path.read_text()
        for block in re.split(r"\n    (?=async def |def )", source):
            if "slot_fail(" not in block:
                continue
            speaks_a_retry = "_generate_slot_retry_response" in block or "generate_recovery_message" in block
            if not speaks_a_retry:
                continue
            if any(marker in block for marker in _WAIT_MARKERS):
                continue
            name = block.strip().split("\n")[0].split("(")[0].replace("async def ", "").replace("def ", "")
            offenders.append(f"{path.parent.name}.{name}")

    assert not offenders, (
        "these burn a retry attempt without recognising a wait — call "
        f"self.wait_ack(state, slot, decision=result) before slot_fail: {offenders}"
    )


# Wait acks written out by hand before wait_ack existed, and left in place:
#   claim_adjustment  the reference_number branch plus its four fallback
#                     sub-flows (claim number, DOS + billed, and two retries)
#   follow_up         two conversational fast paths, which deliberately have no
#                     slot and no retry budget to protect
# Each is one more copy of the same rule to keep in step. They are counted
# rather than rewritten — this test's job is to stop the number growing.
_EXISTING_HAND_WRITTEN_WAIT_ACKS = 7


def test_wait_ack_is_the_only_implementation():
    copies = sum(path.read_text().count("pick(MSG_WAIT_ACK)") for path in _agent_sources())
    assert copies <= _EXISTING_HAND_WRITTEN_WAIT_ACKS, (
        f"{copies} hand-written wait acks in agents — new wait handling belongs in "
        "SlotManagerMixin.wait_ack, not another copy"
    )
