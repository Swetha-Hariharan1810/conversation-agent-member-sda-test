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
    """Every module that can answer a caller turn — handlers.py included.

    The first version of this test read only agents/*/agent.py, and four gaps
    hid in the files it skipped (intake's handle_unclear_intent among them).
    """
    import agent as agent_pkg

    root = pathlib.Path(agent_pkg.__file__).parent
    paths = [p for p in sorted(root.glob("agents/**/*.py")) if p.name != "__init__.py"]
    assert paths, "no agent modules found"
    return paths


# Sites that burn an attempt and speak, but that a wait cannot reach. Each is
# here with the reason it cannot, so the list is checkable rather than a mute
# exception:
#
#   _reroute_unhandled_request  gates on detect_request(), which returns None
#                               for a wait — and detect_wait_request itself
#                               defers to detect_request, so the two cannot
#                               both fire.
#   redirect_off_topic          reached only from the OFFTOPIC_AGENT guard
#                               branch, and run_conversation_guards now defers
#                               a wait before any soft guard branch runs.
_UNREACHABLE_BY_A_WAIT = frozenset({"_reroute_unhandled_request", "redirect_off_topic"})


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
        for block in re.split(r"\n(?:    )?(?=async def |def )", source):
            if "slot_fail(" not in block:
                continue
            # "Speaks" is deliberately broad: a retry that re-asks from a static
            # pool costs the caller an attempt exactly as one that generates a
            # sentence does. Reading only the generated form is what let the
            # delivery_management and benefits offers through.
            speaks = (
                "_generate_slot_retry_response" in block
                or "generate_recovery_message" in block
                or "ask_member(" in block
            )
            if not speaks:
                continue
            if any(marker in block for marker in _WAIT_MARKERS):
                continue
            name = block.strip().split("\n")[0].split("(")[0].replace("async def ", "").replace("def ", "")
            if name in _UNREACHABLE_BY_A_WAIT:
                continue
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


# ── a wait is not a decline of the value we just read back ───────────────────


def test_a_wait_on_a_read_back_is_not_a_decline():
    """Worse than a burned retry, and the reason is_not_an_answer had to read
    the words rather than the label:

        AI      Just to be sure — your fax number is 231-555-3211, correct?
        Caller  hold on, let me dig out the letter... one second
        AI      No problem — what is the correct fax number?

    The extractor returned a plain ANSWERED with nothing extracted, so the WAIT
    branch never fired and the turn fell through as "took no position" = False
    — read as the caller declining the number already on file.
    """
    from agent.core.confirmation import is_not_an_answer
    from agent.llm.schema import WorkerResult

    answered_but_waiting = WorkerResult(extracted={})
    assert (
        is_not_an_answer(
            answered_but_waiting,
            "hold on, let me dig out the letter... one second",
            owned_slots=("fax", "fax_confirmed"),
        )
        is True
    )


def test_a_real_decline_is_still_a_decline():
    """The fix must not turn every unclear turn into a wait."""
    from agent.core.confirmation import is_not_an_answer
    from agent.llm.schema import WorkerResult

    declined = is_not_an_answer(
        WorkerResult(extracted={}), "that's my old fax", owned_slots=("fax", "fax_confirmed")
    )
    assert declined is False


# ── the guard layer must not take a wait off the agent ───────────────────────


@pytest.mark.parametrize("guard", ["INTERRUPTION", "OFFTOPIC_GLOBAL", "OFFTOPIC_AGENT"])
async def test_a_soft_guard_defers_to_the_wait(guard):
    """These three take the whole turn, so a wait read as one of them never
    reaches the agent's wait handling — and spends the deflection budget on the
    way past."""
    from agent.core.agent import BaseAgent
    from agent.llm.schema import GuardType, WorkerResult

    class _Agent(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, state):
            return await self.run_conversation_guards(
                state, user_text=state["messages"][-1]["content"], result=self._result
            ) or self.ask_member(state, "reached the agent")

    agent = _Agent()
    agent._result = WorkerResult(guard=GuardType(guard), guard_confidence=0.95)
    state = _state("hold on, let me dig out the letter... one second")

    result = await agent.execute(state)

    assert result["messages"]["content"] == "reached the agent"
    assert (result.get("slot_attempts") or {}).get("deflected_turns", {}).get("attempt_count", 0) == 0


@pytest.mark.parametrize("guard", ["TRANSFER_REQUEST", "ABUSE", "SELF_HARM"])
async def test_a_hard_guard_still_wins_over_a_wait(guard):
    """ "hold on, get me a human" is a transfer. A wait phrase in front of abuse
    or a safety signal changes nothing."""
    from agent.core.agent import BaseAgent
    from agent.core.signal import AgentStatus
    from agent.llm.schema import GuardType, WorkerResult

    class _Agent(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, state):
            return await self.run_conversation_guards(
                state, user_text=state["messages"][-1]["content"], result=self._result
            ) or self.ask_member(state, "reached the agent")

    agent = _Agent()
    agent._result = WorkerResult(guard=GuardType(guard), guard_confidence=0.95)

    result = await agent.execute(_state("hold on — actually just get me a human"))

    status = (result.get("last_agent_signal") or {}).get("status")
    assert str(getattr(status, "value", status)) == AgentStatus.ESCALATE.value
