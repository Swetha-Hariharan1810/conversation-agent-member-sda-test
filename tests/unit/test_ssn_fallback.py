"""SSN fallback: offer it when the Member ID is unavailable, and let the caller back out.

Two production transcripts drive these tests:

  1. "I do not have a member ID" escalated straight to a representative. The
     denial phrase list had "don't have" but not the expanded "do not have",
     so the fallback gate never fired and the generic cannot-provide escalation
     in _collect_slot won the turn.

  2. Inside the SSN gate the caller said "i think i have the memberid now"
     three times and got "Do you have the SSN?" back verbatim each time —
     no intent covered the pivot, and the ambiguous branch had no attempt
     counter, so it could never end.
"""

from __future__ import annotations

import pytest

from agent.agents.verification import agent as verification_agent
from agent.agents.verification.agent import VerificationAgent
from agent.llm.schema import SsnFallbackResult, SsnIntent
from agent.utils import detect_cannot_provide


def _text(result: dict) -> str:
    """The spoken message out of an interrupt or escalation result."""
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    content = spoken.get("content", "") if isinstance(spoken, dict) else ""
    return str(content or result.get("escalation_pre_message") or "")


@pytest.fixture
def agent(monkeypatch):
    """A VerificationAgent with the extraction LLM stubbed out."""
    monkeypatch.setattr(verification_agent, "get_extraction_llm", lambda: object(), raising=False)
    monkeypatch.setattr("agent.llm.config.get_extraction_llm", lambda: object(), raising=False)
    return VerificationAgent()


def _stub_intent(monkeypatch, intent: SsnIntent, ssn: str | None = None):
    async def _fake(*_args, **_kwargs):
        return SsnFallbackResult(ssn_intent=intent, ssn=ssn)

    monkeypatch.setattr(verification_agent, "extract_ssn_decision", _fake)


# ── Bug 1: the fallback has to be offered at all ─────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "I do not have a member ID",  # the transcript — expanded form
        "I don't have a member ID",
        "I do not know my member id",
        "I never received one",
        "I lost my card",
        "I can't find it",
        "I haven't got it",
        "no idea",
    ],
)
def test_member_id_unavailable_triggers_the_ssn_offer(utterance):
    """The gate is phrase-list OR cannot-provide; either one must open the fallback."""
    assert VerificationAgent._is_member_id_denial(utterance) or detect_cannot_provide(utterance)


@pytest.mark.parametrize("utterance", ["M907503", "my member id is M907503", "yes", "sure"])
def test_a_real_answer_does_not_trigger_the_ssn_offer(utterance):
    assert not (VerificationAgent._is_member_id_denial(utterance) or detect_cannot_provide(utterance))


# ── Bug 2: the caller can pivot back to the Member ID ────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "i think i have the memberid now",  # the transcript
        "now i have the member id can you use that",  # the transcript
        "actually I found my member ID",
        "wait, found it",
        "can you use my member id instead",
        "here's my member id",
    ],
)
def test_pivot_back_to_member_id_is_detected(utterance):
    assert VerificationAgent._wants_member_id(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "I don't have my member id",  # denial, not a pivot
        "I never received my member id",
        "yes",
        "no",
        "527-41-3820",
        "um",
    ],
)
def test_non_pivots_are_not_mistaken_for_one(utterance):
    assert VerificationAgent._wants_member_id(utterance) is False


@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("now i have the member id can you use that, it is M907503", "M907503"),
        ("i found it: m 907 503", "M907503"),
        ("i found it, m-907-503", "M907503"),
        ("now i have the member id can you use that", ""),
        ("my ssn is 527-41-3820", ""),
    ],
)
def test_inline_member_id_is_picked_up_on_the_pivot(utterance, expected):
    assert VerificationAgent._extract_member_id_from_text(utterance) == expected


async def test_ssn_ask_pivot_leaves_the_fallback(agent, monkeypatch):
    """The transcript: the third repeat must not happen."""
    _stub_intent(monkeypatch, SsnIntent.AMBIGUOUS)  # model misses it; regex catches it
    last_user = "now i have the member id can you use that"

    result = await agent._ssn_ask_stage({"ssn_fallback_stage": "ssn_ask"}, last_user, [])

    assert result["ssn_fallback_stage"] == ""
    assert result["awaiting_slot"] == "member_id"
    assert "SSN" not in _text(result)


async def test_ssn_ask_pivot_via_the_extraction_flag(agent, monkeypatch):
    _stub_intent(monkeypatch, SsnIntent.HAS_MEMBER_ID)

    result = await agent._ssn_ask_stage({"ssn_fallback_stage": "ssn_ask"}, "found it", [])

    assert result["ssn_fallback_stage"] == ""
    assert result["awaiting_slot"] == "member_id"


async def test_pivot_with_the_id_inline_skips_the_re_ask(agent, monkeypatch):
    _stub_intent(monkeypatch, SsnIntent.HAS_MEMBER_ID)

    result = await agent._ssn_ask_stage({"ssn_fallback_stage": "ssn_ask"}, "found it — M907503", [])

    assert result["member_id"] == "M907503"
    assert result["ssn_fallback_stage"] == ""
    assert result["is_interrupt"] is False


async def test_pivot_works_from_the_collecting_stage_too(agent, monkeypatch):
    _stub_intent(monkeypatch, SsnIntent.AMBIGUOUS)

    result = await agent._ssn_collecting_stage(
        {"ssn_fallback_stage": "ssn_collecting"}, "actually I found my member ID", []
    )

    assert result["ssn_fallback_stage"] == ""
    assert result["awaiting_slot"] == "member_id"


async def test_unresolvable_ssn_gate_escalates_instead_of_looping(agent, monkeypatch):
    """Three verbatim repeats was the bug; the gate is bounded now."""
    _stub_intent(monkeypatch, SsnIntent.AMBIGUOUS)

    state = {"ssn_fallback_stage": "ssn_ask"}
    seen = []
    for _ in range(3):
        result = await agent._ssn_ask_stage(state, "mmhm", [])
        seen.append(result)
        state = {**state, "slot_attempts": result.get("slot_attempts") or {}}

    assert seen[0]["ssn_fallback_stage"] == "ssn_ask"
    assert seen[1]["ssn_fallback_stage"] == "ssn_ask"
    assert (
        seen[2].get("escalate") or seen[2].get("escalation_reason") or not seen[2].get("ssn_fallback_stage")
    ), "third unresolved turn must stop asking"
