"""
test_ssn_fallback.py — Unit tests for Member ID denial → SSN fallback flow.

These tests are fully deterministic: no LLM calls, no Salesforce calls.
They exercise the VerificationAgent helper methods directly.

Run:
    pytest src/agent/tests/test_ssn_fallback.py -v
"""

from __future__ import annotations

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.agents.verification.constants import (
    MSG_SSN_COLLECT,
    MSG_SSN_EITHER,
    MSG_SSN_INVALID,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> VerificationAgent:
    return VerificationAgent()


def _base_state(**kwargs) -> dict:
    return {
        "app_run_id": "test-run",
        "messages": [],
        "slot_attempts": {},
        "ssn_fallback_stage": "",
        "ssn": "",
        "member_id": "",
        "first_name": "Emily",
        "last_name": "Carter",
        **kwargs,
    }


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# _is_member_id_denial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I don't have it", True),
        ("I don't have my member ID", True),
        ("don't know", True),
        ("can't find it", True),
        ("no member id", True),
        ("I lost it", True),
        ("M907503", False),
        ("yes", False),
        ("Emily Carter", False),
    ],
)
def test_is_member_id_denial(text, expected):
    agent = _make_agent()
    assert agent._is_member_id_denial(text) == expected


# ---------------------------------------------------------------------------
# _ssn_ask_stage — Case 1: YES_WITH_SSN (inline)
# ---------------------------------------------------------------------------


def test_ssn_ask_yes_with_ssn_inline():
    """User replies 'yes, my ssn is 527-41-3820' — accept without asking again."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, "yes, my ssn is 527-41-3820")

    assert result.get("ssn") == "527-41-3820"
    assert result.get("ssn_fallback_stage") == "ssn_lookup"
    assert result.get("is_interrupt") is False


def test_ssn_ask_bare_ssn_inline():
    """User replies with just the SSN number '527-41-3820'."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, "527-41-3820")

    assert result.get("ssn") == "527-41-3820"
    assert result.get("ssn_fallback_stage") == "ssn_lookup"


# ---------------------------------------------------------------------------
# _ssn_ask_stage — Case 1: YES_INTENT
# ---------------------------------------------------------------------------


def test_ssn_ask_yes_intent():
    """User says 'yes' — agent asks for SSN."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, "yes")

    content = result["messages"]["content"] if isinstance(result.get("messages"), dict) else ""
    assert any(m in content for m in MSG_SSN_COLLECT)
    assert result.get("ssn_fallback_stage") == "ssn_collecting"
    assert result.get("is_interrupt") is True


@pytest.mark.parametrize("phrase", ["yeah", "yep", "sure", "okay"])
def test_ssn_ask_yes_variants(phrase):
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, phrase)
    assert result.get("ssn_fallback_stage") == "ssn_collecting"


# ---------------------------------------------------------------------------
# _ssn_ask_stage — Case 2: NO_INTENT (soft no — do not escalate)
# ---------------------------------------------------------------------------


def test_ssn_ask_no_intent():
    """User says 'no' — agent prompts for either Member ID or SSN."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, "no")

    content = result["messages"]["content"] if isinstance(result.get("messages"), dict) else ""
    assert MSG_SSN_EITHER in content
    assert result.get("ssn_fallback_stage") == "ssn_or_mid_retry"
    assert result.get("is_interrupt") is True
    # Must NOT escalate
    assert result.get("next_node") != "escalation_agent"


# ---------------------------------------------------------------------------
# _ssn_ask_stage — Case 3: NO_SSN_AVAILABLE (escalate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "I don't have it",
        "I don't have my ssn",
        "neither",
        "don't know it",
        "can't access it",
        "lost it",
    ],
)
def test_ssn_ask_no_ssn_available_escalates(phrase):
    """User indicates they have no SSN — escalate."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_ask")
    result = agent._ssn_ask_stage(state, phrase)

    # Signal escalate routes to escalation_agent and is not an interrupt
    from agent.orchestration.orchestration import AgentNode

    assert result.get("next_node") == AgentNode.ESCALATION.value
    assert result.get("is_interrupt") is False


# ---------------------------------------------------------------------------
# _ssn_collecting_stage — valid SSN
# ---------------------------------------------------------------------------


def test_ssn_collecting_valid():
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_collecting")
    result = agent._ssn_collecting_stage(state, "527-41-3820")

    assert result.get("ssn") == "527-41-3820"
    assert result.get("ssn_fallback_stage") == "ssn_lookup"
    assert result.get("is_interrupt") is False


def test_ssn_collecting_digits_only():
    """9 raw digits should be normalized to XXX-XX-XXXX."""
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_collecting")
    result = agent._ssn_collecting_stage(state, "527413820")

    assert result.get("ssn") == "527-41-3820"


def test_ssn_collecting_invalid_retries():
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_collecting")
    result = agent._ssn_collecting_stage(state, "not-an-ssn")

    content = result["messages"]["content"] if isinstance(result.get("messages"), dict) else ""
    assert MSG_SSN_INVALID in content
    assert result.get("ssn_fallback_stage") == "ssn_collecting"


def test_ssn_collecting_exhausted_escalates():
    agent = _make_agent()
    state = _base_state(
        ssn_fallback_stage="ssn_collecting",
        slot_attempts={"ssn": {"attempt_count": 2}},
    )
    result = agent._ssn_collecting_stage(state, "bad-input")

    from agent.orchestration.orchestration import AgentNode

    assert result.get("next_node") == AgentNode.ESCALATION.value


# ---------------------------------------------------------------------------
# _ssn_or_mid_retry_stage — Member ID provided
# ---------------------------------------------------------------------------


def test_ssn_or_mid_retry_member_id_provided():
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_or_mid_retry")
    result = agent._ssn_or_mid_retry_stage(state, "M907503")

    assert result.get("member_id") == "M907503"
    assert result.get("ssn_fallback_stage") == ""


def test_ssn_or_mid_retry_ssn_provided():
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_or_mid_retry")
    result = agent._ssn_or_mid_retry_stage(state, "527-41-3820")

    assert result.get("ssn") == "527-41-3820"
    assert result.get("ssn_fallback_stage") == "ssn_lookup"


def test_ssn_or_mid_retry_neither_escalates():
    agent = _make_agent()
    state = _base_state(ssn_fallback_stage="ssn_or_mid_retry")
    result = agent._ssn_or_mid_retry_stage(state, "I don't have either")

    from agent.orchestration.orchestration import AgentNode

    assert result.get("next_node") == AgentNode.ESCALATION.value


# ---------------------------------------------------------------------------
# Denial detection integration — correct stage transition
# ---------------------------------------------------------------------------


def test_member_id_denial_triggers_ssn_ask_stage():
    """_is_member_id_denial returns True for canonical phrases."""
    agent = _make_agent()
    for phrase in [
        "I don't have it",
        "don't have my member ID",
        "I don't know",
        "can't find it",
    ]:
        assert agent._is_member_id_denial(phrase), f"Expected denial for: {phrase!r}"


def test_member_id_provided_does_not_trigger_denial():
    """Valid member ID input must not be flagged as denial."""
    agent = _make_agent()
    assert not agent._is_member_id_denial("M907503")
    assert not agent._is_member_id_denial("em nine zero seven five zero three")
