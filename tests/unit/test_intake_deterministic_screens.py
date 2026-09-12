"""A named specialty and a named appeal end the call the same way every time.

    Caller  Hi, I'm trying to find a neurologist covered under my plan.
    AI      Of course — and what type of provider are you looking for today?

    Caller  I want to appeal my claim denial
    AI      Claim appeals are handled on a different line. How can I help you
            today?

Both should have been a static handoff. Both instead got a generated question,
and the first is the worst sentence available: the caller said "neurologist"
one turn earlier.

Neither failure was in the handlers. handle_unsupported_provider_type and
handle_out_of_scope_intent are both correct, and both read tables that already
knew the answer — _UNSUPPORTED_KEYWORDS has "neurologist", and
OUT_OF_SCOPE_KEYWORD_ROUTING opens with ("appeal", "our appeals team", …).
The failure was that those tables were consulted only AFTER the extraction LLM
had already produced the right tag:

  * the specialty check was gated on intent == provider_services, so a
    specialty misread as "unclear" fell through to handle_unclear_intent,
    which generated an open re-ask;
  * the appeals entry was only reachable from handle_out_of_scope_intent, so a
    misread appeal never got there at all — and the call carried on instead of
    transferring, which is the loop.

intake.md names both cases explicitly ("I want to appeal my claim" →
out_of_scope), so this is not a prompt gap; it is a classification the code
trusted without checking against words it could read for itself.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.agents.intake import agent as intake_module
from agent.agents.intake.handlers import screen_out_of_scope, screen_unsupported_provider_type
from agent.agents.intake.models import IntentTag
from agent.core.signal import AgentStatus
from agent.llm.schema import WorkerResult

NEUROLOGIST = "Hi, I'm trying to find a neurologist covered under my plan."
APPEAL = "I want to appeal my claim denial"

# Every tag the classifier could plausibly return for these two utterances.
_TAGS = [t.value for t in IntentTag]


# ── the screens themselves ───────────────────────────────────────────────────


@pytest.mark.parametrize("tag", ["provider_services", "unclear"])
def test_a_named_specialty_is_read_from_the_words(tag):
    assert screen_unsupported_provider_type(tag, NEUROLOGIST) == "Neurologist"


@pytest.mark.parametrize("tag", ["claim_services", "out_of_scope"])
def test_a_specialty_named_inside_other_business_is_left_alone(tag):
    """ "check my claim for the neurologist visit" names a specialty without
    asking us to search for one."""
    assert screen_unsupported_provider_type(tag, "Checking my claim for the neurologist visit") == ""


@pytest.mark.parametrize(
    "utterance",
    [
        "I need to find a cardiologist",
        "I'm looking for a pediatrician",
        "Can you find me a dermatologist",
        "I need an orthopedic specialist",
        "I need a primary care physician",
        "I need a doctor",  # generic — provider_search asks for the type
    ],
)
def test_a_supported_type_is_never_screened_out(utterance):
    assert screen_unsupported_provider_type("provider_services", utterance) == ""
    assert screen_unsupported_provider_type("unclear", utterance) == ""


@pytest.mark.parametrize("tag", ["unclear", "claim_services", "provider_services"])
def test_an_appeal_is_read_from_the_words(tag):
    assert screen_out_of_scope(tag, APPEAL) is True


def test_an_appeal_already_classified_is_left_to_the_handler():
    assert screen_out_of_scope("out_of_scope", APPEAL) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "I want to check on my claim",
        "I need to find a cardiologist",
        "My claim was denied, what's the status",  # "denied" is not "denial"
        "What's my deductible",
    ],
)
def test_ordinary_business_is_not_screened_as_an_appeal(utterance):
    assert screen_out_of_scope("claim_services", utterance) is False


# ── end to end, through the real agent ───────────────────────────────────────


def _state(said: str) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "How can I help today?"},
            {"role": "user", "content": said},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
    }


async def _turn(said: str, tag: str) -> dict:
    async def _extract(*_a, **_k):
        return WorkerResult(extracted={"intent": tag})

    async def _generate(**_kw):
        raise AssertionError("a deterministic handoff must not call the generation LLM")

    state = _state(said)
    with (
        patch.object(intake_module, "extract_intake_intent", _extract),
        patch.object(intake_module, "get_extraction_llm", lambda: object()),
        patch("agent.llm.response_generator.generate_recovery_message", _generate),
    ):
        return await intake_module.IntakeAgent.from_state(state).execute(state)


def _spoken(result: dict) -> str:
    message = (result.get("messages") or {}).get("content") or ""
    return message or result.get("escalation_pre_message") or ""


@pytest.mark.parametrize("tag", ["provider_services", "unclear", "provider_type_unsupported"])
async def test_the_neurologist_call_always_escalates(tag):
    """Whatever the classifier said, the caller hears the static unsupported-type
    message and is transferred — never asked what type of provider they want."""
    result = await _turn(NEUROLOGIST, tag)

    status = (result.get("last_agent_signal") or {}).get("status")
    assert str(getattr(status, "value", status)) == AgentStatus.ESCALATE.value
    assert result["next_node"] == "escalation_agent"

    spoken = _spoken(result)
    assert "Neurologist" in spoken
    assert "what type of provider" not in spoken.lower()
    assert "Primary Care Physicians" in spoken or "Primary Care Physician" in spoken


@pytest.mark.parametrize("tag", _TAGS)
async def test_the_appeal_call_always_reaches_the_appeals_team(tag):
    result = await _turn(APPEAL, tag)

    assert result["next_node"] == "END"
    assert result["escalation_reason"]
    spoken = _spoken(result)
    assert "appeals team" in spoken
    assert "1-800-555-0105" in spoken
    # The loop: the old reply declined and then re-opened the call.
    assert "how can i help" not in spoken.lower()


@pytest.mark.parametrize("tag", ["claim_services"])
async def test_a_claim_naming_a_specialty_still_goes_to_verification(tag):
    """The screen must not take real work away from the flow that handles it."""
    result = await _turn("Checking my claim for the neurologist visit", tag)

    assert result["next_node"] == "verification_agent"
    assert result["call_intent"] == "claim_services"


@pytest.mark.parametrize(
    "said, tag",
    [
        ("I need to find a cardiologist", "provider_services"),
        ("I want to check on my claim", "claim_services"),
        ("I need a doctor", "provider_services"),
    ],
)
async def test_ordinary_calls_are_untouched(said, tag):
    result = await _turn(said, tag)

    assert result["next_node"] == "verification_agent"
    assert "your first name" in _spoken(result)


# ── an unsupported type the keyword list cannot name ─────────────────────────


async def test_an_unnamed_unsupported_type_still_speaks_english():
    """The classifier says unsupported, the words name nothing the keyword list
    knows ("proctologist"). The decision is still right, but naming it anyway
    rendered the sentinel into the sentence:

        "I can see you're looking for a this provider type."
    """
    result = await _turn("I need to see a proctologist", "provider_type_unsupported")

    spoken = _spoken(result)
    assert "this provider type" not in spoken
    assert "Primary Care Physicians" in spoken
    status = (result.get("last_agent_signal") or {}).get("status")
    assert str(getattr(status, "value", status)) == AgentStatus.ESCALATE.value


def test_the_sentinel_has_one_definition():
    """The screen and the message both branch on "could the type be named?", so
    they read the same constant rather than repeating the string."""
    import inspect

    from agent.agents.intake import handlers
    from agent.agents.intake.constants import PROVIDER_TYPE_UNKNOWN

    source = inspect.getsource(handlers)
    body = source[source.index("def screen_unsupported_provider_type") :]
    assert "PROVIDER_TYPE_UNKNOWN" in body
    assert PROVIDER_TYPE_UNKNOWN == "this provider type"


# ── the suffix rule: the list no longer has to anticipate the caller ─────────


@pytest.mark.parametrize(
    "utterance, named",
    [
        ("I need to see a proctologist", "Proctologist"),  # not on the hand-written list
        ("looking for a hepatologist", "Hepatologist"),
        ("I need a nephrologist", "Nephrologist"),
        ("I need an orthodontist", "Orthodontist"),
        ("need an optometrist", "Optometrist"),
    ],
)
def test_a_specialty_the_list_never_heard_of_is_still_caught(utterance, named):
    from agent.agents.intake.handlers import _extract_provider_type_from_utterance

    assert _extract_provider_type_from_utterance(utterance) == named
    assert screen_unsupported_provider_type("provider_services", utterance) == named
    assert screen_unsupported_provider_type("unclear", utterance) == named


@pytest.mark.parametrize(
    "utterance",
    [
        # Cardiologist and Dermatologist end in "ologist" and Pediatrician in
        # "iatrician" — the suffix rule must not escalate the five it serves.
        "I need a cardiologist",
        "find a dermatologist",
        "I need a pediatrician",
        "I need an orthopedic specialist",
        "a primary care physician",
        "I need a doctor",
        "my kids doctor",
        "heart specialist",
    ],
)
def test_the_suffix_rule_never_catches_a_supported_type(utterance):
    from agent.agents.intake.constants import PROVIDER_TYPE_UNKNOWN
    from agent.agents.intake.handlers import _extract_provider_type_from_utterance

    assert _extract_provider_type_from_utterance(utterance) == PROVIDER_TYPE_UNKNOWN
    assert screen_unsupported_provider_type("provider_services", utterance) == ""


async def test_the_proctologist_call_now_escalates_by_name():
    result = await _turn("I need to see a proctologist", "unclear")

    status = (result.get("last_agent_signal") or {}).get("status")
    assert str(getattr(status, "value", status)) == AgentStatus.ESCALATE.value
    assert "Proctologist" in _spoken(result)
