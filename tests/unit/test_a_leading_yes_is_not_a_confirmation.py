"""The word a turn opens with does not decide the turn.

    ai    Just to confirm — your ZIP code is 16783?
    human yeah. Actually, you know what? I want to update the ZIP code
          because I moved to a new address. So can I do that now?
    ai    Yes, you can update your ZIP code now, and then we'll get that
          provider list to you. Alright — I have a list of in-network
          providers ready for you. Would you like that sent via fax or email?
    human One six seven eight three.
    ai    Just to confirm — fax or email?

The caller asked to change the ZIP and was moved on to delivery with the ZIP
they were replacing still on file. Their next utterance — the ZIP they were
never asked for — landed in the fax-or-email question and parsed as nothing.

The extractor returned zip_confirmed "yes", reading only the opening "yeah".
provider_search acts on a bare "yes" at once, several branches ABOVE
is_not_an_answer, so nothing downstream could undo it: _signal_done, hand off
to delivery.

Two causes, and the first is the one that matters:

  1. The rule against this was written in ONE prompt. delivery_management.md
     had a "Leading-affirmative compound" section, added when a caller was
     read their own stale fax back. Six other prompts collect a confirmation
     slot and none of them had it, so the same "yeah, but actually…" was a
     decline on the fax and a confirmation on the ZIP. It now lives in
     extraction/_confirmation_contract.md, composed into every extraction
     prompt by all three builders, and is no longer stated anywhere else.

  2. A prompt rule cannot be regression-tested, and the cost of missing this
     one is silent. confirms_value is the backstop: a "yes" that arrives with
     change intent in the caller's own words is not a confirmation, and the
     branch falls through to the decline path that asks for the current value.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.core.confirmation import confirms_value
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult

ZIP_ON_FILE = "16783"
SAID = (
    "yeah. Actually, you know what? I want to update the ZIP code "
    "because I moved to a new address. So can I do that now?"
)
ZIP_OWNED = ("zip_code", "zip_confirmed")


# ── confirms_value ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "said",
    [
        SAID,
        "Yeah. That's kind of an old fax number. I'll give you a new number.",
        "Yes, but that number has changed.",
        "Right, although I'd want to update that.",
        "yeah, but can you use a different one?",
        "Sure — though I moved last spring.",
    ],
)
def test_a_qualified_yes_is_not_a_confirmation(said):
    assert confirms_value("yes", said, owned_slots=ZIP_OWNED) is False


@pytest.mark.parametrize(
    "said",
    ["yes", "yeah", "yep", "correct", "that's right", "Yeah, that's the one.", "Yes please."],
)
def test_a_plain_yes_still_confirms(said):
    assert confirms_value("yes", said, owned_slots=ZIP_OWNED) is True


@pytest.mark.parametrize("verdict", ["no", "", "maybe"])
def test_only_a_yes_can_confirm(verdict):
    assert confirms_value(verdict, "yes", owned_slots=ZIP_OWNED) is False


def test_without_owned_slots_the_words_are_never_consulted():
    assert confirms_value("yes", SAID) is True


# ── through the agent ────────────────────────────────────────────────────────


def _text(result: dict) -> str:
    spoken = result.get("messages") or {}
    if isinstance(spoken, list):
        spoken = spoken[-1] if spoken else {}
    return str(spoken.get("content", "") if isinstance(spoken, dict) else "")


async def _zip_turn(extracted: dict, said: str = SAID) -> dict:
    from agent.agents.provider_search import agent as ps
    from agent.core.request_detection import reconcile_worker_result

    async def _extract(*_a, **_k):
        return reconcile_worker_result(
            WorkerResult(
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                followup_disposition=FollowupDisposition.ANSWER,
                followup_query="I want to update the ZIP code. So can I do that now?",
                extracted=dict(extracted),
            ),
            said,
        )

    async def _gen(**_k):
        return "Yes, you can update your ZIP code now."

    async def _save(*_a, **_k):
        return None

    async def _noguard(*_a, **_k):
        return None

    state = {
        "app_run_id": "r",
        "slot_attempts": {},
        "member_id": "M451982",
        "zip_code": ZIP_ON_FILE,
        "awaiting_slot": "zip_confirmed",
        "provider_type": "Primary Care Physician",
        "first_name": "Emily",
        "member_status_verify": True,
        "messages": [
            {"role": "assistant", "content": f"Just to confirm — your ZIP code is {ZIP_ON_FILE}?"},
            {"role": "user", "content": said},
        ],
    }
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(ps, "extract_provider_search_decision", _extract))
        st.enter_context(patch.object(ps, "get_extraction_llm", lambda: object()))
        st.enter_context(patch.object(ps, "update_zip_in_salesforce", _save))
        st.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _gen))
        st.enter_context(patch.object(ps.ProviderSearchAgent, "run_conversation_guards", _noguard))
        return await ps.ProviderSearchAgent.from_state(state).run(state)


async def test_the_reported_turn_asks_for_the_updated_zip():
    result = await _zip_turn({"zip_confirmed": "yes"})

    assert result["awaiting_slot"] == "zip_code", "the caller is asked for the new ZIP"
    assert result.get("next_node") != "delivery_management_agent", "the call does not move on"
    assert "fax or email" not in _text(result).lower()


async def test_a_plain_yes_still_moves_the_call_on():
    """The guard must not block a caller who simply agrees."""
    result = await _zip_turn({"zip_confirmed": "yes"}, said="Yeah, that's right.")

    assert result.get("next_node") == "delivery_management_agent"


# ── the contract reaches every extraction prompt ─────────────────────────────


@pytest.mark.parametrize(
    "builder, agent_file",
    [
        ("build_extraction_prompt", "extraction/records_coordination.md"),
        ("build_extraction_prompt_core", "extraction/benefits.md"),
        ("build_extraction_prompt_extraction", "extraction/provider_search.md"),
        ("build_extraction_prompt_extraction", "extraction/delivery_management.md"),
        ("build_extraction_prompt", "extraction/notification_setup.md"),
    ],
)
def test_every_builder_composes_the_read_back_contract(builder, agent_file):
    """The drift this fixes was one prompt having the rule and six not. A
    builder that stops composing it puts that back."""
    import agent.utils as utils

    prompt = getattr(utils, builder)(agent_file)
    assert "THE READ-BACK CONTRACT" in prompt
    assert "The leading affirmative does not decide the turn" in prompt


def test_the_rule_is_stated_in_exactly_one_file():
    """ "Do not restate any of this in a header or an agent file — a second
    copy is how the drift started.\""""
    from pathlib import Path

    root = Path("src/agent/prompts/extraction")
    carriers = [
        f.name
        for f in sorted(root.glob("*.md"))
        if "The leading affirmative does not decide the turn" in f.read_text()
    ]
    assert carriers == ["_confirmation_contract.md"], carriers
