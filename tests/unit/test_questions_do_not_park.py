"""A side question is handled in the turn it is asked. Only updates park.

Parking was built to answer a question later in the call: a caller who asked
"will I get this by email?" while giving their ZIP heard "I'll get to that in
a moment", and the question went into parked_followups.

Two things had since made that promise hollow:

  - "Coming up:" gave the generation LLM the steps still ahead, so the question
    is answerable in the sentence the caller is already getting. The park path
    downgraded to FOLLOWUP_RESPOND whenever that list was non-empty — which is
    every slot but the last one in a pipeline.

  - follow_up stopped answering parked questions at all (stale answers read as
    misleading), so what still parked was dropped with a warning. Worse, a
    caller who asked about it was told by _match_promised_item that it was
    "queued and will be answered" — a second promise nothing kept.

So questions no longer park anywhere. What the payload cannot answer is
declined gracefully where it is asked; FOLLOWUP_RESPOND already self-triages
against Confirmed:, Coming up:, and call scope.

kind="action" parking is untouched — an update aimed at a slot another flow
owns in_flow is real work to carry to its owner, and follow_up routes it
through the ownership registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.core.slot_manager import SlotManagerMixin


def resolve_park_guard(guard: str, *, parks_as_action: bool) -> str:
    """Looked up at call time so the source/prompt tests below still run when
    the helper is absent, instead of the whole module failing to import."""
    return SlotManagerMixin.resolve_park_guard(guard, parks_as_action=parks_as_action)


# ── the rule ─────────────────────────────────────────────────────────────────


def test_a_question_never_parks():
    assert resolve_park_guard("FOLLOWUP_PARK", parks_as_action=False) == "FOLLOWUP_RESPOND"


def test_an_update_another_flow_owns_still_parks():
    assert resolve_park_guard("FOLLOWUP_PARK", parks_as_action=True) == "FOLLOWUP_PARK"


def test_the_downgrade_does_not_depend_on_having_steps_ahead():
    """The old rule kept the park when "Coming up:" was empty — the last slot
    of a pipeline, where a promise was most likely to be the final word the
    caller heard on it."""
    assert resolve_park_guard("FOLLOWUP_PARK", parks_as_action=False) == "FOLLOWUP_RESPOND"


@pytest.mark.parametrize(
    "guard", ["FOLLOWUP_RESPOND", "FOLLOWUP_ANSWER", "CORRECTION_ACK", "RETRY", "OFFTOPIC_AGENT"]
)
def test_no_other_guard_is_touched(guard):
    assert resolve_park_guard(guard, parks_as_action=False) == guard
    assert resolve_park_guard(guard, parks_as_action=True) == guard


# ── nothing writes a question item any more ──────────────────────────────────


def test_no_source_file_parks_a_question():
    """Writing kind="question" survives only in normalize_parked_followups,
    which coerces legacy checkpoint entries to that shape on read."""
    writers = []
    for path in Path("src/agent").rglob("*.py"):
        if path.name == "state.py":
            continue
        for line in path.read_text().split("\n"):
            if re.search(r'"kind"\s*:\s*"question"', line):
                writers.append(f"{path.name}: {line.strip()}")
    assert not writers, f"these still park a question: {writers}"


def test_the_benefits_degrade_path_parks_an_action_with_its_target():
    """An unknown redo/replay topic is a registry gap. Parked as an action it
    reaches follow_up's ownership routing, which escalates an unknown target to
    a representative; parked as a question it was dropped, and the "I'll come
    back to that" the caller hears was never honoured."""
    body = Path("src/agent/agents/benefits/agent.py").read_text()
    assert '"kind": "action"' in body
    assert '"target": request_target' in body


# ── the promise helper stops vouching for dropped questions ──────────────────


def _promise(parked: list[dict], query: str) -> str:
    return SlotManagerMixin._match_promised_item(None, {"parked_followups": parked}, query)


def test_a_parked_action_is_still_promised():
    parked = [{"query": "update my email", "kind": "action", "target": "email"}]
    assert "email" in _promise(parked, "when will you update my email?")


def test_a_legacy_parked_question_is_not_promised():
    """follow_up drops it, so telling the caller it is queued is a second
    promise nothing keeps."""
    parked = [{"query": "will I get a text about my claim?", "kind": "question", "target": ""}]
    assert _promise(parked, "what about that text about my claim?") == ""


def test_a_legacy_question_does_not_mask_a_real_action_behind_it():
    """The question used to return early — an action later in the list was
    never reached."""
    parked = [
        {"query": "will I get a text about my claim?", "kind": "question", "target": ""},
        {"query": "update my email", "kind": "action", "target": "email"},
    ]
    assert "email" in _promise(parked, "when will you update my email address?")


# ── the prompts no longer teach a disposition nothing parks ──────────────────


def _extraction(name: str) -> str:
    return " ".join(Path(f"src/agent/prompts/extraction/{name}.md").read_text().lower().split())


@pytest.mark.parametrize("name", ["header", "header_extraction"])
def test_park_is_not_offered_as_a_disposition(name):
    body = _extraction(name)
    assert '"answer" | "none"' in body
    assert "park —" not in body  # the option line, whitespace collapsed


@pytest.mark.parametrize(
    "question",
    [
        "will I get a text/notification when it's sent?",
        "how long will delivery take?",
        "when will I hear back about this?",
    ],
)
def test_the_delivery_questions_are_answered_not_parked(question):
    """These three were the whole case for parking. Read the disposition out of
    the quick-example row rather than scanning the file for the word."""
    rows = [
        line
        for line in Path("src/agent/prompts/extraction/header.md").read_text().split("\n")
        if question in line and line.strip().startswith("|")
    ]
    assert rows, f"quick-example row for {question!r} is gone"
    for row in rows:
        disposition = row.strip().strip("|").split("|")[-1].strip()
        assert disposition == "answer", f"{question!r} still says {disposition!r}"


def test_provider_search_no_longer_forces_park_for_delivery_questions():
    assert '"park"' not in _extraction("provider_search")


def test_the_park_generation_prompt_is_about_updates_only():
    body = " ".join(Path("src/agent/prompts/generation/events/followup_park.md").read_text().lower().split())
    assert "for one thing only" in body
    assert "every side question is answered where it is asked" in body
