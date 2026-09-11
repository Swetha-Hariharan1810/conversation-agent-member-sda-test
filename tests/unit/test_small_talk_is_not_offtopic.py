"""A caller being friendly is not raising a topic.

    AI    I understand you'd like to check your claim status, and I can help
          with that, but first I need your first name.
    User  How are you doing today?
    AI    That's a question for our pharmacy team rather than this line, but
          first, could I get your first name?

Nobody mentioned pharmacy. That sentence is the first of four example declines
in global_generation.md, offered as a menu of phrasings for an out-of-scope
question — and a model handed a list will take an item off it. What sent the
turn to that list at all is that small talk fitted nothing else: the scope
section has a "handles" bucket and a "cannot help with" bucket, a pleasantry
is in neither, and the extraction guard reads OFFTOPIC_GLOBAL as "unrelated to
healthcare member services", which a greeting literally is.

So it was declined, and — mid-collection, past the first attempt — the
off-topic branch in guards.py also spends one of the caller's retry attempts
on it, moving them toward an escalation for the crime of saying hello.

Small talk is now a case in its own right in both layers: guard NONE at
extraction, answered in a few words at generation, then straight on with what
was being collected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPTS = Path("src/agent/prompts")


def _flat(path: str) -> str:
    return " ".join((PROMPTS / path).read_text().lower().split())


# ── extraction: it is not a guard ────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "extraction/header.md",
        "extraction/header_core.md",
        "extraction/header_extraction.md",
        "extraction/intake.md",
        "extraction/follow_up.md",
        "extraction/follow_up_claims.md",
    ],
)
def test_every_prompt_that_defines_offtopic_also_excludes_small_talk(path):
    """The definition and the exception have to travel together — a prompt
    carrying "unrelated to healthcare" without it classifies a greeting as a
    topic."""
    body = _flat(path)
    assert "offtopic_global" in body, f"{path} no longer defines the guard — update this test"
    assert "small talk is not off-topic" in body
    assert "how are you doing today?" in body


def test_the_cost_of_getting_it_wrong_is_written_down():
    """Not just "don't" — why: the off-topic branch spends a retry attempt."""
    assert "retry attempts" in _flat("extraction/header.md")


# ── generation: it is answered, not declined ─────────────────────────────────


def test_the_shared_prompt_treats_small_talk_as_in_scope():
    body = _flat("system/global_generation.md")
    assert "small talk is not out of scope" in body
    assert "i'm doing well, thank you for asking" in body


def test_the_pharmacy_line_is_bound_to_pharmacy():
    """It was one of four interchangeable examples; it is now the answer to a
    question about medication and to nothing else."""
    body = _flat("system/global_generation.md")
    assert "asked about a prescription or medication:" in body
    assert "never borrow one of the lines above for a topic it does not mention" in body


def test_an_unlisted_topic_is_declined_in_our_own_words():
    """The failure mode is reaching for the nearest listed line. Say there is
    no line and what to do instead."""
    assert "naming what they asked about" in _flat("system/global_generation.md")


def test_the_offtopic_event_knows_there_is_nothing_to_decline():
    body = _flat("generation/events/offtopic_agent.md")
    assert "small talk rather than a request" in body
    assert "there is nothing to decline" in body


def test_the_offtopic_event_names_the_wrong_answer():
    """The exact sentence the caller got, pinned as WRONG where the model
    reading this prompt will see it."""
    assert "pharmacy team rather than this line" in _flat("generation/events/offtopic_agent.md")


# ── the rule reaches every generated turn ────────────────────────────────────


@pytest.mark.parametrize("guard", ["RETRY", "CLARIFY", "OFFTOPIC_AGENT", "FOLLOWUP_RESPOND", "INTERRUPTION"])
def test_small_talk_guidance_is_in_every_guard_prompt(guard):
    """global_generation.md is included in every assembled generation prompt,
    so the rule holds whichever guard the turn lands on — which matters here,
    because the same greeting has been seen to arrive as both RETRY and
    OFFTOPIC_AGENT."""
    from agent.utils import build_generation_prompt

    assert "Small talk is not out of scope" in build_generation_prompt(guard)
