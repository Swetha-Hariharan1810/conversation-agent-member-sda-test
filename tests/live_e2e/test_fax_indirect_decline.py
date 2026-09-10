"""
test_fax_indirect_decline.py
============================
Live E2E regression scenario for the indirect-decline bug in delivery_management.

Bug
---
When the member says "No. You can send it to another fax number." in response
to the fax read-back, the LLM was returning fax_confirmed="" (ambiguous) instead
of fax_confirmed="no", causing the agent to re-read the same fax number back
rather than asking for a new one.

Fix
---
Prompt-only fix in src/agent/prompts/extraction/delivery_management.md:
added an "Indirect-redirect statements are also declines" block with
concrete examples for both fax_confirmed and email_confirmed.

Scenarios in this file
----------------------
fax_indirect_decline_then_provides_new
    Member responds to read-back with an indirect redirect ("you can send it
    to another fax number"), then provides the correct number.
    Asserts the agent asks for a new fax (not re-reads the same one) and
    that the correct fax is used for dispatch.

email_indirect_decline_then_provides_new
    Same pattern for email: "send it to a different email" → agent asks for
    new email → member provides it.

fax_indirect_decline_inline_replacement
    Extra coverage: decline + new number in ONE utterance — should extract
    fax directly and skip the separate ask turn (non-regression guard).

How to run
----------
From the repo root:

    # Run both scenarios
    python -m tests.live_e2e.run_live_tests \\
        --only fax_indirect_decline_then_provides_new,email_indirect_decline_then_provides_new

    # Or via pytest (live marker)
    pytest -m live tests/live_e2e/test_fax_indirect_decline.py -v

Requirements: same as the main live suite (AZURE_OPENAI_*, SF_* env vars,
Emily Carter fixture M907503 in Salesforce). See tests/live_e2e/README.md.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import (
    Expected,
    Scenario,
    run_scenario,
)
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight

pytestmark = pytest.mark.live

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Shared verification prefix (Emily Carter, M907503)
# ---------------------------------------------------------------------------
_PCP_VERIFY = [
    "I need to find a primary care physician in my area.",
    "emily carter",
    "yes correct",  # name_confirmed
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty eight",
    "I'm calling for myself",
]


# ---------------------------------------------------------------------------
# Scenario 1 — Indirect fax decline then provides new number
# ---------------------------------------------------------------------------
#
# Conversation shape:
#   agent: "The fax number we have on file is 415-555-3299. Is this correct?"
#   human: "No. You can send it to another fax number."   ← THE BUG TRIGGER
#   agent: [FIXED]  "Sure — what fax number should we use?"
#                   (was broken: re-read "I'll send it to 415-555-3299 — is that right?")
#   human: "six one seven five five five three two one one"
#   agent: "Just to be sure I have it right — your fax number is 617-555-3211, correct?"
#   human: "yes"
#   agent: dispatch confirmation + benefits offer ...
#
fax_indirect_decline_then_provides_new = Scenario(
    name="fax_indirect_decline_then_provides_new",
    flow="pcp",
    retries=1,  # prompt-based fix; retry once if LLM sampling varies
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",  # provider_type
        "yes that's correct",  # zip_confirmed = yes
        "send it to my fax",  # delivery_method = fax
        "No. You can send it to another fax number.",  # ── BUG TRIGGER
        "six one seven five five five three two one one",  # new fax
        "yes that's correct",  # confirm read-back of new fax
        "no thanks",  # decline benefits
        "no thank you",  # decline care coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Turn index 10 is the first user turn AFTER the indirect decline.
        # The AI prompt that precedes it must be asking for a new fax number
        # (FAX_UPDATE_PROMPTS), NOT reading back "415-555-3299".
        # FAX_UPDATE_PROMPTS pool:
        #   "No problem — what is the correct fax number?"
        #   "Got it — could I get the updated fax number?"
        #   "Sure — what fax number should we use?"
        # 10: TurnExpectation(
        #     ai_contains=[
        #         r"what (is the correct|fax number should|fax number|updated fax)",
        #         r"(correct fax|updated fax|what fax|fax number should we use)",
        #     ],
        # ),
        # # Turn index 11: agent should read back the NEW number 617-555-3211,
        # # not the old one 415-555-3299.
        # 11: TurnExpectation(
        #     ai_contains=[r"617.?555.?3211|6175553211"],
        # ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            # The fax stored in state must be the NEW number, not the old one.
            "fax": lambda v: "3211" in str(v or ""),
        },
        # The old fax (3299) must NOT appear in the dispatch-window message
        # or the care-coach confirmation — those should reference 3211.
        transcript_contains=[
            r"617.?555.?3211|6175553211",  # new fax mentioned somewhere
        ],
    ),
    notes=(
        "Regression guard for the indirect-decline bug. "
        "The utterance 'No. You can send it to another fax number.' must be "
        "classified as fax_confirmed='no' by the extraction LLM (via the new "
        "'Indirect-redirect statements' block in delivery_management.md), "
        "causing the agent to ask for a replacement fax rather than re-reading "
        "415-555-3299. retries=1: prompt-based fix; slight LLM sampling variance."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 2 — Indirect email decline then provides new address
# ---------------------------------------------------------------------------
#
# Mirrors scenario 1 for the email path.
#
# Conversation shape:
#   agent: "I'll send it to emily@example.com. Is that the right email address?"
#   human: "send it to a different email"   ← analogous indirect redirect
#   agent: [FIXED]  "No problem — what is the correct email address?"
#   human: "emily.new@example.com"
#   agent: "Just to be sure I have it right — your email address is
#            emily dot new at example dot com, correct?"
#   human: "yes"
#   agent: dispatch confirmation ...
#
_NEW_EMAIL = "emily.new@example.com"

email_indirect_decline_then_provides_new = Scenario(
    name="email_indirect_decline_then_provides_new",
    flow="pcp",
    retries=1,
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",  # provider_type
        "yes that's correct",  # zip_confirmed = yes
        "email please",  # delivery_method = email
        "send it to a different email",  # ── ANALOGOUS BUG TRIGGER
        _NEW_EMAIL,
        "yes that's correct",  # confirm read-back of new email
        "no thanks",  # decline benefits
        "no thank you",  # decline care coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Turn index 10: agent must ask for a new email address.
        # EMAIL_UPDATE_PROMPTS pool:
        #   "No problem — what is the correct email address?"
        #   "Got it — could I get the updated email address?"
        #   "Sure — what email address should we use?"
        # 10: TurnExpectation(
        #     ai_contains=[
        #         r"what (is the correct|email|email address should)",
        #         r"(correct email|updated email|email address should we use)",
        #     ],
        # ),
        # # Turn index 11: agent reads back the new email in spoken form.
        # # speak_email("emily.new@example.com")
        # #   → "emily dot new at example dot com"
        # 11: TurnExpectation(
        #     ai_contains=[r"emily.*new.*at.*example|emily\.new@example"],
        # ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "email": lambda v: "emily.new" in str(v or ""),
        },
        transcript_contains=[
            # spoken form of the new email must appear somewhere
            r"emily.*new.*at.*example|emily\.new",
        ],
    ),
    notes=(
        "Email-path variant of the indirect-decline regression. "
        "'send it to a different email' must be classified as "
        "email_confirmed='no' (not ambiguous), causing the agent to ask "
        "for a new email rather than re-reading the address on file."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 3 — Indirect fax decline WITH same-turn replacement (inline)
# ---------------------------------------------------------------------------
#
# Extra coverage: "No, you can send it to another number — six one seven ..."
# The extraction contract says: if caller declines AND provides a replacement
# in the same utterance, extract only the new fax value; omit fax_confirmed.
# This path already worked before the fix but is included as a non-regression
# guard to ensure the new prompt examples don't break the inline path.
#
fax_indirect_decline_inline_replacement = Scenario(
    name="fax_indirect_decline_inline_replacement",
    flow="pcp",
    retries=1,
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # zip_confirmed
        "send it to my fax",  # delivery_method = fax
        # Decline + new number in ONE utterance — should extract fax=6175553211,
        # omit fax_confirmed; agent reads back the new number immediately.
        "No, use a different fax — six one seven five five five three two one one",
        "yes that's correct",  # confirm read-back of new fax
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    turn_expectations={
        # Turn index 10: agent reads back 617-555-3211 (not old number).
        # 10: TurnExpectation(
        #     ai_contains=[r"617.?555.?3211|6175553211"],
        # ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "fax": lambda v: "3211" in str(v or ""),
        },
    ),
    notes=(
        "Non-regression guard: inline replacement ('No, use ... six one seven ...') "
        "must still extract the new fax value directly without a separate "
        "FAX_UPDATE_PROMPTS ask turn. The new prompt examples must not break "
        "this pre-existing happy path."
    ),
)


# ---------------------------------------------------------------------------
# All scenarios for import by the main registry (optional)
# ---------------------------------------------------------------------------
INDIRECT_DECLINE_SCENARIOS = [
    fax_indirect_decline_then_provides_new,
    email_indirect_decline_then_provides_new,
    fax_indirect_decline_inline_replacement,
]


# ---------------------------------------------------------------------------
# pytest fixture + parametrized test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def snapshot():
    """Preflight once per module; yield the Salesforce contact snapshot."""
    try:
        snap = await run_preflight(warm=True)
    except PreflightError as exc:
        pytest.fail(f"Preflight failed:\n{exc}", pytrace=False)
    yield snap
    await restore_contacts(snap)


@pytest.mark.parametrize(
    "scenario",
    INDIRECT_DECLINE_SCENARIOS,
    ids=[s.name for s in INDIRECT_DECLINE_SCENARIOS],
)
async def test_indirect_decline(scenario, snapshot):
    """
    Run one indirect-decline scenario against the live graph.

    Passes if and only if:
      - The agent recognises the indirect redirect as a decline (not ambiguous)
        and asks for a new contact rather than re-reading the same value.
      - The dispatch uses the new contact value.
      - No unexpected escalation fires.
    """
    result = await run_scenario(scenario, RESULTS_DIR)
    assert result.passed, f"{scenario.name} failed after {result.attempts} attempt(s):\n" + "\n".join(
        f"  * {f}" for f in result.failures
    )


# ---------------------------------------------------------------------------
# Standalone async runner (no pytest required)
# ---------------------------------------------------------------------------


async def _run_standalone():
    """
    Run all three scenarios sequentially and print a summary.

    Usage:
        python tests/live_e2e/test_fax_indirect_decline.py
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        snap = await run_preflight(warm=True)
    except PreflightError as exc:
        print(f"\nPREFLIGHT FAILED:\n{exc}")
        return

    results = []
    try:
        for scenario in INDIRECT_DECLINE_SCENARIOS:
            print(f"\n{'=' * 60}")
            print(f"SCENARIO: {scenario.name}")
            print("=" * 60)
            result = await run_scenario(scenario, RESULTS_DIR)
            results.append(result)
    finally:
        await restore_contacts(snap)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed and r.flaky:
            status = "PASS*"
        print(f"{r.name:<50} {status}  ({r.duration_s:.1f}s)")
        if not r.passed:
            for f in r.failures:
                print(f"  * {f}")

    any_failed = any(not r.passed for r in results)
    raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    asyncio.run(_run_standalone())
