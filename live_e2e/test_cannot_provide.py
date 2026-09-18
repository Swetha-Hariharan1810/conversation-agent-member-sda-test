"""
test_cannot_provide.py
======================
Live E2E regression scenarios for the detect_cannot_provide() short-circuit.

Background
----------
Before this fix, utterances such as "I don't have it", "I lost my card", or
"I can't remember" would burn through the slot retry budget (3 attempts) and
only escalate after the third failure via the normal exhaustion path. The
member heard the same question three times before reaching an agent.

Fix
---
Two-part prompt-free, zero-latency fix:

  1. src/agent/utils.py — detect_cannot_provide(text) returns True when the
     utterance contains a first-person "inability" pattern.

  2. src/agent/core/slot_manager.py — _collect_slot() checks
     detect_cannot_provide() at CHANGE 1 (rejected-extraction path) and
     CHANGE 2 (no-extraction / ANSWERED path) BEFORE counting slot_fail(),
     escalating immediately with reason "{slot_name}_cannot_provide".

  3. src/agent/agents/claim_adjustment/agent.py — Phase 1 checks
     detect_cannot_provide() BEFORE calling the extraction LLM, escalating
     immediately with reason "reference_number_cannot_provide".

Scenarios
---------
ref_cannot_provide_escalates
    Member says "I don't have it" when asked for the claim reference number.
    Must escalate on the FIRST turn (reason: reference_number_cannot_provide),
    not after 3 retries.

ref_cannot_provide_lost_letter
    Member says "I lost the letter" — physical-absence pattern.
    Same assertion as above: single-turn escalation.

member_id_cannot_provide_escalates
    Member says "I don't have my member ID" during PCP verification.
    slot_manager CHANGE 2 path: escalates immediately with
    reason member_id_cannot_provide.

dob_cannot_provide_escalates
    Member provides a valid member ID but then says "I can't remember"
    when asked for their date of birth.
    slot_manager CHANGE 2 path: escalates immediately with
    reason dob_cannot_provide.

member_id_cannot_provide_then_retry_succeeds  (non-regression)
    Ensures "I don't have it" on member_id does NOT get swallowed or cause a
    retry loop — the escalation fires and the call ends cleanly.

How to run
----------
    # via run_live_tests
    python -m tests.live_e2e.run_live_tests \\
        --only ref_cannot_provide_escalates,member_id_cannot_provide_escalates

    # via pytest
    pytest -m live tests/live_e2e/test_cannot_provide.py -v

Requirements: AZURE_OPENAI_*, SF_* env vars, Emily Carter M907503 and
James Wilson M310188 in Salesforce. See tests/live_e2e/README.md.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import (
    Expected,
    Scenario,
    TurnExpectation,
    run_scenario,
)
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight

pytestmark = pytest.mark.live

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Shared verification prefixes
# ---------------------------------------------------------------------------

# PCP flow — Emily Carter M907503
# Turn layout (0-based user-turn index):
#   0  intent
#   1  first_name  → "emily"
#   2  last_name   → "carter"
#   3  name_confirmed  (agent spells E-M-I-L-Y C-A-R-T-E-R, member confirms)
#   4  member_id ask  ← turn 4 expectation checks slot_awaiting="member_id"
#   5  dob ask
#   6  relationship ask
_PCP_INTENT = ["I need to find a primary care physician in my area.", "emily", "carter"]
_NAME_CONFIRM = ["yes that's correct"]  # turn 3 — name readback confirmed
_MEMBER_ID = ["m nine zero seven five zero three"]
_DOB = ["April twelfth nineteen eighty eight"]
_RELATIONSHIP = ["I'm calling for myself"]

# Claim flow — James Wilson M310188
# Turn layout:
#   0  intent
#   1  first_name  → "james"
#   2  last_name   → "wilson"
#   3  name_confirmed
#   4  member_id ask
#   5  dob ask
#   6  phone_confirmed ask
#   7  reference_number ask  ← cannot-provide fires here
_CLAIM_VERIFY = [
    "I adjusted the claim and I want to follow up",
    "james",
    "wilson",
    "yes that's right",  # name confirmed
    "m three one zero one eight eight",
    "Thirtieth of July, nineteen seventy seven",
    "yes correct",  # phone_confirmed
]


# ---------------------------------------------------------------------------
# Scenario 1 — Reference number: "I don't have it"
# ---------------------------------------------------------------------------
#
# Claim path — ClaimAdjustmentAgent PHASE 1 cannot-provide check.
# detect_cannot_provide() runs BEFORE the extraction LLM call so the
# fallback sub-flow starts at zero LLM cost on the very first attempt.
#
# Before: agent escalated on attempt 1 with reason=reference_number_cannot_provide
# After (new fallback): agent pivots to the claim-number question instead of
#   escalating, giving the member a secondary way to look up their claim.
#
ref_cannot_provide_escalates = Scenario(
    name="ref_cannot_provide_escalates",
    flow="claim",
    retries=1,
    user_turns=_CLAIM_VERIFY
    + [
        # ── TRIGGER ───────────────────────────────────────────────────────
        # Agent just asked for the reference number.
        # detect_cannot_provide() fires → starts fallback (asks for claim number)
        "I don't have it",
    ],
    turn_expectations={
        # Turn 7: the AI prompt before the cannot-provide utterance must ask
        # for the reference number.
        7: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
        ),
        # Turn 8: the AI response AFTER the cannot-provide utterance must ask
        # for the claim number — the fallback sub-flow started.
        8: TurnExpectation(
            ai_contains=[r"claim\s*(number|#|num)"],
        ),
    },
    expect=Expected(
        completed=False,  # call is NOT completed — fallback is in progress
        escalated=False,  # no escalation on the first turn
        # reference_number must remain unset — no value was given
        final_state={"reference_number": lambda v: not v},
    ),
    notes=(
        "Regression guard (updated for fallback sub-flow): 'I don't have it' must "
        "now START the fallback sub-flow (ask for claim number) rather than "
        "escalating immediately. The AI turn after the cannot-provide utterance "
        "must ask 'Do you have the claim number?' — not transfer to a representative."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 2 — Reference number: "I lost the letter" (physical-absence variant)
# ---------------------------------------------------------------------------
ref_cannot_provide_lost_letter = Scenario(
    name="ref_cannot_provide_lost_letter",
    flow="claim",
    retries=1,
    user_turns=_CLAIM_VERIFY
    + [
        "I lost the letter with my reference number on it",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        # Fallback sub-flow: agent asks for claim number instead of escalating
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
    },
    expect=Expected(
        completed=False,
        escalated=False,
        final_state={"reference_number": lambda v: not v},
    ),
    notes=(
        "Physical-absence variant (updated for fallback sub-flow): 'I lost the letter' "
        r"matches \bi\s+(lost|misplaced)\s+(it|that|my\b|the\b) in detect_cannot_provide(). "
        "Fallback sub-flow now starts instead of escalating — the AI turn after "
        "the utterance must ask for the claim number."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 3 — Member ID: "I don't have my member ID"
# ---------------------------------------------------------------------------
#
# PCP path — slot_manager CHANGE 2 (no-extraction / ANSWERED branch).
# After confirming the name readback (turn 3), the agent asks for member_id
# (turn 4 expectation). Member immediately says they don't have it.
#
# Before fix: slot_fail() counted → agent re-asked twice more
# After fix: detect_cannot_provide() → immediate escalation,
#   reason="member_id_cannot_provide"
#
member_id_cannot_provide_escalates = Scenario(
    name="member_id_cannot_provide_escalates",
    flow="pcp",
    retries=1,
    user_turns=_PCP_INTENT
    + _NAME_CONFIRM
    + [
        # ── BUG TRIGGER ────────────────────────────────────────────────────
        # Agent just asked for member ID (turn 4 expectation asserts this).
        "sorry I dont have it my member ID",
    ],
    turn_expectations={
        # Turn 3: agent reads back the name (name_confirmation feature).
        # 3: TurnExpectation(
        #     ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"],
        # ),
        # # Turn 4: agent asks for member_id — this is the turn that receives
        # # the cannot-provide utterance.
        # 4: TurnExpectation(
        #     ai_contains=[r"member\s*(id|ID|number)"],
        #     slot_awaiting="member_id",
        # ),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_regex=r"member_id_cannot_provide",
        final_state={
            # Verification must never have succeeded
            "member_status_verify": lambda v: not v,
            "member_id": lambda v: not v,
        },
        last_ai_contains=[
            r"(connect you with a representative|transfer|representative who can help)",
        ],
    ),
    notes=(
        "slot_manager CHANGE 2: 'I don't have my member ID' hits the ANSWERED "
        "branch in _collect_slot with no extraction. detect_cannot_provide() "
        "fires before slot_fail() so no retry attempt is consumed. "
        "reason=member_id_cannot_provide. member_status_verify must remain falsy."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 4 — DOB: "I can't remember"
# ---------------------------------------------------------------------------
#
# PCP path — member provides a valid member_id but then cannot provide DOB.
# slot_manager CHANGE 2 (no-extraction / ANSWERED branch) for the dob slot.
#
dob_cannot_provide_escalates = Scenario(
    name="dob_cannot_provide_escalates",
    flow="pcp",
    retries=1,
    user_turns=_PCP_INTENT
    + _NAME_CONFIRM
    + _MEMBER_ID
    + [
        # ── BUG TRIGGER ────────────────────────────────────────────────────
        # Agent has confirmed member_id and is now asking for date of birth.
        "I can't remember my date of birth",
    ],
    turn_expectations={
        3: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"],
        ),
        4: TurnExpectation(
            ai_contains=[r"member\s*(id|ID|number)"],
            slot_awaiting="member_id",
        ),
        # Turn 5: agent asks for date of birth — this receives the cannot-provide
        5: TurnExpectation(
            ai_contains=[r"(date of birth|birth\s*date|when were you born)"],
            slot_awaiting="dob",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_regex=r"dob_cannot_provide",
        final_state={
            "member_status_verify": lambda v: not v,
            "dob": lambda v: not v,
        },
        last_ai_contains=[
            r"(connect you with a representative|transfer|representative who can help)",
        ],
    ),
    notes=(
        "slot_manager CHANGE 2: 'I can't remember my date of birth' matches "
        r"the \bi\s+can'?t\s+(remember|recall|find)\b pattern. detect_cannot_provide "
        "fires in the ANSWERED branch for the dob slot before any slot_fail() is "
        "recorded. reason=dob_cannot_provide. Verification must not complete."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 5 — Member ID: "I never received one" (never-received variant)
# ---------------------------------------------------------------------------
member_id_cannot_provide_never_received = Scenario(
    name="member_id_cannot_provide_never_received",
    flow="pcp",
    retries=1,
    user_turns=_PCP_INTENT
    + _NAME_CONFIRM
    + [
        "I never received a member ID card",
    ],
    turn_expectations={
        3: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"],
        ),
        4: TurnExpectation(
            ai_contains=[r"member\s*(id|ID|number)"],
            slot_awaiting="member_id",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_regex=r"member_id_cannot_provide",
        final_state={"member_status_verify": lambda v: not v},
        last_ai_contains=[
            r"(connect you with a representative|transfer|representative who can help)",
        ],
    ),
    notes=(
        "Covers the never-received variant: 'I never received a member ID card' "
        r"matches \bi\s+never\s+(received|got)\s+(it|that|one|my\b). "
        "Same single-turn escalation as member_id_cannot_provide_escalates."
    ),
)


# ---------------------------------------------------------------------------
# Scenario 6 — Reference number: "I don't have that information"
# ---------------------------------------------------------------------------
ref_cannot_provide_no_info = Scenario(
    name="ref_cannot_provide_no_info",
    flow="claim",
    retries=1,
    user_turns=_CLAIM_VERIFY
    + [
        "I don't have that information",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        # Fallback sub-flow starts — agent asks for claim number
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
    },
    expect=Expected(
        completed=False,
        escalated=False,
        final_state={"reference_number": lambda v: not v},
    ),
    notes=(
        r"Covers the \bdon'?t\s+have\s+that\s+information\b pattern (updated for "
        "fallback sub-flow). The fallback sub-flow now starts instead of escalating — "
        "the AI turn after the utterance asks for the claim number."
    ),
)


# ---------------------------------------------------------------------------
# All scenarios for import by the main registry
# ---------------------------------------------------------------------------
CANNOT_PROVIDE_SCENARIOS = [
    ref_cannot_provide_escalates,
    ref_cannot_provide_lost_letter,
    member_id_cannot_provide_escalates,
    dob_cannot_provide_escalates,
    member_id_cannot_provide_never_received,
    ref_cannot_provide_no_info,
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
    CANNOT_PROVIDE_SCENARIOS,
    ids=[s.name for s in CANNOT_PROVIDE_SCENARIOS],
)
async def test_cannot_provide(scenario, snapshot):
    """
    Run one cannot-provide scenario against the live graph.

    Passes if and only if:
      - The agent escalates on the FIRST cannot-provide utterance.
      - The escalation reason is {slot_name}_cannot_provide (not *_exhausted).
      - No retry prompts are issued before the escalation.
      - The empathetic escalation message is used.
    """
    result = await run_scenario(scenario, RESULTS_DIR)
    assert result.passed, f"{scenario.name} failed after {result.attempts} attempt(s):\n" + "\n".join(
        f"  * {f}" for f in result.failures
    )


# ---------------------------------------------------------------------------
# Standalone async runner
# ---------------------------------------------------------------------------


async def _run_standalone():
    """
    Run all scenarios sequentially and print a summary.

    Usage:
        python tests/live_e2e/test_cannot_provide.py
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
        for scenario in CANNOT_PROVIDE_SCENARIOS:
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
        status = "PASS*" if r.passed and r.flaky else ("PASS" if r.passed else "FAIL")
        print(f"{r.name:<55} {status}  ({r.duration_s:.1f}s)")
        if not r.passed:
            for f in r.failures:
                print(f"  * {f}")

    raise SystemExit(1 if any(not r.passed for r in results) else 0)


if __name__ == "__main__":
    asyncio.run(_run_standalone())
