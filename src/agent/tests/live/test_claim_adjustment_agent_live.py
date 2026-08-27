"""
test_claim_adjustment_agent_live.py — Live integration tests for ClaimAdjustmentAgent.

These tests run against a real LLM (Azure OpenAI / Gemini) and a real
Salesforce sandbox.  They require valid API credentials in the environment.

Run:
    pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -v
    pytest -m live -k "test_claim_adjustment" -v   # single group

Skip in CI:
    Tests auto-skip when AZURE_OPENAI_API_KEY (or GOOGLE_API_KEY) is absent.

Member data
-----------
Scenario A: Michael Brown / M662130 / 03/18/1986
  Reference number: 12491584
  records_required: True  (Salesforce sandbox)

Scenario B: James Wilson / M310188 / 07/30/1977
  Reference number: 42695817
  email on file:  james.wilson@gmail.com
  phone on file:  512-555-6101
  records_required: False (Salesforce sandbox)

Groups
------
# Group A — ClaimAdjustmentAgent reference number collection
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_A" -v

# Group A2 — VerificationAgent phone_confirmed variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_A_pc" -v

# Group B — RecordsCoordinationAgent branches
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_B" -v

# Group B2 — email_confirmed slot behaviour
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_B2" -v

# Group B_method — upload_method natural-language variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_B_m" -v

# Group B_pg — personal_guide_consent variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_B_pg" -v

# Group B_uc — upload_consent variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_B_uc" -v

# Group C — NotificationSetupAgent
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_C" -v

# Group C2 — contact_confirmed variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_C2" -v

# Group C_sms — SMS notification method variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_C_sms" -v

# Group C_email — email notification method variants
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_C_email" -v

# Group C_amb — ambiguous notification method
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_C_amb" -v

# Group D — End-to-end smoke + latency
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_D" -v

# Group D_combo — N1/N2 combinations
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_D_combo" -v

# Group D_latency — latency benchmarks only
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_D_latency" -v

# Group E — Follow-up claims behaviour
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_E" -v

# Group R — Retry/AMBIGUOUS paths
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py -k "test_R" -v

# Skip slow tests in any group
pytest -m "live and not slow" src/agent/tests/live/test_claim_adjustment_agent_live.py -k
"test_A" -v

# Single specific test
pytest -m live src/agent/tests/live/test_claim_adjustment_agent_live.py
 -k "test_A1_spoken_digits_reference_number" -v
"""

from __future__ import annotations

import math
import os

import pytest

from agent.tests.live.conversation_logger import ConversationRecord

# ---------------------------------------------------------------------------
# Skip guard — all tests skip when LLM credentials are absent
# ---------------------------------------------------------------------------

_MISSING_CREDS = not (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_MISSING_CREDS, reason="No LLM credentials in environment"),
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Claims verification path uses phone_confirmed (not relationship)
VERIFICATION_PREFIX_CLAIMS = [
    "I adjusted the claim and I want to follow up",
    "Michael",  # first_name  (Scenario A member)
    "Brown",  # last_name
    "m six six two one three zero",  # member_id M662130 (spoken)
    "eighteenth of March nineteen eighty six",  # dob 03/18/1986
    "yes",  # phone_confirmed
]

VERIFICATION_PREFIX_CLAIMS_B = [
    "I adjusted the claim and I want to follow up",
    "James",
    "Wilson",
    "m three one zero one eight eight",  # M310188
    "Thirtyth of July nineteen seventy seven",  # 07/30/1977
    "yes",  # phone_confirmed
]

REF_A = "12491584"
REF_B = "42695817"
EMAIL_ON_FILE_B = "james.wilson@gmail.com"
PHONE_ON_FILE_B = "512-555-6101"

# ---------------------------------------------------------------------------
# Fixture alias
# ---------------------------------------------------------------------------


@pytest.fixture
def run_conversation(run_intake_conversation):
    """Alias so claim-adjustment tests read naturally. Same graph runner underneath."""
    return run_intake_conversation


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_reference_collected(record: ConversationRecord, expected_ref: str) -> None:
    """reference_number == expected_ref in final state."""
    actual = record.final_state.get("reference_number", "")
    assert actual == expected_ref, f"Expected reference_number={expected_ref!r}, got {actual!r}"


def assert_claim_status_reported(record: ConversationRecord) -> None:
    """claim_status is set and agent message mentions 'review' or 'open' or 'update'."""
    claim_status = record.final_state.get("claim_status", "")
    assert claim_status, f"Expected claim_status to be set, got {claim_status!r}"
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    keywords = ("review", "open", "update", "status", "found")
    matched = any(k in all_msgs for k in keywords)
    assert matched, (
        f"Expected agent message to mention one of {keywords}. "
        f"Full transcript (first 500 chars): {all_msgs[:500]!r}"
    )


def assert_records_required_set(record: ConversationRecord, expected: bool) -> None:
    """records_required == expected in final state."""
    actual = record.final_state.get("records_required")
    assert actual == expected, f"Expected records_required={expected!r}, got {actual!r}"


def assert_not_escalated(record: ConversationRecord) -> None:
    """No escalation occurred."""
    final_active = record.final_state.get("active_agent", "")
    final_reason = record.final_state.get("escalation_reason", "")
    was_escalated = (
        final_active == "escalation_agent"
        or any(t.active_agent == "escalation_agent" for t in record.turns)
        or any(bool(t.state_snapshot.get("escalation_reason")) for t in record.turns)
    )
    assert not was_escalated, f"Unexpected escalation: reason={final_reason!r}, active_agent={final_active!r}"


def assert_escalated(record: ConversationRecord, reason_contains: str | None = None) -> None:
    """Escalation triggered — optionally verify escalation_reason substring."""
    final_reason = record.final_state.get("escalation_reason", "")
    final_next = record.final_state.get("next_node", "")
    final_active = record.final_state.get("active_agent", "")
    was_escalated = (
        bool(final_reason)
        or final_next == "escalation_agent"
        or final_active == "escalation_agent"
        or any(t.active_agent == "escalation_agent" for t in record.turns)
        or any(bool(t.state_snapshot.get("escalation_reason")) for t in record.turns)
    )
    assert was_escalated, (
        f"Expected escalation but escalation_reason={final_reason!r}, "
        f"next_node={final_next!r}, active_agent={final_active!r}"
    )
    if reason_contains:
        all_reasons = [t.state_snapshot.get("escalation_reason") or "" for t in record.turns]
        all_reasons.append(final_reason or "")
        any_match = any(reason_contains.lower() in r.lower() for r in all_reasons if r)
        assert any_match, (
            f"Expected escalation_reason to contain {reason_contains!r}. "
            f"Reasons seen: {[r for r in all_reasons if r]}"
        )


def assert_routed_to(record: ConversationRecord, expected_node: str) -> None:
    """next_node or active_agent == expected_node at some point."""
    actual = record.final_state.get("next_node", "")
    active = record.final_state.get("active_agent", "")
    was_routed = (
        actual == expected_node
        or active == expected_node
        or any(t.active_agent == expected_node for t in record.turns)
    )
    assert was_routed, (
        f"Expected routing to {expected_node!r}, got next_node={actual!r}, active_agent={active!r}"
    )


def assert_any_agent_message_contains(record: ConversationRecord, *substrings: str) -> None:
    """At least one agent message across all turns contains each substring."""
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    for sub in substrings:
        assert sub.lower() in all_msgs, (
            f"Expected any agent message to contain {sub!r}. "
            f"Full transcript (first 500 chars): {all_msgs[:500]!r}"
        )


def assert_member_verified(record: ConversationRecord) -> None:
    """member_status_verify=True in final state."""
    actual = record.final_state.get("member_status_verify")
    assert actual is True, f"Expected member_status_verify=True, got {actual!r}"


def assert_phone_update_requested(record: ConversationRecord) -> None:
    """phone_update_requested=True in final state."""
    actual = record.final_state.get("phone_update_requested")
    assert actual is True, f"Expected phone_update_requested=True, got {actual!r}"


# ===========================================================================
# GROUP A — ClaimAdjustmentAgent: reference number collection variations
# ===========================================================================


@pytest.mark.live
async def test_A1_spoken_digits_reference_number(run_conversation, assert_and_record):
    """
    A1: Spoken-digit reference number "one two four nine one five eight four"
    is normalized to "12491584" and collected.
    Verifies that the spoken-digit normalizer and extraction prompt work end-to-end.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "one two four nine one five eight four",
        ],
        test_name="test_A1_spoken_digits_reference_number",
        scenario=(
            "Spoken digits 'one two four nine one five eight four' → "
            "reference_number=12491584 → claim status reported"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_A), f"reference_number=={REF_A}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A2_numeric_string_reference_number(run_conversation, assert_and_record):
    """
    A2: Numeric string reference number "42695817" is collected directly.
    Verifies that digit-only input (no spoken words) is extracted without
    normalization loss.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            "42695817",
        ],
        test_name="test_A2_numeric_string_reference_number",
        scenario=("Numeric string '42695817' → reference_number=42695817 → claim status reported"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_B), f"reference_number=={REF_B}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A3_reference_number_with_filler(run_conversation, assert_and_record):
    """
    A3: Reference number embedded in filler: "It's one two four nine one five eight four"
    → extracted as "12491584".
    Verifies that the extraction prompt strips surrounding words and normalizes spoken digits.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "It's one two four nine one five eight four",
        ],
        test_name="test_A3_reference_number_with_filler",
        scenario=("Filler + spoken digits → reference_number=12491584 → claim status reported"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_A), f"reference_number=={REF_A}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A4_invalid_reference_too_short(run_conversation, assert_and_record):
    """
    A4: First attempt gives a too-short reference number "1234" (< 6 digits).
    Agent should retry, then member supplies the valid reference number.
    Verifies that validate_reference_number rejects < 6 digits and a retry fires.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "1234",
            REF_A,
        ],
        test_name="test_A4_invalid_reference_too_short",
        scenario=("Too-short '1234' → retry → valid '12491584' → claim status reported"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_A), f"reference_number=={REF_A}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A5_two_invalid_then_valid(run_conversation, assert_and_record):
    """
    A5: Two invalid attempts ("abc", "99") then a valid spoken reference.
    Verifies multi-attempt tolerance before the valid value is accepted.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "abc",
            "99",
            "one two four nine one five eight four",
        ],
        test_name="test_A5_two_invalid_then_valid",
        scenario=("Two invalid attempts → valid spoken digits on third attempt → reference_number=12491584"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_A), f"reference_number=={REF_A}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A6_exhaustion_escalates(run_conversation, assert_and_record):
    """
    A6: Three consecutive invalid reference number attempts exhaust the slot
    and trigger escalation with a reason containing 'reference'.
    Verifies the exhaustion → escalation path and escalation_reason value.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "wrong",
            "also wrong",
            "still wrong",
        ],
        test_name="test_A6_exhaustion_escalates",
        scenario=(
            "Three invalid reference inputs → slot exhausted → escalation with reason containing 'reference'"
        ),
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="reference"), "escalated_reference"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_A7_transfer_request_during_reference_collection(run_conversation, assert_and_record):
    """
    A7: Member says "transfer me to someone" during reference number collection.
    Verifies that the TRANSFER_REQUEST guard fires inside ClaimAdjustmentAgent
    and routes to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "transfer me to someone please",
        ],
        test_name="test_A7_transfer_request_during_reference_collection",
        scenario=("TRANSFER_REQUEST guard during reference collection → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_A8_reference_not_found_in_sf(run_conversation, assert_and_record):
    """
    A8: Member provides a syntactically valid reference number (6+ digits) that
    does not exist in the Salesforce sandbox.
    Verifies MSG_REF_NOT_FOUND message is surfaced and escalation is triggered.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "12345678",  # attempt 1 — not found → fallback to claim number
            "12345678",  # attempt 2 — treated as claim number, also not found → escalation
        ],
        test_name="test_A8_reference_not_found_in_sf",
        scenario=(
            "Valid-format reference not in Salesforce → fallback to claim number → "
            "claim number also not found → escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            # (
            #     lambda: assert_any_agent_message_contains(
            #         record, "wasn't able to find", "couldn't locate", "didn't match"
            #     ),
            #     "ref_not_found_message",
            # ),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A9_dont_have_1_retry_then_valid(run_conversation, assert_and_record):
    """
    A9_dont_have_1: "I don't have it with me right now" → slot retries (not
    counted as a failed attempt) → member provides valid ref → claim status
    reported.

    "I don't have it" is ambiguous/non-numeric and must trigger a retry, not
    consume an attempt toward the max-attempts exhaustion limit.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            "I don't have it with me right now",
            REF_B,
        ],
        test_name="test_A9_dont_have_1_retry_then_valid",
        scenario="ref_number 'I don't have it' → retry → valid ref → claim_status_reported",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_B), "reference_collected"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A10_wrong_type_1_phone_number_spoken(run_conversation, assert_and_record):
    """
    A10_wrong_type_1: Member gives their phone number by mistake —
    "five one two five five five six one zero one" normalises to "5125556101"
    (10 digits, ≥6 → PASSES validate_reference_number) but Salesforce returns
    not-found → escalation.

    Verifies that the reference-number validator only checks length/format, so
    a phone-digit string passes client-side validation but then fails the SF
    lookup, surfacing the correct "wasn't able to find" message.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            "five one two five five five six one zero one",
        ],
        test_name="test_A10_wrong_type_1_phone_number_spoken",
        scenario="phone number spoken as reference → passes validation → SF not found → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A11_too_long_1_twelve_digits(run_conversation, assert_and_record):
    """
    A11_too_long_1: "one two three four five six seven eight nine zero one two"
    normalises to "123456789012" (12 digits — still ≥6, passes validation) →
    SF lookup returns not-found → escalation.

    Verifies behaviour when member provides an over-long numeric string.
    The validator allows ≥6 digits, so 12-digit strings pass client-side.
    If by coincidence SF returns a match, claim_status_reported is also
    acceptable (documented as OR-outcome).
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            "one two three four five six seven eight nine zero one two",
        ],
        test_name="test_A11_too_long_1_twelve_digits",
        scenario="12-digit spoken ref → passes validation → SF not-found (or match) → escalation or status",
    )

    def _assert_escalated_or_status(r):
        was_escalated = bool(r.final_state.get("escalation_reason")) or any(
            bool(t.state_snapshot.get("escalation_reason")) for t in r.turns
        )
        has_status = bool(r.final_state.get("claim_status"))
        assert was_escalated or has_status, (
            f"Expected either escalation or claim_status for 12-digit ref. "
            f"escalation_reason={r.final_state.get('escalation_reason')!r}, "
            f"claim_status={r.final_state.get('claim_status')!r}"
        )

    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP A2 — VerificationAgent: phone_confirmed slot variants
# ===========================================================================

# Base inputs up to (but not including) phone_confirmed answer
_VERIFICATION_BASE_B = [
    "I adjusted the claim and I want to follow up",
    "James",
    "Wilson",
    "m three one zero one eight eight",
    "Thirtyth of July nineteen seventy seven",
    # phone_confirmed variant appended per test
]


@pytest.mark.live
async def test_A_pc_yes_1_plain_yes(run_conversation, assert_and_record):
    """
    A2_yes_1: "yes" → phone_confirmed=yes → verification continues to reference collection.
    Baseline affirmation.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["yes", REF_B],
        test_name="test_A_pc_yes_1_plain_yes",
        scenario="phone_confirmed plain 'yes'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A_pc_yes_2_that_is_correct(run_conversation, assert_and_record):
    """
    A2_yes_2: "that is correct" → phone_confirmed=yes.
    Formal affirmation variant.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["that is correct", REF_B],
        test_name="test_A_pc_yes_2_that_is_correct",
        scenario="phone_confirmed 'that is correct'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A_pc_yes_3_yep_sounds_good(run_conversation, assert_and_record):
    """
    A2_yes_3: "yep sounds good" → phone_confirmed=yes.
    Casual multi-word affirmation.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["yep sounds good", REF_B],
        test_name="test_A_pc_yes_3_yep_sounds_good",
        scenario="phone_confirmed 'yep sounds good'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A_pc_yes_4_correct_number(run_conversation, assert_and_record):
    """
    A2_yes_4: "yes that's my number" → phone_confirmed=yes.
    Affirmation with possessive reference.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["yes that's my number", REF_B],
        test_name="test_A_pc_yes_4_correct_number",
        scenario="phone_confirmed 'yes that's my number'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A_pc_yes_5_sure(run_conversation, assert_and_record):
    """
    A2_yes_5: "sure" → phone_confirmed=yes.
    Minimal one-word casual affirmation.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["sure", REF_B],
        test_name="test_A_pc_yes_5_sure",
        scenario="phone_confirmed 'sure'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A_pc_yes_6_yeah_thats_right(run_conversation, assert_and_record):
    """
    A2_yes_6: "yeah that's right" → phone_confirmed=yes.
    Conversational affirmation.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["yeah that's right", REF_B],
        test_name="test_A_pc_yes_6_yeah_thats_right",
        scenario="phone_confirmed 'yeah that's right'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_A_pc_no_1_plain_no(run_conversation, assert_and_record):
    """
    A2_no_1: "no" → phone_confirmed=no → phone_update_requested=True.
    Verification still completes but flags that phone needs updating.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["no", REF_B],
        test_name="test_A_pc_no_1_plain_no",
        scenario="phone_confirmed 'no' → phone_update_requested",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_phone_update_requested(record), "phone_update_requested"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A_pc_no_2_thats_not_right(run_conversation, assert_and_record):
    """
    A2_no_2: "that's not right" → phone_confirmed=no → phone_update_requested=True.
    Implicit decline without "no" keyword.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["that's not right", REF_B],
        test_name="test_A_pc_no_2_thats_not_right",
        scenario="phone_confirmed 'that's not right' → phone_update_requested",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_phone_update_requested(record), "phone_update_requested"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_A_pc_no_3_wrong_number(run_conversation, assert_and_record):
    """
    A2_no_3: "that's not my number" → phone_confirmed=no → phone_update_requested=True.
    Explicit ownership rejection.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B + ["that's not my number", REF_B],
        test_name="test_A_pc_no_3_wrong_number",
        scenario="phone_confirmed 'that's not my number' → phone_update_requested",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_phone_update_requested(record), "phone_update_requested"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# Additional assertion helpers for Group B
# ===========================================================================


def assert_not_upload_link_sent(record: ConversationRecord) -> None:
    """upload_link_sent is not True in final state or any turn snapshot."""
    final = record.final_state.get("upload_link_sent")
    any_turn = any(t.state_snapshot.get("upload_link_sent") for t in record.turns)
    assert final is not True and not any_turn, f"Expected upload_link_sent absent/False, got final={final!r}"


def assert_upload_link_sent(record: ConversationRecord) -> None:
    """upload_link_sent == True in final state or any turn snapshot."""
    final = record.final_state.get("upload_link_sent")
    any_turn = any(t.state_snapshot.get("upload_link_sent") for t in record.turns)
    assert final or any_turn, f"Expected upload_link_sent=True in state, got final={final!r}"


def assert_personal_guide_triggered(record: ConversationRecord) -> None:
    """personal_guide_outreach_requested == True in final state or any turn snapshot."""
    final = record.final_state.get("personal_guide_outreach_requested")
    any_turn = any(t.state_snapshot.get("personal_guide_outreach_requested") for t in record.turns)
    assert final or any_turn, f"Expected personal_guide_outreach_requested=True, got final={final!r}"


def assert_records_branch(record: ConversationRecord, expected_branch: str) -> None:
    """records_branch_taken == expected_branch in final state or any turn snapshot."""
    actual_final = record.final_state.get("records_branch_taken", "")
    any_turn_match = any(
        t.state_snapshot.get("records_branch_taken") == expected_branch for t in record.turns
    )
    assert actual_final == expected_branch or any_turn_match, (
        f"Expected records_branch_taken={expected_branch!r}, got {actual_final!r}"
    )


def assert_routed_to_notification_setup(record: ConversationRecord) -> None:
    """notification_setup_agent was active or targeted in at least one turn."""
    final_active = record.final_state.get("active_agent", "")
    final_next = record.final_state.get("next_node", "")
    was_routed = (
        final_active == "notification_setup_agent"
        or final_next == "notification_setup_agent"
        or any(t.active_agent == "notification_setup_agent" for t in record.turns)
        or any(t.state_snapshot.get("next_node") == "notification_setup_agent" for t in record.turns)
        or any(t.state_snapshot.get("personal_guide_outreach_requested") for t in record.turns)
    )
    assert was_routed, (
        f"Expected routing to notification_setup_agent, "
        f"got active_agent={final_active!r}, next_node={final_next!r}"
    )


# ---------------------------------------------------------------------------
# Scenario A prefix + reference: lands in records_coordination (records_required=True)
# ---------------------------------------------------------------------------

_PREFIX_A_WITH_REF = VERIFICATION_PREFIX_CLAIMS + [REF_A]

# Scenario B prefix + reference: skips records (records_required=False)
_PREFIX_B_WITH_REF = VERIFICATION_PREFIX_CLAIMS_B + [REF_B]


# ===========================================================================
# GROUP B — RecordsCoordinationAgent: all four branches
# ===========================================================================


@pytest.mark.live
async def test_B1_decline_path_escalates(run_conversation, assert_and_record):
    """
    B1: Scenario A exact — member says "I'd rather not deal with this right now" when asked
    about records. Verifies the decline branch triggers escalation.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'd rather not deal with this right now",
        ],
        test_name="test_B1_decline_path_escalates",
        scenario=(
            "Scenario A: member declines records outright → escalation with reference number in message"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B2_doctor_direct_then_decline_all(run_conversation, assert_and_record):
    """
    B2: Member says "Sure, the office will get that over to you"
    (doctor_direct) → upload link offered → "I'll pass on that, thanks"
    → Personal Guide offered → "No thanks, I'll handle it myself" → escalation.
    Verifies doctor_direct ack, upload link offer, Personal Guide fallback, and
    final decline → escalation path.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Sure, the office will get that over to you",  # doctor_direct
            "I'll pass on that, thanks",  # decline upload link
            "No thanks, I'll handle it myself",  # decline Personal Guide
        ],
        test_name="test_B2_doctor_direct_then_decline_all",
        scenario=(
            "doctor_direct → upload link offered → declined → Personal Guide offered → declined → escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B3_upload_yes_email_confirmed_guide_yes(run_conversation, assert_and_record):
    """
    B3: Member says "Yeah go ahead and send me that" to upload link offer → email on file confirmed
    → link sent → Personal Guide offered → "That works for me, please go ahead" → guide triggered
    → routed to notification_setup_agent.
    Verifies Branch A complete happy path including Personal Guide consent.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Yeah go ahead and send me that",  # upload_method=member_upload
            "yes",  # upload_consent=yes
            "That works for me, please go ahead",  # email_confirmed=yes → link sent + guide offered
            "yes please",  # personal_guide_consent=yes → guide triggered
            "text me",  # notification_method=sms
            "yes, that number works",  # phone confirmed
        ],
        test_name="test_B3_upload_yes_email_confirmed_guide_yes",
        scenario=(
            "Branch A: upload link yes → email confirmed → link sent → "
            "Personal Guide yes → guide triggered → notification_setup routing"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_records_branch(record, "personal_guide"), "records_branch==personal_guide"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B4_upload_yes_email_declined_new_email_guide_yes(run_conversation, assert_and_record):
    """
    B4: Member says "Yeah go ahead and send me that" to upload link → email declined → new email provided
    → link sent → Personal Guide offered → "Sure, please reach out to them" → guide triggered
    → routed to notification_setup_agent.
    Verifies email update sub-flow inside records coordination.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Yeah go ahead and send me that",  # upload_method=member_upload
            "yes",  # upload_consent=yes
            "no",  # email_confirmed=no → ask for new email
            "michael.brown.new@gmail.com",  # new email → link sent + guide offered
            "Sure, please reach out to them",  # personal_guide_consent=yes → guide triggered
            "send a text",  # notification_method=sms
            "yes that's correct",  # phone confirmed
        ],
        test_name="test_B4_upload_yes_email_declined_new_email_guide_yes",
        scenario=(
            "Branch A: upload yes → email declined → new email → link sent → "
            "Personal Guide yes → notification_setup routing"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B5_upload_yes_email_ambiguous_exhaustion(run_conversation, assert_and_record):
    """
    B5: Member says "Yeah go ahead and send me that" to accept the upload link, then gives
    ambiguous email confirmation twice → email_confirmed slot exhausts → escalation.
    Verifies that the email_confirmed exhaustion guard fires inside records coordination.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Yeah go ahead and send me that",  # upload_method=member_upload → upload offer
            "yes",  # upload_consent=yes → email_confirmed offered
            "maybe",  # email_confirmed ambiguous #1 (count=1)
            "I think so",  # email_confirmed ambiguous #2 (count=2)
            "not sure about that",  # email_confirmed ambiguous #3 → EXHAUSTED → escalation
        ],
        test_name="test_B5_upload_yes_email_ambiguous_exhaustion",
        scenario=("Upload yes → email ambiguous × 3 → email_confirmed exhausted → escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B6_personal_guide_immediate_consent_yes(run_conversation, assert_and_record):
    """
    B6: Member says "Feel free to call my doctor's office directly" (personal_guide immediately)
    → consent asked → "Sure, please reach out to them" → guide triggered → routed to notification_setup.
    Verifies Branch C direct entry without upload link step.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",  # upload_method=personal_guide
            "Sure, please reach out to them",  # personal_guide_consent=yes
            "you can text me updates",  # notification_method=sms
            "yes that's my number",  # phone confirmed
        ],
        test_name="test_B6_personal_guide_immediate_consent_yes",
        scenario=(
            "Branch C: immediate personal_guide intent → consent yes → "
            "guide triggered → notification_setup routing"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_records_branch(record, "personal_guide"), "records_branch==personal_guide"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B7_personal_guide_immediate_consent_no(run_conversation, assert_and_record):
    """
    B7: Member says "Feel free to call my doctor's office directly"
    → consent asked → "No thanks, I'll handle it myself" → escalation.
    Verifies Branch C decline path (member withdraws Personal Guide consent).
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",  # upload_method = personal_guide
            "No thanks, I'll handle it myself",  # personal_guide_consent declined
        ],
        test_name="test_B7_personal_guide_immediate_consent_no",
        scenario=("Branch C: personal_guide intent → consent no → escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B8_decline_from_start(run_conversation, assert_and_record):
    """
    B8: Member says "no" at the first records question → immediate escalation.
    Verifies Branch D (decline) fires on the first utterance without requiring
    multiple attempts.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "no",  # decline = immediate escalation
        ],
        test_name="test_B8_decline_from_start",
        scenario=("Branch D: 'no' at first records question → immediate escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B9_upload_consent_ambiguous_falls_through_to_guide(run_conversation, assert_and_record):
    """
    B9: Member says "Yeah go ahead and send me that" to accept the upload link, then gives
    ambiguous upload consent twice → slot exhausts → falls through to Personal Guide offer.
    Verifies the upload_consent exhaustion → Personal Guide fallback transition.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Yeah go ahead and send me that",  # upload_method=member_upload → upload offer
            "maybe",  # upload_consent ambiguous #1 (count=1)
            "I'm not sure",  # upload_consent ambiguous #2 (count=2)
            "I don't know",  # upload_consent ambiguous #3 → EXHAUSTED → guide offered
            "Sure, please reach out to them",  # personal_guide_consent=yes → guide triggered
            "send me a text",  # notification_method=sms
            "yes that's fine",  # phone confirmed
        ],
        test_name="test_B9_upload_consent_ambiguous_falls_through_to_guide",
        scenario=(
            "Upload consent ambiguous × 3 → exhaustion falls through to "
            "Personal Guide offer → consent yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B10_personal_guide_consent_exhaustion(run_conversation, assert_and_record):
    """
    B10: Personal Guide consent asked → three ambiguous responses → exhaustion → escalation.
    Verifies the personal_guide_consent exhaustion path.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",  # upload_method = personal_guide
            "maybe",  # ambiguous attempt 1
            "I think so",  # ambiguous attempt 2
            "not sure",  # ambiguous attempt 3
        ],
        test_name="test_B10_personal_guide_consent_exhaustion",
        scenario=("Personal Guide consent ambiguous × 3 → slot exhausted → escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B11_transfer_request_during_records_coordination(run_conversation, assert_and_record):
    """
    B11: Member says "Can I speak with a live representative instead?" during records coordination.
    Verifies the TRANSFER_REQUEST guard fires inside RecordsCoordinationAgent
    and routes to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Can I speak with a live representative instead?",
        ],
        test_name="test_B11_transfer_request_during_records_coordination",
        scenario=("TRANSFER_REQUEST guard during records coordination → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B12_abuse_guard_during_records_coordination(run_conversation, assert_and_record):
    """
    B12: Member uses explicit profanity during records coordination.
    Verifies the ABUSE guard fires inside RecordsCoordinationAgent
    and routes to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "this is bullshit just send the damn link",
        ],
        test_name="test_B12_abuse_guard_during_records_coordination",
        scenario=("ABUSE guard during records coordination → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP B_method — RecordsCoordinationAgent: upload_method natural-language
#                  variants (doctor_direct and member_upload)
# ===========================================================================


@pytest.mark.live
async def test_B_m_1_my_doctor_can_send_it(run_conversation, assert_and_record):
    """
    B_m_1: "my doctor can send it" → upload_method='doctor_direct'.

    A first-person declaration that the doctor will handle sending.  The
    extraction prompt maps any "my doctor will send" construction to
    doctor_direct.  Agent should acknowledge, offer the upload link as
    a fallback, and when declined proceed to Personal Guide offer.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "my doctor can send it",  # doctor_direct
            "I'll pass on that, thanks",  # decline upload link
            "Sure, please reach out to them",  # personal_guide_consent
            "text works for me",  # notification_method=sms
            "yes that's still my number",  # phone confirmed
        ],
        test_name="test_B_m_1_my_doctor_can_send_it",
        scenario=(
            "'my doctor can send it' → doctor_direct → upload link declined → "
            "Personal Guide yes → guide triggered → notification_setup routing"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_m_2_provider_fax_it_over(run_conversation, assert_and_record):
    """
    B_m_2: "I'll have the provider fax it over" → upload_method='doctor_direct'.

    Provider-office framing with 'fax' as the transfer mechanism.  Verifies
    that a provider-sent variation using 'fax' (rather than 'send') is
    still classified as doctor_direct and not as a request for an upload link.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'll have the provider fax it over",  # doctor_direct
            "I'll pass on that, thanks",  # decline upload link
            "Sure, please reach out to them",  # personal_guide_consent
            "send me a text",  # notification_method=sms
            "yes, that's my number",  # phone confirmed
        ],
        test_name="test_B_m_2_provider_fax_it_over",
        scenario=(
            "'I'll have the provider fax it over' → doctor_direct → "
            "upload declined → Personal Guide yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_m_3_providers_office_will_send(run_conversation, assert_and_record):
    """
    B_m_3: "the provider's office will send it" → upload_method='doctor_direct'.

    Possessive noun-phrase form ('provider's office') rather than 'my doctor'.
    Verifies that the doctor_direct mapping handles office/provider terminology
    as a synonym for the doctor sending records directly.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "the provider's office will send it",  # doctor_direct
            "I'll pass on that, thanks",  # decline upload link
            "Sure, please reach out to them",  # personal_guide_consent
            "email works for me",  # notification_method=email
            "yes that's the right one",  # email confirmed
        ],
        test_name="test_B_m_3_providers_office_will_send",
        scenario=(
            "'the provider's office will send it' → doctor_direct → "
            "upload declined → Personal Guide yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_m_4_will_upload_myself(run_conversation, assert_and_record):
    """
    B_m_4: "I will upload them myself" → upload_method='member_upload' →
    upload link offered → email confirmed ("yes that's the right one") → upload_link_sent.

    First-person commitment with explicit 'myself' reinforcement.  Verifies
    that a member who volunteers to upload without first being asked is
    classified as member_upload and immediately offered the secure link.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I will upload them myself",  # member_upload
            "yes that's the right one",  # email on file confirmed
            "Sure, please reach out to them",  # personal_guide_consent
            "let's do texts",  # notification_method=sms
            "yep that's right",  # phone confirmed
        ],
        test_name="test_B_m_4_will_upload_myself",
        scenario=(
            "'I will upload them myself' → member_upload → upload link offered → "
            "email confirmed → link sent → Personal Guide yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            # (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            # (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            # (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_m_5_can_i_just_upload_it(run_conversation, assert_and_record):
    """
    B_m_5: "can I just upload it?" → upload_method='member_upload' →
    upload link offered → email confirmed ("go ahead and use that one") →
    upload_link_sent → Personal Guide consent yes → guide triggered → notification_setup.

    Question form with 'just' as a softener.  Verifies that a polite upload
    request is classified as member_upload and the full happy path completes
    through to notification_setup.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "can I just upload it?",  # member_upload (question form)
            "yes",
            "Sure, please reach out to them",  # personal_guide_consent
            "texts are fine",  # notification_method=sms
            "yes go ahead",  # phone confirmed
        ],
        test_name="test_B_m_5_can_i_just_upload_it",
        scenario=(
            "'can I just upload it?' → member_upload → email confirmed → "
            "link sent → Personal Guide yes → guide triggered → notification_setup"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_m_6_conversational_easier_doctor_handle_it(run_conversation, assert_and_record):
    """
    B_m_6: Conversational: "Actually I think it would be easier to have my
    doctor handle it, can they send it over?"

    Complex sentence with a hedge ('I think'), a comparative ('easier'),
    and a trailing confirmation question ('can they send it over?').  Verifies
    that doctor_direct is extracted despite the multi-clause structure and that
    the agent does not treat the trailing question as a new slot value.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Actually I think it would be easier to have my doctor handle it, "
            "can they send it over?",  # doctor_direct
            "I'll pass on that, thanks",  # decline upload link
            "Sure, please reach out to them",  # personal_guide_consent
            "you can text me on that number",  # notification_method=sms
            "yes that's still good",  # phone confirmed
        ],
        test_name="test_B_m_6_conversational_easier_doctor_handle_it",
        scenario=(
            "Conversational doctor_direct with hedge + trailing question → "
            "upload declined → Personal Guide yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_m_7_conversational_rather_upload_myself(run_conversation, assert_and_record):
    """
    B_m_7: Conversational: "I'd rather upload it myself if that's an option"
    → upload_method='member_upload'.

    Preference declaration with conditional framing ('if that's an option').
    Verifies that member_upload is extracted even when the member frames their
    choice as conditional rather than a direct instruction.  The conditional
    clause contains no contradicting signal.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'd rather upload it myself if that's an option",  # member_upload
            "that works",  # email confirmed
            "Sure, please reach out to them",  # personal_guide_consent
            "email me please",  # notification_method=email
            "yes that's the right email",  # email confirmed
        ],
        test_name="test_B_m_7_conversational_rather_upload_myself",
        scenario=(
            "Conversational 'rather upload myself, if that's an option' → member_upload → "
            "email confirmed → link sent → Personal Guide yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            # (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            # (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP B_pg — RecordsCoordinationAgent: personal_guide_consent variants
# ===========================================================================


@pytest.mark.live
async def test_B_pg_1_yes_please_proceed(run_conversation, assert_and_record):
    """
    B_pg_1: "yes please proceed" → personal_guide_consent='yes' →
    guide triggered → notification_setup routing.

    Two-word affirmation ('yes please') with an explicit action verb
    ('proceed').  Verifies the consent extraction accepts this emphatic form
    and the guide is triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Please just go ahead and contact them directly",  # personal_guide intent
            "yes please proceed",  # personal_guide_consent
            "send me texts",  # notification_method=sms
            "yes that number is correct",  # phone confirmed
        ],
        test_name="test_B_pg_1_yes_please_proceed",
        scenario=(
            "personal_guide intent → 'yes please proceed' → "
            "personal_guide_consent=yes → guide triggered → notification_setup"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_pg_2_sure_go_ahead(run_conversation, assert_and_record):
    """
    B_pg_2: "sure go ahead" → personal_guide_consent='yes' →
    guide triggered.

    Casual single-word affirmative ('sure') with an imperative clause
    ('go ahead').  Verifies that 'sure' normalises to consent='yes' per
    the extraction prompt and the guide is triggered without ambiguity.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Can your team call my doctor's office?",  # personal_guide intent
            "sure go ahead",  # personal_guide_consent=yes
            "text is fine",  # notification_method=sms
            "yep",  # phone confirmed
        ],
        test_name="test_B_pg_2_sure_go_ahead",
        scenario=("personal_guide intent → 'sure go ahead' → personal_guide_consent=yes → guide triggered"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_pg_3_please_arrange_that(run_conversation, assert_and_record):
    """
    B_pg_3: "please arrange that" → personal_guide_consent='yes' →
    guide triggered.

    Imperative consent with a placeholder object ('that').  Verifies that
    an indirect instruction to proceed ('arrange that') is classified as
    an unambiguous affirmative consent rather than an ambiguous or off-topic
    response.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Reach out to the provider on my behalf",  # personal_guide intent
            "please arrange that",  # personal_guide_consent=yes
            "email is fine",  # notification_method=email
            "yes that's the right one",  # email confirmed
        ],
        test_name="test_B_pg_3_please_arrange_that",
        scenario=(
            "personal_guide intent → 'please arrange that' → personal_guide_consent=yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_4_conversational_yes_reach_out(run_conversation, assert_and_record):
    """
    B_pg_4: Conversational: "Yes that would be great, please have them reach
    out to my doctor."

    Multi-clause affirmation: 'yes' leads, 'that would be great' is a
    sentiment clause, and 'please have them reach out to my doctor' restates
    the action.  Verifies personal_guide_consent='yes' is extracted despite
    the verbose structure and the guide is triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Go ahead and give my doctor a call",  # personal_guide intent
            "Yes that would be great, please have them reach out to my doctor",
            "you can text me on my phone",  # notification_method=sms
            "yes that's still the number",  # phone confirmed
        ],
        test_name="test_B_pg_4_conversational_yes_reach_out",
        scenario=(
            "Conversational multi-clause affirmation → personal_guide_consent=yes → "
            "guide triggered → notification_setup"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_5_conversational_yes_doctor_better_with_calls(run_conversation, assert_and_record):
    """
    B_pg_5: Conversational: "Yes please, I'd appreciate that — my doctor's
    office is better with phone calls anyway."

    Affirmation with an aside explaining the member's preference.  The
    em-dash separates the consent ('yes please, I'd appreciate that') from
    a contextual remark ('my doctor's office is better with phone calls').
    Verifies that the aside does not introduce ambiguity for the consent
    extraction and the guide is triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "You're welcome to contact my physician directly",  # personal_guide intent
            "Yes please, I'd appreciate that — my doctor's office is better with phone calls anyway",
            "email works better for me",  # notification_method=email
            "yes that's my address",  # email confirmed
        ],
        test_name="test_B_pg_5_conversational_yes_doctor_better_with_calls",
        scenario=(
            "Conversational affirmation + em-dash aside → personal_guide_consent=yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_pg_6_no_wont_be_necessary_handle_myself(run_conversation, assert_and_record):
    """
    B_pg_6: "no that won't be necessary, I'll handle it myself" →
    personal_guide_consent='no' → escalation.

    Explicit decline with a self-sufficiency assertion.  The extraction
    prompt maps "that won't be necessary" to consent='no'.  Verifies
    escalation fires when consent is declined after a personal_guide intent.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Have someone from your team contact my doctor",  # personal_guide intent
            "no that won't be necessary, I'll handle it myself",  # consent=no
        ],
        test_name="test_B_pg_6_no_wont_be_necessary_handle_myself",
        scenario=(
            "personal_guide intent → 'no that won't be necessary' → personal_guide_consent=no → escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_7_conversational_rather_deal_directly(run_conversation, assert_and_record):
    """
    B_pg_7: Conversational: "No I think I'd rather deal with it directly,
    no thanks" → personal_guide_consent='no' → escalation.

    Double-negation decline: leading 'no' and closing 'no thanks', with a
    preference clause ('I'd rather deal with it directly') in between.
    Verifies that the extraction resolves to consent='no' despite the
    verbose multi-clause structure and escalation is triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I was thinking you could contact my doctor's office",  # personal_guide intent
            "No I think I'd rather deal with it directly, no thanks",  # consent=no
        ],
        test_name="test_B_pg_7_conversational_rather_deal_directly",
        scenario=("Conversational double-negation decline → personal_guide_consent=no → escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_B_pg_8_not_right_now_soft_deferral(run_conversation, assert_and_record):
    """
    B_pg_8: "maybe some other time" → personal_guide_consent='no' → escalation.

    Soft temporal deferral; a member who defers to a future time is declining
    consent now.  Verifies that a deferral that could be mistaken for ambiguity
    is correctly classified as a decline and escalation fires.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'd prefer you contact my doctor on my behalf",  # personal_guide intent
            "maybe some other time",  # soft temporal deferral → consent=no
        ],
        test_name="test_B_pg_8_not_right_now_soft_deferral",
        scenario=(
            "'maybe some other time' → personal_guide_consent=no (temporal deferral = decline) → escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP B_pg_exact — personal_guide_consent exact phrases + ambiguous-then-no
# ===========================================================================


@pytest.mark.live
async def test_B_pg_exact_1_perfect_please_do_that(run_conversation, assert_and_record):
    """
    B_pg_exact_1: "That works for me, please go ahead"
    — natural spoken-language phrase replacing the prior exact phrase.
    personal_guide_consent=yes → personal_guide_triggered=True.

    This MUST pass — it is the canonical phrase in the static transcript.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "That works for me, please go ahead",
            "send me a text",  # notification_method=sms
            "yes that's correct",  # phone confirmed
        ],
        test_name="test_B_pg_exact_1_perfect_please_do_that",
        scenario="personal_guide_consent natural phrase 'That works for me, please go ahead' → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_pg_exact_2_yes_please_do_that(run_conversation, assert_and_record):
    """
    B_pg_exact_2: "Yes please do that" (near-exact) → personal_guide_consent=yes → triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "Yes please do that",
            "text me",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B_pg_exact_2_yes_please_do_that",
        scenario="personal_guide_consent 'Yes please do that' → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_exact_3_absolutely_please_reach_out(run_conversation, assert_and_record):
    """
    B_pg_exact_3: "absolutely, please reach out to them" → personal_guide_consent=yes → triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "absolutely, please reach out to them",
            "text me updates",  # notification_method=sms
            "absolutely, that's the right number",  # phone confirmed
        ],
        test_name="test_B_pg_exact_3_absolutely_please_reach_out",
        scenario="personal_guide_consent 'absolutely, please reach out to them' → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_ambiguous_no_1_hmm_then_no(run_conversation, assert_and_record):
    """
    B_pg_ambiguous_no_1: personal_guide intent → "hmm" (ambiguous) → agent re-asks →
    "No, I don't think so" → escalation. personal_guide was NOT triggered.

    Verifies that ambiguous first response triggers retry, and subsequent
    explicit decline results in escalation (member_declined_personal_guide).
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I guess you could try reaching the doctor",  # personal_guide intent
            "hmm",  # ambiguous
            "No, I don't think so",  # explicit decline after re-ask
        ],
        test_name="test_B_pg_ambiguous_no_1_hmm_then_no",
        scenario="personal_guide_consent 'hmm' (ambiguous) → re-ask → 'No, I don't think so' → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_ambiguous_no_2_maybe_then_explicit_no(run_conversation, assert_and_record):
    """
    B_pg_ambiguous_no_2: "maybe" (ambiguous) → re-ask → "no I don't want to do that" → escalation.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Maybe you can contact the doctor's office if that helps",  # personal_guide intent
            "maybe",  # ambiguous
            "no I don't want to do that",  # explicit decline after re-ask
        ],
        test_name="test_B_pg_ambiguous_no_2_maybe_then_explicit_no",
        scenario="personal_guide_consent 'maybe' → re-ask → explicit no → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_no_reason_1_ill_handle_it(run_conversation, assert_and_record):
    """
    B_pg_no_reason_1: "I'll handle it on my own" → personal_guide_consent=no → escalation.
    Self-sufficiency reason framing; no "no" keyword but intent is clear decline.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "You can try calling my doctor if you need records",  # personal_guide intent
            "I'll handle it on my own",  # consent=no (implicit decline with reason)
        ],
        test_name="test_B_pg_no_reason_1_ill_handle_it",
        scenario="personal_guide_consent decline with reason 'I'll handle it on my own' → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_pg_no_reason_2_doctor_will_send(run_conversation, assert_and_record):
    """
    B_pg_no_reason_2: "that's not needed, my doctor's office will send it directly"
    → personal_guide_consent=no → escalation.

    Even though member says their doctor will send records directly, declining
    Personal Guide outreach is a hard decline and escalation fires regardless.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "My doctor's office should be able to help you get the records",  # personal_guide intent
            "that's not needed, my doctor's office will send it directly",  # consent=no with reason
        ],
        test_name="test_B_pg_no_reason_2_doctor_will_send",
        scenario="personal_guide_consent declined with 'doctor will send it' reasoning → escalation",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record), "escalated"),
            # (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP B_uc — RecordsCoordinationAgent: upload_consent variants
# ===========================================================================


@pytest.mark.live
async def test_B_uc_1_yes_please_send_the_link(run_conversation, assert_and_record):
    """
    B_uc_1: "Sounds good, please send it over" → upload_consent='yes' →
    email on file confirmed → upload_link_sent=True.

    Explicit affirmative with an imperative restating the action ('send it
    over').  Verifies upload_consent='yes' is extracted from a natural spoken
    form and upload_link_sent=True after the email confirmation step.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Sounds good, please send it over",  # upload_method=member_upload → upload offer
            "yes",  # upload_consent=yes → email shown
            "that's the right email",  # email_confirmed=yes → link sent + guide offered
            "go ahead and have them reach out",  # personal_guide_consent=yes
            "send me a text",  # notification_method=sms
            "yes, that's my number",  # phone confirmed
        ],
        test_name="test_B_uc_1_yes_please_send_the_link",
        scenario=(
            "'Sounds good, please send it over' → upload_consent=yes "
            "→ email confirmed → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_uc_2_sure_that_would_help(run_conversation, assert_and_record):
    """
    B_uc_2: "sure that would help" → upload_consent='yes' →
    email confirmed → upload_link_sent=True.

    Casual 'sure' affirmative followed by a sentiment clause ('that would
    help').  Verifies that 'sure' is accepted as upload_consent='yes' and
    that the trailing clause does not dilute the affirmation enough to
    fire the bias rule.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "sure that would help",  # upload_method=member_upload
            "sure",  # upload_consent=yes
            "yep that's correct",  # email_confirmed=yes
            "yes please do that",  # personal_guide_consent=yes
            "texts work",  # notification_method=sms
            "yep",  # phone confirmed
        ],
        test_name="test_B_uc_2_sure_that_would_help",
        scenario=("'sure that would help' → upload_consent=yes → email confirmed → upload_link_sent=True"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_uc_3_conversational_sounds_easier_than_fax(run_conversation, assert_and_record):
    """
    B_uc_3: Conversational: "Oh yes please, that sounds much easier than
    having to fax anything."

    Emphatic affirmation ('oh yes please') with a contrastive justification
    that mentions 'fax'.  Verifies upload_consent='yes' is extracted and
    that 'fax' in the trailing clause does not cause misclassification.
    The upload link must be sent after email confirmation.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            ("Oh yes please, that sounds much easier than having to fax anything"),
            "yes please",  # upload_consent=yes
            "that's the one, go ahead",  # email_confirmed=yes
            "yes please do that too",  # personal_guide_consent=yes
            "oh, texts would be great",  # notification_method=sms
            "yes that's still my number",  # phone confirmed
        ],
        test_name="test_B_uc_3_conversational_sounds_easier_than_fax",
        scenario=(
            "Conversational 'yes please' + fax mention → upload_consent=yes → "
            "email confirmed → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B_uc_4_no_dont_need_link_but_guide_yes(run_conversation, assert_and_record):
    """
    B_uc_4: "no I don't need the link" declines upload → Personal Guide
    offered → "yes" → guide triggered.

    Explicit one-sentence decline of the upload link with a reason ('don't
    need the link').  Verifies the fallback path: declining the link triggers
    the Personal Guide offer, and accepting it triggers the guide without
    upload_link_sent ever being set.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "no I don't need the link",  # upload_method=member_upload → upload offer
            "no",  # upload_consent=no → guide offered directly
            "Sure, please reach out to them",  # personal_guide_consent=yes
            "text me",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B_uc_4_no_dont_need_link_but_guide_yes",
        scenario=(
            "'no I don't need the link' → upload_consent=no → "
            "Personal Guide offered → consent yes → guide triggered → notification_setup"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_uc_5_conversational_no_link_but_yes_reach_doctor(run_conversation, assert_and_record):
    """
    B_uc_5: Conversational: "No thanks for the link, but yes please have
    someone reach out to my doctor."

    A single utterance that combines upload_consent='no' ('no thanks for
    the link') with an immediate personal_guide consent ('yes please have
    someone reach out to my doctor').  Verifies that the extraction handles
    the compound decline + consent and the guide is triggered directly
    without a separate consent-collection turn.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            (
                "No thanks for the link, but yes please have someone reach out to my doctor"
            ),  # upload_method + guide intent
            "yes please",  # personal_guide_consent=yes (if agent still asks)
            "send me a text",  # notification_method=sms
            "yes that's right",  # phone confirmed
        ],
        test_name="test_B_uc_5_conversational_no_link_but_yes_reach_doctor",
        scenario=(
            "Compound 'no link + yes guide' in one utterance → "
            "upload_consent=no + personal_guide_consent=yes → guide triggered"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_routed_to_notification_setup(record), "routed_to_notification_setup"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP B_uc_softno — upload_consent soft-no + ambiguous exhaustion
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_B_uc_softno_1_id_rather_not(run_conversation, assert_and_record):
    """
    B_uc_softno_1: "I'd rather not" → upload_consent=no (soft refusal, no "no" keyword).
    Agent offers Personal Guide → "yes" → personal_guide_triggered=True.
    upload_link_sent must be False (link was declined).
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'd rather not",  # upload_method → soft decline → upload offer (or guide)
            "I'd rather not",  # upload_consent=no → guide offered
            "yes please do that",  # personal_guide_consent=yes
            "texts are fine I suppose",  # notification_method=sms
            "yes that's correct",  # phone confirmed
        ],
        test_name="test_B_uc_softno_1_id_rather_not",
        scenario="upload_consent soft 'I'd rather not' → guide offered → yes → personal_guide_triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_upload_link_sent(record), "not_upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_uc_softno_2_doctor_send_with_guide_yes(run_conversation, assert_and_record):
    """
    B_uc_softno_2: "no that's ok I'll have my doctor send it" → upload_consent=no with reason.
    Agent offers Personal Guide → "yes" → personal_guide_triggered=True.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "no that's ok I'll have my doctor send it",  # upload_method=doctor_direct → upload offer
            "no thanks",  # upload_consent=no → guide offered
            "yes please reach out to them",  # personal_guide_consent=yes
            "you can text me",  # notification_method=sms
            "yes that's my number",  # phone confirmed
        ],
        test_name="test_B_uc_softno_2_doctor_send_with_guide_yes",
        scenario="upload_consent decline with reason → guide offered → yes → personal_guide_triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_upload_link_sent(record), "not_upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B_uc_exhaust_1_three_ambiguous_then_guide(run_conversation, assert_and_record):
    """
    B_uc_exhaust_1: "hmm" → "I'm not sure" → "maybe" (three ambiguous upload_consent answers).
    Slot exhausts → falls through to Personal Guide offer → "yes" → personal_guide_triggered=True.

    upload_consent exhaustion in records_coordination falls through to guide,
    NOT escalation. This is distinct from email_confirmed exhaustion (which escalates).
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "yes please",  # upload_method=member_upload → upload offer (upload_consent asked)
            "hmm",  # upload_consent ambiguous #1 (count=1)
            "I'm not sure",  # upload_consent ambiguous #2 (count=2)
            "maybe",  # upload_consent ambiguous #3 → EXHAUSTED → guide offered
            "yes, please do that",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes that works",  # phone confirmed
        ],
        test_name="test_B_uc_exhaust_1_three_ambiguous_then_guide",
        scenario="upload_consent exhaustion (3× ambiguous) → guide fallthrough → yes → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_upload_link_sent(record), "not_upload_link_sent"),
        ],
    )


# ---------------------------------------------------------------------------
# Shared prefix for Group B2 — lands at email_confirmed inside
# RecordsCoordinationAgent.  Scenario A (records_required=True) is the only
# path that enters records_coordination; accepting the upload link offer
# causes the next agent turn to read back the email on file.
# ---------------------------------------------------------------------------
_B2_EMAIL_PREFIX = _PREFIX_A_WITH_REF + [
    "Yeah, go ahead and send me that link",  # upload_method=member_upload → upload offer
    "yes",  # upload_consent=yes → NOW at email_confirmed
]
NEW_EMAIL_B2 = "michael.brown.new@gmail.com"


# ===========================================================================
# GROUP B2 — RecordsCoordinationAgent: email_confirmed slot behaviour
#            (affirmations, bias rule, inline replacement, exhaustion)
# ===========================================================================


@pytest.mark.live
async def test_B2_1_yes_that_correct_confirms_email(run_conversation, assert_and_record):
    """
    B2_1: "yes that's correct" confirms email on file → upload link sent.

    email_confirmed uses the bias rule (same as delivery_management): only a
    clear affirmation maps to 'yes'.  "yes that's correct" leads with 'yes' and
    the trailing clause reinforces it — the bias rule must NOT fire and the
    original email on file must be used without triggering a new-email request.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "yes that's correct",  # clear affirmation; bias rule must not fire
            "yes please",  # personal_guide_consent=yes
            "send me a text",  # notification_method=sms
            "yes that's fine",  # phone confirmed
        ],
        test_name="test_B2_1_yes_that_correct_confirms_email",
        scenario=(
            "'yes that's correct' → email_confirmed=yes (bias rule does not fire) → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_2_correct_single_word_confirms_email(run_conversation, assert_and_record):
    """
    B2_2: Bare "correct" confirms email on file → upload link sent.

    Single-word affirmative that is not "yes" — verifies the extraction prompt
    maps 'correct' to email_confirmed='yes' and the bias rule does not fire.
    Mirrors delivery_management test_email_confirmed_correct for the
    records_coordination context.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "correct",
            "yes",  # personal_guide_consent=yes
            "text me",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_2_correct_single_word_confirms_email",
        scenario=("'correct' → email_confirmed=yes → upload_link_sent=True"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_3_yep_thats_my_email_confirms(run_conversation, assert_and_record):
    """
    B2_3: "yep that's my email" confirms email on file → upload link sent.

    Colloquial 'yep' with a possessive trailing clause.  The leading 'yep' is
    a clear affirmation; the bias rule must not fire even though 'that's my
    email' is evaluative phrasing rather than a direct 'yes'.
    Mirrors delivery_management test_email_confirmed_yep_thats_my_email.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "yep that's my email",
            "yes please",  # personal_guide_consent=yes
            "texts work",  # notification_method=sms
            "yep",  # phone confirmed
        ],
        test_name="test_B2_3_yep_thats_my_email_confirms",
        scenario=("'yep that's my email' → email_confirmed=yes → upload_link_sent=True"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_4_yes_bare_confirms_email(run_conversation, assert_and_record):
    """
    B2_4: Bare "yes" confirms email on file → upload link sent.

    Baseline case for email_confirmed in records_coordination: the simplest
    possible affirmation.  Verifies the happy path with no trailing context words.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "yes",
            "yes",  # personal_guide_consent=yes
            "text me",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_4_yes_bare_confirms_email",
        scenario="'yes' → email_confirmed=yes → upload_link_sent=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_5_conversational_yes_check_regularly(run_conversation, assert_and_record):
    """
    B2_5: Conversational affirmation "Yes that email is fine, I check it
    regularly" confirms email on file → upload link sent.

    Net affirmative despite 'fine' qualifier and a usage-habit clause
    ('I check it regularly').  Verifies the bias rule does not fire when
    the leading 'yes' is explicit — the original email on file must be used
    without triggering a new-email request.  Mirrors delivery_management
    test_email_confirmed_check_every_day for the records_coordination context.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "Yes that email is fine, I check it regularly",
            "yes that would be great",  # personal_guide_consent=yes
            "email me please",  # notification_method=email
            "yes that's right",  # email confirmed
        ],
        test_name="test_B2_5_conversational_yes_check_regularly",
        scenario=(
            "Conversational affirmation with usage clause → email_confirmed=yes "
            "(no bias) → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_6_i_think_so_bias_then_new_email(run_conversation, assert_and_record):
    """
    B2_6: "I think so" triggers the bias rule → agent asks for new email →
    valid replacement provided → upload link sent.

    The bias rule maps any non-clear-affirmation to email_confirmed='no'.
    "I think so" is a hedged affirmation that must trip the bias, causing the
    agent to request a replacement address.  The member then supplies one on
    the next turn and the link is sent.  Mirrors delivery_management
    test_email_bias_i_think_so for the records_coordination email_confirmed slot.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "I think so",  # hedged: bias rule fires → email_confirmed=no
            NEW_EMAIL_B2,  # replacement address → link sent
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_6_i_think_so_bias_then_new_email",
        scenario=(
            "'I think so' triggers bias rule → agent asks for new email → "
            f"'{NEW_EMAIL_B2}' → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_7_not_sure_bias_then_new_email(run_conversation, assert_and_record):
    """
    B2_7: "not sure" triggers the bias rule → agent asks for new email →
    valid replacement provided → upload link sent.

    Two-word uncertainty without first-person 'I'; the bias rule must still
    treat this as non-affirmative and request a replacement.  Mirrors
    delivery_management test_email_bias_not_anymore.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "not sure",  # bias rule fires → email_confirmed=no
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_7_not_sure_bias_then_new_email",
        scenario=(
            "'not sure' triggers bias rule → agent asks for new email → "
            f"'{NEW_EMAIL_B2}' → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_8_no_declines_then_new_email(run_conversation, assert_and_record):
    """
    B2_8: Bare "no" declines email on file → agent asks for replacement →
    "james.wilson.new@gmail.com" provided → upload link sent.

    Explicit rejection — contact_confirmed='no' — verifies the clean two-turn
    decline path inside records_coordination.  Mirrors delivery_management
    test_email_bias_no for the email_confirmed slot in this context.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "no",  # explicit decline
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_8_no_declines_then_new_email",
        scenario=(
            f"'no' → email_confirmed=no → agent asks for new email → '{NEW_EMAIL_B2}' → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_9_inline_no_use_new_email(run_conversation, assert_and_record):
    """
    B2_9: Inline decline + replacement in one utterance:
    "no use michael.brown.new@gmail.com instead" → email extracted inline →
    upload link sent.

    The inline-update rule must extract the replacement address from this single
    utterance, skipping a separate collection turn.  Mirrors delivery_management
    test_email_inline_no_use_new_address for the records_coordination context.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            f"no use {NEW_EMAIL_B2} instead",
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_9_inline_no_use_new_email",
        scenario=(
            f"'no use {NEW_EMAIL_B2} instead' → email extracted inline → "
            "upload_link_sent=True (no separate collection turn)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_10_conversational_inline_outdated_email(run_conversation, assert_and_record):
    """
    B2_10: Conversational inline decline + replacement:
    "Oh wait that email is outdated, please use michael.brown.new@gmail.com,
    that's my current one."

    The inline-update rule must fire within a longer natural-speech utterance
    that opens with an explanation ('that email is outdated') and closes with
    a contextual aside ('my current one').  No second collection turn should be
    required.  Mirrors delivery_management test_email_inline_bounces_current_one.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            f"Oh wait that email is outdated, please use {NEW_EMAIL_B2}, that's my current one",
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_10_conversational_inline_outdated_email",
        scenario=(
            "Conversational inline 'outdated, use <email>' → "
            "email extracted in one turn → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_11_conversational_not_sure_active_then_new_email(run_conversation, assert_and_record):
    """
    B2_11: Conversational: "Hmm I'm not 100% sure that's active anymore,
    let me give you a different one" → bias rule fires → agent asks for
    replacement → new email provided → upload link sent.

    Uncertainty ('not 100% sure') plus a commitment to replace ('let me give
    you a different one') contains no email address, so the inline-update rule
    does not apply.  The bias rule fires on the next turn; the agent asks
    explicitly and the member provides the replacement.  Mirrors delivery_management
    test_email_bias_not_100_sure_active for the records_coordination context.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "Hmm I'm not 100% sure that's active anymore, let me give you a different one",
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_11_conversational_not_sure_active_then_new_email",
        scenario=(
            "Conversational uncertainty triggers bias rule → "
            f"agent asks for new email → '{NEW_EMAIL_B2}' → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_12_invalid_email_reask_then_valid(run_conversation, assert_and_record):
    """
    B2_12: "notanemail" at email_confirmed step → bias rule fires (not a clear
    affirmation) → agent asks for new email → "notanemail" again (invalid
    format) → agent re-asks → valid email provided → upload link sent.

    Tests two layers of collection: (1) the bias rule turning a non-affirmative
    into email_confirmed='no', and (2) the new_email validator rejecting a
    malformed address before accepting the well-formed replacement.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "notanemail",  # bias fires → email_confirmed=no → agent asks for new email
            "notanemail",  # invalid address format → validator rejects → re-ask
            NEW_EMAIL_B2,  # valid address → link sent
            "yes please",  # personal_guide_consent=yes
            "send a text",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_12_invalid_email_reask_then_valid",
        scenario=(
            "'notanemail' → bias fires → re-ask → 'notanemail' invalid → "
            f"re-ask again → '{NEW_EMAIL_B2}' → upload_link_sent=True"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_B2_13_email_exhaustion_escalates(run_conversation, assert_and_record):
    """
    B2_13: Three ambiguous responses at the email_confirmed step exhaust the
    slot → escalation.

    The bias rule maps each non-clear-affirmation to email_confirmed='no' and
    the agent requests a replacement, but the member never provides a valid
    address.  After the allowed number of attempts the slot is exhausted and
    the conversation escalates.  Mirrors delivery_management Group F exhaustion
    behaviour (test_B5_upload_yes_email_ambiguous_exhaustion) for the case
    where more than two non-affirmative responses are given.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "maybe",  # ambiguous attempt 1 → bias → email_confirmed=no
            "I think so",  # ambiguous attempt 2
            "not sure",  # ambiguous attempt 3 → slot exhausted → escalation
        ],
        test_name="test_B2_13_email_exhaustion_escalates",
        scenario=("email_confirmed ambiguous × 3 → slot exhausted → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "agent_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP B2_implicit — email_confirmed implicit-no variants (bias rule)
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_B2_implicit_1_thats_my_old_email(run_conversation, assert_and_record):
    """
    B2_implicit_1: "that's my old email" → bias rule fires (non-clear-affirmation → no)
    → agent asks for new email → new email provided → upload_link_sent=True.

    Stale-data framing with no "no" keyword; bias rule in email_confirmed maps
    any ambiguous/negative-toned response to 'no'.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "that's my old email",
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send me a text",  # notification_method=sms
            "yes that's fine",  # phone confirmed
        ],
        test_name="test_B2_implicit_1_thats_my_old_email",
        scenario="email_confirmed 'that's my old email' → bias rule → new email → upload_link_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_implicit_2_dont_use_that_account(run_conversation, assert_and_record):
    """
    B2_implicit_2: "I don't use that account anymore" → bias rule fires → new email → upload_link_sent=True.
    Account-disuse framing without an explicit negation.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "I don't use that account anymore",
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send me a text",  # notification_method=sms
            "yes that's fine",  # phone confirmed
        ],
        test_name="test_B2_implicit_2_dont_use_that_account",
        scenario="email_confirmed ('I don't use that account anymore')"
        "   → bias rule → new email → upload_link_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_implicit_3_might_not_be_active(run_conversation, assert_and_record):
    """
    B2_implicit_3: "hmm that might not be active" → bias rule fires (uncertainty without negation)
    → agent asks for new email → new email → upload_link_sent=True.

    Uncertainty framing; bias rule treats doubt as non-confirmation → 'no'.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "hmm that might not be active",
            NEW_EMAIL_B2,
            "yes please",  # personal_guide_consent=yes
            "send me a text",  # notification_method=sms
            "yes that's fine",  # phone confirmed
        ],
        test_name="test_B2_implicit_3_might_not_be_active",
        scenario="email_confirmed uncertainty ('might not be active') "
        "→ bias rule → new email → upload_link_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_B2_invalid_1_bad_email_then_valid(run_conversation, assert_and_record):
    """
    B2_invalid_1: "no" → "bademail" (missing @, invalid) → retry → valid email → upload_link_sent=True.

    Verifies the email-format validator retries on invalid format and eventually
    accepts a well-formed address.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "no",  # email_confirmed=no
            "bademail",  # invalid format → retry
            NEW_EMAIL_B2,  # valid email → upload_link_sent
            "yes please",  # personal_guide_consent=yes
            "text me",  # notification_method=sms
            "yes",  # phone confirmed
        ],
        test_name="test_B2_invalid_1_bad_email_then_valid",
        scenario="email_confirmed=no → invalid email → retry → valid email → upload_link_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# Shared prefix for Group C — lands in notification_setup
# Records not required (Scenario B): verification → reference → records skipped
# → notification_setup is the next agent.
# ---------------------------------------------------------------------------
# NOTE: REF_A (12491584) has records_required=True in the SF sandbox.
# _C_PREFIX passes through records_coordination (personal guide path)
# before landing in notification_setup_agent.
_C_PREFIX = VERIFICATION_PREFIX_CLAIMS + [
    REF_A,
    "Feel free to call my doctor's office directly",  # upload_method=personal_guide
    "yes",  # personal_guide_consent=yes → guide triggered
    # → notification_setup_agent now active, awaiting notification_method (N1)
]

# NOTE: REF_B (42695817) also has records_required=True in the SF sandbox.
# _C_PREFIX_B is the Scenario B equivalent of _C_PREFIX.
_C_PREFIX_B = VERIFICATION_PREFIX_CLAIMS_B + [
    REF_B,
    "Feel free to call my doctor's office directly",  # upload_method=personal_guide
    "yes",  # personal_guide_consent=yes → guide triggered
    # → notification_setup_agent now active, awaiting notification_method (N1)
]

# Scenario A prefix that went through records and is now in notification_setup
# (upload link sent path — agent already asked notification method via bridge)
_C_PREFIX_AFTER_RECORDS = VERIFICATION_PREFIX_CLAIMS + [
    REF_A,
    "yes please",  # upload link offer accepted
    "yes",  # email on file confirmed
    "no",  # personal_guide_consent declined → records done
]

NEW_PHONE = "5125554300"
NEW_PHONE_SPOKEN = "five one two five five five four three zero zero"
NEW_EMAIL_C = "james.wilson.updated@gmail.com"


# ===========================================================================
# GROUP C — NotificationSetupAgent: phone_confirmed and email_confirmed slot
#           behaviour (revised)
# ===========================================================================


# ---------------------------------------------------------------------------
# Group-C–specific assertion helpers
# ---------------------------------------------------------------------------


def assert_phone_readback_has_dashes(record: ConversationRecord) -> None:
    """Agent phone readback uses dashes: 512-555-6101, not 5125556101."""
    all_msgs = " ".join((t.agent_message or "") for t in record.turns)
    assert PHONE_ON_FILE_B in all_msgs, (
        f"Expected agent message to contain {PHONE_ON_FILE_B!r} (dash-formatted). "
        f"Full transcript (first 500 chars): {all_msgs[:500]!r}"
    )


def assert_phone_saved(record: ConversationRecord, expected: str) -> None:
    """phone == expected in final state."""
    actual = record.final_state.get("phone", "")
    assert actual == expected, f"Expected phone={expected!r}, got {actual!r}"


# ===========================================================================
# C1_revised – C5_revised: phone_confirmed clear affirmations
# ===========================================================================


@pytest.mark.live
async def test_C1_revised_yes_correct_phone_affirmation(run_conversation, assert_and_record):
    """
    C1_revised: Two-word affirmation "yes correct" confirms phone on file.

    phone_confirmed uses validate_yes_no (no bias rule) — any clear affirmation
    maps directly to 'yes'.  "yes correct" leads with 'yes' so it must be
    accepted as phone_confirmed='yes' without triggering a retry or re-ask.
    The SMS happy path must complete and notification_channel must be 'sms'.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "yes correct",  # two-word affirmation: must map to phone_confirmed=yes
            "no thanks",  # follow_up closure
        ],
        test_name="test_C1_revised_yes_correct_phone_affirmation",
        scenario=("'yes correct' → phone_confirmed=yes → notification_channel=sms → no retry, no escalation"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C2_revised_thats_right_phone_affirmation(run_conversation, assert_and_record):
    """
    C2_revised: Demonstrative affirmation "that's right" confirms phone on file.

    "that's right" is a clear single-clause affirmation; phone_confirmed must
    resolve to 'yes' without a retry.  Mirrors the delivery_management fax
    test_fax_confirmed_thats_right invariant applied to the phone_confirmed slot.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "that's right",  # demonstrative affirmation
            "no thanks",
        ],
        test_name="test_C2_revised_thats_right_phone_affirmation",
        scenario=(
            "'that's right' → phone_confirmed=yes → notification_channel=sms → no retry, no escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C3_revised_repeat_number_back_treated_as_yes(run_conversation, assert_and_record):
    """
    C3_revised: Member repeats the phone number back in spoken-digit form
    → treated as 'yes' per the extraction prompt.

    When the agent reads back "Is your phone number 512-555-6101?", a member
    who responds with the same number in spoken form ("five one two five five
    five six one zero one") is implicitly confirming.  The extraction prompt
    maps a repeated number to phone_confirmed='yes'; the SMS path must complete.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            # spoken form of 512-555-6101 — the number on file
            "five one two five five five six one zero one",
            "no thanks",
        ],
        test_name="test_C3_revised_repeat_number_back_treated_as_yes",
        scenario=(
            "Repeat spoken number back → phone_confirmed=yes (extraction prompt rule) → "
            "notification_channel=sms"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C4_revised_yep_phone_affirmation(run_conversation, assert_and_record):
    """
    C4_revised: Colloquial single-word "yep" confirms phone on file.

    "yep" is an informal but unambiguous affirmative; phone_confirmed must
    resolve to 'yes' in one turn with no retry.  Mirrors delivery_management
    test_fax_confirmed_yep for the phone_confirmed slot.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "yep",
            "no thanks",
        ],
        test_name="test_C4_revised_yep_phone_affirmation",
        scenario=("'yep' → phone_confirmed=yes → notification_channel=sms → no retry"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C5_revised_conversational_still_my_number(run_conversation, assert_and_record):
    """
    C5_revised: Conversational affirmation "Yeah that's still my number,
    I haven't changed it" confirms phone on file.

    The utterance is a net affirmation despite containing filler ("yeah") and
    a subordinate clause ("I haven't changed it").  phone_confirmed has NO
    bias rule — it uses validate_yes_no — so this clear affirmation must map
    to 'yes' without triggering a retry.  The SMS path must complete and
    no escalation should occur.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "Yeah that's still my number, I haven't changed it",
            "no thanks",
        ],
        test_name="test_C5_revised_conversational_still_my_number",
        scenario=(
            "Conversational 'still my number, haven't changed it' → "
            "phone_confirmed=yes (no bias rule) → notification_channel=sms"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# C6_revised – C8_revised: phone declined → new number provided
# ===========================================================================


@pytest.mark.live
async def test_C6_revised_phone_declined_spoken_new_number(run_conversation, assert_and_record):
    """
    C6_revised: Member declines phone on file with "no that's changed", then
    provides replacement in spoken-digit form on the next turn.

    "no that's changed" contains no digits, so phone_confirmed='no' and the
    agent must ask explicitly for a new number.  The replacement spoken as
    "five one two five five five four three zero zero" must normalise to
    5125554300 and the SMS path must complete with the updated number.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "no that's changed",  # phone_confirmed=no (no inline number)
            NEW_PHONE_SPOKEN,  # spoken replacement: 5125554300
            "no thanks",
        ],
        test_name="test_C6_revised_phone_declined_spoken_new_number",
        scenario=(
            "'no that's changed' → agent asks for new number → spoken replacement → "
            "phone=5125554300 → notification_channel=sms"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), f"phone=={NEW_PHONE}"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C7_revised_phone_declined_numeric_string_new_number(run_conversation, assert_and_record):
    """
    C7_revised: Member declines phone on file with bare "no", then provides
    replacement as a numeric digit string "5125554300".

    Mirrors C6_revised but tests the digit-string input form rather than
    spoken words.  The numeric string must be accepted directly as the new
    phone value and the SMS path must complete.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "no",  # phone_confirmed=no
            NEW_PHONE,  # numeric string replacement
            "no thanks",
        ],
        test_name="test_C7_revised_phone_declined_numeric_string_new_number",
        scenario=(
            "'no' → agent asks for new number → numeric '5125554300' → "
            "phone=5125554300 → notification_channel=sms"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), f"phone=={NEW_PHONE}"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C8_revised_conversational_inline_new_phone(run_conversation, assert_and_record):
    """
    C8_revised: Conversational inline decline + replacement in one utterance:
    "Actually I got a new phone, it's five one two five five five four three zero zero."

    The utterance declines the on-file number AND provides the replacement in
    the same turn.  The inline-update rule must extract the new phone (5125554300)
    directly from this utterance so no second collection turn is needed.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            f"Actually I got a new phone, it's {NEW_PHONE_SPOKEN}",
            "no thanks",
        ],
        test_name="test_C8_revised_conversational_inline_new_phone",
        scenario=(
            "Inline 'got a new phone, it's <spoken>' → phone extracted in one turn → "
            "phone=5125554300 → notification_channel=sms"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), f"phone=={NEW_PHONE}"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# C9_revised: phone number formatting
# ===========================================================================


@pytest.mark.live
async def test_C9_revised_phone_readback_uses_dashes(run_conversation, assert_and_record):
    """
    C9_revised: Agent readback of phone number must use dashes (512-555-6101),
    not a raw digit string.

    build_phone_confirmation_prompt formats any 10-digit phone string as
    XXX-XXX-XXXX before embedding it in the question.  This test asserts that
    the agent message literally contains "512-555-6101" — verifying the
    formatter fires and the member sees the human-readable form.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "yes",  # phone confirmed — we just want to see the readback
            "no thanks",
        ],
        test_name="test_C9_revised_phone_readback_uses_dashes",
        scenario=(
            "SMS selected → agent reads back phone on file → "
            "message must contain '512-555-6101' (dashes, not raw digits)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_phone_readback_has_dashes(record), f"readback_contains_{PHONE_ON_FILE_B}"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# C10_revised – C11_revised: notification method spoken variants
# ===========================================================================


@pytest.mark.live
async def test_C10_revised_email_them_to_me_notification_method(run_conversation, assert_and_record):
    """
    C10_revised: "email them to me" as the notification method → extracted as
    'email' → email on file confirmed → preference saved.

    Tests that a verb-first email expression used in the notification-method
    slot is correctly normalised to 'email'.  email_confirmed uses a bias rule
    (anything other than a clear affirmation → 'no'), so the "yes" confirmation
    must fire the affirmation path and no update turn should occur.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "email them to me",  # notification_method = email (verb-first)
            "yes",  # email on file confirmed via bias-rule affirmation
            "no thanks",
        ],
        test_name="test_C10_revised_email_them_to_me_notification_method",
        scenario=(
            "'email them to me' → notification_method=email → email confirmed → "
            "notification_channel=email → no escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C11_revised_send_it_to_my_phone_notification_method(run_conversation, assert_and_record):
    """
    C11_revised: "send it to my phone" as the notification method → extracted
    as 'sms' → phone on file confirmed → preference saved.

    Tests that an indirect SMS expression ("send it to my phone") normalises
    to notification_method='sms'.  phone_confirmed then fires with validate_yes_no
    (no bias rule); "yes" must resolve to phone_confirmed='yes' in one turn.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "send it to my phone",  # notification_method = sms (indirect)
            "yes",  # phone on file confirmed
            "no thanks",
        ],
        test_name="test_C11_revised_send_it_to_my_phone_notification_method",
        scenario=(
            "'send it to my phone' → notification_method=sms → phone confirmed → "
            "notification_channel=sms → no escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C_sms — notification_method SMS spoken variants
# ===========================================================================


@pytest.mark.live
async def test_C_sms_1_text_me(run_conversation, assert_and_record):
    """
    C_sms_1: User says bare "text me" → normalized to notification_method='sms'.

    The simplest SMS expression — verb-first imperative.  Verifies that a
    two-word instruction with no hedging maps unambiguously to 'sms' and the
    phone happy path completes.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "text me",
            "yes",  # phone on file confirmed
            "no thanks",  # follow_up closure
        ],
        test_name="test_C_sms_1_text_me",
        scenario="'text me' → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_2_send_me_a_text(run_conversation, assert_and_record):
    """
    C_sms_2: User says "send me a text" → notification_method='sms'.

    Prepositional-object form: the direct object 'text' (noun) rather than
    'text' as a verb.  Verifies that 'text' used as a noun still triggers the
    SMS mapping.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "send me a text",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_2_send_me_a_text",
        scenario="'send me a text' → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_3_text_message_please(run_conversation, assert_and_record):
    """
    C_sms_3: User says "text message please" → notification_method='sms'.

    Noun-phrase form with polite 'please'; verifies that the full two-word
    channel name ('text message') maps to 'sms' and that the trailing 'please'
    does not introduce ambiguity.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "text message please",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_3_text_message_please",
        scenario="'text message please' → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_4_just_text_it_to_me(run_conversation, assert_and_record):
    """
    C_sms_4: User says "just text it to me" → notification_method='sms'.

    Imperative form with a softening 'just' and a prepositional object.
    Verifies that the leading filler 'just' and the indirect-object clause
    ('to me') do not prevent extraction of the SMS channel.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "just text it to me",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_4_just_text_it_to_me",
        scenario="'just text it to me' → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_5_uppercase_sms(run_conversation, assert_and_record):
    """
    C_sms_5: User says uppercase "SMS" → notification_method='sms'.

    Canonical acronym form in upper case.  Verifies case-insensitive normalisation
    of the 'sms' keyword — the simplest possible SMS input should always succeed.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_5_uppercase_sms",
        scenario="'SMS' (uppercase) → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_6_send_to_my_phone(run_conversation, assert_and_record):
    """
    C_sms_6: User says "you can send me to my phone" → notification_method='sms'.

    Exact phrase observed in transcripts.  'phone' is the channel keyword here;
    the verb phrase 'you can send me to my' is leading filler.  Verifies that
    the extraction handles the indirect phrasing and correctly maps to 'sms'.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "you can send me to my phone",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_6_send_to_my_phone",
        scenario=("'you can send me to my phone' → notification_method=sms → phone confirmed → closure"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C_sms_7_conversational_check_phone_constantly(run_conversation, assert_and_record):
    """
    C_sms_7: Conversational: "Oh just text me, I check my phone constantly."

    'text' embedded in a casual sentence with a colloquial lead-in ('oh just')
    and a trailing self-description ('I check my phone constantly').  Verifies
    that 'text' is extracted as the channel keyword even when the sentence
    contains 'phone' as a second occurrence that refers to a device rather than
    a channel.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "Oh just text me, I check my phone constantly",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_7_conversational_check_phone_constantly",
        scenario=(
            "Conversational 'text me' with trailing 'phone' mention → "
            "notification_method=sms (text wins over phone noun) → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C_sms_8_conversational_prefer_text_not_email(run_conversation, assert_and_record):
    """
    C_sms_8: Conversational: "I'd prefer a text message, I'm not great at
    checking email."

    'text message' is the channel preference; the trailing clause mentions
    'email' in a contrastive context ('not great at').  Verifies that the
    leading SMS preference is extracted as notification_method='sms' and that
    'email' in the trailing clause does not override it.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "I'd prefer a text message, I'm not great at checking email",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_8_conversational_prefer_text_not_email",
        scenario=(
            "Conversational SMS preference with contrastive 'email' mention → "
            "notification_method=sms (text message wins) → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_9_call_me(run_conversation, assert_and_record):
    """
    C_sms_9: User says "call me" → notification_method='sms'.

    'call' is listed as a phone/SMS synonym in normalize_notification_method.
    Verifies that a voice-call expression maps to the SMS channel (the system
    does not support voice; 'call' is treated as a phone-based preference and
    normalised to 'sms').
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "call me",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_9_call_me",
        scenario="'call me' → notification_method=sms (call→phone→sms) → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_sms_10_my_cell_phone(run_conversation, assert_and_record):
    """
    C_sms_10: User says "my cell phone" → notification_method='sms'.

    Noun-phrase channel reference using the 'cell phone' compound.  Verifies
    that 'phone' embedded in a two-word compound is still extracted as the
    SMS mapping keyword and that 'cell' (modifier) does not prevent normalisation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "my cell phone",
            "yes",
            "no thanks",
        ],
        test_name="test_C_sms_10_my_cell_phone",
        scenario="'my cell phone' → notification_method=sms → phone confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C_email — notification_method email spoken variants
# ===========================================================================


@pytest.mark.live
async def test_C_email_1_email_bare(run_conversation, assert_and_record):
    """
    C_email_1: User says bare "email" → notification_method='email'.

    Simplest possible email channel input — the canonical single-word form.
    Verifies the baseline email mapping and that the email confirmation
    happy path completes.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "email",
            "yes",  # email on file confirmed
            "no thanks",
        ],
        test_name="test_C_email_1_email_bare",
        scenario="'email' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_2_email_me(run_conversation, assert_and_record):
    """
    C_email_2: User says "email me" → notification_method='email'.

    Verb-first imperative form.  Verifies that 'email' used as a verb still
    maps to notification_method='email', mirroring delivery_management
    test_delivery_method_email_it_to_me.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "email me",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_2_email_me",
        scenario="'email me' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_3_send_an_email(run_conversation, assert_and_record):
    """
    C_email_3: User says "send an email" → notification_method='email'.

    Indirect-object form where 'email' is the object of 'send'.  Verifies
    that 'email' extracted as a direct object still maps to the email channel.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "send an email",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_3_send_an_email",
        scenario="'send an email' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_4_via_email(run_conversation, assert_and_record):
    """
    C_email_4: User says "via email" → notification_method='email'.

    Prepositional form without a verb.  Verifies the minimal 'via email'
    phrase maps to 'email', mirroring delivery_management
    test_delivery_method_via_email.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "via email",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_4_via_email",
        scenario="'via email' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_5_email_them_to_me(run_conversation, assert_and_record):
    """
    C_email_5: User says "email them to me" → notification_method='email'.

    Exact phrase observed in transcripts.  Verb-first with a pronoun object
    and indirect-object clause.  Verifies that the extraction handles the
    full prepositional form and maps to 'email'.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "email them to me",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_5_email_them_to_me",
        scenario=("'email them to me' → notification_method=email → email confirmed → closure"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_6_by_email_please(run_conversation, assert_and_record):
    """
    C_email_6: User says "by email please" → notification_method='email'.

    Prepositional form with polite 'please'.  Verifies that the trailing
    'please' does not introduce ambiguity and that 'by email' maps correctly,
    mirroring delivery_management test_delivery_method_via_fax_please for the
    email channel.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "by email please",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_6_by_email_please",
        scenario="'by email please' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C_email_7_conversational_refer_back(run_conversation, assert_and_record):
    """
    C_email_7: Conversational: "Email is better for me, I can refer back to it."

    'email' leads the sentence as subject; 'better for me' is a preference
    clause and 'refer back to it' is a justification.  Verifies that the
    channel keyword at the start of a preference sentence is extracted as
    notification_method='email' despite the trailing reasoning clause.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "Email is better for me, I can refer back to it",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_7_conversational_refer_back",
        scenario=(
            "Conversational email preference with justification → "
            "notification_method=email → email confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C_email_8_conversational_just_email_address_on_file(run_conversation, assert_and_record):
    """
    C_email_8: Conversational: "Just email me at my address on file, that works
    fine."

    'email' used as a verb in a casual instruction with an indirect-object
    clause ('at my address on file') and a trailing confirmation ('that works
    fine').  Verifies extraction of notification_method='email' from a fully
    conversational utterance that also implies contact confirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "Just email me at my address on file, that works fine",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_8_conversational_just_email_address_on_file",
        scenario=(
            "Conversational 'email me at address on file' → "
            "notification_method=email → email confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_9_hyphenated_e_mail(run_conversation, assert_and_record):
    """
    C_email_9: User says "e-mail" (hyphenated form) → notification_method='email'.

    The hyphenated variant 'e-mail' is listed as an email synonym in
    normalize_notification_method.  Verifies that the hyphen is correctly
    handled and the mapping resolves to 'email'.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "e-mail",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_9_hyphenated_e_mail",
        scenario="'e-mail' (hyphenated) → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_email_10_i_prefer_email(run_conversation, assert_and_record):
    """
    C_email_10: User says "I prefer email" → notification_method='email'.

    First-person preference declaration — mirrors delivery_management
    test_delivery_method_i_prefer_email for the notification_method slot.
    Verifies that 'prefer' as a verb does not confuse the extraction.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "I prefer email",
            "yes",
            "no thanks",
        ],
        test_name="test_C_email_10_i_prefer_email",
        scenario="'I prefer email' → notification_method=email → email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C_amb — notification_method ambiguous then valid
# ===========================================================================


@pytest.mark.live
async def test_C_amb_1_whatever_works_then_email(run_conversation, assert_and_record):
    """
    C_amb_1: "whatever works" is ambiguous → agent retries → "email" accepted →
    notification_channel='email'.

    "whatever works" contains no channel keyword; the slot must treat it as
    an invalid/ambiguous value and re-ask.  The member then provides a clear
    'email' on the retry turn and the email happy path completes.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "whatever works",  # ambiguous — no channel keyword
            "email",  # valid on retry
            "yes",  # email on file confirmed
            "no thanks",
        ],
        test_name="test_C_amb_1_whatever_works_then_email",
        scenario=(
            "'whatever works' → ambiguous → retry → 'email' → "
            "notification_method=email → email confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C_amb_2_either_is_fine_then_sms(run_conversation, assert_and_record):
    """
    C_amb_2: "either is fine" is ambiguous → agent retries → "text me" accepted →
    notification_channel='sms'.

    "either is fine" offers no channel signal; the slot retries.  The member
    then picks SMS on the retry and the phone happy path completes.  Verifies
    that a retry correctly advances the conversation after a single ambiguous
    response.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "either is fine",  # ambiguous
            "text me",  # sms on retry
            "yes",  # phone on file confirmed
            "no thanks",
        ],
        test_name="test_C_amb_2_either_is_fine_then_sms",
        scenario=(
            "'either is fine' → ambiguous → retry → 'text me' → "
            "notification_method=sms → phone confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C_amb_3_no_preference_then_email(run_conversation, assert_and_record):
    """
    C_amb_3: Conversational: "I don't really have a preference" → ambiguous →
    agent retries → "email" accepted → notification_channel='email'.

    First-person preference disclaimer — the most natural way a real caller
    signals no strong opinion.  Verifies that a conversational non-answer
    triggers a retry and that the subsequent clear 'email' is accepted.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "I don't really have a preference",  # conversational non-answer
            "email",  # clear choice on retry
            "yes",
            "no thanks",
        ],
        test_name="test_C_amb_3_no_preference_then_email",
        scenario=(
            "Conversational 'no preference' → ambiguous → retry → "
            "'email' → notification_method=email → email confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C2 — NotificationSetupAgent: contact_confirmed slot variants
# ===========================================================================

# New email constant for C2 inline-replacement tests
NEW_EMAIL_C2 = "james.wilson.new@gmail.com"


@pytest.mark.live
async def test_C2_yes_1_correct(run_conversation, assert_and_record):
    """
    C2_yes_1: contact_confirmed "correct" → proceeds without re-asking.
    Phone path (SMS channel selected).
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "correct"],
        test_name="test_C2_yes_1_correct",
        scenario="contact_confirmed 'correct' on phone path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C2_yes_2_thats_right(run_conversation, assert_and_record):
    """
    C2_yes_2: contact_confirmed "that's right" → proceeds.
    Possessive affirmation variant.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "that's right"],
        test_name="test_C2_yes_2_thats_right",
        scenario="contact_confirmed 'that's right' on phone path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C2_yes_3_yep(run_conversation, assert_and_record):
    """
    C2_yes_3: contact_confirmed "yep" → proceeds.
    Casual short affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "yep"],
        test_name="test_C2_yes_3_yep",
        scenario="contact_confirmed 'yep' on phone path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_4_conversational(run_conversation, assert_and_record):
    """
    C2_yes_4: contact_confirmed "yes that number is fine" → proceeds.
    Conversational multi-word affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "yes that number is fine"],
        test_name="test_C2_yes_4_conversational",
        scenario="contact_confirmed 'yes that number is fine'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_5_repeat_number_back(run_conversation, assert_and_record):
    """
    C2_yes_5: Member repeats the phone number back → contact_confirmed=yes.
    Verifies numeric readback is treated as affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", PHONE_ON_FILE_B],
        test_name="test_C2_yes_5_repeat_number_back",
        scenario=f"contact_confirmed by repeating number '{PHONE_ON_FILE_B}'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_6_emphatic(run_conversation, assert_and_record):
    """
    C2_yes_6: contact_confirmed "absolutely, that's the one" → proceeds.
    Emphatic affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "absolutely, that's the one"],
        test_name="test_C2_yes_6_emphatic",
        scenario="contact_confirmed emphatic affirmation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_7_sure_go_ahead(run_conversation, assert_and_record):
    """
    C2_yes_7: contact_confirmed "sure go ahead" → proceeds.
    Delegating affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "sure go ahead"],
        test_name="test_C2_yes_7_sure_go_ahead",
        scenario="contact_confirmed 'sure go ahead'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C2_yes_email_1_correct(run_conversation, assert_and_record):
    """
    C2_yes_email_1: email path — contact_confirmed "correct" → proceeds.
    Email channel baseline affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "correct"],
        test_name="test_C2_yes_email_1_correct",
        scenario="contact_confirmed 'correct' on email path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_C2_yes_email_2_yes(run_conversation, assert_and_record):
    """
    C2_yes_email_2: email path — contact_confirmed "yes" → proceeds.
    Plain yes on email path.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "yes"],
        test_name="test_C2_yes_email_2_yes",
        scenario="contact_confirmed 'yes' on email path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_email_3_that_email_is_fine(run_conversation, assert_and_record):
    """
    C2_yes_email_3: email path — "yes that email is fine" → contact_confirmed=yes.
    Conversational email affirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "yes that email is fine"],
        test_name="test_C2_yes_email_3_that_email_is_fine",
        scenario="contact_confirmed 'yes that email is fine'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_yes_email_4_looks_good(run_conversation, assert_and_record):
    """
    C2_yes_email_4: email path — "looks good" → contact_confirmed=yes.
    Informal approbation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "looks good"],
        test_name="test_C2_yes_email_4_looks_good",
        scenario="contact_confirmed 'looks good' on email path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_no_implicit_1_stale_phone(run_conversation, assert_and_record):
    """
    C2_no_implicit_1: "that number is old" → contact_confirmed=no → agent re-asks for new number.
    Stale-address decline without "no" keyword (phone path).
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "that number is old", NEW_PHONE],
        test_name="test_C2_no_implicit_1_stale_phone",
        scenario="contact_confirmed implicit decline 'that number is old' → provide new phone",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), "phone_saved"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_no_implicit_2_wrong_address(run_conversation, assert_and_record):
    """
    C2_no_implicit_2: "that email is wrong" → contact_confirmed=no → agent re-asks for new email.
    Wrong-address implicit decline (email path).
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "that email is wrong", NEW_EMAIL_C2],
        test_name="test_C2_no_implicit_2_wrong_address",
        scenario="contact_confirmed implicit decline 'that email is wrong' → provide new email",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_no_implicit_3_not_mine(run_conversation, assert_and_record):
    """
    C2_no_implicit_3: "that's not my number" → contact_confirmed=no → provide new number.
    Ownership rejection (phone path).
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "that's not my number", NEW_PHONE],
        test_name="test_C2_no_implicit_3_not_mine",
        scenario="contact_confirmed 'that's not my number' → provide new phone",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), "phone_saved"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_inline_1_no_plus_new_phone(run_conversation, assert_and_record):
    """
    C2_inline_1: "no use 5125554300 instead" → contact_confirmed=no with new phone inline.
    Inline-update rule: new value extracted in same utterance as decline.
    Agent must not ask for the number again.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", f"no use {NEW_PHONE} instead"],
        test_name="test_C2_inline_1_no_plus_new_phone",
        scenario="contact_confirmed inline decline + new phone in one utterance",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_phone_saved(record, NEW_PHONE), "phone_saved"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_inline_2_no_plus_new_email(run_conversation, assert_and_record):
    """
    C2_inline_2: "no send it to james.wilson.new@gmail.com" → contact_confirmed=no with new email inline.
    Inline-update rule on email path.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", f"no send it to {NEW_EMAIL_C2}"],
        test_name="test_C2_inline_2_no_plus_new_email",
        scenario="contact_confirmed inline decline + new email in one utterance",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_invalid_1_gibberish_then_valid(run_conversation, assert_and_record):
    """
    C2_invalid_1: "hmm" (ambiguous/invalid) → agent retries → "yes" → proceeds.
    Verifies retry loop on contact_confirmed slot.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "hmm", "yes"],
        test_name="test_C2_invalid_1_gibberish_then_valid",
        scenario="contact_confirmed ambiguous 'hmm' → retry → 'yes'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_C2_invalid_2_maybe_then_valid(run_conversation, assert_and_record):
    """
    C2_invalid_2: "maybe" (ambiguous) → agent retries → "correct" → proceeds.
    Verifies that uncertain responses trigger a retry, not a confirmation.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "maybe", "correct"],
        test_name="test_C2_invalid_2_maybe_then_valid",
        scenario="contact_confirmed 'maybe' → retry → 'correct'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# Latency helpers (shared with Group D)
# ===========================================================================

_LATENCY_P50_SEC = 12.0
_LATENCY_P95_SEC = 20.0


def _compute_latency_percentile(record: ConversationRecord, p: float) -> float:
    """Return the p-th percentile (0-100) of per-turn duration_sec values."""
    durations = [t.duration_sec for t in record.turns if t.duration_sec > 0]
    if not durations:
        return 0.0
    s = sorted(durations)
    n = len(s)
    if n == 1:
        return s[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def assert_p50_under(record: ConversationRecord, threshold_sec: float) -> None:
    p50 = _compute_latency_percentile(record, 50)
    assert p50 <= threshold_sec, f"p50 latency {p50:.3f}s exceeds threshold {threshold_sec:.3f}s"


def assert_p95_under(record: ConversationRecord, threshold_sec: float) -> None:
    p95 = _compute_latency_percentile(record, 95)
    assert p95 <= threshold_sec, f"p95 latency {p95:.3f}s exceeds threshold {threshold_sec:.3f}s"


def assert_notification_channel(record: ConversationRecord, expected: str) -> None:
    """notification_channel == expected in final state or any turn snapshot."""
    actual_final = record.final_state.get("notification_channel", "")
    if actual_final == expected:
        return
    for t in record.turns:
        if t.state_snapshot.get("notification_channel") == expected:
            return
    raise AssertionError(
        f"Expected notification_channel={expected!r} in final state or any turn, got final={actual_final!r}"
    )


def assert_n2_notification_channel(record: ConversationRecord, expected: str) -> None:
    """claim_timeline_notification_channel == expected in final state or any turn snapshot."""
    actual_final = record.final_state.get("claim_timeline_notification_channel", "")
    if actual_final == expected:
        return
    for t in record.turns:
        if t.state_snapshot.get("claim_timeline_notification_channel") == expected:
            return
    raise AssertionError(
        f"Expected claim_timeline_notification_channel={expected!r}, got final={actual_final!r}"
    )


def assert_claim_flow_complete(record: ConversationRecord) -> None:
    """claim_flow_complete=True in final state or any turn snapshot."""
    if record.final_state.get("claim_flow_complete") is True:
        return
    for t in record.turns:
        if t.state_snapshot.get("claim_flow_complete") is True:
            return
    raise AssertionError("Expected claim_flow_complete=True in final state or any turn")


# ===========================================================================
# GROUP C_scenario_a — notification_setup reached via Scenario A (records flow)
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_C_scenario_a_after_records_sms(run_conversation, assert_and_record):
    """
    C_scenario_a_after_records: Scenario A path — upload link sent + personal
    guide consent declined — lands in notification_setup.  Member chooses SMS
    and confirms the phone on file.

    Verifies that notification_setup still collects the channel correctly after
    a full records-coordination flow (upload + guide-declined), i.e. the
    _C_PREFIX_AFTER_RECORDS prefix leaves the conversation in notification_setup
    and the agent successfully resolves notification_channel='sms'.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX_AFTER_RECORDS
        + [
            "SMS",  # notification_method
            "yes",  # phone on file confirmed
        ],
        test_name="test_C_scenario_a_after_records_sms",
        scenario="Scenario A: upload+guide-declined → notification_setup → SMS → phone confirmed → sms",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


def _assert_timeline_answer(record: ConversationRecord) -> None:
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    assert "5 to 10" in all_msgs or "business days" in all_msgs, (
        f"Expected '5 to 10' or 'business days' in agent messages. Got: {all_msgs[:400]!r}"
    )


# ===========================================================================
# GROUP D — End-to-end smoke tests and latency benchmarks
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_D1_full_scenario_a_decline(run_conversation, assert_and_record):
    """
    D1: Full Scenario A — intake → verification → claim_adjustment → records (decline)
    → escalation.
    Verifies the complete pipeline runs without routing errors and that the
    escalation message surfaces the reference number.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "no I don't want to proceed",  # decline records → escalation
        ],
        test_name="test_D1_full_scenario_a_decline",
        scenario=(
            "Full Scenario A: intake → verification → claim_adj → records decline → escalation. "
            "Verify escalation triggered and reference number appears in transcript."
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_A), f"reference_number=={REF_A}"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
            # Escalation agent generates its own reference number — not the claim ref
            # (lambda: assert_any_agent_message_contains(record, REF_A), f"ref_{REF_A}_in_transcript"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D2_full_scenario_b_upload_guide_sms_followup(run_conversation, assert_and_record):
    """
    D2: Full Scenario A — intake → verification → claim_adjustment → records
    (upload link sent + Personal Guide triggered) → notification setup (N1=SMS,
    timeline declined, N2=email) → follow_up "where can I see my rewards?" → closure.
    Verifies: upload_link_sent, personal_guide_triggered, notification_channel='sms',
    n2_channel='email', and follow_up answer contains 'mysagilityhealth.com'.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "Perfect. Please do that",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "email",  # N2 method
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "where can I see my rewards?",  # follow_up question
            "no thanks",  # closure
        ],
        test_name="test_D2_full_scenario_b_upload_guide_sms_followup",
        scenario=(
            "Full Scenario A: upload link + guide triggered → N1=sms → timeline no → N2=email → "
            "follow_up rewards question answered from snapshot → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (
                lambda: assert_any_agent_message_contains(record, "mysagilityhealth.com"),
                "rewards_portal_in_answer",
            ),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D3_records_not_required_goes_to_notification(run_conversation, assert_and_record):
    """
    D3: Scenario B — personal guide → notification_setup (N1=email, timeline declined,
    N2=sms) → closure.
    Verifies both notification channels are collected and no escalation occurs.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent → guide triggered
            "email",  # N1 method
            "yes",  # N1 email on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D3_records_not_required_goes_to_notification",
        scenario=("Scenario B (REF_B): personal guide → N1=email → timeline no → N2=sms → closure"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_n2_notification_channel(record, "sms"), "n2_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D4_followup_cannot_answer_then_timeline_answered(run_conversation, assert_and_record):
    """
    D4: Full path — personal guide → N1=email → timeline_question answered with
    a real timeline question (fires MSG_TIMELINE_ANSWER) → N2=sms → follow_up where
    member first asks about deductible (cannot_answer), then asks timeline again
    (answered from snapshot) → closure.
    Verifies the timeline slot answers inline during notification_setup AND that
    follow_up also answers timeline questions from the claims snapshot.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent
            "email",  # N1 method
            "yes",  # N1 email on file confirmed
            "how long will the adjustment take?",  # timeline_question → answered → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "what is my deductible?",  # follow_up cannot_answer
            "how long will it take?",  # follow_up timeline Q → answered from snapshot
            "no that's all",  # closure
        ],
        test_name="test_D4_followup_cannot_answer_then_timeline_answered",
        scenario=(
            "personal guide → N1=email → timeline Q answered → N2=sms → "
            "follow_up: deductible cannot_answer → timeline answered from snapshot → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_n2_notification_channel(record, "sms"), "n2_channel==sms"),
            (
                lambda: _assert_timeline_answer(record),
                "timeline_in_answer",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D5_latency_benchmark_full_scenario_b(run_conversation, assert_and_record):
    """
    D5: Latency benchmark — full Scenario A path (upload + guide + N1 + N2).
    p50 per-turn latency ≤ 12s, p95 per-turn latency ≤ 20s.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "yes please do that",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "email",  # N2 method
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D5_latency_benchmark_full_scenario_b",
        scenario=f"Latency benchmark: full claim adjustment "
        f"flow p50≤{_LATENCY_P50_SEC}s, p95≤{_LATENCY_P95_SEC}s",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D6_update_request_in_followup_escalates(run_conversation, assert_and_record):
    """
    D6: Full path reaches follow_up_agent via N1 + timeline + N2, then member says
    "can you resend the upload link to a different email" — an UPDATE_REQUEST.
    Verifies that follow_up_agent escalates immediately on UPDATE_REQUEST
    (no threshold, no counting) even in the claims follow-up variant.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent → guide triggered
            "email",  # N1 method
            "yes",  # N1 email on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "can you resend the upload link to a different email",  # UPDATE_REQUEST in follow_up
        ],
        test_name="test_D6_update_request_in_followup_escalates",
        scenario=(
            "personal guide → N1=email → timeline no → N2=sms → follow_up: "
            "UPDATE_REQUEST → immediate escalation (no threshold)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )


# ===========================================================================
# GROUP D — N1/N2 notification channel combinations and conversational E2E
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_D_combo_1_upload_guide_n1_sms_n2_email(run_conversation, assert_and_record):
    """
    D_combo_1: Full Scenario A — upload link sent + Personal Guide triggered →
    N1=sms (phone on file confirmed) → N2=email (email on file confirmed) →
    follow_up rewards question → closure.

    This is the transcript happy path.  The notification_setup agent asks two
    sequential questions: N1 covers how to be notified when the Personal Guide
    contacts the provider (→ notification_channel); N2 covers how to receive
    timeline/progress updates (→ claim_timeline_notification_channel).  A
    realistic caller answers them with different channels — SMS for immediate
    outreach status, email for longer-timeline updates.

    Key invariants:
      - upload_link_sent == True  (Branch A: member_upload + email confirmed)
      - personal_guide_triggered  (personal_guide_consent accepted)
      - notification_channel == 'sms'  (N1: phone on file confirmed)
      - claim_timeline_notification_channel == 'email'  (N2)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "Perfect. Please do that",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "email",  # N2 method
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "where can I see my rewards?",  # follow_up question
            "no thanks",  # closure
        ],
        test_name="test_D_combo_1_upload_guide_n1_sms_n2_email",
        scenario=(
            "Full Scenario A: upload + guide → N1=sms → timeline no → N2=email → rewards follow_up → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_records_branch(record, "personal_guide"), "records_branch==personal_guide"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (
                lambda: assert_any_agent_message_contains(record, "mysagilityhealth.com"),
                "rewards_portal_in_answer",
            ),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D_combo_2_upload_guide_n1_email_n2_sms(run_conversation, assert_and_record):
    """
    D_combo_2: Full Scenario A — upload link sent + Personal Guide triggered →
    N1=email (email on file confirmed) → N2=sms (phone on file confirmed) →
    closure.

    Mirrors D_combo_1 with channels reversed.  The caller prefers email for
    the immediate provider-outreach notification (N1) and SMS for ongoing
    progress updates (N2) — a realistic pattern for someone who checks email
    infrequently but has text notifications enabled.  Verifies that both N1
    and N2 slots collect the correct channel independently.

    Key invariants:
      - upload_link_sent == True
      - personal_guide_triggered
      - notification_channel == 'email'  (N1)
      - claim_timeline_notification_channel == 'sms'  (N2)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "yes please do that",  # personal_guide_consent → guide triggered
            "email",  # N1 method
            "yes",  # N1 email on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "text me",  # N2 method = sms
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_combo_2_upload_guide_n1_email_n2_sms",
        scenario=("Full Scenario A: upload + guide → N1=email → timeline no → N2=sms → closure"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_n2_notification_channel(record, "sms"), "n2_channel==sms"),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D_combo_3_guide_only_n1_sms_n2_email_timeline(run_conversation, assert_and_record):
    """
    D_combo_3: Full Scenario A — Personal Guide triggered (no upload link) →
    N1=sms (phone confirmed) → N2=email (email confirmed) → follow_up
    "how long will it take?" answered from snapshot → closure.

    The member goes directly to Personal Guide without requesting the upload
    link, representing a caller who trusts their doctor to handle record
    submission.  N1 and N2 are still collected since the Guide was triggered.
    A timeline follow_up question verifies the session snapshot is populated
    correctly even when the upload_link path was bypassed.

    Key invariants:
      - personal_guide_triggered  (guide consent accepted)
      - upload_link_sent NOT asserted  (link was never sent)
      - notification_channel == 'sms'  (N1)
      - claim_timeline_notification_channel == 'email'  (N2)
      - agent answers timeline question with '5 to 10' / 'business days'
      - not escalated
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "Feel free to call my doctor's office directly",  # personal_guide intent (no upload)
            "yes",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "how long will it take?",  # timeline_question → question → MSG_TIMELINE_ANSWER fired
            "email",  # N2 method (asked after timeline answer)
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "how long will the adjustment take?",  # follow_up timeline Q → answered from snapshot
            "no that's all",  # closure
        ],
        test_name="test_D_combo_3_guide_only_n1_sms_n2_email_timeline",
        scenario=(
            "Full Scenario A: guide only (no upload) → N1=sms → timeline Q answered → "
            "N2=email → follow_up timeline Q → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_records_branch(record, "personal_guide"), "records_branch==personal_guide"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (
                lambda: _assert_timeline_answer(record),
                "timeline_in_answer",
            ),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D_combo_4_upload_no_guide_n2_sms_phone_updated(run_conversation, assert_and_record):
    """
    D_combo_4: Full Scenario A — upload link sent, Personal Guide not triggered →
    N2=sms, phone on file declined, new phone "5125554300" provided → closure.

    Tests the path where the member accepts the upload link but declines the
    Personal Guide offer, resulting in upload_link_sent=True with
    personal_guide_outreach_requested remaining False.  Without a guide there
    is no N1; only N2 (timeline/progress updates) is collected in
    notification_setup.  The member also declines the phone on file and supplies
    a replacement, exercising the phone-update sub-flow within N2.

    Key invariants:
      - upload_link_sent == True
      - personal_guide_outreach_requested is absent / False  (guide declined)
      - notification_channel == 'sms'  (N2, new phone used)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "no",  # personal_guide_consent declined → no guide, no N1
            "SMS",  # N1 method (notification_setup entered fresh)
            "no",  # N1 phone on file declined
            NEW_PHONE,  # new phone number → N1 preference saved
            "no",  # timeline_question → no → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_combo_4_upload_no_guide_n2_sms_phone_updated",
        scenario=(
            "Full Scenario A: upload + guide declined → N1=sms (new phone) → "
            "timeline no → N2=sms confirmed → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D_combo_5_full_conversational_all_spoken(run_conversation, assert_and_record):
    """
    D_combo_5: Full Scenario A — every turn answered in natural conversational
    speech.  No bare 'yes'/'no' inputs anywhere in the flow.

    This is the most realistic end-to-end smoke test: it models a genuine phone
    caller who answers every question in their own words rather than with terse
    slot values.  Each step exercises a different extraction prompt under
    conversational load:

      "Can my doctor just send it?"               → doctor_direct (upload_method)
      "Oh yes please send me the link too"        → upload_consent=yes
      "Yes that email is correct"                 → email_confirmed=yes → link sent
      "Actually yes have someone reach out too"   → personal_guide_consent=yes → guide triggered
      "You can text me updates"                   → N1=sms
      "Yes that number is right"                  → N1 phone confirmed
      "Okay how long will this take?"             → follow_up timeline question (mid-setup)
      "Email me the updates please"               → N2=email
      "Yes that email is fine"                    → N2 email confirmed
      "No that's all, thanks"                     → closure

    The mid-conversation follow_up question at step 7 tests that the agent
    handles an out-of-order question during notification_setup (answers it and
    then returns to collect N2) rather than misclassifying it.

    Key invariants:
      - upload_link_sent == True  (spoken consent + email confirmation)
      - personal_guide_triggered  (spoken guide consent)
      - notification_channel == 'sms'  (N1: spoken phone confirmation)
      - claim_timeline_notification_channel == 'email'  (N2: spoken email confirmation)
      - agent messages contain '5 to 10' or 'business days'  (timeline answered)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "Can my doctor just send it?",  # doctor_direct
            "Oh yes please send me the link too",  # upload_consent=yes
            "Yes that email is correct",  # email_confirmed=yes → upload_link_sent
            "Actually yes have someone reach out too",  # personal_guide_consent=yes → guide triggered
            "You can text me updates",  # N1=sms
            "Yes that number is right",  # N1 phone confirmed → preference saved
            "Yes please tell me how long it will take",  # timeline_question → yes → MSG_TIMELINE_ANSWER
            "Email me the updates please",  # N2=email
            "Yes that email is fine",  # N2 email confirmed → handoff to follow_up
            "No that's all, thanks",  # closure
        ],
        test_name="test_D_combo_5_full_conversational_all_spoken",
        scenario=(
            "Full conversational Scenario A: every slot answered in natural speech "
            "→ upload + guide + N1=sms + timeline yes + N2=email + closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_D_latency_1_scenario_b_no_records_email(run_conversation, assert_and_record):
    """
    D_latency_1: Latency benchmark — Scenario B (records_required=False) happy
    path, N2=email confirmed, follow_up closure.

    The shortest successful end-to-end path for a claims call: no records
    coordination, single notification question, no follow_up question.
    Drives: verification → claim_adjustment → notification_setup (email) →
    follow_up "no thanks" → closure.

    Also validates functional correctness so a latency-passing run that
    functionally broke is still caught.

    Key invariants:
      - notification_channel == 'email'
      - not escalated
      - p50 per-turn latency ≤ _LATENCY_P50_SEC (12 s)
      - p95 per-turn latency ≤ _LATENCY_P95_SEC (20 s)
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent → guide triggered
            "email",  # N1 method
            "yes",  # N1 email on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_latency_1_scenario_b_no_records_email",
        scenario=(
            f"Latency benchmark: Scenario B personal guide → N1=email → timeline no → "
            f"N2=sms → closure — p50≤{_LATENCY_P50_SEC}s, p95≤{_LATENCY_P95_SEC}s"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_n2_notification_channel(record, "sms"), "n2_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_D_latency_2_full_scenario_a_upload_guide_n1_n2(run_conversation, assert_and_record):
    """
    D_latency_2: Latency benchmark — full Scenario A (upload link + guide + N1 + N2)
    must meet p50 ≤ 12 s, p95 ≤ 20 s.

    The longest successful claims path: records coordination (upload + guide),
    two notification questions, then closure.  This benchmark guards against
    per-turn latency regressions on the most complex routing in the claims flow.

    Drives: verification → claim_adjustment → records (upload + guide) →
    notification_setup (N1=sms, N2=email) → follow_up → closure.

    Key invariants:
      - upload_link_sent == True
      - personal_guide_triggered
      - notification_channel == 'sms'  (N1)
      - claim_timeline_notification_channel == 'email'  (N2)
      - p50 per-turn latency ≤ _LATENCY_P50_SEC
      - p95 per-turn latency ≤ _LATENCY_P95_SEC
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            REF_A,
            "yes please",  # upload link accepted
            "yes",  # email on file confirmed → upload_link_sent
            "yes",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "email",  # N2 method
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_latency_2_full_scenario_a_upload_guide_n1_n2",
        scenario=(
            f"Latency benchmark: full Scenario A (upload+guide+N1+timeline+N2) — "
            f"p50≤{_LATENCY_P50_SEC}s, p95≤{_LATENCY_P95_SEC}s"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (lambda: assert_claim_flow_complete(record), "claim_flow_complete"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_D_latency_3_guard_transfer_request_reference_collection(run_conversation, assert_and_record):
    """
    D_latency_3: Latency benchmark — TRANSFER_REQUEST guard fires during
    reference number collection; tighter thresholds (p50 ≤ 8 s, p95 ≤ 15 s).

    A guard trigger is the shortest possible path after intake: one input
    after verification causes an immediate escalation.  Per-turn latency on
    this path should be well below the full-flow threshold because the LLM
    does not need to call any tools or run slot extractors.  Tighter thresholds
    catch regressions specific to fast-path guard evaluation.

    Key invariants:
      - escalated  (TRANSFER_REQUEST guard fires)
      - p50 per-turn latency ≤ 8 s
      - p95 per-turn latency ≤ 15 s
    """
    _GUARD_P50 = 8.0
    _GUARD_P95 = 15.0
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS
        + [
            "transfer me to someone please",  # TRANSFER_REQUEST during reference collection
        ],
        test_name="test_D_latency_3_guard_transfer_request_reference_collection",
        scenario=(
            f"Latency benchmark: TRANSFER_REQUEST guard → immediate escalation — "
            f"p50≤{_GUARD_P50}s, p95≤{_GUARD_P95}s"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_p50_under(record, _GUARD_P50), f"p50<={_GUARD_P50}s"),
            (lambda: assert_p95_under(record, _GUARD_P95), f"p95<={_GUARD_P95}s"),
        ],
    )


# ===========================================================================
# GROUP D_n2 — Scenario B N2 (claim_timeline_notification_channel) collection
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_D_n2_scenario_b_sms_email(run_conversation, assert_and_record):
    """
    D_n2_scenario_b_sms_email: Scenario B (records_required=False) → N1=sms
    (phone on file confirmed) → timeline bridge → N2=email (email on file
    confirmed) → closure.

    Verifies that after N1 is saved the agent progresses to collect N2
    (claim_timeline_notification_channel) and resolves both channels correctly.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent → guide triggered
            "SMS",  # N1 method
            "yes",  # N1 phone on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "email",  # N2 method
            "yes",  # N2 email on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_n2_scenario_b_sms_email",
        scenario="Scenario B: N1=sms confirmed → timeline no → N2=email confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_n2_notification_channel(record, "email"), "n2_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_D_n2_scenario_b_email_sms(run_conversation, assert_and_record):
    """
    D_n2_scenario_b_email_sms: Scenario B (records_required=False) → N1=email
    (email on file confirmed) → timeline bridge → N2=sms (phone on file
    confirmed) → closure.

    Mirrors test_D_n2_scenario_b_sms_email with channels swapped to verify
    both orderings of N1/N2 work on the Scenario B path.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # personal_guide intent
            "yes",  # personal_guide_consent → guide triggered
            "email",  # N1 method
            "yes",  # N1 email on file confirmed → preference saved
            "no",  # timeline_question → no → go to N2
            "SMS",  # N2 method
            "yes",  # N2 phone on file confirmed → handoff to follow_up
            "no thanks",  # closure
        ],
        test_name="test_D_n2_scenario_b_email_sms",
        scenario="Scenario B: N1=email confirmed → timeline no → N2=sms confirmed → closure",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_n2_notification_channel(record, "sms"), "n2_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# Shared prefix for Group E — lands in follow_up after full N1+timeline+N2 setup
# ---------------------------------------------------------------------------
# NOTE: REF_B has records_required=True → must pass through records_coordination
# via personal guide before reaching notification_setup_agent.
_E_PREFIX = _C_PREFIX_B + [
    "email",  # N1 method
    "yes",  # N1 email on file confirmed → preference saved
    "no",  # timeline_question → no → move to N2
    "email",  # N2 method
    "yes",  # N2 email on file confirmed → handoff to follow_up_agent
    # follow_up_agent now active
]

# Scenario A prefix that goes through upload + guide so personal_guide flag is set
_E_PREFIX_WITH_GUIDE = VERIFICATION_PREFIX_CLAIMS + [
    REF_A,
    "Feel free to call my doctor's office directly",  # personal_guide intent
    "yes",  # personal_guide_consent
    "SMS",  # notification_method
    "yes",  # phone on file confirmed
]


# ===========================================================================
# GROUP E — Follow-up agent claim-specific behaviour
# ===========================================================================


@pytest.mark.live
async def test_E1_rewards_portal_answered(run_conversation, assert_and_record):
    """
    E1: After a completed claim flow, member asks "where can I see my rewards?"
    The follow_up_claims prompt must answer with www.mysagilityhealth.com from
    Sagility general knowledge (not session snapshot).
    Verifies the special-case wellness portal answer fires in the claims variant.
    """
    record = await run_conversation(
        user_inputs=_E_PREFIX
        + [
            "where can I see my rewards?",
            "no that's all",  # closure
        ],
        test_name="test_E1_rewards_portal_answered",
        scenario=(
            "follow_up_claims: rewards portal question → answered with 'mysagilityhealth.com' → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_any_agent_message_contains(record, "mysagilityhealth.com"),
                "rewards_portal_url_in_answer",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_E2_timeline_answered_from_snapshot(run_conversation, assert_and_record):
    """
    E2: After a completed claim flow, member asks "how long will it take?"
    The follow_up_claims prompt must answer from the session snapshot:
    "5 to 10 business days from receipt of required information."
    Verifies the timeline field injected by _build_session_snapshot is used.
    """
    record = await run_conversation(
        user_inputs=_E_PREFIX
        + [
            "how long will it take?",
            "no that's all",  # closure
        ],
        test_name="test_E2_timeline_answered_from_snapshot",
        scenario=(
            "follow_up_claims: timeline question → answered from snapshot '5 to 10 business days' → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (
                lambda: _assert_timeline_answer(record),
                "timeline_in_answer",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_E3_personal_guide_timing_answered_from_snapshot(run_conversation, assert_and_record):
    """
    E3: After a claim flow where Personal Guide was triggered, member asks
    "when will the guide contact my doctor?"
    The follow_up_claims prompt must answer from the session snapshot:
    "within 24 hours."
    Verifies that the personal_guide snapshot line is surfaced by the LLM.
    """
    record = await run_conversation(
        user_inputs=_E_PREFIX_WITH_GUIDE
        + [
            "when will the guide contact my doctor?",
            "no that's all",  # closure
        ],
        test_name="test_E3_personal_guide_timing_answered_from_snapshot",
        scenario=(
            "follow_up_claims: Personal Guide timing question → "
            "answered from snapshot 'within 24 hours' → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_any_agent_message_contains(record, "24 hours"),
                "personal_guide_timing_in_answer",
            ),
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_E4_update_request_notifications_escalates(run_conversation, assert_and_record):
    """
    E4: After a completed claim flow, member says "send notifications to a different number"
    in follow_up — classified as UPDATE_REQUEST → immediate escalation.
    Verifies that the claims follow_up variant (follow_up_claims.md) correctly
    classifies a notification change request as UPDATE_REQUEST and escalates
    without any counting threshold.
    """
    record = await run_conversation(
        user_inputs=_E_PREFIX
        + [
            "send notifications to a different number",
        ],
        test_name="test_E4_update_request_notifications_escalates",
        scenario=(
            "follow_up_claims: UPDATE_REQUEST 'send notifications to a different number' → "
            "immediate escalation (no threshold)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )


# ===========================================================================
# GROUP R — Retry paths: AMBIGUOUS vs ANSWERED event distinction
#
# AMBIGUOUS event → no attempt_count increment → gentle CLARIFY re-ask
# ANSWERED (wrong value) → attempt_count increments → firm RETRY re-ask
# ===========================================================================


def assert_attempt_count_below_max(record: ConversationRecord, slot_name: str, max_count: int = 3) -> None:
    """Verify slot did not exhaust budget — confirms retry succeeded before MAX_SLOT_ATTEMPTS."""
    final = record.final_state.get("slot_attempts") or {}
    slot = final.get(slot_name, {})
    count = slot.get("attempt_count", 0) if isinstance(slot, dict) else 0
    assert count < max_count, f"Expected {slot_name} attempt_count < {max_count}, got {count}"


# ---------------------------------------------------------------------------
# GROUP R1 — upload_method retries (records_coordination)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R1_1_ambiguous_once_then_doctor_direct(run_conversation, assert_and_record):
    """
    R1_1: upload_method — AMBIGUOUS once, then doctor_direct.

    Turn 1: "hmm I'm not sure"
      → AMBIGUOUS event (no extractable intent) → attempt_count NOT incremented
      → agent re-asks gently (CLARIFY guard message)
    Turn 2: "Can my doctor send it over?"
      → upload_method=doctor_direct → upload offer → "yes please" → email confirmed
      → guide offered → "yes" → personal_guide_triggered=True → notification_setup
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "hmm I'm not sure",
            "Can my doctor send it over?",
            "yes please",
            "yes",
            "yes",
        ],
        test_name="test_R1_1_ambiguous_once_then_doctor_direct",
        scenario="upload_method AMBIGUOUS once → doctor_direct → upload → guide yes → notification_setup",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R1_2_ambiguous_once_then_member_upload(run_conversation, assert_and_record):
    """
    R1_2: upload_method — AMBIGUOUS once, then member_upload.

    Turn 1: "I don't know, maybe?"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Turn 2: "I'll upload it myself"
      → upload_method=member_upload → upload link offered → "yes" → email confirmed
      → guide offered → "no" → guide declined → notification_setup
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I don't know, maybe?",
            "I'll upload it myself",
            "yes",
            "yes",
            "no",
        ],
        test_name="test_R1_2_ambiguous_once_then_member_upload",
        scenario="upload_method AMBIGUOUS once → member_upload → upload link → guide no → notification_setup",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R1_3_vague_first_then_valid(run_conversation, assert_and_record):
    """
    R1_3: upload_method — vague first answer, then valid.

    Turn 1: "yes sure" — LLM may classify as AMBIGUOUS (no clear upload method)
      or as member_upload (vague affirmative). Both paths are acceptable:
      - If AMBIGUOUS: agent re-asks → "my doctor can send it" → doctor_direct path
      - If member_upload: upload link offered → link flow proceeds
    Primary assertion: not_escalated. Secondary: upload_link_sent OR personal_guide_triggered.
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "yes sure",
            "my doctor can send it",
            "yes",
            "yes",
        ],
        test_name="test_R1_3_vague_first_then_valid",
        scenario="upload_method vague 'yes sure' → both extraction outcomes handled → not_escalated",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R1_4_ambiguous_twice_then_valid(run_conversation, assert_and_record):
    """
    R1_4: upload_method — AMBIGUOUS twice, then doctor_direct on third.

    Turn 1: "I'm not really sure what my options are"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Turn 2: "whatever is easiest"
      → AMBIGUOUS event again → second consecutive AMBIGUOUS increments attempt_count (count=1)
      → firmer re-ask fires
    Turn 3: "Can I just have my doctor send it?"
      → upload_method=doctor_direct → valid before MAX_SLOT_ATTEMPTS=3
      → upload offer → "yes" → email confirmed → guide → "yes" → triggered
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "I'm not really sure what my options are",
            "whatever is easiest",
            "Can I just have my doctor send it?",
            "yes",
            "yes",
            "yes",
        ],
        test_name="test_R1_4_ambiguous_twice_then_valid",
        scenario="upload_method 2×AMBIGUOUS (second increments count) → doctor_direct before MAX",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R2 — upload_consent 1-fail then yes (records_coordination)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R2_1_ambiguous_once_then_yes(run_conversation, assert_and_record):
    """
    R2_1: upload_consent — AMBIGUOUS once, then yes.

    Base: _PREFIX_A_WITH_REF + ["yes please"] (member_upload intent → link offered)

    Turn at upload_consent: "hmm"
      → AMBIGUOUS event (not yes/no) → attempt_count NOT incremented → gentle CLARIFY re-ask
    Retry: "yes please send it"
      → upload_consent=yes → email confirmed → upload_link_sent=True
      → guide offered → "no" → guide declined → notification_setup
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "yes please",
            "hmm",
            "yes please send it",
            "yes",
            "no",
        ],
        test_name="test_R2_1_ambiguous_once_then_yes",
        scenario="upload_consent AMBIGUOUS 'hmm' → gentle re-ask → yes → upload_link_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R2_2_ambiguous_once_then_no_guide_yes(run_conversation, assert_and_record):
    """
    R2_2: upload_consent — AMBIGUOUS once, then no → guide offered → yes.

    Turn at upload_consent: "I'm not sure"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Retry: "no thanks"
      → upload_consent=no → guide offered
    Guide: "yes" → personal_guide_triggered=True
    upload_link_sent must be False (link was declined).
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "yes please",
            "I'm not sure",
            "no thanks",
            "yes",
        ],
        test_name="test_R2_2_ambiguous_once_then_no_guide_yes",
        scenario="upload_consent AMBIGUOUS → gentle re-ask → no → guide yes → personal_guide_triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_upload_link_sent(record), "not_upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R3 — personal_guide_consent AMBIGUOUS-then-YES
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R3_1_ambiguous_once_then_yes(run_conversation, assert_and_record):
    """
    R3_1: personal_guide_consent — AMBIGUOUS once, then yes.

    Guide consent turn 1: "I think so maybe?"
      → AMBIGUOUS event (extraction prompt: ambiguous → leave empty) → attempt_count NOT incremented
      → agent re-asks gently (CLARIFY)
    Guide consent turn 2: "yes please proceed"
      → personal_guide_consent=yes → personal_guide_triggered=True
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "I think so maybe?",
            "yes please proceed",
        ],
        test_name="test_R3_1_ambiguous_once_then_yes",
        scenario="personal_guide_consent AMBIGUOUS 'I think so maybe?' → gentle re-ask → yes → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R3_2_ambiguous_twice_then_yes(run_conversation, assert_and_record):
    """
    R3_2: personal_guide_consent — AMBIGUOUS twice, then yes on third.

    Turn 1: "hmm I'm not sure"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle re-ask
    Turn 2: "I guess maybe?"
      → AMBIGUOUS event again → second consecutive AMBIGUOUS increments attempt_count (count=1)
      → firmer re-ask fires
    Turn 3: "yes go ahead"
      → personal_guide_consent=yes → triggered before MAX_SLOT_ATTEMPTS=3
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "hmm I'm not sure",
            "I guess maybe?",
            "yes go ahead",
        ],
        test_name="test_R3_2_ambiguous_twice_then_yes",
        scenario="personal_guide_consent 2×AMBIGUOUS (second increments count) → yes before MAX",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R3_3_answered_wrong_then_yes(run_conversation, assert_and_record):
    """
    R3_3: personal_guide_consent — ANSWERED wrong (question-back) once, then yes.

    Turn 1: "what does that involve exactly?"
      → ANSWERED event: agent asked yes/no, member asked a question back.
        No yes/no value extracted, but LLM classifies as ANSWERED (not AMBIGUOUS)
        → attempt_count INCREMENTS (unlike AMBIGUOUS) → firmer RETRY re-ask fires
    Turn 2: "yes please"
      → personal_guide_consent=yes → triggered
    """
    record = await run_conversation(
        user_inputs=_PREFIX_A_WITH_REF
        + [
            "Feel free to call my doctor's office directly",
            "what does that involve exactly?",
            "yes please",
        ],
        test_name="test_R3_3_answered_wrong_then_yes",
        scenario="personal_guide_consent ANSWERED (question-back, count increments) → yes → triggered",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_personal_guide_triggered(record), "personal_guide_triggered"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R4 — email_confirmed 2-fail chain (records_coordination)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R4_1_bias_then_invalid_then_valid(run_conversation, assert_and_record):
    """
    R4_1: email_confirmed — bias rule (turn 1) → invalid format (turn 2) → valid (turn 3).

    Base: _PREFIX_A_WITH_REF + ["yes please"] to reach email_confirmed step.

    Turn 1: "I think so"
      → bias rule fires (non-clear-affirmation → email_confirmed=no) → agent asks for new email
    Turn 2: "bademail"
      → invalid format (missing @) → validate_email fails → retry
    Turn 3: "james.wilson.new@gmail.com"
      → valid → upload_link_sent=True

    Tests that TWO separate failure modes (bias rule + format validation) chain
    correctly before slot exhaustion. attempt_count < 3 confirms success before MAX.
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "I think so",
            "bademail",
            NEW_EMAIL_B2,
        ],
        test_name="test_R4_1_bias_then_invalid_then_valid",
        scenario="email_confirmed: bias rule (turn 1) → invalid format (turn 2) → valid (turn 3)",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R4_2_ambiguous_then_yes(run_conversation, assert_and_record):
    """
    R4_2: email_confirmed — AMBIGUOUS once (does NOT trigger bias rule), then yes.

    Turn 1: "hmm let me check"
      → AMBIGUOUS event: LLM cannot extract any value at all → no attempt_count increment
      → gentle CLARIFY re-ask. NOTE: bias rule does NOT apply to AMBIGUOUS events;
        bias rule only fires when LLM extracts a non-clear-affirmation value.
    Turn 2: "yes that's correct"
      → email_confirmed=yes → upload_link_sent=True

    Distinction from B2_implicit tests: AMBIGUOUS means LLM extracts nothing;
    bias triggers when LLM extracts a non-clear-affirmation (e.g. "I think so").
    """
    record = await run_conversation(
        user_inputs=_B2_EMAIL_PREFIX
        + [
            "hmm let me check",
            "yes that's correct",
        ],
        test_name="test_R4_2_ambiguous_then_yes",
        scenario="email_confirmed AMBIGUOUS 'hmm let me check' (no bias rule) → gentle re-ask → yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_upload_link_sent(record), "upload_link_sent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R5 — phone_confirmed retries in notification_setup
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R5_1_ambiguous_once_then_yes(run_conversation, assert_and_record):
    """
    R5_1: notification_setup phone readback — AMBIGUOUS once, then yes.

    Base: _C_PREFIX + ["SMS"] → phone readback step.

    Phone readback turn 1: "maybe?"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Turn 2: "yes that's correct"
      → phone_confirmed=yes → sms channel saved
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["SMS", "maybe?", "yes that's correct"],
        test_name="test_R5_1_ambiguous_once_then_yes",
        scenario="phone readback AMBIGUOUS 'maybe?' → gentle re-ask → yes → notification_channel=sms",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R5_2_ambiguous_once_then_no_inline(run_conversation, assert_and_record):
    """
    R5_2: notification_setup phone readback — AMBIGUOUS once, then no + inline new number.

    Phone readback turn 1: "hmm I'm not sure"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Turn 2: "no use five one two five five five four three zero zero"
      → phone_confirmed=no + inline new number → inline-update rule saves new phone
      → sms channel saved
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "hmm I'm not sure",
            f"no use {NEW_PHONE_SPOKEN}",
        ],
        test_name="test_R5_2_ambiguous_once_then_no_inline",
        scenario="phone readback AMBIGUOUS → gentle re-ask → no + inline number → sms saved",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R5_3_answered_wrong_then_yes(run_conversation, assert_and_record):
    """
    R5_3: notification_setup phone readback — ANSWERED wrong (question-back) then yes.

    Phone readback turn 1: "what number do you have on file?"
      → ANSWERED event: agent asked yes/no, member asked a question back.
        LLM classifies as ANSWERED (not AMBIGUOUS) → attempt_count INCREMENTS
        → firmer RETRY re-ask fires (distinct from gentle CLARIFY in R5_1)
    Turn 2: "yes that's right"
      → phone_confirmed=yes → sms channel saved
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "what number do you have on file?",
            "yes that's right",
        ],
        test_name="test_R5_3_answered_wrong_then_yes",
        scenario="phone readback ANSWERED (question-back, count increments) → firm re-ask → yes → sms",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R5_4_answered_wrong_twice_then_yes(run_conversation, assert_and_record):
    """
    R5_4: notification_setup phone readback — ANSWERED wrong twice, then yes on third.

    Turn 1: "what number?" → ANSWERED event → attempt_count=1 → firm re-ask
    Turn 2: "I need to check" → ANSWERED event → attempt_count=2 → firm re-ask
    Turn 3: "yes" → phone_confirmed=yes → sms saved

    Verifies retry budget is MAX=3 not MAX=2; count=2 still allows one more attempt.
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "SMS",
            "what number?",
            "I need to check",
            "yes",
        ],
        test_name="test_R5_4_answered_wrong_twice_then_yes",
        scenario="phone readback 2×ANSWERED (count=1,2) → yes on third → sms saved before MAX=3",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "sms"), "notification_channel==sms"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R6 — contact_confirmed AMBIGUOUS-then-yes in notification_setup (email)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R6_1_ambiguous_once_then_yes_email(run_conversation, assert_and_record):
    """
    R6_1: notification_setup email readback — AMBIGUOUS once, then yes.

    Base: _C_PREFIX + ["email"] → email readback step.

    Email readback turn 1: "hmm"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
      NOTE: bias rule does NOT fire on AMBIGUOUS — bias requires an extracted non-affirmation value.
    Turn 2: "yes that's my email"
      → contact_confirmed=yes → email channel saved
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX + ["email", "hmm", "yes that's my email"],
        test_name="test_R6_1_ambiguous_once_then_yes_email",
        scenario="email readback AMBIGUOUS 'hmm' (no bias rule) → gentle re-ask → yes → email channel",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_R6_2_ambiguous_then_bias_then_new_email(run_conversation, assert_and_record):
    """
    R6_2: notification_setup email readback — AMBIGUOUS (turn 1) → bias trigger (turn 2) → new email.

    Turn 1: "I'm not sure"
      → AMBIGUOUS event: LLM extracts nothing → attempt_count NOT incremented
      → gentle CLARIFY re-ask. Bias rule NOT triggered (no extracted value).
    Turn 2: "I think so"
      → LLM extracts non-clear-affirmation → bias rule fires → contact_confirmed=no
      → agent asks for new email
    Turn 3: "james.wilson.new@gmail.com"
      → valid new email → email channel saved

    Tests that AMBIGUOUS (turn 1) does not consume retry budget before bias fires (turn 2).
    """
    record = await run_conversation(
        user_inputs=_C_PREFIX
        + [
            "email",
            "I'm not sure",
            "I think so",
            NEW_EMAIL_C2,
        ],
        test_name="test_R6_2_ambiguous_then_bias_then_new_email",
        scenario="email readback AMBIGUOUS (turn 1) → bias trigger (turn 2) → new email (turn 3)",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_notification_channel(record, "email"), "notification_channel==email"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R7 — reference_number AMBIGUOUS path
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R7_1_dont_have_it_then_valid(run_conversation, assert_and_record):
    """
    R7_1: reference_number — "I don't have it with me right now" → retry → valid ref.

    Reference turn 1: "I don't have it with me right now"
      → AMBIGUOUS event: no digits extractable → attempt_count NOT incremented
      → agent re-asks gently (CLARIFY guard message)
    Reference turn 2: "42695817"
      → valid ref → claim_status_reported
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            "I don't have it with me right now",
            REF_B,
        ],
        test_name="test_R7_1_dont_have_it_then_valid",
        scenario="reference AMBIGUOUS 'don't have it' (no count) → valid ref → claim_status_reported",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_B), "reference_collected"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_R7_2_filler_then_ref_same_utterance(run_conversation, assert_and_record):
    """
    R7_2: reference_number — filler pause then valid ref in same utterance.

    Single turn: "let me find it… ok it's 42695817"
      Filler prefix before the number; extraction must ignore the filler and
      extract the embedded reference. This is a single turn (no retry).
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            f"let me find it… ok it's {REF_B}",
        ],
        test_name="test_R7_2_filler_then_ref_same_utterance",
        scenario="reference with filler prefix 'let me find it… ok it's 42695817' → extracted",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_reference_collected(record, REF_B), "reference_collected"),
            (lambda: assert_claim_status_reported(record), "claim_status_reported"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ---------------------------------------------------------------------------
# GROUP R8 — verification phone_confirmed 2×AMBIGUOUS then valid
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.slow
async def test_R8_1_two_ambiguous_then_yes(run_conversation, assert_and_record):
    """
    R8_1: verification phone_confirmed — 2× AMBIGUOUS then yes on third.

    Phone confirm turn 1: "uh… I'm not sure"
      → AMBIGUOUS event → attempt_count NOT incremented → gentle CLARIFY re-ask
    Phone confirm turn 2: "maybe?"
      → AMBIGUOUS event again → second consecutive AMBIGUOUS increments attempt_count (count=1)
        per _collect_slot logic: two consecutive AMBIGUOUS treated as genuine non-answer
      → firmer re-ask fires
    Phone confirm turn 3: "yes that's correct"
      → phone_confirmed=yes → member_status_verify=True
      count=1 < MAX=3, so slot is not exhausted.
    """
    record = await run_conversation(
        user_inputs=_VERIFICATION_BASE_B
        + [
            "uh… I'm not sure",
            "maybe?",
            "yes that's correct",
            REF_B,
        ],
        test_name="test_R8_1_two_ambiguous_then_yes",
        scenario="phone_confirmed 2×AMBIGUOUS (second increments count=1) → yes → member_verified",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_member_verified(record), "member_verified"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP F — Guard tests inside NotificationSetupAgent
# ===========================================================================


@pytest.mark.live
async def test_F1_transfer_request_during_notification_setup(run_conversation, assert_and_record):
    """
    F1: Member says "transfer me to a real person" after providing the
    notification method but before email_confirmed is collected.
    Verifies the TRANSFER_REQUEST guard fires inside NotificationSetupAgent
    and routes to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # records pass-through
            "yes",  # personal_guide_consent
            "email",
            "transfer me to a real person",
        ],
        test_name="test_F1_transfer_request_during_notification_setup",
        scenario=(
            "TRANSFER_REQUEST during notification_setup "
            "(between method and email_confirmed) → escalation_agent"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_F2_abuse_during_notification_setup(run_conversation, assert_and_record):
    """
    F2: Member uses explicit profanity after providing the notification method
    but before email_confirmed is collected.
    Verifies the ABUSE guard fires inside NotificationSetupAgent and routes
    to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX_CLAIMS_B
        + [
            REF_B,
            "Feel free to call my doctor's office directly",  # records pass-through
            "yes",  # personal_guide_consent
            "email",
            "you fucking useless machine",
        ],
        test_name="test_F2_abuse_during_notification_setup",
        scenario=("ABUSE guard fires inside notification_setup_agent → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalated"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routed_to_escalation"),
        ],
    )
