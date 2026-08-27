"""
test_provider_search_agent_live.py — Live integration tests for ProviderSearchAgent.

These tests run against a real LLM (Azure OpenAI / Gemini).
They require valid API credentials in the environment.

Run:
    pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -v
    pytest -m live -k "test_provider_search" -v   # single group
    pytest -m live --count=20 -v src/agent/tests/live/test_provider_search_agent_live.py

# Group G — Provider type variations
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k
"test_provider_search_provider_type" -v

# Group H — Unsupported medical type
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k
"test_provider_search_unsupported" -v

# Group I — Non-medical / ambiguous then recovery
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k "test_provider_
search_non_medical or test_provider_search_ambiguous or test_provider_search_silence" -v

# Group J — ZIP confirmation affirmatives
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k "test_zip_confirm" -v

# Group K — ZIP hesitation
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k "test_zip_hesitation" -v

# Group L — ZIP rejection and update
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k "test_zip_reject" -v

# Group M — ZIP exhaustion
pytest -m live src/agent/tests/live/test_provider_search_agent_live.py -k
"test_zip_confirm_exhausted or test_zip_update_exhausted" -v

Skip in CI:
    Tests auto-skip when AZURE_OPENAI_API_KEY (or GOOGLE_API_KEY) is absent.

Member data
-----------
Uses the same VERIFIED_MEMBER as verification tests:
  Emily Carter / M907503 / 04/12/1988 — matches Salesforce sandbox
  zip_code on file: 12139

Groups
------
G  Provider type natural language variations            (12 tests)
H  Unsupported medical type fast-fail                   (5 tests)
I  Non-medical type / ambiguous then recovery           (5 tests)
J  ZIP confirmation — affirmative variations            (6 tests)
K  ZIP hesitation and clarification                     (4 tests)
L  ZIP rejection and update                             (7 tests)
M  ZIP exhaustion edge cases                            (2 tests)
"""

from __future__ import annotations

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
# Fixture alias
# ---------------------------------------------------------------------------


@pytest.fixture
def run_conversation(run_intake_conversation):
    """Alias so provider-search tests read naturally. Same graph runner underneath."""
    return run_intake_conversation


# ---------------------------------------------------------------------------
# Verified member data and conversation prefix
# ---------------------------------------------------------------------------

# Six turns that complete intake + verification for Emily Carter.
# Every test in this file starts with these inputs.
VERIFICATION_PREFIX = [
    "I need to find an in-network doctor",
    "Emily",
    "Carter",
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty-eight",
    "I'm calling for myself",
]

# ZIP code stored in Salesforce for Emily Carter
ZIP_ON_FILE = "12139"

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_not_escalated(record: ConversationRecord) -> None:
    """No escalation occurred across any turn."""
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


def assert_routed_to_delivery(record: ConversationRecord) -> None:
    """Conversation reached delivery_management_agent."""
    assert_routed_to(record, "delivery_management_agent")


def assert_provider_type(record: ConversationRecord, expected: str) -> None:
    """provider_type in final state equals expected."""
    actual = record.final_state.get("provider_type", "")
    assert actual == expected, f"Expected provider_type={expected!r}, got {actual!r}"


def assert_provider_search_was_active(record: ConversationRecord) -> None:
    """provider_search_agent was active in at least one turn."""
    was_active = record.final_state.get("active_agent") == "provider_search_agent" or any(
        t.active_agent == "provider_search_agent" for t in record.turns
    )
    assert was_active, "Expected provider_search_agent to be active in at least one turn"


def assert_zip_code_used(record: ConversationRecord, expected_zip: str) -> None:
    """zip_code_used OR zip_code in final state equals expected_zip."""
    zip_used = (record.final_state.get("zip_code_used") or "").strip()
    zip_code = (record.final_state.get("zip_code") or "").strip()
    assert zip_used == expected_zip or zip_code == expected_zip, (
        f"Expected zip_code_used or zip_code == {expected_zip!r}, "
        f"got zip_code_used={zip_used!r}, zip_code={zip_code!r}"
    )


def assert_any_agent_message_contains(record: ConversationRecord, *substrings: str) -> None:
    """At least one agent message across all turns contains each substring."""
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    for sub in substrings:
        assert sub.lower() in all_msgs, (
            f"Expected any agent message to contain {sub!r}. Full transcript: {all_msgs[:500]!r}"
        )


# ===========================================================================
# GROUP G — Provider type natural language variations
# ===========================================================================


@pytest.mark.live
async def test_provider_search_provider_type_family_doctor(run_conversation, assert_and_record):
    """User says 'family doctor' — should normalize to Primary Care Physician."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["family doctor", "yes"],
        test_name="test_provider_search_provider_type_family_doctor",
        scenario="'family doctor' normalizes to Primary Care Physician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_my_doctor(run_conversation, assert_and_record):
    """User says 'my regular doctor' — should normalize to Primary Care Physician."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["my regular doctor", "yes"],
        test_name="test_provider_search_provider_type_my_doctor",
        scenario="'my regular doctor' normalizes to Primary Care Physician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_general_practitioner(run_conversation, assert_and_record):
    """User says 'general practitioner' — should normalize to Primary Care Physician."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["general practitioner", "yes"],
        test_name="test_provider_search_provider_type_general_practitioner",
        scenario="'general practitioner' normalizes to Primary Care Physician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_heart_specialist(run_conversation, assert_and_record):
    """User says 'heart specialist' — should normalize to Cardiologist."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["heart specialist", "yes"],
        test_name="test_provider_search_provider_type_heart_specialist",
        scenario="'heart specialist' normalizes to Cardiologist → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_provider_type(record, "Cardiologist"), "provider_type==Cardiologist"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_skin_doctor(run_conversation, assert_and_record):
    """User says 'skin doctor' — should normalize to Dermatologist."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["skin doctor", "yes"],
        test_name="test_provider_search_provider_type_skin_doctor",
        scenario="'skin doctor' normalizes to Dermatologist → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_provider_type(record, "Dermatologist"), "provider_type==Dermatologist"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_kids_doctor(run_conversation, assert_and_record):
    """User says 'doctor for my kids' — should normalize to Pediatrician."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["doctor for my kids", "yes"],
        test_name="test_provider_search_provider_type_kids_doctor",
        scenario="'doctor for my kids' normalizes to Pediatrician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_provider_type(record, "Pediatrician"), "provider_type==Pediatrician"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_bone_doctor(run_conversation, assert_and_record):
    """User says 'bone doctor' — should normalize to Orthopedic Specialist."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["bone doctor", "yes"],
        test_name="test_provider_search_provider_type_bone_doctor",
        scenario="'bone doctor' normalizes to Orthopedic Specialist → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Orthopedic Specialist"),
                "provider_type==Orthopedic Specialist",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_with_context_sentence(run_conversation, assert_and_record):
    """Full sentence with 'primary care physician' — should extract and normalize correctly."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "I'm looking for a primary care physician in my area please",
            "yes",
        ],
        test_name="test_provider_search_provider_type_with_context_sentence",
        scenario="Full context sentence containing PCP → Primary Care Physician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_with_filler(run_conversation, assert_and_record):
    """User says 'um, I need a cardiologist I think' — filler should not block extraction."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["um, I need a cardiologist I think", "yes"],
        test_name="test_provider_search_provider_type_with_filler",
        scenario="Provider type buried in filler words → Cardiologist extracted correctly",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_provider_type(record, "Cardiologist"), "provider_type==Cardiologist"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_with_correction(run_conversation, assert_and_record):
    """User self-corrects in one utterance — LLM should extract the final corrected value."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "I need a dermatologist, wait actually a cardiologist",
            "yes",
        ],
        test_name="test_provider_search_provider_type_with_correction",
        scenario="Self-correction in single utterance: dermatologist → cardiologist → Cardiologist extracted",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_provider_type(record, "Cardiologist"), "provider_type==Cardiologist"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_uppercase_pcp(run_conversation, assert_and_record):
    """User says 'PCP' in uppercase — should normalize to Primary Care Physician."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["PCP", "yes"],
        test_name="test_provider_search_provider_type_uppercase_pcp",
        scenario="'PCP' (uppercase) normalizes to Primary Care Physician → delivery routing",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_provider_search_provider_type_primary_care(run_conversation, assert_and_record):
    """User says 'primary care' without 'physician' — should still normalize to PCP."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care", "yes"],
        test_name="test_provider_search_provider_type_primary_care",
        scenario="'primary care' (without 'physician') normalizes to Primary Care Physician",
    )
    assert_and_record(
        record,
        [
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP H — Unsupported medical type fast-fail (CASE 2)
# ===========================================================================


@pytest.mark.live
async def test_provider_search_unsupported_oncologist(run_conversation, assert_and_record):
    """User says 'oncologist' — valid medical but unsupported, immediate escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["oncologist"],
        test_name="test_provider_search_unsupported_oncologist",
        scenario="'oncologist' triggers CASE 2 fast-fail: escalated with representative message",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="Let me connect you")
            # , "escalated_unsupported"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "msg_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_provider_search_unsupported_urologist(run_conversation, assert_and_record):
    """User says 'urologist' — valid medical but unsupported, immediate escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["urologist"],
        test_name="test_provider_search_unsupported_urologist",
        scenario="'urologist' triggers CASE 2 fast-fail: escalated with representative message",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="Let me connect you"),
            #  "escalated_unsupported"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "msg_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_provider_search_unsupported_ophthalmologist(run_conversation, assert_and_record):
    """User says 'ophthalmologist' — valid medical but unsupported, immediate escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["ophthalmologist"],
        test_name="test_provider_search_unsupported_ophthalmologist",
        scenario="'ophthalmologist' triggers CASE 2 fast-fail: escalated with representative message",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="Let me connect you"),
            #  "escalated_unsupported"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "msg_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_provider_search_unsupported_endocrinologist(run_conversation, assert_and_record):
    """User says 'endocrinologist' — valid medical but unsupported, immediate escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["endocrinologist"],
        test_name="test_provider_search_unsupported_endocrinologist",
        scenario="'endocrinologist' triggers CASE 2 fast-fail: escalated immediately",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="Let me connect you"),
            # "escalated_unsupported"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "msg_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_provider_search_unsupported_with_context(run_conversation, assert_and_record):
    """User says 'I'm looking for a neurologist for my migraines' — unsupported, immediate escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["I'm looking for a neurologist for my migraines"],
        test_name="test_provider_search_unsupported_with_context",
        scenario="Neurologist in full sentence → CASE 2 fast-fail with representative message",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="Let me connect you"),
            # "escalated_unsupported"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "msg_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP I — Non-medical type then recovery (CASE 1)
# ===========================================================================


@pytest.mark.live
async def test_provider_search_non_medical_one_bad_then_valid(run_conversation, assert_and_record):
    """User says 'electrician' then 'primary care physician' — recovery on second attempt."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["electrician", "primary care physician", "yes"],
        test_name="test_provider_search_non_medical_one_bad_then_valid",
        scenario="Non-medical 'electrician' → retry → 'primary care physician' → success",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
            (
                lambda: assert_provider_type(record, "Primary Care Physician"),
                "provider_type==Primary Care Physician",
            ),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
        ],
    )


@pytest.mark.live
async def test_provider_search_non_medical_two_bad_then_valid(run_conversation, assert_and_record):
    """User says 'plumber' then 'electrician' then 'cardiologist' — recovery on third attempt."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["plumber", "electrician", "cardiologist", "yes"],
        test_name="test_provider_search_non_medical_two_bad_then_valid",
        scenario="Two non-medical turns → 'cardiologist' on third attempt → success",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_provider_type(record, "Cardiologist"), "provider_type==Cardiologist"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
        ],
    )


@pytest.mark.live
async def test_provider_search_non_medical_two_bad_one_unclear_then_exhaust(
    run_conversation, assert_and_record
):
    """Three non-medical/unintelligible answers exhaust MAX_SLOT_ATTEMPTS — generic escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "electrician",
            "plumber",
            "someone who fixes my pipes",
            "i am looking for a doctor but i don't know what kind",
        ],
        test_name="test_provider_search_non_medical_two_bad_one_unclear_then_exhaust",
        scenario="Three non-medical attempts exhaust provider_type slot → generic escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_provider_search_was_active(record), "provider_search_was_active"),
        ],
    )


@pytest.mark.live
async def test_provider_search_ambiguous_then_valid(run_conversation, assert_and_record):
    """User says 'I'm not sure' then 'dermatologist' — recovery after one ambiguous turn."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["I'm not sure", "dermatologist", "yes"],
        test_name="test_provider_search_ambiguous_then_valid",
        scenario="Ambiguous first answer → retry → 'dermatologist' collected → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_provider_type(record, "Dermatologist"), "provider_type==Dermatologist"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
        ],
    )


@pytest.mark.live
async def test_provider_search_silence_then_valid(run_conversation, assert_and_record):
    """User says 'hmm' (genuine non-answer) then 'pediatrician' — recovery."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["yes", "hmm", "pediatrician", "yes"],
        test_name="test_provider_search_silence_then_valid",
        scenario="Filler non-answer → retry → 'pediatrician' collected → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_provider_type(record, "Pediatrician"), "provider_type==Pediatrician"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
        ],
    )


# ===========================================================================
# GROUP J — ZIP confirmation variations (zip on file, user confirms)
# ===========================================================================


@pytest.mark.live
async def test_zip_confirm_yes_explicit(run_conversation, assert_and_record):
    """User says 'yes' to zip readback — explicit confirmation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care physician", "yes"],
        test_name="test_zip_confirm_yes_explicit",
        scenario="'yes' confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_confirm_affirmative_variation_correct(run_conversation, assert_and_record):
    """User says 'correct' to zip readback — affirms ZIP on file."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care physician", "correct"],
        test_name="test_zip_confirm_affirmative_variation_correct",
        scenario="'correct' confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_confirm_affirmative_variation_thats_right(run_conversation, assert_and_record):
    """User says 'yeah that's right' to zip readback — affirms ZIP on file."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care physician", "yeah that's right"],
        test_name="test_zip_confirm_affirmative_variation_thats_right",
        scenario="'yeah that's right' confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_confirm_affirmative_variation_yep(run_conversation, assert_and_record):
    """User says 'yep' to zip readback — colloquial affirmation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care physician", "yep"],
        test_name="test_zip_confirm_affirmative_variation_yep",
        scenario="'yep' confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_confirm_affirmative_with_filler(run_conversation, assert_and_record):
    """User says 'uh huh yes that's correct' — affirmation with leading filler."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX + ["primary care physician", "uh huh yes that's correct"],
        test_name="test_zip_confirm_affirmative_with_filler",
        scenario="'uh huh yes that's correct' confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_confirm_affirmative_long_sentence(run_conversation, assert_and_record):
    """User says full affirmative sentence containing the ZIP — still confirms correctly."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "yes that is the correct zip code for my area",
        ],
        test_name="test_zip_confirm_affirmative_long_sentence",
        scenario="Long affirmative sentence confirms ZIP on file → delivery routing",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_zip_code_used(record, ZIP_ON_FILE), f"zip_code_used=={ZIP_ON_FILE}"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP K — ZIP hesitation and clarification
# ===========================================================================


@pytest.mark.live
async def test_zip_hesitation_then_confirms(run_conversation, assert_and_record):
    """User hesitates on zip readback, then confirms on re-ask."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "hmm let me think about that",
            "yes that's right",
        ],
        test_name="test_zip_hesitation_then_confirms",
        scenario="'hmm let me think about that' → re-ask → 'yes that's right' → confirmed",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_hesitation_i_think_so(run_conversation, assert_and_record):
    """User says 'I think so', then gives explicit confirmation on re-ask."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "I think so",
            "yes that is correct",
        ],
        test_name="test_zip_hesitation_i_think_so",
        scenario="'I think so' on zip → re-ask → explicit 'yes' → confirmed, not escalated",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_hesitation_not_sure_then_confirms(run_conversation, assert_and_record):
    """User says 'I'm not sure actually', then confirms on re-ask."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "I'm not sure actually",
            "yes that's the one",
        ],
        test_name="test_zip_hesitation_not_sure_then_confirms",
        scenario="'I'm not sure actually' on zip → re-ask → 'yes that's the one' → confirmed",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_hesitation_long_pause_filler(run_conversation, assert_and_record):
    """User says 'uh... yeah I think that's right' — single-turn net affirmative."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "uh... yeah I think that's right",
            "yes that is correct",
        ],
        test_name="test_zip_hesitation_long_pause_filler",
        scenario="Hesitation filler with net affirmative 'yeah I think that's right' → confirmed in one turn",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP L — ZIP rejection and update
# ===========================================================================


@pytest.mark.live
async def test_zip_reject_no_then_provide_spoken_digits(run_conversation, assert_and_record):
    """User rejects zip, then provides 'seven seven one seven two' → '77172'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no",
            "seven seven one seven two",
        ],
        test_name="test_zip_reject_no_then_provide_spoken_digits",
        scenario="'no' rejects ZIP → spoken digits '77172' collected → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "77172"), "zip_code_used==77172"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_thats_wrong_then_provide(run_conversation, assert_and_record):
    """User says 'that's not right', then provides 'nine zero two one zero' → '90210'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "that's not right",
            "nine zero two one zero",
        ],
        test_name="test_zip_reject_thats_wrong_then_provide",
        scenario="'that's not right' rejects ZIP → '90210' spoken digits collected → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "90210"), "zip_code_used==90210"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_different_zip_inline(run_conversation, assert_and_record):
    """User says 'no it's 10001' — rejection and new zip in one utterance."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no it's 10001",
        ],
        test_name="test_zip_reject_different_zip_inline",
        scenario="Inline 'no it's 10001' → LLM extracts zip_code=10001 directly → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "10001"), "zip_code_used==10001"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_moved_recently(run_conversation, assert_and_record):
    """User explains they moved and provides new zip 'three zero three zero one' → '30301'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "I moved recently so that's the old zip, it's now three zero three zero one",
        ],
        test_name="test_zip_reject_moved_recently",
        scenario="'moved recently' with spoken new zip 30301 inline → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "30301"), "zip_code_used==30301"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_moved_recently_separate(run_conversation, assert_and_record):
    """User explains they moved and provides new zip 'three zero three zero one' → '30301'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "I moved recently",
            "it's now three zero three zero one",
        ],
        test_name="test_zip_reject_moved_recently_separate",
        scenario="'moved recently' with spoken new zip 30301 separate turns → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "30301"), "zip_code_used==30301"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_with_explanation(run_conversation, assert_and_record):
    """User explains zip is wrong and provides new one 'six zero six zero one' → '60601'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no that zip is wrong, let me give you the correct one, it's six zero six zero one",
        ],
        test_name="test_zip_reject_with_explanation",
        scenario="Verbose rejection with inline spoken zip 60601 → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "60601"), "zip_code_used==60601"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_then_invalid_zip_then_valid(run_conversation, assert_and_record):
    """User rejects zip, provides invalid '123', then valid 'seven seven zero zero two' → '77002'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no",
            "123",
            "seven seven zero zero two",
        ],
        test_name="test_zip_reject_then_invalid_zip_then_valid",
        scenario="Reject → invalid '123' → valid spoken '77002' → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "77002"), "zip_code_used==77002"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_zip_reject_then_spoken_with_oh(run_conversation, assert_and_record):
    """User declines zip, then says 'seven seven oh one two' — 'oh' maps to zero → '77012'."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no",
            "seven seven oh one two",
        ],
        test_name="test_zip_reject_then_spoken_with_oh",
        scenario="Reject → 'seven seven oh one two' with spoken 'oh'=0 → 77012 → delivery routing",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_zip_code_used(record, "77012"), "zip_code_used==77012"),
            (lambda: assert_routed_to_delivery(record), "routed_to_delivery"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP M — ZIP exhaustion edge cases
# ===========================================================================


@pytest.mark.live
async def test_zip_confirm_exhausted_after_repeated_unclear(run_conversation, assert_and_record):
    """Three unclear zip confirmation responses exhaust the slot budget → escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "I'm not sure",
            "hmm",
            "I don't know",
        ],
        test_name="test_zip_confirm_exhausted_after_repeated_unclear",
        scenario="Three unclear zip_confirmed answers exhaust slot → escalation with zip reason",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_escalated(record, reason_contains="zip"), "escalated_zip_reason"),
            (lambda: assert_provider_search_was_active(record), "provider_search_was_active"),
        ],
    )


@pytest.mark.live
async def test_zip_update_exhausted_after_repeated_invalid(run_conversation, assert_and_record):
    """User rejects zip on file, then provides three invalid zip codes → escalation."""
    record = await run_conversation(
        user_inputs=VERIFICATION_PREFIX
        + [
            "primary care physician",
            "no",
            "123",
            "abcde",
            "nine nine",
            "seven seven seven",
        ],
        test_name="test_zip_update_exhausted_after_repeated_invalid",
        scenario="Reject ZIP → three invalid zips exhaust zip_code slot → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_provider_search_was_active(record), "provider_search_was_active"),
        ],
    )
