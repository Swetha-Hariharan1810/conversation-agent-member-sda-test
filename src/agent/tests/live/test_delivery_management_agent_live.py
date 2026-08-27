"""
test_delivery_management_agent_live.py — Live integration tests for DeliveryManagementAgent.

These tests run against a real LLM (Azure OpenAI / Gemini) and a real
Salesforce sandbox.  They require valid API credentials in the environment.

Run:
    pytest -m live src/agent/tests/live/test_delivery_management_agent_live.py -v
    pytest -m live -k "test_delivery_management" -v   # single group

Skip in CI:
    Tests auto-skip when AZURE_OPENAI_API_KEY (or GOOGLE_API_KEY) is absent.

Member data
-----------
Emily Carter / M907503 / 04/12/1988 — matches Salesforce sandbox
  fax on file:   617-555-4199
  email on file: emily.carter@gmail.com
  zip on file:   60601
  relationship:  plan_holder

Groups
------
A  Delivery method fax variations                                         (10 tests)
B  Delivery method email/mail variations                                  (11 tests)
C  Fax confirmed — clear affirmations                                     (10 tests)
D  Fax bias rule → two-turn update                                        ( 9 tests)
E  Fax inline rejection + replacement in one utterance                    ( 6 tests)
F  Email bias rule → two-turn update                                      ( 9 tests)
G  Email inline rejection + replacement in one utterance                  ( 5 tests)
H  Benefits offer response variations                                     (10 tests)
I  Delivery method ambiguous then valid                                   ( 5 tests)
J  Guard triggers inside DeliveryManagementAgent                          ( 4 tests)
K  Invalid contact then valid                                             ( 4 tests)
L  Contact slot exhaustion → escalation                                   ( 3 tests)
M  Delivery method slot exhaustion → escalation                          ( 1 test)
N  Email confirmed — clear affirmations                                   ( 8 tests)
O  Latency benchmarks                                                     ( 3 tests)
"""

from __future__ import annotations

import math
import os
import re

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

FULL_PREFIX = [
    "I need to find an in-network doctor",
    "Emily",
    "Carter",
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty-eight",
    "I'm calling for myself",
    "primary care physician",
    "yes",
]

FAX_ON_FILE = "617-555-4199"
EMAIL_ON_FILE = "emily.carter@gmail.com"
NEW_FAX = "6175554200"
NEW_FAX_SPOKEN = "six one seven five five five four two zero zero"
NEW_EMAIL = "emily.carter.new@gmail.com"

_LATENCY_P50_SEC = 8.0
_LATENCY_P95_SEC = 15.0

# ---------------------------------------------------------------------------
# Fixture alias
# ---------------------------------------------------------------------------


@pytest.fixture
def run_conversation(run_intake_conversation):
    """Alias so delivery-management tests read naturally. Same graph runner underneath."""
    return run_intake_conversation


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_provider_list_sent(record: ConversationRecord) -> None:
    """provider_list_sent=True in final state."""
    actual = record.final_state.get("provider_list_sent")
    assert actual is True, f"Expected provider_list_sent=True, got {actual!r}"


def assert_delivery_method(record: ConversationRecord, expected: str) -> None:
    """delivery_method == expected in final state."""
    actual = record.final_state.get("delivery_method", "")
    assert actual == expected, f"Expected delivery_method={expected!r}, got {actual!r}"


def assert_fax_used(record: ConversationRecord, expected_fax: str) -> None:
    """fax == expected_fax in final state (format-insensitive)."""
    actual = re.sub(r"\D", "", record.final_state.get("fax", ""))
    expected = re.sub(r"\D", "", expected_fax)
    assert actual == expected, f"Expected fax={expected_fax!r}, got {record.final_state.get('fax', '')!r}"


def assert_email_used(record: ConversationRecord, expected_email: str) -> None:
    """email == expected_email in final state."""
    actual = record.final_state.get("email", "")
    assert actual == expected_email, f"Expected email={expected_email!r}, got {actual!r}"


def assert_benefits_offer_made(record: ConversationRecord) -> None:
    """benefits_offer_made=True in final state."""
    actual = record.final_state.get("benefits_offer_made")
    assert actual is True, f"Expected benefits_offer_made=True, got {actual!r}"


def assert_proactive_offer_available(record: ConversationRecord, expected: bool) -> None:
    """proactive_offer_available == expected in final state."""
    actual = record.final_state.get("proactive_offer_available")
    assert actual == expected, f"Expected proactive_offer_available={expected!r}, got {actual!r}"


def assert_delivery_management_was_active(record: ConversationRecord) -> None:
    """delivery_management_agent was active in at least one turn."""
    was_active = record.final_state.get("active_agent") == "delivery_management_agent" or any(
        t.active_agent == "delivery_management_agent" for t in record.turns
    )
    assert was_active, "Expected delivery_management_agent to be active in at least one turn"


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


def assert_routed_to_benefits(record: ConversationRecord) -> None:
    """Conversation reached benefits_agent."""
    was_routed = (
        record.final_state.get("next_node") == "benefits_agent"
        or record.final_state.get("active_agent") == "benefits_agent"
        or any(t.active_agent == "benefits_agent" for t in record.turns)
    )
    assert was_routed, (
        f"Expected routing to 'benefits_agent', got "
        f"next_node={record.final_state.get('next_node')!r}, "
        f"active_agent={record.final_state.get('active_agent')!r}"
    )


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


def assert_any_agent_message_contains(record: ConversationRecord, *substrings: str) -> None:
    """At least one agent message across all turns contains each substring."""
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    for sub in substrings:
        assert sub.lower() in all_msgs, (
            f"Expected any agent message to contain {sub!r}. Full transcript: {all_msgs[:500]!r}"
        )


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


# ===========================================================================
# GROUP A — Delivery method fax variations
# ===========================================================================


@pytest.mark.live
async def test_delivery_method_fax_bare(run_conversation, assert_and_record):
    """
    User says bare 'fax' — the simplest possible delivery_method input.
    Verifies that the single-word canonical form is extracted as 'fax' with
    no ambiguity and that the full fax happy path completes.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "yes"],
        test_name="test_delivery_method_fax_bare",
        scenario="'fax' → delivery_method=fax → confirm on file → benefits yes → complete",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_send_by_fax(run_conversation, assert_and_record):
    """
    User says 'send it by fax' — a prepositional phrase form.
    Verifies that 'by fax' is correctly parsed as delivery_method='fax'
    even when embedded in a short instruction sentence.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["send it by fax", "yes", "yes"],
        test_name="test_delivery_method_send_by_fax",
        scenario="'send it by fax' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_fax_it_to_me(run_conversation, assert_and_record):
    """
    User says 'fax it to me' — verb-first imperative form.
    Verifies that the extraction handles 'fax' as a verb (imperative) and
    still maps to delivery_method='fax', not ambiguous.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax it to me", "yes", "yes"],
        test_name="test_delivery_method_fax_it_to_me",
        scenario="'fax it to me' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_via_fax_please(run_conversation, assert_and_record):
    """
    User says 'via fax please' — polite prepositional form.
    Verifies 'via fax' is extracted as 'fax' and that the trailing 'please'
    does not confuse the extraction.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["via fax please", "yes", "yes"],
        test_name="test_delivery_method_via_fax_please",
        scenario="'via fax please' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_fax_would_be_great(run_conversation, assert_and_record):
    """
    User says 'fax would be great' — preference statement with filler.
    Verifies that 'would be great' padding after the channel keyword does
    not prevent extraction of delivery_method='fax'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax would be great", "yes", "yes"],
        test_name="test_delivery_method_fax_would_be_great",
        scenario="'fax would be great' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_i_prefer_fax(run_conversation, assert_and_record):
    """
    User says 'I prefer fax' — explicit preference declaration.
    Verifies that the first-person preference form ('I prefer X') maps
    cleanly to delivery_method='fax'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["I prefer fax", "yes", "yes"],
        test_name="test_delivery_method_i_prefer_fax",
        scenario="'I prefer fax' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_use_the_fax(run_conversation, assert_and_record):
    """
    User says 'use the fax' — definite-article imperative form.
    Verifies that the article 'the' before 'fax' (referring to the fax machine)
    is handled correctly and does not produce an ambiguous result.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["use the fax", "yes", "yes"],
        test_name="test_delivery_method_use_the_fax",
        scenario="'use the fax' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_fax_number(run_conversation, assert_and_record):
    """
    User says 'fax number' — noun phrase referring to the channel by its
    contact-detail name.  Verifies that 'fax number' (without a verb) is
    still extracted as delivery_method='fax' rather than treated as ambiguous.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax number", "yes", "yes"],
        test_name="test_delivery_method_fax_number",
        scenario="'fax number' → delivery_method=fax → confirm on file → benefits yes",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_fax_paper_trail(run_conversation, assert_and_record):
    """
    Conversational: 'Actually you know what, just fax it over, that way I'll have a paper trail.'
    Verifies that 'fax' embedded mid-sentence with colloquial lead-in ('you know what')
    and trailing justification is still extracted as delivery_method='fax'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "Actually you know what, just fax it over, that way I'll have a paper trail",
            "yes",
            "yes",
        ],
        test_name="test_delivery_method_fax_paper_trail",
        scenario="Conversational fax with justification → delivery_method=fax → full happy path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_fax_at_office(run_conversation, assert_and_record):
    """
    Conversational: 'I think fax is probably easier for me since I'm at the office right now.'
    Verifies that 'fax' buried after 'I think' (hedge) and followed by a contextual
    explanation is still unambiguously extracted as delivery_method='fax'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "I think fax is probably easier for me since I'm at the office right now",
            "yes",
            "yes",
        ],
        test_name="test_delivery_method_fax_at_office",
        scenario="Conversational hedged fax preference → delivery_method=fax → full happy path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP B — Delivery method email variations
# ===========================================================================


@pytest.mark.live
async def test_delivery_method_email_bare(run_conversation, assert_and_record):
    """
    User says bare 'email' — the simplest possible email delivery input.
    Verifies that the single-word canonical form is extracted as 'email' and
    the full email happy path completes (benefits declined with 'no').
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "no"],
        test_name="test_delivery_method_email_bare",
        scenario="'email' → delivery_method=email → confirm on file → benefits no → complete",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_send_by_email(run_conversation, assert_and_record):
    """
    User says 'send it by email' — prepositional phrase mirroring the fax variant.
    Verifies 'by email' extracts as 'email' and the email happy path completes.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["send it by email", "yes", "no"],
        test_name="test_delivery_method_send_by_email",
        scenario="'send it by email' → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_email_it_to_me(run_conversation, assert_and_record):
    """
    User says 'email it to me' — verb-first imperative form for email.
    Verifies that 'email' used as a verb still maps to delivery_method='email'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email it to me", "yes", "no"],
        test_name="test_delivery_method_email_it_to_me",
        scenario="'email it to me' → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_via_email(run_conversation, assert_and_record):
    """
    User says 'via email' — prepositional form without 'please'.
    Verifies the minimal 'via email' phrase extracts as delivery_method='email'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["via email", "yes", "no"],
        test_name="test_delivery_method_via_email",
        scenario="'via email' → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_by_mail(run_conversation, assert_and_record):
    """
    User says 'by mail' — a mail variant that must map to 'email' per the
    extraction prompt rule ('all mail variants → email').
    Verifies the mail→email mapping is applied correctly.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["by mail", "yes", "no"],
        test_name="test_delivery_method_by_mail",
        scenario="'by mail' (mail variant) → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_mail_it(run_conversation, assert_and_record):
    """
    User says 'mail it' — imperative mail variant that must map to 'email'.
    Verifies that 'mail' used as a verb still triggers the mail→email rule
    from the extraction prompt.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["mail it", "yes", "no"],
        test_name="test_delivery_method_mail_it",
        scenario="'mail it' (mail variant) → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_send_in_the_mail(run_conversation, assert_and_record):
    """
    User says 'send it in the mail' — full noun-phrase mail variant.
    Verifies that the longer 'in the mail' form also triggers the mail→email
    rule and does not produce an ambiguous result.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["send it in the mail", "yes", "no"],
        test_name="test_delivery_method_send_in_the_mail",
        scenario="'send it in the mail' (mail variant) → delivery_method=email → confirm on file",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_i_prefer_email(run_conversation, assert_and_record):
    """
    User says 'I'd prefer email' — contraction preference form.
    Verifies that the contracted 'I'd prefer' still extracts 'email'
    cleanly, mirroring the fax 'I prefer fax' test.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["I'd prefer email", "yes", "no"],
        test_name="test_delivery_method_i_prefer_email",
        scenario="'I'd prefer email' → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_electronic(run_conversation, assert_and_record):
    """
    User says 'electronic' — informal synonym for email delivery.
    Verifies that 'electronic' (without the word 'email') is understood in
    context as delivery_method='email' given the agent just asked fax or email.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["electronic", "yes", "no"],
        test_name="test_delivery_method_electronic",
        scenario="'electronic' → delivery_method=email → confirm on file → benefits no",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_email_shoot_it_over(run_conversation, assert_and_record):
    """
    Conversational: 'Oh just shoot it over to my email, that's way easier for me to deal with.'
    Verifies that 'email' embedded in a casual sentence with colloquial
    phrasing ('shoot it over') and trailing filler is still extracted as
    delivery_method='email'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "Oh just shoot it over to my email, that's way easier for me to deal with",
            "yes",
            "no",
        ],
        test_name="test_delivery_method_email_shoot_it_over",
        scenario="Conversational email preference → delivery_method=email → full happy path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_email_check_more_often(run_conversation, assert_and_record):
    """
    Conversational: 'Email works better for me honestly, I check it more often than my fax machine.'
    Verifies that 'email' is extracted as delivery_method='email' when followed by
    a contrastive clause that also mentions 'fax machine', ensuring 'fax' in the
    trailing clause does not override the leading preference.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "Email works better for me honestly, I check it more often than my fax machine",
            "yes",
            "no",
        ],
        test_name="test_delivery_method_email_check_more_often",
        scenario="Conversational email with contrastive fax mention → delivery_method=email wins",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C — Fax confirmed clear affirmations
# ===========================================================================


@pytest.mark.live
async def test_fax_confirmed_yes(run_conversation, assert_and_record):
    """
    User says bare 'yes' to confirm fax on file.
    Simplest affirmation — verifies the base case of the fax_confirmed
    slot: contact_confirmed='yes' → no SF update, proceeds to dispatch.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "yes"],
        test_name="test_fax_confirmed_yes",
        scenario="'yes' confirms fax on file → fax==FAX_ON_FILE, dispatched, no update",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_correct(run_conversation, assert_and_record):
    """
    User says 'correct' to confirm fax on file.
    Verifies that the single-word affirmative 'correct' normalizes to
    contact_confirmed='yes' and the original fax is used without update.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "correct", "yes"],
        test_name="test_fax_confirmed_correct",
        scenario="'correct' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_thats_right(run_conversation, assert_and_record):
    """
    User says 'that's right' to confirm fax on file.
    Verifies that the demonstrative affirmative ('that's right') maps to
    contact_confirmed='yes' without triggering the bias rule.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "that's right", "yes"],
        test_name="test_fax_confirmed_thats_right",
        scenario="'that's right' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_yep(run_conversation, assert_and_record):
    """
    User says 'yep' to confirm fax on file.
    Verifies that the colloquial affirmative 'yep' normalizes to
    contact_confirmed='yes' and the fax flow completes correctly.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yep", "yes"],
        test_name="test_fax_confirmed_yep",
        scenario="'yep' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_yes_thats_the_one(run_conversation, assert_and_record):
    """
    User says 'yes that's the one' to confirm fax on file.
    Verifies that an affirmation with a trailing demonstrative clause
    still maps to contact_confirmed='yes' without ambiguity.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes that's the one", "yes"],
        test_name="test_fax_confirmed_yes_thats_the_one",
        scenario="'yes that's the one' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_uh_huh_yes(run_conversation, assert_and_record):
    """
    User says 'uh huh yes' to confirm fax on file.
    Verifies that a filler-prefixed affirmation ('uh huh') does not block
    extraction of contact_confirmed='yes'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "uh huh yes", "yes"],
        test_name="test_fax_confirmed_uh_huh_yes",
        scenario="'uh huh yes' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_absolutely(run_conversation, assert_and_record):
    """
    User says 'absolutely' to confirm fax on file.
    Verifies that the emphatic single-word affirmative maps to
    contact_confirmed='yes' and the fax flow completes.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "absolutely", "yes"],
        test_name="test_fax_confirmed_absolutely",
        scenario="'absolutely' confirms fax on file → fax==FAX_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_confirmed_office_fax_line(run_conversation, assert_and_record):
    """
    Conversational: 'Yeah that's the fax number we use here at the office, go ahead and send it there.'
    User explicitly names the number as theirs and instructs dispatch.
    Verifies that a verbose but clear affirmation maps to contact_confirmed='yes'
    and the original fax on file is used without any SF update.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "Yeah that's the fax number we use here at the office, go ahead and send it there",
            "yes",
        ],
        test_name="test_fax_confirmed_office_fax_line",
        scenario="Verbose affirmation confirms fax on file → fax==FAX_ON_FILE, no SF update",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_confirmed_pretty_sure_main_fax(run_conversation, assert_and_record):
    """
    Conversational: 'That sounds right to me, I'm pretty sure that's our main fax line.'
    Net affirmative despite mild hedging ('pretty sure').  The contact-confirmation
    bias rule only fires for non-affirmations; 'pretty sure that's our main fax line'
    is still a clear enough affirmation that contact_confirmed='yes' is correct —
    the original fax on file must be used without triggering a new-contact request.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "That sounds right to me, I'm pretty sure that's our main fax line",
            "yes",
        ],
        test_name="test_fax_confirmed_pretty_sure_main_fax",
        scenario="Mildly hedged but net-affirmative → contact_confirmed=yes → fax==FAX_ON_FILE",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_confirmed_yes_exactly_send_whenever(run_conversation, assert_and_record):
    """
    Conversational: 'Yes please, that's exactly the number, feel free to send it whenever you're ready.'
    Emphatic multi-clause confirmation with a dispatch instruction.
    Verifies that the emphatic 'exactly' plus the caller's own dispatch instruction
    map to contact_confirmed='yes' and the original fax is used with no update.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "Yes please, that's exactly the number, feel free to send it whenever you're ready",
            "yes",
        ],
        test_name="test_fax_confirmed_yes_exactly_send_whenever",
        scenario="Emphatic multi-clause confirmation → contact_confirmed=yes → fax==FAX_ON_FILE",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, FAX_ON_FILE), f"fax=={FAX_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP D — Fax: bias rule fires → two-turn update flow
# ===========================================================================


@pytest.mark.live
async def test_fax_bias_i_think_so(run_conversation, assert_and_record):
    """
    User says 'I think so' to the fax readback — hedged affirmation.
    The contact-confirmation bias rule ('anything other than a clear affirmation
    → no') fires, causing the agent to ask for a new fax number.  Caller then
    provides NEW_FAX_SPOKEN on the next turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "I think so", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_i_think_so",
        scenario="'I think so' triggers bias rule → agent asks for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_maybe(run_conversation, assert_and_record):
    """
    User says 'maybe' to the fax readback — uncertain single-word response.
    The bias rule fires: 'maybe' is not a clear affirmation, so the agent
    requests a new fax number.  Caller supplies NEW_FAX_SPOKEN next.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "maybe", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_maybe",
        scenario="'maybe' triggers bias rule → agent asks for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_not_sure(run_conversation, assert_and_record):
    """
    User says 'not sure' to the fax readback — explicit uncertainty.
    The bias rule fires and the agent collects a replacement fax number.
    Verifies that 'not sure' (two words, no 'I') is still treated as non-affirmative.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "not sure", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_not_sure",
        scenario="'not sure' triggers bias rule → agent asks for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_no(run_conversation, assert_and_record):
    """
    User says bare 'no' to the fax readback — explicit rejection.
    contact_confirmed='no' → agent asks for a new fax number.
    Caller provides NEW_FAX_SPOKEN.  Verifies the clean two-turn decline path.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_no",
        scenario="'no' declines fax on file → agent asks for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_no_thats_wrong(run_conversation, assert_and_record):
    """
    User says 'no that's wrong' to the fax readback — negation with reinforcement.
    The negation is unambiguous; the bias rule fires on 'no' and the trailing
    clause provides no new number, so the agent must ask explicitly.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no that's wrong", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_no_thats_wrong",
        scenario="'no that's wrong' declines → agent asks for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_thats_the_old_one(run_conversation, assert_and_record):
    """
    User says 'that's the old one' to the fax readback — implicit decline via
    stale-data framing.  The bias rule should treat this as contact_confirmed='no'
    (not a clear affirmation) and prompt for a replacement.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "that's the old one", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_thats_the_old_one",
        scenario="'that's the old one' triggers bias rule → ask for new fax → NEW_FAX spoken → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_bias_actually_no(run_conversation, assert_and_record):
    """
    User says 'actually no' to the fax readback — emphatic retraction.
    Verifies that the leading 'actually' does not soften the 'no' enough
    to bypass the bias rule; contact_confirmed='no' and agent asks for new fax.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "actually no", NEW_FAX_SPOKEN, "yes"],
        test_name="test_fax_bias_actually_no",
        scenario="'actually no' declines fax on file → agent asks for new fax → NEW_FAX spoken",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_bias_not_entirely_sure(run_conversation, assert_and_record):
    """
    Conversational: 'Hmm I'm not entirely sure that's still the right number, we may have changed it.'
    Uncertainty ('not entirely sure') plus a possibility clause ('may have changed it').
    The bias rule must fire — this is not a clear affirmation — and the agent
    asks for a replacement.  Caller then provides NEW_FAX_SPOKEN.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "Hmm I'm not entirely sure that's still the right number, we may have changed it",
            NEW_FAX_SPOKEN,
            "yes",
        ],
        test_name="test_fax_bias_not_entirely_sure",
        scenario="Conversational uncertainty triggers bias rule → two-turn update → fax==NEW_FAX",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_bias_not_active_anymore(run_conversation, assert_and_record):
    """
    Conversational: 'You know what I don't think that fax number is active anymore,
    let me give you a better one.'
    Explicit declaration that the number is inactive plus a signal to replace it.
    The bias rule fires; the agent asks for the new number.  The trailing 'let me
    give you a better one' contains no digits, so the replacement must come from
    the following turn (NEW_FAX_SPOKEN).
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "You know what I don't think that fax number is active anymore, let me give you a better one",
            NEW_FAX_SPOKEN,
            "yes",
        ],
        test_name="test_fax_bias_not_active_anymore",
        scenario="Conversational 'not active anymore' triggers bias → two-turn update → fax==NEW_FAX",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP E — Fax: inline rejection + new value in SAME utterance
# ===========================================================================


@pytest.mark.live
async def test_fax_inline_no_spoken_digits(run_conversation, assert_and_record):
    """
    User says 'no it's six one seven five five five four two zero zero' — decline and
    spoken replacement in one utterance.  The inline-update rule must extract
    fax=NEW_FAX directly from this turn, skipping a separate ask.  The agent
    should proceed straight to dispatch with no second collection turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no it's six one seven five five five four two zero zero", "yes"],
        test_name="test_fax_inline_no_spoken_digits",
        scenario="Inline 'no it's <spoken>' → fax extracted in one turn → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_inline_actually_numeric(run_conversation, assert_and_record):
    """
    User says 'actually the fax should be 6175554200' — decline with numeric replacement.
    Verifies the inline-update rule handles a digit-string form (not spoken words)
    of the new fax number, extracting fax=NEW_FAX in a single turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "actually the fax should be 6175554200", "yes"],
        test_name="test_fax_inline_actually_numeric",
        scenario="Inline 'actually the fax should be 6175554200' → fax extracted → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_inline_wrong_correct_number_spoken(run_conversation, assert_and_record):
    """
    User says 'that's wrong, the correct number is six one seven five five five four two zero zero.'
    Verifies the inline-update rule with a longer preamble ('that's wrong, the correct
    number is') before the spoken replacement digits.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "that's wrong, the correct number is six one seven five five five four two zero zero",
            "yes",
        ],
        test_name="test_fax_inline_wrong_correct_number_spoken",
        scenario="'that's wrong, the correct number is <spoken>' → fax extracted in one turn",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_fax_inline_use_this_one_instead_numeric(run_conversation, assert_and_record):
    """
    User says 'use this one instead: 6175554200' — imperative redirect with numeric value.
    Verifies the inline-update rule handles a colon-separated numeric replacement,
    extracting fax=NEW_FAX without a second collection turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "use this one instead: 6175554200", "yes"],
        test_name="test_fax_inline_use_this_one_instead_numeric",
        scenario="'use this one instead: 6175554200' → fax extracted inline → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_inline_direct_line_spoken(run_conversation, assert_and_record):
    """
    Conversational: 'Oh wait no that's not right, the fax you want is
    six one seven five five five four two zero zero, that's my direct line.'
    Verifies the inline-update rule fires within a longer natural-speech utterance
    that opens with 'oh wait no' and closes with a contextual aside ('my direct line').
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "Oh wait no that's not right, the fax you want is six one "
            "seven five five five four two zero zero, that's my direct line",
            "yes",
        ],
        test_name="test_fax_inline_direct_line_spoken",
        scenario="Conversational inline decline + spoken replacement → fax==NEW_FAX in one turn",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_inline_new_machine_number_changed(run_conversation, assert_and_record):
    """
    Conversational: 'Actually we got a new fax machine so the number changed,
    it's now six one seven five five five four two zero zero.'
    Verifies the inline-update rule handles a contextual explanation ('new fax
    machine, number changed') followed by the spoken replacement.  The explanation
    contains no valid fax digits; only the trailing value should be extracted.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "fax",
            "Actually we got a new fax machine so the number changed, "
            "it's now six one seven five five five four two zero zero",
            "yes",
        ],
        test_name="test_fax_inline_new_machine_number_changed",
        scenario="Conversational 'new machine, number changed, it's now <spoken>' → fax==NEW_FAX inline",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP F — Email: bias rule fires → two-turn update flow
# ===========================================================================


@pytest.mark.live
async def test_email_bias_i_think_so(run_conversation, assert_and_record):
    """
    User says 'I think so' to the email readback — hedged affirmation.
    The contact-confirmation bias rule fires (same rule as fax): anything
    other than a clear affirmation → 'no'.  Agent asks for a new email.
    Caller provides NEW_EMAIL on the next turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "I think so", NEW_EMAIL, "yes"],
        test_name="test_email_bias_i_think_so",
        scenario="'I think so' triggers bias rule on email → ask for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_maybe(run_conversation, assert_and_record):
    """
    User says 'maybe' to the email readback — uncertain single-word response.
    Bias rule fires; agent requests a replacement email address.
    Mirrors the fax 'maybe' test to confirm the rule applies to both channels.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "maybe", NEW_EMAIL, "yes"],
        test_name="test_email_bias_maybe",
        scenario="'maybe' triggers bias rule on email → ask for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_no(run_conversation, assert_and_record):
    """
    User says bare 'no' to the email readback — explicit rejection.
    contact_confirmed='no' → agent asks for a new email address.
    Verifies the clean two-turn decline path for the email channel.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "no", NEW_EMAIL, "yes"],
        test_name="test_email_bias_no",
        scenario="'no' declines email on file → agent asks for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_no_use_different_one(run_conversation, assert_and_record):
    """
    User says 'no use a different one' — rejection with a vague replacement
    instruction but no actual address.  The inline-update rule does not apply
    (no email address present); the agent must ask explicitly for the new email.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "no use a different one", NEW_EMAIL, "yes"],
        test_name="test_email_bias_no_use_different_one",
        scenario="'no use a different one' (no address) → ask for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_thats_my_old_email(run_conversation, assert_and_record):
    """
    User says 'that's my old email' — implicit decline via stale-data framing.
    Mirrors the fax 'that's the old one' test.  The bias rule fires because
    'that's my old email' is not a clear affirmation; agent requests a new address.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "that's my old email", NEW_EMAIL, "yes"],
        test_name="test_email_bias_thats_my_old_email",
        scenario="'that's my old email' triggers bias rule → ask for new email → NEW_EMAIL",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_not_anymore(run_conversation, assert_and_record):
    """
    User says 'not anymore' to the email readback — terse implicit decline.
    Verifies that a two-word stale-data signal with no negation keyword still
    trips the bias rule and causes the agent to request a new email address.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "not anymore", NEW_EMAIL, "yes"],
        test_name="test_email_bias_not_anymore",
        scenario="'not anymore' triggers bias rule → ask for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_bias_actually_no(run_conversation, assert_and_record):
    """
    User says 'actually no' to the email readback — emphatic retraction.
    Mirrors the fax 'actually no' test for the email channel.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "actually no", NEW_EMAIL, "yes"],
        test_name="test_email_bias_actually_no",
        scenario="'actually no' declines email on file → ask for new email → NEW_EMAIL → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_bias_not_100_sure_active(run_conversation, assert_and_record):
    """
    Conversational: 'Hmm I'm not 100% sure that email is still active, I may have changed it.'
    Uncertainty ('not 100% sure') plus a possibility clause ('may have changed it').
    Mirrors the fax D8 test for the email channel.  Bias rule fires; agent
    requests a replacement email address.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            "Hmm I'm not 100% sure that email is still active, I may have changed it",
            NEW_EMAIL,
            "yes",
        ],
        test_name="test_email_bias_not_100_sure_active",
        scenario="Conversational uncertainty triggers "
        "bias rule on email → two-turn update → email==NEW_EMAIL",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_bias_dont_use_that_account(run_conversation, assert_and_record):
    """
    Conversational: 'I don't think I use that email account anymore to be honest,
    can I give you a different one?'
    Explicit statement that the address is no longer used, plus a request to
    substitute.  No address is provided in this utterance, so the bias rule
    fires and the agent must explicitly ask for the new email.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            "I don't think I use that email account anymore to be honest, can I give you a different one?",
            NEW_EMAIL,
            "yes",
        ],
        test_name="test_email_bias_dont_use_that_account",
        scenario="Conversational 'don't use that account' triggers bias → two-turn update → email==NEW_EMAIL",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP G — Email: inline rejection + new value in SAME utterance
# ===========================================================================


@pytest.mark.live
async def test_email_inline_no_use_new_address(run_conversation, assert_and_record):
    """
    User says 'no use emily.carter.new@gmail.com' — bare 'no' followed immediately
    by the replacement address.  The inline-update rule must extract email=NEW_EMAIL
    from this single utterance, skipping a separate ask.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", f"no use {NEW_EMAIL}", "yes"],
        test_name="test_email_inline_no_use_new_address",
        scenario=f"'no use {NEW_EMAIL}' → email extracted inline → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_inline_actually_send_to(run_conversation, assert_and_record):
    """
    User says 'actually send it to emily.carter.new@gmail.com' — redirect
    with the replacement address in one utterance.  Verifies the inline-update
    rule handles the 'actually send it to <email>' pattern.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", f"actually send it to {NEW_EMAIL}", "yes"],
        test_name="test_email_inline_actually_send_to",
        scenario=f"'actually send it to {NEW_EMAIL}' → email extracted inline → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_inline_wrong_use_instead(run_conversation, assert_and_record):
    """
    User says 'that's the wrong email, use emily.carter.new@gmail.com instead' —
    explicit rejection with a replacement and trailing 'instead'.  Verifies the
    inline-update rule extracts the address even with surrounding context words.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", f"that's the wrong email, use {NEW_EMAIL} instead", "yes"],
        test_name="test_email_inline_wrong_use_instead",
        scenario=f"'wrong email, use {NEW_EMAIL} instead' → email extracted inline → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_inline_bounces_current_one(run_conversation, assert_and_record):
    """
    Conversational: 'Oh no that email bounces now, just send it to
    emily.carter.new@gmail.com, that's my current one.'
    Verifies the inline-update rule fires within a longer natural-speech
    utterance that opens with an explanation ('bounces now') and closes
    with a contextual aside ('my current one').
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            f"Oh no that email bounces now, just send it to {NEW_EMAIL}, that's my current one",
            "yes",
        ],
        test_name="test_email_inline_bounces_current_one",
        scenario=f"Conversational 'bounces now, send to {NEW_EMAIL}' → email extracted inline",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_inline_switched_to_new_address(run_conversation, assert_and_record):
    """
    Conversational: 'Actually I switched to a new email address, it's
    emily.carter.new@gmail.com, please use that one.'
    Verifies the inline-update rule handles an explanation-first form
    ('switched to a new email address') followed by the replacement address
    and a polite instruction ('please use that one').
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            f"Actually I switched to a new email address, it's {NEW_EMAIL}, please use that one",
            "yes",
        ],
        test_name="test_email_inline_switched_to_new_address",
        scenario=f"Conversational 'switched to {NEW_EMAIL}' → email extracted inline → dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP H — Benefits offer response variations
# ===========================================================================


@pytest.mark.live
async def test_benefits_fax_yes(run_conversation, assert_and_record):
    """
    Fax happy path; user says bare 'yes' to the benefits offer.
    Baseline test confirming proactive_offer_available=True is written into
    state when the simplest affirmation follows a dispatched fax.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "yes"],
        test_name="test_benefits_fax_yes",
        scenario="Fax confirmed → 'yes' to benefits offer → proactive_offer_available=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, True), "proactive_offer_available==True"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_fax_no(run_conversation, assert_and_record):
    """
    Fax happy path; user says bare 'no' to the benefits offer.
    Baseline test confirming proactive_offer_available=False is written into
    state when the caller declines following a dispatched fax.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "no"],
        test_name="test_benefits_fax_no",
        scenario="Fax confirmed → 'no' to benefits offer → proactive_offer_available=False",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, False), "proactive_offer_available==False"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_email_yes_please(run_conversation, assert_and_record):
    """
    Email happy path; user says 'yes please' to the benefits offer.
    Verifies that a polite affirmation with 'please' still maps to
    benefits_response='yes' and proactive_offer_available=True.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "yes please"],
        test_name="test_benefits_email_yes_please",
        scenario="Email confirmed → 'yes please' to benefits offer → proactive_offer_available=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, True), "proactive_offer_available==True"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_email_no_thank_you(run_conversation, assert_and_record):
    """
    Email happy path; user says 'no thank you' to the benefits offer.
    Verifies that a polite decline ('no thank you') still maps to
    benefits_response='no' and proactive_offer_available=False.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "no thank you"],
        test_name="test_benefits_email_no_thank_you",
        scenario="Email confirmed → 'no thank you' to benefits offer → proactive_offer_available=False",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, False), "proactive_offer_available==False"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_yes_sounds_useful(run_conversation, assert_and_record):
    """
    Conversational yes: 'Oh that sounds really useful actually, yes please send me the details.'
    Verifies that an enthusiastic multi-clause acceptance with 'oh' lead-in and
    a trailing action request ('send me the details') maps to benefits_response='yes'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + ["fax", "yes", "Oh that sounds really useful actually, yes please send me the details"],
        test_name="test_benefits_conversational_yes_sounds_useful",
        scenario="Conversational yes to benefits → proactive_offer_available=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, True), "proactive_offer_available==True"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_no_all_good(run_conversation, assert_and_record):
    """
    Conversational no: 'No I think I'm all good for now, I don't need that right now.'
    Verifies that a verbose decline with hedges ('I think', 'for now') and
    a reason clause still maps unambiguously to benefits_response='no'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + ["email", "yes", "No I think I'm all good for now, I don't need that right now"],
        test_name="test_benefits_conversational_no_all_good",
        scenario="Conversational no to benefits → proactive_offer_available=False",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, False), "proactive_offer_available==False"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_yes_love_to_know(run_conversation, assert_and_record):
    """
    Conversational yes: 'Sure why not, I'd love to know more about what's covered.'
    Verifies that an indirect acceptance ('sure why not') with an elaborating clause
    maps to benefits_response='yes' and proactive_offer_available=True.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "Sure why not, I'd love to know more about what's covered"],
        test_name="test_benefits_conversational_yes_love_to_know",
        scenario="Conversational indirect yes → proactive_offer_available=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, True), "proactive_offer_available==True"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_no_know_benefits(run_conversation, assert_and_record):
    """
    Conversational no: 'Thanks but I already know my benefits pretty well, no need.'
    Verifies that a polite decline with a justification clause and trailing
    'no need' maps to benefits_response='no' and proactive_offer_available=False.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + ["email", "yes", "Thanks but I already know my benefits pretty well, no need"],
        test_name="test_benefits_conversational_no_know_benefits",
        scenario="Conversational polite no with justification → proactive_offer_available=False",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, False), "proactive_offer_available==False"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_ambiguous_clarify_then_yes(run_conversation, assert_and_record):
    """
    User asks 'what does that cover?' (ambiguous — not a yes/no) then 'yes'.
    Verifies the two-turn recovery path: the agent re-asks the benefits offer
    after the clarifying question, and the follow-up 'yes' resolves to True.
    The extraction prompt specifies clarification requests are not yes/no answers.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "what does that cover?", "yes"],
        test_name="test_benefits_ambiguous_clarify_then_yes",
        scenario="Clarifying question on benefits offer → re-ask → 'yes' → proactive_offer_available=True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, True), "proactive_offer_available==True"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_ambiguous_hmm_then_no(run_conversation, assert_and_record):
    """
    User says 'hmm' (genuine non-answer) then 'no'.
    Verifies the two-turn recovery path: 'hmm' triggers neither yes nor no,
    agent re-asks, and the follow-up 'no' resolves to proactive_offer_available=False.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "hmm", "no"],
        test_name="test_benefits_ambiguous_hmm_then_no",
        scenario="'hmm' non-answer on benefits offer → re-ask → 'no' → proactive_offer_available=False",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_proactive_offer_available(record, False), "proactive_offer_available==False"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP I — Delivery method ambiguous then valid
# ===========================================================================


@pytest.mark.live
async def test_delivery_method_ambiguous_hmm_then_fax(run_conversation, assert_and_record):
    """
    User says 'hmm' (genuine non-answer) when asked for delivery method,
    then 'fax' on the next turn.  Verifies the ambiguous-then-valid recovery
    path: agent re-asks after the non-answer, and 'fax' is accepted.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["hmm", "fax", "yes", "yes"],
        test_name="test_delivery_method_ambiguous_hmm_then_fax",
        scenario="'hmm' non-answer → re-ask → 'fax' → delivery_method=fax → full happy path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_delivery_method_ambiguous_not_sure_then_email(run_conversation, assert_and_record):
    """
    User says 'I'm not sure' when asked for delivery method, then 'email'.
    Verifies the ambiguous recovery path for the email channel: agent re-asks
    after the uncertain response and accepts 'email' on the second attempt.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["I'm not sure", "email", "yes", "no"],
        test_name="test_delivery_method_ambiguous_not_sure_then_email",
        scenario="'I'm not sure' → re-ask → 'email' → delivery_method=email → happy path",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_ambiguous_what_are_options_then_fax(run_conversation, assert_and_record):
    """
    User asks 'what are my options?' when asked for delivery method — an
    off-topic-adjacent clarifying question.  After the agent re-asks (or
    answers and re-asks), the caller chooses 'fax'.  Verifies recovery from
    a question response rather than a channel preference.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["what are my options?", "fax", "yes", "yes"],
        test_name="test_delivery_method_ambiguous_what_are_options_then_fax",
        scenario="'what are my options?' → agent re-asks → 'fax' → delivery_method=fax",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "fax"), "delivery_method==fax"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_ambiguous_either_fine_then_email(run_conversation, assert_and_record):
    """
    User says 'either is fine' when asked for delivery method — a non-choice
    that the extraction prompt marks as genuinely indeterminate.  The agent
    must re-ask; the caller then chooses 'email'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["either is fine", "email", "yes", "no"],
        test_name="test_delivery_method_ambiguous_either_fine_then_email",
        scenario="'either is fine' → ambiguous → re-ask → 'email' → delivery_method=email",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_delivery_method_ambiguous_conversational_then_email(run_conversation, assert_and_record):
    """
    Conversational: user says 'Oh gosh I haven't thought about that' then
    'let's go with email I guess.'  Verifies the two-turn recovery path when
    the first response is a genuine non-answer followed by a casual, hedged
    email preference ('let's go with email I guess') on the second turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + ["Oh gosh I haven't thought about that", "let's go with email I guess", "yes", "no"],
        test_name="test_delivery_method_ambiguous_conversational_then_email",
        scenario="Conversational non-answer → hedged 'let's go with email I guess' → delivery_method=email",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP J — Guard triggers
# ===========================================================================


@pytest.mark.live
async def test_guard_transfer_before_method(run_conversation, assert_and_record):
    """
    Transfer request fires before the delivery method is collected.
    After the full prefix the agent asks fax or email; the caller requests
    a transfer instead.  TRANSFER_REQUEST guard has highest priority and
    must escalate immediately regardless of where in the flow the agent is.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["transfer me to a person"],
        test_name="test_guard_transfer_before_method",
        scenario="TRANSFER_REQUEST before delivery_method collected → escalation_agent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_guard_transfer_during_fax_confirmation(run_conversation, assert_and_record):
    """
    Transfer request fires during the fax_confirmed slot (after delivery_method
    collected).  Verifies that TRANSFER_REQUEST guard preempts the contact
    confirmation flow mid-way and escalates immediately.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "actually can I speak to someone please"],
        test_name="test_guard_transfer_during_fax_confirmation",
        scenario="TRANSFER_REQUEST during fax_confirmed collection → escalation_agent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_guard_abuse_during_fax_confirmation(run_conversation, assert_and_record):
    """
    Abusive language during fax_confirmed collection triggers the ABUSE guard.
    Verifies that the guard fires inside DeliveryManagementAgent (not just in
    earlier agents) and routes to escalation_agent.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "you're fucking useless"],
        test_name="test_guard_abuse_during_fax_confirmation",
        scenario="ABUSE guard fires during fax_confirmed collection → escalation_agent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
        ],
    )


@pytest.mark.live
async def test_guard_self_harm_during_email_confirmation(run_conversation, assert_and_record):
    """
    Self-harm signal during email_confirmed collection triggers the SELF_HARM guard.
    Verifies compassionate escalation fires inside DeliveryManagementAgent.
    The agent message must contain 'support' or 'help' — confirming a supportive
    response rather than a generic error message.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "I want to end my life"],
        test_name="test_guard_self_harm_during_email_confirmation",
        scenario="SELF_HARM guard fires during email_confirmed → compassionate escalation",
    )

    def _assert_supportive_message(rec: ConversationRecord) -> None:
        all_msgs = " ".join((t.agent_message or "").lower() for t in rec.turns)
        assert "support" in all_msgs or "help" in all_msgs, (
            f"Expected agent message to contain 'support' or 'help'. Full transcript: {all_msgs[:500]!r}"
        )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: _assert_supportive_message(record), "supportive_message_present"),
        ],
    )


# ===========================================================================
# GROUP K — Invalid contact then valid
# ===========================================================================


@pytest.mark.live
@pytest.mark.slow
async def test_fax_invalid_too_short_then_valid(run_conversation, assert_and_record):
    """
    Caller declines fax on file, provides '123' (too short — fails 10-digit rule),
    then provides NEW_FAX_SPOKEN on the next turn.  Verifies that the fax slot
    rejects the invalid value, re-asks, and accepts the correct spoken digits.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no", "123", NEW_FAX_SPOKEN, "no"],
        test_name="test_fax_invalid_too_short_then_valid",
        scenario="'no' → '123' invalid (too short) → NEW_FAX_SPOKEN → fax==NEW_FAX, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_fax_invalid_letters_then_valid(run_conversation, assert_and_record):
    """
    Caller declines fax on file, provides 'abcde' (non-numeric — fails validation),
    then provides NEW_FAX_SPOKEN on the next turn.  Verifies that alphabetic input
    is rejected by the fax slot validator and a re-ask fires.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no", "abcde", NEW_FAX_SPOKEN, "no"],
        test_name="test_fax_invalid_letters_then_valid",
        scenario="'no' → 'abcde' invalid (letters) → NEW_FAX_SPOKEN → fax==NEW_FAX, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_invalid_no_at_then_valid(run_conversation, assert_and_record):
    """
    Caller declines email on file, provides 'emailwithnoat' (missing '@' — fails
    email validation), then provides NEW_EMAIL.  Verifies that the email slot
    rejects a value without '@' and the correct address is accepted next.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "no", "emailwithnoat", NEW_EMAIL, "no"],
        test_name="test_email_invalid_no_at_then_valid",
        scenario="'no' → 'emailwithnoat' invalid (no @) → NEW_EMAIL → email==NEW_EMAIL, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_invalid_partial_then_valid(run_conversation, assert_and_record):
    """
    Caller declines email on file, provides 'emily@' (missing domain — fails
    email validation), then provides NEW_EMAIL.  Verifies that a partial address
    with '@' but no domain is rejected by the email slot validator.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "no", "emily@", NEW_EMAIL, "no"],
        test_name="test_email_invalid_partial_then_valid",
        scenario="'no' → 'emily@' invalid (no domain) → NEW_EMAIL → email==NEW_EMAIL, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, NEW_EMAIL), f"email=={NEW_EMAIL}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP L — Contact slot exhaustion
# ===========================================================================


@pytest.mark.live
async def test_fax_slot_exhaustion_three_invalid(run_conversation, assert_and_record):
    """
    Caller declines fax on file then provides three consecutive invalid fax numbers
    ('123', 'abc', '99'), exhausting MAX_SLOT_ATTEMPTS for the fax slot.
    Verifies that the slot exhaustion path fires and the agent escalates rather
    than looping indefinitely.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no", "123", "abc", "99", "no benefits"],
        test_name="test_fax_slot_exhaustion_three_invalid",
        scenario="Fax slot: 3 invalid values exhaust MAX_SLOT_ATTEMPTS → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
        ],
    )


@pytest.mark.live
async def test_email_slot_exhaustion_three_invalid(run_conversation, assert_and_record):
    """
    Caller declines email on file then provides three consecutive invalid email
    addresses ('notanemail', 'also@', 'stillbad'), exhausting MAX_SLOT_ATTEMPTS
    for the email slot.  Verifies escalation fires after three failures without
    a valid '@domain' address ever being accepted.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "no", "notanemail", "also@", "stillbad"],
        test_name="test_email_slot_exhaustion_three_invalid",
        scenario="Email slot: 3 invalid addresses exhaust MAX_SLOT_ATTEMPTS → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
        ],
    )


@pytest.mark.live
async def test_fax_confirmed_slot_exhaustion_unclear(run_conversation, assert_and_record):
    """
    Three consecutive unclear responses to the fax readback ('dunno' × 3)
    that are neither yes/no nor a replacement number.  The fax_confirmed slot
    retries twice then exhausts, triggering escalation via MSG_CONTACT_EXHAUST.
    Verifies the exhaustion path in the fax_confirmed branch of agent.py.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "dunno", "dunno", "dunno"],
        test_name="test_fax_confirmed_slot_exhaustion_unclear",
        scenario="fax_confirmed: 3 unclear 'dunno' responses exhaust slot → escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
        ],
    )


# ===========================================================================
# GROUP M — Delivery method slot exhaustion
# ===========================================================================


@pytest.mark.live
async def test_delivery_method_slot_exhaustion(run_conversation, assert_and_record):
    """
    Three consecutive ambiguous non-answers to the delivery method question
    ('I don't know', 'not sure', 'whatever') exhaust MAX_SLOT_ATTEMPTS for the
    delivery_method slot.  Verifies that the agent escalates after three failures
    and that delivery_management_agent was the active agent at the time.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["I don't know", "not sure", "whatever"],
        test_name="test_delivery_method_slot_exhaustion",
        scenario="delivery_method: 3 non-answers exhaust slot → escalation from delivery_management_agent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_delivery_management_was_active(record), "delivery_management_was_active"),
        ],
    )


# ===========================================================================
# GROUP N — Email confirmed clear affirmations
# ===========================================================================


@pytest.mark.live
async def test_email_confirmed_yes(run_conversation, assert_and_record):
    """
    User says bare 'yes' to confirm email on file.
    Baseline test for the email_confirmed slot — mirrors C1 for the email channel.
    Verifies contact_confirmed='yes' leads to dispatch with the original email.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "no"],
        test_name="test_email_confirmed_yes",
        scenario="'yes' confirms email on file → email==EMAIL_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_confirmed_correct(run_conversation, assert_and_record):
    """
    User says 'correct' to confirm email on file.
    Verifies the single-word affirmative normalizes to contact_confirmed='yes'
    for the email channel — mirrors C2.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "correct", "no"],
        test_name="test_email_confirmed_correct",
        scenario="'correct' confirms email on file → email==EMAIL_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_confirmed_thats_right(run_conversation, assert_and_record):
    """
    User says 'that's right' to confirm email on file.
    Verifies the demonstrative affirmative maps to contact_confirmed='yes'
    for the email channel — mirrors C3.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "that's right", "no"],
        test_name="test_email_confirmed_thats_right",
        scenario="'that's right' confirms email on file → email==EMAIL_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_confirmed_yep_thats_my_email(run_conversation, assert_and_record):
    """
    User says 'yep that's my email' to confirm — colloquial affirmative with
    a possessive clause.  Verifies that the trailing 'that's my email' does
    not introduce ambiguity when the leading word is a clear 'yep'.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yep that's my email", "no"],
        test_name="test_email_confirmed_yep_thats_my_email",
        scenario="'yep that's my email' confirms email on file → email==EMAIL_ON_FILE",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_confirmed_yes_that_email_is_fine(run_conversation, assert_and_record):
    """
    User says 'yes that email is fine' — affirmation with a mild qualifier ('fine').
    Verifies that 'fine' as a quality qualifier does not weaken the leading 'yes'
    into ambiguity; the bias rule should not fire.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes that email is fine", "no"],
        test_name="test_email_confirmed_yes_that_email_is_fine",
        scenario="'yes that email is fine' confirms email on file → email==EMAIL_ON_FILE",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_email_confirmed_sure(run_conversation, assert_and_record):
    """
    User says bare 'sure' to confirm email on file — casual single-word affirmative.
    Verifies that 'sure' (which is not 'yes') normalizes to contact_confirmed='yes'
    in the email confirmation context.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "sure", "no"],
        test_name="test_email_confirmed_sure",
        scenario="'sure' confirms email on file → email==EMAIL_ON_FILE, dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_confirmed_check_every_day(run_conversation, assert_and_record):
    """
    Conversational: 'Yes that's my email address, I check it every day so I'll see it right away.'
    Verbose affirmation with a usage-habit clause.  Verifies the original email
    on file is used without update and that the trailing reassurance does not
    introduce ambiguity.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            "Yes that's my email address, I check it every day so I'll see it right away",
            "no",
        ],
        test_name="test_email_confirmed_check_every_day",
        scenario="Verbose affirmation with usage habit → email==EMAIL_ON_FILE, no update",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_email_confirmed_notifications_on_phone(run_conversation, assert_and_record):
    """
    Conversational: 'Yeah that should work fine for me, I'll get notifications on my phone.'
    Net affirmative ('yeah') despite 'should work fine' framing and a trailing
    aside about phone notifications.  Verifies the bias rule does not fire —
    the leading 'yeah' is a clear enough affirmation and email==EMAIL_ON_FILE.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "email",
            "Yeah that should work fine for me, I'll get notifications on my phone",
            "no",
        ],
        test_name="test_email_confirmed_notifications_on_phone",
        scenario="'Yeah that should work fine' → net affirmative → email==EMAIL_ON_FILE, no bias",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_email_used(record, EMAIL_ON_FILE), f"email=={EMAIL_ON_FILE}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
            (lambda: assert_delivery_method(record, "email"), "delivery_method==email"),
            (lambda: assert_benefits_offer_made(record), "benefits_offer_made==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP O — Latency benchmarks
# ===========================================================================


@pytest.mark.live
async def test_latency_fax_happy_path(run_conversation, assert_and_record):
    """
    Latency benchmark for the fax happy path (shortest delivery management flow).
    Drives FULL_PREFIX + fax choice + contact confirm + benefits yes.
    Asserts p50 ≤ _LATENCY_P50_SEC and p95 ≤ _LATENCY_P95_SEC across all turns,
    and that the provider list was actually sent (not a short-circuit).
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "yes", "yes"],
        test_name="test_latency_fax_happy_path",
        scenario="Latency benchmark: fax happy path — p50/p95 within threshold",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
        ],
    )


@pytest.mark.live
async def test_latency_email_happy_path(run_conversation, assert_and_record):
    """
    Latency benchmark for the email happy path.
    Drives FULL_PREFIX + email choice + contact confirm + benefits no.
    Asserts p50 ≤ _LATENCY_P50_SEC and p95 ≤ _LATENCY_P95_SEC across all turns.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["email", "yes", "no"],
        test_name="test_latency_email_happy_path",
        scenario="Latency benchmark: email happy path — p50/p95 within threshold",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
        ],
    )


@pytest.mark.live
async def test_latency_fax_two_turn_update(run_conversation, assert_and_record):
    """
    Latency benchmark for the fax two-turn update path (longer than happy path).
    Drives FULL_PREFIX + fax + decline + NEW_FAX_SPOKEN + benefits yes.
    This path incurs an extra LLM call for the fax slot collection; the p95
    threshold captures the tail latency of the additional turn.
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["fax", "no", NEW_FAX_SPOKEN, "yes"],
        test_name="test_latency_fax_two_turn_update",
        scenario="Latency benchmark: fax two-turn update — p50/p95 within threshold",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<={_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<={_LATENCY_P95_SEC}s"),
            (lambda: assert_fax_used(record, NEW_FAX), f"fax=={NEW_FAX}"),
            (lambda: assert_provider_list_sent(record), "provider_list_sent==True"),
        ],
    )
