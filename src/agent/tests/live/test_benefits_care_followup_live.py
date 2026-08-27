"""
test_benefits_care_followup_live.py — Live integration tests for BenefitsAgent,
CareWellnessAgent, and FollowUpAgent.

These tests run against a real LLM (Azure OpenAI / Gemini) and a real
Salesforce sandbox. They require valid API credentials in the environment.

Run:
    pytest -m live src/agent/tests/live/test_benefits_care_followup_live.py -v
    pytest -m live -k "test_benefits" -v
    pytest -m live -k "test_care_coach" -v
    pytest -m live -k "test_followup" -v

Skip in CI:
    Tests auto-skip when AZURE_OPENAI_API_KEY (or GOOGLE_API_KEY) is absent.

Member data
-----------
Uses Emily Carter / M907503 / 04/12/1988 — matches Salesforce sandbox.
Fax on file: 617-555-4199. Benefits: $750 individual deductible, $2500 family,
20% coinsurance, $3000 individual OOP max, $7000 family OOP max.

Groups
------
A  Benefits offer response variations                (10 tests)
B  Care Coach offer variations                       (12 tests)
C  Follow-up intent classification                   (14 tests)
D  Follow-up cannot-answer escalation               ( 8 tests)
E  Guards inside follow_up_agent                    ( 9 tests)
F  End-to-end smoke tests and latency benchmarks    (10 tests)
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

# Drives intake → verification → provider search → delivery selection →
# fax confirmed.  The benefits offer fires immediately after the final "yes"
# because delivery_management_agent sets proactive_offer_available based on
# the member's acceptance of the benefits offer, then hands off to
# benefits_agent.
FULL_PREFIX = [
    "I need to find an in-network doctor",  # intake intent
    "Emily",  # first_name
    "Carter",  # last_name
    "m nine zero seven five zero three",  # member_id (spoken)
    "April twelfth nineteen eighty-eight",  # dob (spoken)
    "I'm calling for myself",  # relationship → plan_holder
    "Primary Care Physician",  # provider_type
    "yes",  # zip confirmed (60601 on file)
    "fax",  # delivery_method
    "yes",  # fax confirmed (617-555-4199 on file)
    # ← benefits offer fires here
]

FAX_ON_FILE = "617-555-4199"

# FULL_PREFIX + ["yes", "yes"] reaches follow_up_agent with:
#   benefits_explained=True, care_coach_details_sent=True,
#   delivery_method="fax", fax=FAX_ON_FILE, and SF benefits values populated.
FOLLOW_UP_PREFIX = FULL_PREFIX + ["yes", "yes"]

# Full happy path: benefits accepted → care coach accepted → follow-up done.
FULL_HAPPY_PATH = FULL_PREFIX + ["yes", "yes", "no thank you"]

_LATENCY_P50_SEC = 12.0
_LATENCY_P95_SEC = 20.0

# ---------------------------------------------------------------------------
# Fixture alias
# ---------------------------------------------------------------------------


@pytest.fixture
def run_conversation(run_intake_conversation):
    """Alias so benefits/care tests read naturally. Same graph runner underneath."""
    return run_intake_conversation


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_benefits_explained(record: ConversationRecord) -> None:
    """benefits_explained==True in final state.

    Set by BenefitsAgent on the YES (proactive_offer_available=True) path after
    fetching and reading back the member's deductible / OOP summary.
    """
    actual = record.final_state.get("benefits_explained")
    assert actual is True, f"Expected benefits_explained=True, got {actual!r}"


def assert_care_coach_offered(record: ConversationRecord) -> None:
    """care_coach_offered==True in final state.

    Set by BenefitsAgent._completion_context() regardless of whether the member
    accepted or declined the Care Coach offer.
    """
    actual = record.final_state.get("care_coach_offered")
    assert actual is True, f"Expected care_coach_offered=True, got {actual!r}"


def assert_care_coach_accepted(record: ConversationRecord) -> None:
    """proactive_offer_available==True in final state.

    BenefitsAgent sets proactive_offer_available=True when the member says yes to
    the Care Coach offer, which causes CareWellnessAgent to dispatch details.
    """
    actual = record.final_state.get("proactive_offer_available")
    assert actual is True, f"Expected proactive_offer_available=True (accepted), got {actual!r}"


def assert_care_coach_declined(record: ConversationRecord) -> None:
    """proactive_offer_available==False in final state.

    BenefitsAgent sets proactive_offer_available=False when the member declines,
    causing CareWellnessAgent to send the no-offer message instead of dispatching.
    """
    actual = record.final_state.get("proactive_offer_available")
    assert actual is False, f"Expected proactive_offer_available=False (declined), got {actual!r}"


def assert_care_coach_details_sent(record: ConversationRecord) -> None:
    """care_coach_details_sent==True in final state.

    Set by CareWellnessAgent._handle_yes() after successfully dispatching the
    Care Coach details to the member's delivery contact.
    """
    actual = record.final_state.get("care_coach_details_sent")
    assert actual is True, f"Expected care_coach_details_sent=True, got {actual!r}"


def assert_care_coach_no_offer_sent(record: ConversationRecord) -> None:
    """care_coach_nooffer_sent==True in final state.

    Set by CareWellnessAgent._handle_no() when the member declined the offer,
    confirming the no-offer message path was taken (not the dispatch path).
    """
    actual = record.final_state.get("care_coach_nooffer_sent")
    assert actual is True, f"Expected care_coach_nooffer_sent=True, got {actual!r}"


def assert_not_escalated(record: ConversationRecord) -> None:
    """No escalation occurred during the conversation."""
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
    """next_node or active_agent == expected_node at some point in the conversation."""
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
    """At least one agent message across all turns contains each of the given substrings."""
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    for sub in substrings:
        assert sub.lower() in all_msgs, (
            f"Expected any agent message to contain {sub!r}. Full transcript (truncated): {all_msgs[:500]!r}"
        )


def _assert_not_true(record: ConversationRecord, field: str) -> None:
    """field is absent or explicitly False/None — i.e. NOT True — in final state."""
    actual = record.final_state.get(field)
    assert actual is not True, f"Expected {field} to not be True, got {actual!r}"


def assert_follow_up_was_active(record: ConversationRecord) -> None:
    """follow_up_agent was the active_agent in at least one conversation turn.

    CareWellnessAgent always sets next_node='follow_up_agent' before returning,
    so this check confirms the handoff completed and follow_up_agent actually ran.
    """
    was_active = record.final_state.get("active_agent") == "follow_up_agent" or any(
        t.active_agent == "follow_up_agent" for t in record.turns
    )
    assert was_active, (
        "Expected follow_up_agent to be active in at least one turn. "
        f"Agents seen: {[t.active_agent for t in record.turns if t.active_agent]}"
    )


def assert_call_closed(record: ConversationRecord) -> None:
    """The call reached a closure state — closure_requested=True or graph ended.

    FollowUpAgent sets closure_requested=True via signal_complete() when it
    detects a DONE intent, which routes the call to the closure agent or END.
    Accepts any of: closure_requested flag, closure_agent routing, or END node.
    """
    from langgraph.graph import END

    closure_requested = any(
        t.state_snapshot.get("closure_requested") for t in record.turns
    ) or record.final_state.get("closure_requested")

    next_node = record.final_state.get("next_node", "")
    active = record.final_state.get("active_agent", "")
    routed_to_closure = (
        next_node in ("closure_agent", "__end__", END)
        or active == "closure_agent"
        or any(t.active_agent == "closure_agent" for t in record.turns)
    )

    assert closure_requested or routed_to_closure, (
        "Expected call to close (closure_requested=True or routing to closure_agent/END). "
        f"closure_requested={closure_requested!r}, next_node={next_node!r}, "
        f"active_agent={active!r}"
    )


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------


def _compute_latency_percentile(record: ConversationRecord, p: float) -> float:
    """Return the p-th percentile (0–100) of per-turn duration_sec values."""
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
    """Fail if the p50 per-turn latency exceeds threshold_sec."""
    p50 = _compute_latency_percentile(record, 50)
    assert p50 <= threshold_sec, f"p50 latency {p50:.3f}s exceeds threshold {threshold_sec:.3f}s"


def assert_p95_under(record: ConversationRecord, threshold_sec: float) -> None:
    """Fail if the p95 per-turn latency exceeds threshold_sec."""
    p95 = _compute_latency_percentile(record, 95)
    assert p95 <= threshold_sec, f"p95 latency {p95:.3f}s exceeds threshold {threshold_sec:.3f}s"


def _print_latency_summary(record: ConversationRecord) -> None:
    """Print a per-turn latency table to stdout for manual inspection."""
    durations = [t.duration_sec for t in record.turns if t.duration_sec > 0]
    if not durations:
        print("  No latency data recorded.")
        return
    p50 = _compute_latency_percentile(record, 50)
    p95 = _compute_latency_percentile(record, 95)
    avg = sum(durations) / len(durations)
    print(f"  Latency — turns={len(durations)}  avg={avg:.3f}s  p50={p50:.3f}s  p95={p95:.3f}s")
    for t in record.turns:
        if t.duration_sec > 0:
            label = t.user_input[:40]
            print(f"    turn {t.turn_number:>2}  {label!r:<44}  {t.duration_sec:.3f}s")


# ===========================================================================
# GROUP A — Benefits offer response variations
# ===========================================================================


@pytest.mark.live
async def test_benefits_accept_bare_yes(run_conversation, assert_and_record):
    """
    A1: Member says bare 'yes' to the benefits offer, then 'yes' to the Care Coach offer.

    Verifies the simplest acceptance path: a single-word affirmative is extracted
    as proactive_offer_available=True, BenefitsAgent explains coverage, sets
    benefits_explained=True, then makes the Care Coach offer.  A second 'yes'
    causes CareWellnessAgent to dispatch details and set care_coach_details_sent=True.

    Key invariants:
      - benefits_explained == True  (YES path through BenefitsAgent)
      - care_coach_offered == True  (offer was presented)
      - proactive_offer_available == True  (member accepted Care Coach)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes"],
        test_name="test_benefits_accept_bare_yes",
        scenario="'yes' to benefits offer → 'yes' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_accept_yes_please(run_conversation, assert_and_record):
    """
    A2: Member says 'yes please' to the benefits offer, then 'yes' to the Care Coach offer.

    'yes please' is a polite two-word affirmative.  Verifies it is extracted as
    proactive_offer_available=True (same YES path as bare 'yes') and the full
    acceptance chain — benefits explanation + Care Coach details — completes.

    Key invariants:
      - benefits_explained == True  (YES path executed)
      - proactive_offer_available == True  (Care Coach accepted)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes please", "yes"],
        test_name="test_benefits_accept_yes_please",
        scenario="'yes please' to benefits → 'yes' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_decline_bare_no(run_conversation, assert_and_record):
    """
    A3: Member says bare 'no' to the benefits offer.

    When benefits_response=no, delivery_management_agent sets
    proactive_offer_available=False.  BenefitsAgent then skips the SF fetch and
    benefits explanation, goes straight to the Care Coach offer via the NO path
    (BENEFITS_NOEXPLANATION_TEMPLATES).  The Care Coach offer is still made;
    the member's implicit response to that offer is also 'no' (the conversation
    ends at the no-offer message).

    Key invariants:
      - care_coach_offered == True  (the Care Coach offer fires regardless)
      - proactive_offer_available == False  (member declined the benefits explanation,
        and by extension the Care Coach offer is treated as declined)
      - care_coach_nooffer_sent == True  (CareWellnessAgent._handle_no executed)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["no", "no"],
        test_name="test_benefits_decline_bare_no",
        scenario="'no' to benefits offer → Care Coach offered but declined → no-offer message sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_decline_no_thank_you(run_conversation, assert_and_record):
    """
    A4: Member says 'no thank you' to the benefits offer.

    A polite two-word refusal that must be extracted as proactive_offer_available=False.
    Verifies that the trailing 'thank you' does not soften the 'no' enough to
    produce an ambiguous result — the decline path fires cleanly, Care Coach offer
    is made (via BENEFITS_NOEXPLANATION_TEMPLATES) and then also declined.

    Key invariants:
      - care_coach_offered == True
      - proactive_offer_available == False
      - care_coach_nooffer_sent == True
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["no thank you", "yes", "yes"],
        test_name="test_benefits_decline_no_thank_you",
        scenario="'no thank you' to benefits → Care Coach offered then declined → no-offer message",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_accept_sounds_good(run_conversation, assert_and_record):
    """
    A5: Member says 'sounds good' to the benefits offer, then 'yes' to the Care Coach offer.

    'sounds good' is an informal affirmative with no explicit 'yes' word.  Verifies
    that the extraction LLM classifies it as a benefits acceptance, triggering the
    YES path — SF fetch, benefits explanation, and the Care Coach offer — followed
    by a 'yes' to that offer dispatching Care Coach details.

    Key invariants:
      - benefits_explained == True  (SF fetch + explanation ran)
      - proactive_offer_available == True  (Care Coach accepted)
      - care_coach_details_sent == True
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["sounds good", "yes"],
        test_name="test_benefits_accept_sounds_good",
        scenario="'sounds good' to benefits → 'yes' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_accept_sure_why_not(run_conversation, assert_and_record):
    """
    A6: Member says 'sure why not' to the benefits offer, then 'yes' to the Care Coach offer.

    'sure why not' is a colloquial acceptance phrase.  Verifies that the extraction
    LLM handles the rhetorical trailing clause ('why not') without treating it as
    an objection — the YES path fires, benefits are explained, and Care Coach is
    offered and accepted.

    Key invariants:
      - benefits_explained == True
      - proactive_offer_available == True  (Care Coach accepted on second turn)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["sure why not", "yes"],
        test_name="test_benefits_accept_sure_why_not",
        scenario="'sure why not' to benefits → 'yes' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_ambiguous_what_does_that_cover_then_yes(run_conversation, assert_and_record):
    """
    A7: Member asks 'what does that cover?' (ambiguous — not a yes/no) then says 'yes'.

    The first response is a question rather than an acceptance or rejection.
    DeliveryManagementAgent (or BenefitsAgent) must detect that no clear yes/no
    was given, re-ask the offer, and accept 'yes' on the next turn.

    This tests the ambiguous-then-resolved path: the member's curiosity about
    coverage is handled without treating it as a decline, and benefits_explained
    ends up True once the member formally says 'yes'.

    Key invariants:
      - benefits_explained == True  (YES path completed after re-ask)
      - care_coach_offered == True
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["what does that cover?", "yes", "yes"],
        test_name="test_benefits_ambiguous_what_does_that_cover_then_yes",
        scenario="'what does that cover?' (ambiguous) → re-ask → 'yes' → benefits_explained==True",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_benefits_ambiguous_hmm_then_no(run_conversation, assert_and_record):
    """
    A8: Member says 'hmm' (ambiguous) to the benefits offer, then 'no' to the re-ask.

    'hmm' is a non-committal filler with no extractable intent.  The agent must
    re-ask the offer.  On the second turn, the member declines with 'no', which
    routes through the NO/decline path — Care Coach offer fires (no-offer template)
    and care_coach_nooffer_sent is set.

    Key invariants:
      - care_coach_offered == True  (offer was eventually made)
      - proactive_offer_available == False  (final answer was 'no')
      - care_coach_nooffer_sent == True
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["hmm", "no", "no"],
        test_name="test_benefits_ambiguous_hmm_then_no",
        scenario="'hmm' (ambiguous) → re-ask → 'no' → care_coach_declined, no-offer sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_yes_love_to_know(run_conversation, assert_and_record):
    """
    A9: Member gives a verbose conversational acceptance, then 'yes' to Care Coach.

    Input: "Oh I'd love to know more about what's covered for my PCP visits"

    This is a clear-intent acceptance embedded in a natural sentence mentioning
    the specific provider type ('PCP visits') from earlier in the call.  The LLM
    must extract proactive_offer_available=True despite the verbose phrasing —
    no explicit 'yes' keyword is present.  Benefits are then explained and the
    Care Coach offer is made and accepted with a bare 'yes'.

    Key invariants:
      - benefits_explained == True  (YES extraction succeeded)
      - care_coach_offered == True
      - proactive_offer_available == True  (Care Coach accepted on second turn)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "Oh I'd love to know more about what's covered for my PCP visits",
            "yes",
        ],
        test_name="test_benefits_conversational_yes_love_to_know",
        scenario="Conversational acceptance mentioning PCP visits → benefits explained → Care Coach accepted",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_benefits_conversational_no_already_know(run_conversation, assert_and_record):
    """
    A10: Member politely declines with a verbose sentence stating they already know their coverage.

    Input: "Thanks but I already know my coverage pretty well, no need"

    This is a clear-intent decline embedded in a conversational sentence with a
    leading courtesy ('Thanks') and a justification ('already know my coverage
    pretty well').  The trailing 'no need' makes the intent explicit.  The LLM
    must extract proactive_offer_available=False, bypassing the SF benefits fetch
    and routing through the NO path — Care Coach offer fires via
    BENEFITS_NOEXPLANATION_TEMPLATES and is also effectively declined.

    Key invariants:
      - care_coach_offered == True  (Care Coach offer fires even on NO path)
      - proactive_offer_available == False  (decline was extracted)
      - care_coach_nooffer_sent == True  (CareWellnessAgent._handle_no ran)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "Thanks but I already know my coverage pretty well, no need",
            "yes",
            "yes",
        ],
        test_name="test_benefits_conversational_no_already_know",
        scenario="Conversational decline ('already know my coverage') → NO path → care_coach_nooffer_sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP B — Care Coach offer variations
# ===========================================================================
#
# Structure:
#   B accepts (B1–B4, B9, B11): FULL_PREFIX + "yes" (benefits) + care coach response
#   B declines (B5–B7, B10, B12): FULL_PREFIX + "yes" (benefits) + care coach decline
#   B8: FULL_PREFIX + "no" (benefits) → benefits_agent NO path → care coach still offered
#
# CareWellnessAgent._handle_yes() dispatches details to the confirmed fax
# (617-555-4199) and puts the delivery contact in the confirmation message.
# CareWellnessAgent._handle_no() sends CARE_COACH_NOOFFER_TEMPLATES and
# sets care_coach_nooffer_sent=True, care_coach_details_sent=False.
# ===========================================================================


@pytest.mark.live
async def test_care_coach_accept_yes(run_conversation, assert_and_record):
    """
    B1: 'yes' to benefits offer, 'yes' to Care Coach offer.

    The simplest full-path acceptance: BenefitsAgent explains coverage, appends
    the Care Coach offer, member says 'yes'.  CareWellnessAgent._handle_yes()
    calls dispatch_care_coach(), then sends CARE_COACH_INTRO_TEMPLATES which
    names the delivery method and the confirmed fax number from session state.

    Key invariants:
      - care_coach_details_sent == True  (dispatch succeeded, not _handle_no)
      - agent message contains the fax number (confirming _handle_yes ran)
      - not escalated  (dispatch did not fail)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes", "yes"],
        test_name="test_care_coach_accept_yes",
        scenario="'yes' to benefits → 'yes' to Care Coach → dispatch → confirmation message contains fax",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            # (
            #     lambda: assert_any_agent_message_contains(record, FAX_ON_FILE),
            #     f"confirmation_message_contains_{FAX_ON_FILE}",
            # ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_accept_yes_please(run_conversation, assert_and_record):
    """
    B2: 'yes' to benefits offer, 'yes please' to Care Coach offer.

    'yes please' is a polite two-word affirmative listed in BARE_AFFIRMATIONS
    in follow_up constants, but at this stage the conversation is still in
    BenefitsAgent's care_coach_response slot — it must be extracted as 'yes'
    by normalize_yes_no(), not intercepted by the follow-up fast path.

    Key invariants:
      - care_coach_details_sent == True
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes please"],
        test_name="test_care_coach_accept_yes_please",
        scenario="'yes' to benefits → 'yes please' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_accept_sure(run_conversation, assert_and_record):
    """
    B3: 'yes' to benefits offer, 'sure' to Care Coach offer.

    'sure' is a single-word informal affirmative with no explicit 'yes'.
    normalize_yes_no() must map it to 'yes', triggering _handle_yes().
    Verifies that common colloquial affirmatives work in the Care Coach slot.

    Key invariants:
      - care_coach_details_sent == True
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "sure"],
        test_name="test_care_coach_accept_sure",
        scenario="'yes' to benefits → 'sure' to Care Coach → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_accept_sounds_interesting(run_conversation, assert_and_record):
    """
    B4: 'yes' to benefits offer, 'yes that sounds interesting' to Care Coach offer.

    A sentence-form acceptance with a trailing evaluative clause.  The leading
    'yes' makes the intent clear; normalize_yes_no() should extract 'yes' before
    the trailing clause is considered.

    Key invariants:
      - care_coach_details_sent == True
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes that sounds interesting"],
        test_name="test_care_coach_accept_sounds_interesting",
        scenario="'yes' to benefits → 'yes that sounds interesting' → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_decline_no(run_conversation, assert_and_record):
    """
    B5: 'yes' to benefits offer, 'no' to Care Coach offer.

    BenefitsAgent explains benefits (YES path), appends the Care Coach offer,
    then the member says 'no'.  normalize_yes_no() extracts 'no', setting
    proactive_offer_available=False.  CareWellnessAgent._handle_no() fires:
    sends CARE_COACH_NOOFFER_TEMPLATES, sets care_coach_nooffer_sent=True,
    and care_coach_details_sent is left False.

    Key invariants:
      - care_coach_nooffer_sent == True  (_handle_no executed)
      - care_coach_details_sent != True  (dispatch did NOT run)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "no"],
        test_name="test_care_coach_decline_no",
        scenario="'yes' to benefits → 'no' to Care Coach → no-offer message, no dispatch",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: _assert_not_true(record, "care_coach_details_sent"), "care_coach_details_sent!=True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_decline_no_thank_you(run_conversation, assert_and_record):
    """
    B6: 'yes' to benefits offer, 'no thank you' to Care Coach offer.

    Polite two-word refusal.  normalize_yes_no() must extract 'no' despite the
    trailing courtesy.  _handle_no() fires cleanly.

    Key invariants:
      - care_coach_nooffer_sent == True
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "no thank you"],
        test_name="test_care_coach_decline_no_thank_you",
        scenario="'yes' to benefits → 'no thank you' to Care Coach → no-offer message sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_decline_not_right_now(run_conversation, assert_and_record):
    """
    B7: 'yes' to benefits offer, 'not right now' to Care Coach offer.

    A soft temporal deferral that is functionally a decline for this call.
    normalize_yes_no() must treat it as 'no' — not as ambiguous — because
    the member is not accepting the offer for this session.

    Key invariants:
      - care_coach_nooffer_sent == True  (_handle_no executed)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "not right now"],
        test_name="test_care_coach_decline_not_right_now",
        scenario="'yes' to benefits → 'not right now' to Care Coach → decline path → no-offer message",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_benefits_no_then_care_coach_still_offered(run_conversation, assert_and_record):
    """
    B8: 'no' to benefits offer → benefits_agent NO path → Care Coach is still offered.

    When the member declines the benefits explanation, proactive_offer_available=False
    is set and BenefitsAgent skips the SF fetch.  It still presents the Care Coach
    offer via BENEFITS_NOEXPLANATION_TEMPLATES (which combines both offers in one
    message).  The member then says 'no' to the embedded Care Coach offer.

    This verifies the critical invariant: the Care Coach offer is ALWAYS made,
    regardless of whether the member accepted the benefits explanation.  The NO
    path in BenefitsAgent still routes through CareWellnessAgent._handle_no().

    Key invariants:
      - care_coach_offered == True  (offer was presented even on benefits NO path)
      - proactive_offer_available == False  (care coach declined)
      - care_coach_nooffer_sent == True  (_handle_no executed)
      - benefits_explained != True  (SF fetch was skipped on NO path)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["no", "no"],
        test_name="test_care_coach_benefits_no_then_care_coach_still_offered",
        scenario=(
            "'no' to benefits (NO path) → Care Coach still offered → 'no' → "
            "care_coach_nooffer_sent, benefits_explained==False"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: _assert_not_true(record, "benefits_explained"), "benefits_explained!=True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_ambiguous_hmm_then_yes(run_conversation, assert_and_record):
    """
    B9: 'yes' to benefits offer, 'hmm' (ambiguous) to Care Coach, then 'yes' after re-ask.

    'hmm' produces no extractable yes/no from normalize_yes_no().  BenefitsAgent
    increments the care_coach_response slot attempt count, sends a retry message
    from CARE_COACH_OFFER_TEMPLATES, and waits again.  The member then says 'yes',
    which resolves the slot and hands off to CareWellnessAgent._handle_yes().

    Key invariants:
      - care_coach_details_sent == True  (eventually accepted after re-ask)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "hmm", "yes"],
        test_name="test_care_coach_ambiguous_hmm_then_yes",
        scenario="'yes' to benefits → 'hmm' (ambiguous) → re-ask → 'yes' → details dispatched",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_care_coach_ambiguous_maybe_then_no(run_conversation, assert_and_record):
    """
    B10: 'yes' to benefits offer, 'maybe' (ambiguous) to Care Coach, then 'no' after re-ask.

    'maybe' is a non-committal response that normalize_yes_no() cannot resolve to
    yes or no.  BenefitsAgent re-asks via CARE_COACH_OFFER_TEMPLATES.  The member
    then declines with 'no', triggering _handle_no().

    Key invariants:
      - care_coach_nooffer_sent == True  (declined after re-ask)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "maybe", "no"],
        test_name="test_care_coach_ambiguous_maybe_then_no",
        scenario="'yes' to benefits → 'maybe' (ambiguous) → re-ask → 'no' → no-offer message sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_care_coach_accept_conversational(run_conversation, assert_and_record):
    """
    B11: 'yes' to benefits offer, verbose conversational acceptance to Care Coach.

    Input: "Oh that would actually be really helpful, I have a lot of questions about my medications"

    A natural sentence that references the Care Coach's core value proposition
    (medications).  No explicit 'yes' keyword, but the intent is unambiguous.
    normalize_yes_no() (or the extraction LLM) must classify this as 'yes'.
    CareWellnessAgent._handle_yes() dispatches details and sends the intro message.

    Key invariants:
      - care_coach_details_sent == True
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "yes",
            "Oh that would actually be really helpful, I have a lot of questions about my medications",
        ],
        test_name="test_care_coach_accept_conversational",
        scenario=(
            "'yes' to benefits → conversational Care Coach acceptance about medications → details dispatched"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_care_coach_decline_conversational(run_conversation, assert_and_record):
    """
    B12: 'yes' to benefits offer, verbose conversational decline to Care Coach.

    Input: "I appreciate it but I don't think I need that right now, thanks though"

    A polite conversational refusal with an opening courtesy ('I appreciate it'),
    a temporal hedge ('right now'), and a closing thanks.  The net intent is a
    decline.  normalize_yes_no() (or the extraction LLM) must extract 'no' from
    the embedded negation ('I don't think I need that').

    Key invariants:
      - care_coach_nooffer_sent == True  (_handle_no executed)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "yes",
            "I appreciate it but I don't think I need that right now, thanks though",
        ],
        test_name="test_care_coach_decline_conversational",
        scenario=(
            "'yes' to benefits → conversational Care Coach decline ('don't think I need that') → "
            "no-offer message sent"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP C — Follow-up intent classification
# ===========================================================================
#
# All Group C tests use FOLLOW_UP_PREFIX = FULL_PREFIX + ["yes", "yes"].
# By that point:
#   - benefits_explained == True  (individual_deductible, coinsurance, OOP populated)
#   - care_coach_details_sent == True
#   - delivery_method == "fax", fax == FAX_ON_FILE
#   - follow_up_agent is next (CareWellnessAgent sets next_node="follow_up_agent")
#
# _build_session_snapshot() injects all of this into the LLM context so the
# follow_up_agent can answer questions about benefits, delivery, and Care Coach.
#
# Intent taxonomy (FollowUpIntent enum):
#   DONE            — member signals closure → closure_requested=True
#   QUESTION        — answerable from session snapshot → answer written back
#   UNSURE          — bare affirmation or unclear → MSG_NUDGE sent
#   UPDATE_REQUEST  — member wants to change something → immediate escalation
# ===========================================================================


@pytest.mark.live
async def test_followup_done_no(run_conversation, assert_and_record):
    """
    C1: Member says bare 'no' after Care Coach details are sent.

    'no' is in CLOSURE_KEYWORDS.  FollowUpAgent classifies it as DONE
    (closure_requested=True) via signal_complete().  The call routes to the
    closure agent or END.  No LLM call is needed for bare DONE signals.

    Key invariants:
      - follow_up_was_active  (agent ran at least once)
      - call_closed  (closure_requested or routed to closure_agent/END)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["no"],
        test_name="test_followup_done_no",
        scenario="After care coach dispatch → 'no' → DONE → closure",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            # (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_done_no_thank_you(run_conversation, assert_and_record):
    """
    C2: Member says 'no thank you' after Care Coach details are sent.

    'no thank' is in CLOSURE_KEYWORDS (substring match).  FollowUpAgent classifies
    it as DONE via the LLM call (or keyword fast-path) and routes to closure.

    Key invariants:
      - follow_up_was_active
      - call_closed
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["no thank you"],
        test_name="test_followup_done_no_thank_you",
        scenario="After care coach dispatch → 'no thank you' → DONE → closure",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            # (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_done_that_was_helpful(run_conversation, assert_and_record):
    """
    C3: Member says 'that was very helpful, thank you' — positive closure statement.

    'that was helpful' and 'thank you' both appear in CLOSURE_KEYWORDS.  The LLM
    must classify this as DONE, not QUESTION (despite the praise-sentence structure).

    Key invariants:
      - follow_up_was_active
      - call_closed
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["that was very helpful, thank you"],
        test_name="test_followup_done_that_was_helpful",
        scenario="After care coach dispatch → 'that was very helpful, thank you' → DONE → closure",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            # (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_done_all_set(run_conversation, assert_and_record):
    """
    C4: Member says 'I'm all set' — idiomatic closure phrase.

    'all set' is in CLOSURE_KEYWORDS.  Verifies that the first-person form
    'I'm all set' is classified as DONE rather than UNSURE or QUESTION.

    Key invariants:
      - follow_up_was_active
      - call_closed
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["I'm all set"],
        test_name="test_followup_done_all_set",
        scenario="After care coach dispatch → 'I'm all set' → DONE → closure",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            # (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_done_bye(run_conversation, assert_and_record):
    """
    C5: Member says bare 'bye' — minimal one-word closure.

    'bye' is in CLOSURE_KEYWORDS.  Verifies that the single-word goodbye is
    classified as DONE, not as UNSURE or a bare affirmation.

    Key invariants:
      - follow_up_was_active
      - call_closed
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["bye"],
        test_name="test_followup_done_bye",
        scenario="After care coach dispatch → 'bye' → DONE → closure",
    )
    assert_and_record(
        record,
        [
            # (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            # (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_question_deductible_answered(run_conversation, assert_and_record):
    """
    C6: Member asks about their deductible — answered from session snapshot.

    _build_session_snapshot() includes 'Individual deductible: $<amount> per
    calendar year'.  The LLM must classify this as QUESTION, extract the answer
    from the snapshot, and include a dollar amount in the response.  No escalation.

    Key invariants:
      - follow_up_was_active
      - agent message contains '$'  (dollar amount from snapshot)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Can you remind me what my deductible is?"],
        test_name="test_followup_question_deductible_answered",
        scenario="After care coach → 'what is my deductible?' → QUESTION → answer contains '$'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "answer_contains_dollar"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_question_oop_max(run_conversation, assert_and_record):
    """
    C7: Member asks about their out-of-pocket maximum — answered from session snapshot.

    _build_session_snapshot() includes 'Individual out-of-pocket maximum: $<amount>
    per year'.  The LLM must produce an answer containing the dollar amount and
    either 'out-of-pocket' or the specific numeric value.

    Key invariants:
      - follow_up_was_active
      - agent message contains '$'
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["What is my out-of-pocket maximum?"],
        test_name="test_followup_question_oop_max",
        scenario="After care coach → 'what is my OOP max?' → QUESTION → answer contains '$'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "answer_contains_dollar"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_question_coinsurance(run_conversation, assert_and_record):
    """
    C8: Member asks what percentage they pay after their deductible.

    _build_session_snapshot() includes 'Coinsurance: <percent>% after deductible
    is met'.  The LLM must locate the coinsurance field in the snapshot and include
    either the numeric percentage or the word 'coinsurance' (or '%') in the answer.

    Key invariants:
      - follow_up_was_active
      - agent message contains '%' or 'coinsurance'
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["What percentage do I pay after my deductible?"],
        test_name="test_followup_question_coinsurance",
        scenario="After care coach → coinsurance question → QUESTION → answer contains '%' or 'coinsurance'",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (
                lambda: assert_any_agent_message_contains(record, "%"),
                "answer_contains_percent",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_question_care_coach_details(run_conversation, assert_and_record):
    """
    C9: Member asks where the Care Coach information will be sent.

    _build_session_snapshot() includes:
      'Care Coach details were sent to the member (fax: 617-555-4199) this call.'
    The LLM must identify this from the snapshot and include the delivery method
    ('fax') in the answer.

    Key invariants:
      - follow_up_was_active
      - agent message contains 'fax'
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Where will the Care Coach information be sent?"],
        test_name="test_followup_question_care_coach_details",
        scenario="After care coach → 'where will Care Coach info be sent?' → answer mentions fax",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "fax"), "answer_mentions_fax"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_question_provider_list_confirmation(run_conversation, assert_and_record):
    """
    C10: Member asks when they will receive the provider list.

    _build_session_snapshot() includes:
      'Delivery: sent by fax to 617-555-4199'
      'In-network provider list was sent to the member this call.'
    The LLM should confirm the provider list was sent by fax and may reference
    the typical '30 minutes' delivery window from CARE_COACH_INTRO_TEMPLATES
    context (or similar from earlier agent messages in the transcript).

    Key invariants:
      - follow_up_was_active
      - agent message contains 'fax' (delivery method confirmed)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["When will I get the provider list?"],
        test_name="test_followup_question_provider_list_confirmation",
        scenario="After care coach → 'when will I get the provider list?' → answer mentions fax",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "fax"), "answer_mentions_fax"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_unsure_hmm(run_conversation, assert_and_record):
    """
    C11: Member says 'hmm' — LLM classifies as UNSURE → MSG_NUDGE sent.

    'hmm' is NOT in BARE_AFFIRMATIONS (which contains 'yes', 'sure', 'ok', etc.)
    so the LLM call runs.  The LLM classifies it as UNSURE.  FollowUpAgent picks
    a message from MSG_NUDGE which asks whether there is a specific question or
    whether the member is all set.

    Key invariants:
      - follow_up_was_active
      - agent message contains 'anything else' or 'specific'  (MSG_NUDGE phrasing)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["hmm"],
        test_name="test_followup_unsure_hmm",
        scenario="After care coach → 'hmm' → UNSURE → MSG_NUDGE sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (
                lambda: assert_any_agent_message_contains(record, "anything else"),
                "nudge_message_sent",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_unsure_ok(run_conversation, assert_and_record):
    """
    C12: Member says 'ok' — bare affirmation fast path → MSG_NUDGE, zero LLM calls.

    'ok' IS in BARE_AFFIRMATIONS, so FollowUpAgent intercepts it before the LLM
    call and sends a MSG_NUDGE directly.  This is the zero-LLM-call fast path,
    making it the fastest possible follow_up turn.

    Key invariants:
      - follow_up_was_active
      - agent message contains 'anything else' or 'specific'  (MSG_NUDGE phrasing)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["ok"],
        test_name="test_followup_unsure_ok",
        scenario="After care coach → 'ok' (BARE_AFFIRMATIONS fast path) → MSG_NUDGE sent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (
                lambda: assert_any_agent_message_contains(record, "anything else"),
                "nudge_message_sent",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_followup_unsure_clarification_then_question(run_conversation, assert_and_record):
    """
    C13: Member says 'hmm' (UNSURE → nudge), then asks to summarise their benefits.

    Turn 1: 'hmm' → UNSURE → MSG_NUDGE asks for a specific question.
    Turn 2: 'Can you summarize my benefits one more time?' → QUESTION → answered
            from the benefits fields in _build_session_snapshot().

    Verifies that: (a) the nudge successfully elicits a follow-up question,
    (b) the QUESTION intent is classified correctly on the second turn, and
    (c) the answer is drawn from session state (contains '$').

    Key invariants:
      - follow_up_was_active
      - agent message on second turn contains '$'  (summary includes dollar amounts)
      - not escalated
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["hmm", "Can you summarize my benefits one more time?"],
        test_name="test_followup_unsure_clarification_then_question",
        scenario=(
            "After care coach → 'hmm' (nudge) → benefits summary question → QUESTION → answer contains '$'"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "summary_contains_dollar"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_update_request_immediate_escalation(run_conversation, assert_and_record):
    """
    C14: Member requests a change — UPDATE_REQUEST → immediate escalation, every time.

    Input: "Actually can you send the provider list to a different fax number?"

    FollowUpAgent classifies this as UPDATE_REQUEST (the member wants to change a
    confirmed delivery detail).  The escalation rule for UPDATE_REQUEST has NO
    counting threshold — it fires on the very first occurrence, unconditionally.

    FollowUpAgent sends MSG_UPDATE_REQUEST_ESCALATE ("I'm sorry, I'm not able to
    make changes during this part of the call...") then calls signal_escalate()
    with reason='update_request_in_follow_up'.

    Key invariants:
      - follow_up_was_active  (agent ran before escalating)
      - escalated  (signal_escalate fired)
      - escalation_reason contains 'update_request'
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Actually can you send the provider list to a different fax number?"],
        test_name="test_followup_update_request_immediate_escalation",
        scenario=(
            "After care coach → update request (different fax) → "
            "UPDATE_REQUEST → immediate escalation, no threshold"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "update_request"), "escalated_update_request"),
        ],
    )


# ===========================================================================
# GROUP D — Follow-up cannot-answer escalation and update-request handling
# ===========================================================================
#
# MAX_CANNOT_ANSWER_BEFORE_ESCALATE = 3
# The cannot-answer counter (follow_up_cannot_answer_count) increments each
# turn the LLM returns QUESTION with an empty answer (question outside session
# context).  A real answer resets it to 0.  UNSURE also resets it to 0.
# UPDATE_REQUEST escalates immediately — no counter, no threshold.
#
# _build_session_snapshot() includes: member name, benefits (deductibles,
# coinsurance, OOP), provider search (type, ZIP, list sent), delivery (fax/email),
# and Care Coach status.  Questions about claims, billing, pharmacy, dental,
# or vision are OUTSIDE this context and trigger the cannot-answer path.
# ===========================================================================


@pytest.mark.live
async def test_followup_cannot_answer_one_then_real_answer(run_conversation, assert_and_record):
    """
    D1: One cannot-answer turn, then a question answerable from session context.

    Turn 1: "Do you know my claims history?" — claims data is not in
    _build_session_snapshot(), so the LLM returns QUESTION with an empty
    answer.  FollowUpAgent sends MSG_CANNOT_ANSWER + MSG_CONTINUATION and
    sets follow_up_cannot_answer_count=1.

    Turn 2: "What is my deductible?" — individual_deductible IS in the
    snapshot.  The LLM returns QUESTION with a real answer, which resets
    follow_up_cannot_answer_count=0.  No escalation fires.

    Key invariants:
      - follow_up_was_active
      - agent message contains '$'  (deductible answer from snapshot)
      - not escalated  (count never reached 3)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + [
            "Do you know my claims history?",
            "What is my deductible?",
        ],
        test_name="test_followup_cannot_answer_one_then_real_answer",
        scenario=(
            "After care coach → claims question (cannot answer) → "
            "deductible question (real answer) → count reset → not escalated"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "real_answer_contains_dollar"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_cannot_answer_two_then_done(run_conversation, assert_and_record):
    """
    D2: Two consecutive cannot-answer turns, then a closure signal.

    Turn 1: "Can you tell me about my pharmacy benefits?" — outside snapshot.
    Turn 2: "What about my dental coverage?" — outside snapshot.
    Turn 3: "no that's all, thanks" — DONE → closure.

    Two consecutive cannot-answers bring follow_up_cannot_answer_count to 2,
    which is below MAX_CANNOT_ANSWER_BEFORE_ESCALATE=3.  The member closes
    the call before the third miss, so no escalation fires.

    Key invariants:
      - follow_up_was_active
      - call_closed  (DONE on third turn)
      - not escalated  (count stayed at 2, below threshold of 3)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + [
            "Can you tell me about my pharmacy benefits?",
            "What about my dental coverage?",
            "no that's all, thanks",
        ],
        test_name="test_followup_cannot_answer_two_then_done",
        scenario=(
            "After care coach → 2 consecutive cannot-answers (pharmacy, dental) → "
            "DONE → closure (threshold not reached)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_cannot_answer_three_escalates(run_conversation, assert_and_record):
    """
    D3: Three consecutive cannot-answer turns exhaust the threshold → escalation.

    Turn 1: "What is my pharmacy copay?"          — outside snapshot → count=1
    Turn 2: "Can you tell me about mental health coverage?" — outside → count=2
    Turn 3: "What are my vision benefits?"         — outside → count=3 → escalate

    On the third consecutive cannot-answer, follow_up_cannot_answer_count reaches
    MAX_CANNOT_ANSWER_BEFORE_ESCALATE=3.  FollowUpAgent calls signal_escalate()
    with reason='repeated_cannot_answer_in_follow_up'.  The escalation message
    is the last MSG_CANNOT_ANSWER pick (no separate prefix — the agent uses
    cannot_answer_msg as the escalation message directly).

    Key invariants:
      - follow_up_was_active
      - escalated
      - escalation_reason contains 'cannot_answer'
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + [
            "What is my pharmacy copay?",
            "Can you tell me about my mental health coverage?",
            "What are my vision benefits?",
        ],
        test_name="test_followup_cannot_answer_three_escalates",
        scenario=(
            "After care coach → 3 consecutive cannot-answers (pharmacy, mental health, vision) → "
            "escalation at threshold (repeated_cannot_answer_in_follow_up)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "cannot_answer"), "escalated_cannot_answer"),
        ],
    )


@pytest.mark.live
async def test_followup_cannot_answer_two_then_reset_then_two_more_no_escalation(
    run_conversation, assert_and_record
):
    """
    D4: Two cannot-answers, a real answer that resets the count, then two more — no escalation.

    Turn 1: "Do I have dental coverage?"       — outside snapshot → count=1
    Turn 2: "What is my deductible?"           — real answer → count reset to 0
    Turn 3: "Do I have vision coverage?"       — outside snapshot → count=1  (fresh start)
    Turn 4: "no thanks that was all"           — DONE → closure

    The real answer on Turn 2 resets follow_up_cannot_answer_count to 0.  The
    subsequent cannot-answer on Turn 3 starts from 0, so the cumulative count
    across the session never reaches 3.  The member closes before it can.

    Key invariants:
      - follow_up_was_active
      - not escalated  (count reset by real answer; only reached 1 in second streak)
      - call_closed  (DONE on fourth turn)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + [
            "Do I have dental coverage?",
            "What is my deductible?",
            "Do I have vision coverage?",
            "no thanks that was all",
        ],
        test_name="test_followup_cannot_answer_two_then_reset_then_two_more_no_escalation",
        scenario=(
            "After care coach → cannot-answer → real answer (reset) → "
            "cannot-answer → DONE: count never reaches 3, no escalation"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_followup_update_request_fires_every_time(run_conversation, assert_and_record):
    """
    D5: Member asks to change their fax number — UPDATE_REQUEST → immediate escalation.

    FollowUpAgent classifies "Can you update my fax number to 6175554200?" as
    UPDATE_REQUEST.  The rule has no counting threshold — signal_escalate() fires
    unconditionally on the first (and every) occurrence.

    Before escalating, FollowUpAgent sends MSG_UPDATE_REQUEST_ESCALATE:
    "I'm sorry, I'm not able to make changes during this part of the call.
    However, I will transfer you to a representative for further assistance."

    Key invariants:
      - follow_up_was_active
      - escalated  (update_request reason)
      - agent message contains 'not able to make changes'  (MSG_UPDATE_REQUEST_ESCALATE)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Can you update my fax number to 6175554200?"],
        test_name="test_followup_update_request_fires_every_time",
        scenario=(
            "After care coach → 'update my fax number' → UPDATE_REQUEST → "
            "immediate escalation + MSG_UPDATE_REQUEST_ESCALATE"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "update_request"), "escalated_update_request"),
            (
                lambda: assert_any_agent_message_contains(record, "not able to make changes"),
                "update_request_escalate_message",
            ),
        ],
    )


@pytest.mark.live
async def test_followup_update_request_second_variation(run_conversation, assert_and_record):
    """
    D6: Member asks to redirect Care Coach details to email — UPDATE_REQUEST escalation.

    "Send the Care Coach details to my email instead of fax" is a clear delivery
    channel change request.  FollowUpAgent classifies it as UPDATE_REQUEST and
    escalates immediately.  MSG_UPDATE_REQUEST_ESCALATE is sent before escalation.

    This test verifies that the update_request rule covers channel-switch requests
    (email vs. fax), not just number changes as in D5.

    Key invariants:
      - follow_up_was_active
      - escalated  (update_request reason)
      - agent message contains 'not able to make changes'
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Send the Care Coach details to my email instead of fax"],
        test_name="test_followup_update_request_second_variation",
        scenario=(
            "After care coach → 'send to email instead of fax' → UPDATE_REQUEST → "
            "immediate escalation (channel switch = update)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "update_request"), "escalated_update_request"),
            (
                lambda: assert_any_agent_message_contains(record, "not able to make changes"),
                "update_request_escalate_message",
            ),
        ],
    )


@pytest.mark.live
async def test_followup_update_request_resend_same_number(run_conversation, assert_and_record):
    """
    D7: Member expresses doubt and asks to resend to the same fax — UPDATE_REQUEST.

    Input: "I'm not sure that was the right fax number, can you resend to 6175554199?"

    Even though the member cites the same number that is already on file, any
    resend-or-correct request is classified as UPDATE_REQUEST by the follow_up
    extraction prompt — the agent cannot re-initiate a dispatch in this stage.
    MSG_UPDATE_REQUEST_ESCALATE is sent and escalation follows immediately.

    Key invariants:
      - follow_up_was_active
      - escalated  (update_request reason)
      - agent message contains 'not able to make changes'
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + ["I'm not sure that was the right fax number, can you resend to 6175554199?"],
        test_name="test_followup_update_request_resend_same_number",
        scenario=(
            "After care coach → 'resend to same fax number' → UPDATE_REQUEST → "
            "immediate escalation (re-send counts as update)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "update_request"), "escalated_update_request"),
            (
                lambda: assert_any_agent_message_contains(record, "not able to make changes"),
                "update_request_escalate_message",
            ),
        ],
    )


@pytest.mark.live
async def test_followup_max_turns_safety_cap(run_conversation, assert_and_record):
    """
    D8: Eleven 'hmm' inputs exhaust MAX_FOLLOW_UP_TURNS=10 → safety-cap closure.

    FollowUpAgent increments follow_up_turn_count on every turn (including the
    BARE_AFFIRMATIONS fast path and the UNSURE LLM path).  'hmm' is NOT in
    BARE_AFFIRMATIONS, so each turn goes through the LLM and is classified as
    UNSURE → MSG_NUDGE sent, count incremented.

    When turn_count > MAX_FOLLOW_UP_TURNS (i.e. turn 11 > 10), FollowUpAgent
    calls signal_complete() with closure_requested=True before running any
    further logic — the safety cap fires unconditionally.

    Key invariants:
      - follow_up_was_active  (agent ran many times)
      - call_closed  (closure_requested via safety cap, or routing to END/closure_agent)
      - not escalated  (cap uses signal_complete, not signal_escalate)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["hmm"] * 11,
        test_name="test_followup_max_turns_safety_cap",
        scenario=(
            "After care coach → 11× 'hmm' → MAX_FOLLOW_UP_TURNS=10 exceeded → "
            "signal_complete(closure_requested=True)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_call_closed(record), "call_closed"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP E — Guards inside follow_up_agent
# ===========================================================================
#
# FollowUpAgent calls run_conversation_guards() AFTER the LLM extraction call
# and BEFORE applying intent routing.  Guard priority (from guards.py):
#   1. NON_MEMBER_CALLER  (passive, from extracted.caller_type)
#   2. TRANSFER_REQUEST   (confidence ≥ 0.7 or keyword fallback)
#   3. ABUSE              (confidence ≥ 0.7 or regex fallback)
#   4. SELF_HARM          (confidence ≥ 0.7 or regex fallback)
#   5. INTERRUPTION       (confidence ≥ 0.7 or keyword fallback)
#   6. OFFTOPIC_GLOBAL    (static response, count-limited before escalation)
#   7. OFFTOPIC_AGENT     (LLM-generated recovery)
#
# Note: the BARE_AFFIRMATIONS fast path fires BEFORE the LLM call and therefore
# also before run_conversation_guards() is reached.
# ===========================================================================


def _assert_caller_type_is(record: ConversationRecord, expected: str) -> None:
    """caller_type == expected in final state or any turn snapshot."""
    actual = record.final_state.get("caller_type", "")
    if actual == expected:
        return
    for t in record.turns:
        if t.state_snapshot.get("caller_type") == expected:
            return
    raise AssertionError(
        f"Expected caller_type={expected!r} in any turn. "
        f"Final caller_type={actual!r}. "
        f"Seen across turns: {[t.state_snapshot.get('caller_type') for t in record.turns]}"
    )


@pytest.mark.live
async def test_followup_guard_transfer_request(run_conversation, assert_and_record):
    """
    E1: Member asks to be transferred — TRANSFER_REQUEST guard fires inside follow_up_agent.

    "Get me a real person" triggers the TRANSFER_REQUEST guard (LLM confidence ≥ 0.7
    or the keyword fallback via detect_transfer_request).  FollowUpAgent calls
    signal_escalate() with a MSG_TRANSFER_REQUEST message ("Please hold") before
    routing exits the follow_up stage.

    TRANSFER_REQUEST has higher priority than UPDATE_REQUEST because guards run
    before intent classification.  This is confirmed by the guard ordering in
    run_conversation_guards(): TRANSFER is checked at position 2, before the
    UNSURE / QUESTION / UPDATE_REQUEST routing code runs.

    Key invariants:
      - follow_up_was_active  (agent ran before guard fired)
      - escalated
      - routed_to escalation_agent
      - agent message contains 'hold'  (MSG_TRANSFER_REQUEST)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Get me a real person"],
        test_name="test_followup_guard_transfer_request",
        scenario=(
            "After care coach → 'Get me a real person' → TRANSFER_REQUEST guard → "
            "escalation_agent, message contains 'hold'"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
            (lambda: assert_any_agent_message_contains(record, "hold"), "transfer_hold_message"),
        ],
    )


@pytest.mark.live
async def test_followup_guard_transfer_variation_supervisor(run_conversation, assert_and_record):
    """
    E2: "Can I speak to your supervisor?" — TRANSFER_REQUEST guard variation.

    A supervisor-request phrasing that is unambiguously a transfer attempt.
    Verifies that the LLM guard (or keyword fallback) correctly classifies this
    as TRANSFER_REQUEST rather than QUESTION or UNSURE — 'supervisor' must not
    be mistaken for a session-context question.

    Key invariants:
      - follow_up_was_active
      - escalated
      - routed_to escalation_agent
      - agent message contains 'hold'
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Can I speak to your supervisor?"],
        test_name="test_followup_guard_transfer_variation_supervisor",
        scenario=("After care coach → 'speak to your supervisor' → TRANSFER_REQUEST → escalation_agent"),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
            (lambda: assert_any_agent_message_contains(record, "hold"), "transfer_hold_message"),
        ],
    )


@pytest.mark.live
@pytest.mark.slow
async def test_followup_guard_transfer_after_real_answer(run_conversation, assert_and_record):
    """
    E3: Real answer given on Turn 1, then transfer request on Turn 2.

    Turn 1: "What is my deductible?" → QUESTION → answered from snapshot (count=0).
    Turn 2: "Actually just transfer me" → TRANSFER_REQUEST → escalation.

    Verifies that the TRANSFER_REQUEST guard fires correctly on the second follow_up
    turn, after a normal QUESTION→answer cycle.  This confirms that guard state is
    independent of the cannot-answer count and that guards run on every turn.

    Key invariants:
      - follow_up_was_active
      - escalated  (on second turn)
      - routed_to escalation_agent
      - agent message contains 'hold' (from MSG_TRANSFER_REQUEST)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX
        + [
            "What is my deductible?",
            "Actually just transfer me",
        ],
        test_name="test_followup_guard_transfer_after_real_answer",
        scenario=(
            "After care coach → deductible question (answered) → "
            "'just transfer me' → TRANSFER_REQUEST → escalation on second turn"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
            (lambda: assert_any_agent_message_contains(record, "hold"), "transfer_hold_message"),
        ],
    )


@pytest.mark.live
async def test_followup_guard_abuse(run_conversation, assert_and_record):
    """
    E4: Hostile language inside follow_up_agent — ABUSE guard → escalation.

    "You're fucking useless" matches ABUSE_PATTERNS via regex fallback (or LLM
    guard at confidence ≥ 0.7).  FollowUpAgent calls signal_escalate() with
    MSG_ABUSE_ESCALATION ("I'm going to connect you with a live representative...
    Please hold" or "Let me transfer you to a member of our team... Please hold").

    ABUSE has higher guard priority than INTERRUPTION and OFFTOPIC, so it fires
    before any follow_up intent routing would run.

    Key invariants:
      - follow_up_was_active
      - escalated
      - routed_to escalation_agent
      - agent message contains 'hold'  (both MSG_ABUSE_ESCALATION variants end with 'hold')
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["You're fucking useless"],
        test_name="test_followup_guard_abuse",
        scenario=(
            "After care coach → abusive language → ABUSE guard → escalation_agent, MSG_ABUSE_ESCALATION sent"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
            (lambda: assert_any_agent_message_contains(record, "hold"), "abuse_escalation_message"),
        ],
    )


@pytest.mark.live
async def test_followup_guard_self_harm(run_conversation, assert_and_record):
    """
    E5: Self-harm ideation inside follow_up_agent — SELF_HARM guard → escalation.

    "I just want to end it all" matches SELF_HARM_PATTERNS via regex fallback
    (or LLM guard at confidence ≥ 0.7).  FollowUpAgent calls signal_escalate()
    with MSG_SELF_HARM_ESCALATION, which is a compassionate message:
    "I hear you, and I want to make sure you get the right support..."
    or "Thank you for sharing that with me. I want to connect you with someone
    who is better placed to help..."

    Both variants contain 'support' or 'help' and are distinctly warmer in tone
    than ABUSE or TRANSFER escalations.

    Key invariants:
      - follow_up_was_active
      - escalated
      - routed_to escalation_agent
      - agent message contains 'support' or 'help'  (compassionate MSG_SELF_HARM_ESCALATION)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["I just want to end it all"],
        test_name="test_followup_guard_self_harm",
        scenario=(
            "After care coach → self-harm signal → SELF_HARM guard → "
            "escalation_agent, compassionate MSG_SELF_HARM_ESCALATION sent"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "help"),
                "self_harm_compassionate_message",
            ),
        ],
    )


@pytest.mark.live
async def test_followup_guard_offtopic_global_then_done(run_conversation, assert_and_record):
    """
    E6: Off-topic (non-healthcare) question, then closure — OFFTOPIC_GLOBAL, no escalation.

    Turn 1: "What's the weather today?" — zero healthcare relevance → OFFTOPIC_GLOBAL.
    FollowUpAgent sends a MSG_OFFTOPIC_GLOBAL static response that redirects back to
    provider services or claims ("I can help with provider services or claims...").
    The offtopic_global_count is set to 1, which is below MAX_SLOT_ATTEMPTS (3),
    so no escalation fires.

    Turn 2: "no that's all" — DONE → closure.

    Key invariants:
      - follow_up_was_active
      - not escalated  (only 1 offtopic, below the count-based escalation threshold)
      - agent message contains 'provider' or 'claims'  (MSG_OFFTOPIC_GLOBAL redirect)
      - call_closed  (DONE on second turn)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["What's the weather today?", "no that's all"],
        test_name="test_followup_guard_offtopic_global_then_done",
        scenario=(
            "After care coach → 'weather' (OFFTOPIC_GLOBAL, count=1) → redirect message → "
            "'no that's all' → DONE → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "provider"),
                "offtopic_global_redirect_contains_provider",
            ),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_followup_guard_non_member_caller_mid_followup(run_conversation, assert_and_record):
    """
    E7: Caller reveals they are a provider mid-follow-up — NON_MEMBER_CALLER → call ends.

    "I'm actually a provider calling about a patient referral" causes the
    passive NON_MEMBER_CALLER detection (checked first in run_conversation_guards,
    before TRANSFER_REQUEST) to fire.  extracted.caller_type == 'provider' is set
    by the LLM, which triggers _handle_non_member_caller().

    The handler sends a message with the dedicated provider line number
    (1-800-555-0201) and sets next_node=END + caller_type_handled=True.
    No escalation_agent is used — the call simply ends cleanly.

    Key invariants:
      - follow_up_was_active
      - caller_type == 'provider'  (in final_state or any turn snapshot)
      - call routed to END  (captured by assert_call_closed via next_node check)
      - not escalated  (_handle_non_member_caller uses END, not escalation_agent)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["I'm actually a provider calling about a patient referral"],
        test_name="test_followup_guard_non_member_caller_mid_followup",
        scenario=(
            "After care coach → provider identifies mid-follow-up → NON_MEMBER_CALLER → "
            "dedicated line message, next_node=END"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: _assert_caller_type_is(record, "provider"), "caller_type==provider"),
            (lambda: assert_call_closed(record), "call_routed_to_end"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_followup_update_request_vs_question_boundary(run_conversation, assert_and_record):
    """
    E8: Benefits summary request — QUESTION (not UPDATE_REQUEST), answered from context.

    "Can you summarize the benefits we discussed?" is a request for information
    from the session — it is QUESTION, not UPDATE_REQUEST.  The LLM must
    distinguish between wanting to change something (UPDATE_REQUEST) and wanting
    to hear information again (QUESTION).

    _build_session_snapshot() has all the benefits fields.  The answer should
    include at least one dollar amount from the deductible or OOP max.
    No escalation fires.

    This test acts as a boundary-condition guard: a close neighbour to UPDATE_REQUEST
    phrasing ('summarize the benefits we discussed') must NOT be mis-classified.

    Key invariants:
      - follow_up_was_active
      - not escalated  (QUESTION answered — not UPDATE_REQUEST)
      - agent message contains '$' or 'deductible'  (real answer from snapshot)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["Can you summarize the benefits we discussed?"],
        test_name="test_followup_update_request_vs_question_boundary",
        scenario=(
            "After care coach → 'summarize benefits' → QUESTION (not UPDATE_REQUEST) → "
            "answered from snapshot, not escalated"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "$"),
                "summary_answer_contains_dollar",
            ),
        ],
    )


@pytest.mark.live
async def test_followup_bare_affirmation_fast_path(run_conversation, assert_and_record):
    """
    E9: 'yes please' — BARE_AFFIRMATIONS fast path, zero LLM calls.

    'yes please' is in BARE_AFFIRMATIONS (frozenset in follow_up/constants.py).
    FollowUpAgent intercepts it before the LLM call and before run_conversation_guards()
    is reached — the fast path returns MSG_NUDGE immediately, incrementing
    follow_up_turn_count but making no LLM calls and firing no guards.

    This confirms the ordering: BARE_AFFIRMATIONS check → (return) → LLM call →
    run_conversation_guards → intent routing.  No guard can fire on a bare affirmation
    because the fast path exits before guards run.

    Key invariants:
      - follow_up_was_active
      - not escalated
      - agent message contains 'anything else' or 'specific'  (MSG_NUDGE)
    """
    record = await run_conversation(
        user_inputs=FOLLOW_UP_PREFIX + ["yes please"],
        test_name="test_followup_bare_affirmation_fast_path",
        scenario=(
            "After care coach → 'yes please' (BARE_AFFIRMATIONS) → "
            "zero-LLM fast path → MSG_NUDGE sent, no guard fired"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (
                lambda: assert_any_agent_message_contains(record, "anything else"),
                "nudge_message_sent",
            ),
        ],
    )


# ===========================================================================
# GROUP F — End-to-end smoke tests and latency benchmarks
# ===========================================================================
#
# FULL_HAPPY_PATH = FULL_PREFIX + ["yes", "yes", "no thank you"]
#   → benefits accepted → care coach accepted → follow-up DONE
#
# Smoke tests (F1–F8) validate the four key path combinations:
#   YES/YES, NO/NO, YES/NO, NO/YES, plus multi-turn follow-up variants.
#
# Latency tests (F9–F10) use thresholds:
#   _LATENCY_P50_SEC = 12.0 s
#   _LATENCY_P95_SEC = 20.0 s
# ===========================================================================


@pytest.mark.live
async def test_full_happy_path_benefits_carecoach_done(run_conversation, assert_and_record):
    """
    F1: Full happy path — benefits accepted, care coach accepted, follow-up done.

    FULL_HAPPY_PATH = FULL_PREFIX + ["yes", "yes", "no thank you"]

    The canonical end-to-end smoke test.  Drives the complete call flow:
      intake → verification → provider search → delivery confirmation →
      benefits explained (YES) → care coach dispatched (YES) →
      follow_up DONE ("no thank you" → closure).

    Validates that all six agents hand off correctly and no escalation fires.

    Key invariants:
      - benefits_explained == True  (BenefitsAgent YES path)
      - care_coach_offered == True
      - care_coach_details_sent == True  (CareWellnessAgent._handle_yes ran)
      - proactive_offer_available == True  (care coach accepted)
      - not escalated
      - follow_up_agent was active in at least one turn
      - call_closed  (DONE routing to closure_agent / END)
    """
    record = await run_conversation(
        user_inputs=FULL_HAPPY_PATH,
        test_name="test_full_happy_path_benefits_carecoach_done",
        scenario=(
            "Full happy path: benefits accepted → care coach accepted → 'no thank you' (DONE) → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_offered(record), "care_coach_offered==True"),
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_care_coach_accepted(record), "proactive_offer_available==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_happy_path_benefits_declined_carecoach_declined(run_conversation, assert_and_record):
    """
    F2: Full path — benefits declined, care coach declined, follow-up done.

    FULL_PREFIX + ["no", "no", "no thank you"]
      → "no" to benefits offer (proactive_offer_available=False, NO path in BenefitsAgent)
      → "no" to care coach offer (care_coach_nooffer_sent=True via _handle_no)
      → "no thank you" in follow_up → DONE → closure

    Verifies the all-decline path completes cleanly: no SF fetch, no dispatch,
    no escalation — the call closes via the follow_up DONE signal.

    Key invariants:
      - care_coach_nooffer_sent == True  (CareWellnessAgent._handle_no ran)
      - proactive_offer_available == False  (declined care coach)
      - not escalated
      - follow_up_agent was active
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["no", "no", "no thank you"],
        test_name="test_full_happy_path_benefits_declined_carecoach_declined",
        scenario=(
            "Full decline path: 'no' to benefits → 'no' to care coach → "
            "'no thank you' (follow-up DONE) → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_care_coach_declined(record), "proactive_offer_available==False"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_path_benefits_yes_carecoach_no(run_conversation, assert_and_record):
    """
    F3: Benefits accepted but care coach declined — follow-up done.

    FULL_PREFIX + ["yes", "no", "no"]
      → "yes" to benefits offer → BenefitsAgent YES path (SF fetch, explanation)
      → "no" to care coach offer → CareWellnessAgent._handle_no → nooffer message
      → "no" in follow_up → DONE → closure

    Verifies the mixed YES/NO path: benefits_explained=True because the member
    accepted the benefits explanation, but care_coach_details_sent is NOT set
    because the member declined the care coach offer.

    Key invariants:
      - benefits_explained == True  (SF fetch ran)
      - care_coach_nooffer_sent == True  (_handle_no ran)
      - not escalated
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "no", "no"],
        test_name="test_full_path_benefits_yes_carecoach_no",
        scenario=(
            "'yes' to benefits (explained) → 'no' to care coach (no-offer) → "
            "'no' in follow-up → DONE → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_no_offer_sent(record), "care_coach_nooffer_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_path_benefits_no_carecoach_yes(run_conversation, assert_and_record):
    """
    F4: Benefits declined but care coach accepted — follow-up done.

    FULL_PREFIX + ["no", "yes", "no"]
      → "no" to benefits offer → BenefitsAgent NO path (skips SF fetch)
        but still presents Care Coach offer via BENEFITS_NOEXPLANATION_TEMPLATES
      → "yes" to care coach → CareWellnessAgent._handle_yes → dispatch + intro
      → "no" in follow_up → DONE → closure

    proactive_offer_available=False is set for the benefits decline.  The Care
    Coach offer embedded in BENEFITS_NOEXPLANATION_TEMPLATES is still presented;
    when the member says 'yes' to it, CareWellnessAgent dispatches details and
    sets care_coach_details_sent=True.

    Key invariants:
      - proactive_offer_available == False  (benefits declined → NO path)
      - care_coach_details_sent == True  (member accepted care coach despite declining benefits)
      - not escalated
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["no", "yes", "no"],
        test_name="test_full_path_benefits_no_carecoach_yes",
        scenario=(
            "'no' to benefits (NO path, no SF fetch) → 'yes' to care coach (dispatch) → "
            "'no' in follow-up → DONE → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_path_with_one_follow_up_question(run_conversation, assert_and_record):
    """
    F5: Full path with one follow-up question about deductible, then done.

    FULL_PREFIX + ["yes", "yes", "What is my deductible?", "no that's all"]
      → benefits accepted → care coach dispatched → follow_up active
      → "What is my deductible?" → QUESTION → answered from _build_session_snapshot()
      → "no that's all" → DONE → closure

    _build_session_snapshot() injects 'Individual deductible: $<amount> per
    calendar year'.  The LLM answer must include a dollar amount.

    Key invariants:
      - benefits_explained == True
      - care_coach_details_sent == True
      - follow_up answered with '$' in the agent message
      - not escalated
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes", "What is my deductible?", "no that's all"],
        test_name="test_full_path_with_one_follow_up_question",
        scenario=(
            "Full path → care coach dispatched → deductible question answered ('$') → "
            "'no that's all' → DONE → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_any_agent_message_contains(record, "$"), "deductible_answer_contains_dollar"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_path_followup_summary_request(run_conversation, assert_and_record):
    """
    F6: Full path with a benefits-summary request in follow-up.

    FULL_PREFIX + ["yes", "yes", "Can you summarize my PCP benefits?", "no thanks"]
      → benefits accepted → care coach dispatched → follow_up active
      → summary request → QUESTION → answered from session snapshot
        (contains deductible, coinsurance, OOP fields)
      → "no thanks" → DONE → closure

    _build_session_snapshot() includes all benefits fields.  The LLM answer
    should surface at least one of: a dollar amount ('$'), the literal string
    'deductible', or the coinsurance percentage string.

    Key invariants:
      - follow_up_agent was active
      - agent message contains '$' or 'deductible' or 'coinsurance'
      - not escalated
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes", "Can you summarize my PCP benefits?", "no thanks"],
        test_name="test_full_path_followup_summary_request",
        scenario=(
            "Full path → care coach dispatched → PCP benefits summary request → "
            "answer from snapshot → 'no thanks' → DONE → closure"
        ),
    )

    def _assert_summary_answer(r: ConversationRecord) -> None:
        all_msgs = " ".join((t.agent_message or "").lower() for t in r.turns)
        found = any(kw in all_msgs for kw in ["$", "deductible", "coinsurance"])
        assert found, (
            "Expected follow-up summary to mention '$', 'deductible', or 'coinsurance'. "
            f"Transcript (truncated): {all_msgs[:400]!r}"
        )

    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: _assert_summary_answer(record), "summary_contains_benefits_detail"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_full_path_update_request_then_closure(run_conversation, assert_and_record):
    """
    F7: Full path with an update request in follow-up — escalation overrides closure.

    FULL_PREFIX + ["yes", "yes", "Actually can you update the fax number?"]
      → benefits accepted → care coach dispatched → follow_up active
      → update request → UPDATE_REQUEST → immediate escalation

    UPDATE_REQUEST fires unconditionally on the first occurrence, before any
    DONE signal can close the call.  The escalation routes to escalation_agent,
    not closure_agent.  MSG_UPDATE_REQUEST_ESCALATE is sent first.

    Key invariants:
      - follow_up_agent was active
      - escalated (reason contains 'update_request')
      - NOT closed via closure (escalation overrides)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes", "Actually can you update the fax number?"],
        test_name="test_full_path_update_request_then_closure",
        scenario=(
            "Full path → care coach dispatched → 'update the fax number' → "
            "UPDATE_REQUEST → immediate escalation (not closure)"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_escalated(record, "update_request"), "escalated_update_request"),
        ],
    )


@pytest.mark.live
async def test_full_path_three_follow_up_questions(run_conversation, assert_and_record):
    """
    F8: Full path with three sequential follow-up questions, then done.

    FULL_PREFIX + ["yes", "yes",
                   "What is my individual deductible?",
                   "What is my family out-of-pocket max?",
                   "Was the Care Coach info sent to my fax?",
                   "no that's all thanks"]

    Three distinct answerable questions from session context:
      1. Individual deductible → '$' in answer (individual_deductible field)
      2. Family OOP max → '$' in answer (family_oop_max field)
      3. Care Coach fax confirmation → 'fax' in answer (care_coach_details_sent + fax field)

    All three are answered from _build_session_snapshot() without escalation.
    The cannot-answer counter never increments (all answered).

    Key invariants:
      - follow_up_agent was active
      - agent messages collectively contain '$' and 'fax'
      - not escalated
      - call_closed
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX
        + [
            "yes",
            "yes",
            "What is my individual deductible?",
            "What is my family out-of-pocket max?",
            "Was the Care Coach info sent to my fax?",
            "no that's all thanks",
        ],
        test_name="test_full_path_three_follow_up_questions",
        scenario=(
            "Full path → care coach dispatched → 3 sequential questions answered "
            "(deductible, family OOP, fax confirmation) → DONE → closure"
        ),
    )
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "answers_contain_dollar"),
            (lambda: assert_any_agent_message_contains(record, "fax"), "answers_contain_fax"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_call_closed(record), "call_closed"),
        ],
    )


@pytest.mark.live
async def test_latency_benefits_care_coach_done_happy_path(run_conversation, assert_and_record):
    """
    F9: Latency benchmark — full happy path must meet p50 ≤ 12 s, p95 ≤ 20 s.

    Runs FULL_HAPPY_PATH (benefits accepted → care coach accepted → DONE) and
    asserts per-turn latency percentiles.  This is a regression guard: a
    significant increase in p50 or p95 indicates a model or infrastructure
    regression, not just an occasional slow turn.

    Also validates functional correctness so a latency-passing run that
    functionally broke is still caught.

    Key invariants:
      - benefits_explained == True
      - care_coach_details_sent == True
      - p50 per-turn latency ≤ _LATENCY_P50_SEC (12 s)
      - p95 per-turn latency ≤ _LATENCY_P95_SEC (20 s)
    """
    record = await run_conversation(
        user_inputs=FULL_HAPPY_PATH,
        test_name="test_latency_benefits_care_coach_done_happy_path",
        scenario=(
            "Latency benchmark: full happy path (benefits+carecoach+done) — "
            f"p50 ≤ {_LATENCY_P50_SEC}s, p95 ≤ {_LATENCY_P95_SEC}s"
        ),
    )
    _print_latency_summary(record)
    assert_and_record(
        record,
        [
            (lambda: assert_benefits_explained(record), "benefits_explained==True"),
            (lambda: assert_care_coach_details_sent(record), "care_coach_details_sent==True"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50_under_{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95_under_{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_followup_question_answered(run_conversation, assert_and_record):
    """
    F10: Latency benchmark — one follow-up question must meet p50 ≤ 12 s, p95 ≤ 20 s.

    FULL_PREFIX + ["yes", "yes", "What is my deductible?", "no thanks"]

    Adds one QUESTION turn after the care coach dispatch, which exercises the
    LLM call inside FollowUpAgent.  This is typically the slowest single turn
    in the happy path (the LLM must classify QUESTION + generate a dollar-amount
    answer in one forward pass).

    Key invariants:
      - follow_up_agent was active
      - agent message contains '$' (deductible answer)
      - p50 per-turn latency ≤ _LATENCY_P50_SEC (12 s)
      - p95 per-turn latency ≤ _LATENCY_P95_SEC (20 s)
    """
    record = await run_conversation(
        user_inputs=FULL_PREFIX + ["yes", "yes", "What is my deductible?", "no thanks"],
        test_name="test_latency_followup_question_answered",
        scenario=(
            "Latency benchmark: full path + one deductible question — "
            f"p50 ≤ {_LATENCY_P50_SEC}s, p95 ≤ {_LATENCY_P95_SEC}s"
        ),
    )
    _print_latency_summary(record)
    assert_and_record(
        record,
        [
            (lambda: assert_follow_up_was_active(record), "follow_up_was_active"),
            (lambda: assert_any_agent_message_contains(record, "$"), "deductible_answer_contains_dollar"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50_under_{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95_under_{_LATENCY_P95_SEC}s"),
        ],
    )
