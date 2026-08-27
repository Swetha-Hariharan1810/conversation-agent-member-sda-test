"""
test_intake_agent_live.py — Live integration tests for IntakeAgent.

These tests run against a real LLM (Azure OpenAI / Gemini).
They require valid API credentials in the environment.

Run:
    pytest -m live src/agent/tests/live/test_intake_agent_live.py -v
    pytest -m live -k "test_intake_provider" -v   # single test
    pytest -m live --count=30 -v src/agent/tests/live/test_intake_agent_live.py

Skip in CI:
    Tests auto-skip when AZURE_OPENAI_API_KEY (or GOOGLE_API_KEY) is absent.

Groups
------
A  Happy-path intent classification (4 tests)
B  Unclear intent + retry flow (3 tests)
C  Guard triggers (6 tests)
D  Caller type detection (3 tests)
E  Edge cases and combinations (4 tests)
F  Conversation continuity (2 tests)
G  Latency benchmarks (5 tests)
F  Conversation continuity (2 tests)
"""

from __future__ import annotations

import math
import os
import statistics

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
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_intent(record: ConversationRecord, expected: str) -> None:
    actual = record.final_state.get("call_intent", "")
    assert actual == expected, f"Expected call_intent={expected!r}, got {actual!r}"


def assert_routed_to(record: ConversationRecord, expected_node: str) -> None:
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


def assert_escalated(record: ConversationRecord, reason_contains: str | None = None) -> None:
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
            f"Reasons seen across turns: {[r for r in all_reasons if r]}"
        )


def assert_not_escalated(record: ConversationRecord) -> None:
    final_active = record.final_state.get("active_agent", "")
    final_reason = record.final_state.get("escalation_reason", "")
    was_escalated = (
        final_active == "escalation_agent"
        or any(t.active_agent == "escalation_agent" for t in record.turns)
        or any(bool(t.state_snapshot.get("escalation_reason")) for t in record.turns)
    )
    assert not was_escalated, f"Unexpected escalation: reason={final_reason!r}, active_agent={final_active!r}"


def assert_agent_message_contains(
    record: ConversationRecord,
    *substrings: str,
    turn: int = -1,
) -> None:
    """Check that the agent message at the given turn contains all substrings."""
    turns = record.turns
    if not turns:
        raise AssertionError("No turns recorded — conversation did not run")
    target_turn = turns[turn]
    msg = (target_turn.agent_message or "").lower()
    for sub in substrings:
        assert sub.lower() in msg, (
            f"Expected agent message to contain {sub!r}. "
            f"Actual turn {target_turn.turn_number} message: {target_turn.agent_message!r}"
        )


def assert_call_ended(record: ConversationRecord) -> None:
    from langgraph.graph import END

    next_node = record.final_state.get("next_node", "")
    is_interrupt = record.final_state.get("is_interrupt", True)
    assert next_node in (END, "__end__"), f"Expected call to END, got next_node={next_node!r}"
    assert not is_interrupt, "Expected is_interrupt=False when call ends"


def assert_caller_type(record: ConversationRecord, expected_type: str) -> None:
    actual = record.final_state.get("caller_type", "")
    assert actual == expected_type, f"Expected caller_type={expected_type!r}, got {actual!r}"


def assert_turn_count(
    record: ConversationRecord,
    min_turns: int | None = None,
    max_turns: int | None = None,
) -> None:
    n = record.total_turns
    if min_turns is not None:
        assert n >= min_turns, f"Expected >= {min_turns} turns, got {n}"
    if max_turns is not None:
        assert n <= max_turns, f"Expected <= {max_turns} turns, got {n}"


def assert_any_agent_message_contains(
    record: ConversationRecord,
    *substrings: str,
) -> None:
    """Check that at least one agent message across all turns contains each substring."""
    all_msgs = " ".join((t.agent_message or "").lower() for t in record.turns)
    for sub in substrings:
        assert sub.lower() in all_msgs, (
            f"Expected any agent message to contain {sub!r}. Full transcript: {all_msgs[:500]!r}"
        )


def _is_escalated_to_escalation_agent(record: ConversationRecord) -> bool:
    return record.final_state.get("next_node") == "escalation_agent" or bool(
        record.final_state.get("escalation_reason")
    )


# ===========================================================================
# GROUP A — Happy Path Intent Classification
# ===========================================================================


@pytest.mark.live
async def test_intake_provider_services_happy_path(run_intake_conversation, assert_and_record):
    """
    Happy path: user clearly states they need to find a doctor.
    IntakeAgent should classify intent as 'provider_services', send the
    bridge message asking for first name, and route to verification_agent.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to find a doctor in my network"],
        test_name="test_intake_provider_services_happy_path",
        scenario="User clearly requests provider services — expects clean"
        " classification and verification routing",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (
                lambda: assert_any_agent_message_contains(record, "first name"),
                "bridge_message_asks_first_name",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_claim_services_happy_path(run_intake_conversation, assert_and_record):
    """
    Happy path: user asks about a claim they submitted.
    Intent should be classified as 'claim_services' and routed to verification.
    """
    record = await run_intake_conversation(
        user_inputs=["I want to check on a claim I submitted last month"],
        test_name="test_intake_claim_services_happy_path",
        scenario="User requests claim follow-up — expects claim_services intent and verification routing",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_out_of_scope_billing(run_intake_conversation, assert_and_record):
    """
    Out-of-scope: user mentions billing.
    IntakeAgent should classify as out_of_scope, deliver a routing message
    that mentions the billing team and phone number, and route to END.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to pay my bill"],
        test_name="test_intake_out_of_scope_billing",
        scenario="User asks about billing — expects out-of-scope routing to END with billing team info",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_call_ended(record), "call_ended_cleanly"),
            (
                lambda: assert_any_agent_message_contains(record, "billing"),
                "message_mentions_billing",
            ),
            (
                lambda: assert_any_agent_message_contains(record, "1-800"),
                "message_contains_phone_number",
            ),
        ],
    )


@pytest.mark.live
async def test_intake_out_of_scope_pharmacy(run_intake_conversation, assert_and_record):
    """
    Out-of-scope: user asks about a prescription.
    Agent should route to END and mention pharmacy/pharmacy benefits team.
    """
    record = await run_intake_conversation(
        user_inputs=["I have a question about my prescription"],
        test_name="test_intake_out_of_scope_pharmacy",
        scenario="User asks about prescription — expects pharmacy team routing to END",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_call_ended(record), "call_ended_cleanly"),
            (
                lambda: assert_any_agent_message_contains(record, "pharmacy"),
                "message_mentions_pharmacy",
            ),
        ],
    )


# ===========================================================================
# GROUP B — Unclear Intent + Retry Flow
# ===========================================================================


@pytest.mark.live
async def test_intake_unclear_first_attempt(run_intake_conversation, assert_and_record):
    """
    First utterance is too vague to classify ('I have a question').
    Agent should ask for clarification without classifying intent yet.
    call_intent must remain unset after one unclear turn.
    """
    record = await run_intake_conversation(
        user_inputs=["Hello, I have a question"],
        test_name="test_intake_unclear_first_attempt",
        scenario="Vague first utterance — agent should ask for clarification, not classify intent",
    )

    assert_and_record(
        record,
        [
            (
                lambda: _assert_no_intent(record),
                "call_intent_not_yet_set",
            ),
            (
                lambda: _assert_still_in_intake(record),
                "still_in_intake_agent",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_unclear_then_clarifies(run_intake_conversation, assert_and_record):
    """
    Turn 1: vague greeting.  Turn 2: user clarifies they want a provider.
    After clarification intent should be provider_services and route to verification.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Hi",
            "I'm looking for a provider near me",
        ],
        test_name="test_intake_unclear_then_clarifies",
        scenario="Unclear first turn, clarifies in second turn — expects provider_services on turn 2",
    )

    assert_and_record(
        record,
        [
            (
                lambda: assert_intent(record, "provider_services"),
                "intent==provider_services_after_clarification",
            ),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_max_clarification_then_escalate(run_intake_conversation, assert_and_record):
    """
    User repeatedly gives unclear responses.
    After MAX_CLARIFICATION_ATTEMPTS (2) the agent should escalate rather
    than looping indefinitely.  The escalation message should be warm and
    non-punishing.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "I don't know",
            "Not sure what I need",
            "I'm confused",
        ],
        test_name="test_intake_max_clarification_then_escalate",
        scenario="Repeated unclear utterances — expects escalation after MAX_CLARIFICATION_ATTEMPTS",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "escalation_message_mentions_representative",
            ),
        ],
    )


# ===========================================================================
# GROUP C — Guard Triggers
# ===========================================================================


@pytest.mark.live
async def test_intake_guard_transfer_request(run_intake_conversation, assert_and_record):
    """
    User explicitly asks to speak to a real person.
    TRANSFER_REQUEST guard should fire and route to escalation_agent.
    The message must be a warm handoff (mention 'hold' or 'transfer').
    """
    record = await run_intake_conversation(
        user_inputs=["I want to speak to a real person please"],
        test_name="test_intake_guard_transfer_request",
        scenario="User requests human agent — expects TRANSFER_REQUEST guard → escalation_agent",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation_agent"),
            (
                lambda: assert_any_agent_message_contains(record, "hold"),
                "warm_transfer_message_contains_hold",
            ),
        ],
    )


@pytest.mark.live
async def test_intake_guard_transfer_request_variation(run_intake_conversation, assert_and_record):
    """
    Variation of transfer request phrasing: 'Can I talk to a human agent?'
    Same guard and routing expectations as the canonical transfer request.
    """
    record = await run_intake_conversation(
        user_inputs=["Can I talk to a human agent?"],
        test_name="test_intake_guard_transfer_request_variation",
        scenario="Alternate transfer phrasing — same guard should fire",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation_agent"),
        ],
    )


@pytest.mark.live
async def test_intake_guard_abuse(run_intake_conversation, assert_and_record):
    """
    User uses abusive language.
    ABUSE guard should fire (LLM or keyword fallback), triggering escalation
    with a message from the MSG_ABUSE_ESCALATION pool.
    """
    record = await run_intake_conversation(
        user_inputs=["You're fucking useless"],
        test_name="test_intake_guard_abuse",
        scenario="Abusive language — expects ABUSE guard → escalation with MSG_ABUSE_ESCALATION",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation_agent"),
            (
                lambda: assert_any_agent_message_contains(record, "representative"),
                "message_mentions_representative",
            ),
        ],
    )


@pytest.mark.live
async def test_intake_guard_self_harm(run_intake_conversation, assert_and_record):
    """
    User expresses self-harm ideation.
    SELF_HARM guard must fire with a compassionate message from
    MSG_SELF_HARM_ESCALATION.  The message must convey support.
    """
    record = await run_intake_conversation(
        user_inputs=["I want to kill myself"],
        test_name="test_intake_guard_self_harm",
        scenario="Self-harm signal — expects SELF_HARM guard → compassionate escalation message",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation_agent"),
            # MSG_SELF_HARM_ESCALATION contains "support" or "help"
            # (
            #     lambda: assert_any_agent_message_contains(record, "hold"),
            #     "compassionate_message_contains_support",
            # ),
        ],
    )


@pytest.mark.live
async def test_intake_guard_offtopic_global(run_intake_conversation, assert_and_record):
    """
    Single off-topic utterance (weather question).
    OFFTOPIC_GLOBAL guard fires → agent steers back to healthcare topics.
    Must NOT escalate on the first off-topic occurrence.
    """
    record = await run_intake_conversation(
        user_inputs=["What's the weather like today?"],
        test_name="test_intake_guard_offtopic_global",
        scenario="Single off-topic utterance — agent redirects, does NOT escalate on first occurrence",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_not_escalated(record), "no_escalation_on_first_offtopic"),
            (
                lambda: assert_any_agent_message_contains(record, "provider", "claim"),
                "agent_redirects_to_healthcare",
            ),
        ],
    )


@pytest.mark.live
async def test_intake_guard_offtopic_max_then_escalate(run_intake_conversation, assert_and_record):
    """
    Repeated off-topic requests exhaust MAX_SLOT_ATTEMPTS (3).
    After the threshold the agent should escalate, not loop forever.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Tell me a joke",
            "Who won the game last night?",
            "What's the best pizza place nearby?",
        ],
        test_name="test_intake_guard_offtopic_max_then_escalate",
        scenario="Three consecutive off-topic turns — expects escalation after MAX_SLOT_ATTEMPTS",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered_after_max_offtopic"),
        ],
    )


@pytest.mark.live
async def test_intake_guard_offtopic_then_valid_intent(run_intake_conversation, assert_and_record):
    """
    Turn 1: off-topic (weather) → agent redirects, no escalation.
    Turn 2: valid provider_services intent → classified correctly, routed to verification.
    The offtopic_global_count resets on a valid answer, so no escalation should fire.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "What's the weather like today?",
            "I need to find a doctor in my network",
        ],
        test_name="test_intake_guard_offtopic_then_valid_intent",
        scenario="Off-topic turn 1 redirected, turn 2 gives valid provider intent — no escalation",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_turn_count(record, min_turns=3), "at_least_3_turns"),
        ],
    )


@pytest.mark.live
async def test_intake_guard_two_offtopic_then_valid_intent(run_intake_conversation, assert_and_record):
    """
    Turn 1: off-topic (sports) → redirect.
    Turn 2: off-topic (food) → redirect (count=2, still under threshold of 3).
    Turn 3: valid claim_services intent → classified, no escalation.
    Verifies the counter does not escalate until MAX_SLOT_ATTEMPTS (3) is truly hit.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Who won the game last night?",
            "What's a good pizza place nearby?",
            "I want to follow up on a claim I submitted",
        ],
        test_name="test_intake_guard_two_offtopic_then_valid_intent",
        scenario="Two off-topic turns then valid claim intent before escalation threshold",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_guard_offtopic_then_out_of_scope(run_intake_conversation, assert_and_record):
    """
    Turn 1: off-topic (joke) → redirect.
    Turn 2: out-of-scope (billing) → routed cleanly to END with billing team info.
    Verifies that out_of_scope still resolves correctly after one off-topic turn.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Tell me a joke",
            "I need to pay my insurance bill",
        ],
        test_name="test_intake_guard_offtopic_then_out_of_scope",
        scenario="Off-topic turn 1 then out-of-scope billing — should route to END cleanly",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_call_ended(record), "call_ended_cleanly"),
            (
                lambda: assert_any_agent_message_contains(record, "billing"),
                "message_mentions_billing",
            ),
            (lambda: assert_escalated(record), "escalation_triggered"),
        ],
    )


@pytest.mark.live
async def test_intake_guard_offtopic_then_transfer_request(run_intake_conversation, assert_and_record):
    """
    Turn 1: off-topic (weather) → redirect.
    Turn 2: transfer request → TRANSFER_REQUEST guard fires, escalation happens.
    Verifies guard priority: TRANSFER_REQUEST always wins regardless of offtopic state.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "What's the weather?",
            "Actually just transfer me to a real person",
        ],
        test_name="test_intake_guard_offtopic_then_transfer_request",
        scenario="Off-topic then transfer request — TRANSFER_REQUEST guard wins, escalation fires",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_routed_to(record, "escalation_agent"), "routes_to_escalation_agent"),
        ],
    )


# ===========================================================================
# GROUP D — Caller Type Detection
# ===========================================================================


@pytest.mark.live
async def test_intake_caller_type_provider(run_intake_conversation, assert_and_record):
    """
    Caller identifies as a healthcare provider.
    Non-member routing: caller_type='provider', caller_type_handled=True,
    message contains provider phone number, call routes to END.
    """
    record = await run_intake_conversation(
        user_inputs=["Hi, I'm a doctor calling about a patient referral"],
        test_name="test_intake_caller_type_provider",
        scenario="Provider caller — expects non-member routing to END with provider line number",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_caller_type(record, "provider"), "caller_type==provider"),
            (
                lambda: _assert_caller_type_handled(record),
                "caller_type_handled==True",
            ),
            (lambda: assert_call_ended(record), "call_routed_to_END"),
            (
                lambda: assert_any_agent_message_contains(record, "1-800"),
                "message_contains_phone_number",
            ),
        ],
    )


@pytest.mark.live
async def test_intake_caller_type_employer(run_intake_conversation, assert_and_record):
    """
    Caller identifies as an employer calling about a group plan.
    Non-member routing: caller_type='employer_group', call ends cleanly.
    """
    record = await run_intake_conversation(
        user_inputs=["I'm calling about our group plan for employees"],
        test_name="test_intake_caller_type_employer",
        scenario="Employer group caller — expects non-member routing to END",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_caller_type(record, "employer_group"), "caller_type==employer_group"),
            (lambda: assert_call_ended(record), "call_routed_to_END"),
        ],
    )


@pytest.mark.live
async def test_intake_caller_identifies_as_member(run_intake_conversation, assert_and_record):
    """
    Caller explicitly says 'I am a member'.
    Should NOT be treated as a non-member caller; intent should still be
    extracted normally and routed to verification.
    """
    record = await run_intake_conversation(
        user_inputs=["I am a member and need help finding a doctor"],
        test_name="test_intake_caller_identifies_as_member",
        scenario="Member self-identifies + states intent — expects normal routing, not non-member path",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification"),
            (
                lambda: _assert_not_caller_type(record, "provider"),
                "not_treated_as_provider",
            ),
            (
                lambda: _assert_not_caller_type(record, "employer_group"),
                "not_treated_as_employer_group",
            ),
        ],
    )


# ===========================================================================
# GROUP E — Edge Cases and Combinations
# ===========================================================================


@pytest.mark.live
async def test_intake_answered_with_followup(run_intake_conversation, assert_and_record):
    """
    User gives the intent AND asks a follow-up ('can you repeat that?').
    Intent should be classified (provider_services) without triggering a
    second round of intent clarification.  EventType.ANSWERED_WITH_FOLLOWUP
    path in IntakeAgent should handle the follow-up gracefully.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to find a doctor — can you repeat what you just said?"],
        test_name="test_intake_answered_with_followup",
        scenario="Intent + follow-up question in same utterance — intent classified, no double-ask",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_intent_with_side_question(run_intake_conversation, assert_and_record):
    """
    User gives claim intent plus an unrelated side question ('Do you speak Spanish?').
    Agent should extract the primary intent (claim_services) correctly;
    the side question should not break classification or cause escalation.
    """
    record = await run_intake_conversation(
        user_inputs=["I want to check on my claim. Do you speak Spanish?"],
        test_name="test_intake_intent_with_side_question",
        scenario="Claim intent + language side question — intent extracted despite noise",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_very_verbose_user(run_intake_conversation, assert_and_record):
    """
    User provides a very long, conversational explanation.
    The LLM should still extract the core intent (provider_services) correctly
    from a verbose multi-clause utterance.
    """
    verbose_input = (
        "Yeah so I've been having trouble with my knee and my doctor said "
        "I need to see a provider and I'm not sure if the orthopedic doctor "
        "I want to see is covered by my insurance so I wanted to check that"
    )

    record = await run_intake_conversation(
        user_inputs=[verbose_input],
        test_name="test_intake_very_verbose_user",
        scenario="Long verbose utterance — intent (provider_services) should still be extracted correctly",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_greeting_then_explicit_intent(run_intake_conversation, assert_and_record):
    """
    Turn 1: pure greeting ('Good morning!').
    Turn 2: explicit claim follow-up intent.
    After the greeting the agent should ask what the caller needs;
    after turn 2 it should classify claim_services.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Good morning!",
            "I need to file a claim follow-up",
        ],
        test_name="test_intake_greeting_then_explicit_intent",
        scenario="Greeting-only first turn then explicit intent — expects claim_services on turn 2",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services_after_greeting"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_offtopic_sandwiched_between_valid(run_intake_conversation, assert_and_record):
    """
    Turn 1: unclear/vague → agent asks for clarification.
    Turn 2: off-topic (pizza) → redirected.
    Turn 3: valid provider_services intent → classified correctly.
    Tests that a single offtopic turn sandwiched in a clarification flow
    does not break intent classification.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "I have a question",
            "What is the best pizza place?",
            "I need to find an in-network provider",
        ],
        test_name="test_intake_offtopic_sandwiched_between_valid",
        scenario="Unclear → off-topic sandwich → valid intent: should classify provider_services",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


@pytest.mark.live
async def test_intake_repeated_offtopic_exact_threshold_minus_one(run_intake_conversation, assert_and_record):
    """
    Send MAX_SLOT_ATTEMPTS-1 (2) off-topic messages then a valid intent.
    Verifies the system does NOT escalate when the caller recovers on the
    turn immediately before the threshold would fire (count reaches 2, not 3).
    This is the boundary condition test for offtopic_global_count.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Tell me a joke",
            "What's the capital of France?",
            "I need to check on a health insurance claim",
        ],
        test_name="test_intake_repeated_offtopic_exact_threshold_minus_one",
        scenario="Two off-topic turns (threshold-1) then valid claim — must NOT escalate",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_not_escalated(record), "no_escalation_at_boundary"),
        ],
    )


@pytest.mark.live
async def test_intake_offtopic_recovery_with_answered_with_followup(
    run_intake_conversation, assert_and_record
):
    """
    Turn 1: off-topic → redirected.
    Turn 2: valid intent PLUS a side question in the same utterance
            (EventType.ANSWERED_WITH_FOLLOWUP path).
    Verifies the ANSWERED_WITH_FOLLOWUP branch handles post-offtopic recovery correctly.
    """
    record = await run_intake_conversation(
        user_inputs=[
            "Who won the game last night?",
            "I need to find a doctor — sorry, can you repeat what you just said?",
            "Actually, I wanted to find a doctor?",
        ],
        test_name="test_intake_offtopic_recovery_with_answered_with_followup",
        scenario="Off-topic then intent+followup in same utterance — intent classified correctly",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_not_escalated(record), "no_escalation"),
        ],
    )


# ===========================================================================
# GROUP F — Conversation Continuity
# ===========================================================================


@pytest.mark.live
async def test_intake_re_entry_guard(run_intake_conversation, assert_and_record):
    """
    If call_intent is already set before IntakeAgent runs, the early-exit
    guard at the top of IntakeAgent.run() should fire immediately.

    We simulate this by running a full happy-path first (which sets the
    intent), then confirming the record reflects clean short-circuit.
    In practice the guard fires on re-entry from the orchestrator.

    Here we test it indirectly: after intent classification the state has
    call_intent set; a second call with the same thread would short-circuit.
    The assertion verifies the happy-path state correctly has call_intent set
    so re-entry would work.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to check on my health insurance claim"],
        test_name="test_intake_re_entry_guard",
        scenario="Verify call_intent is set so re-entry guard would short-circuit on next entry",
    )

    assert_and_record(
        record,
        [
            (
                lambda: _assert_intent_is_set(record),
                "call_intent_is_set_enabling_re_entry_guard",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: assert_turn_count(record, max_turns=4), "completed_within_4_turns"),
        ],
    )


@pytest.mark.live
async def test_intake_full_happy_path_end_to_end(run_intake_conversation, assert_and_record):
    """
    Complete realistic conversation:
      1. System start (greeting received)
      2. User provides clear provider_services intent
      3. Intent classified → verification routing
      4. Bridge message (INTENT_BRIDGE_MSGS) asks for first name

    Asserts all key state fields and saves full transcript.
    This is the canonical 'everything works' smoke test.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to find an in-network primary care doctor"],
        test_name="test_intake_full_happy_path_end_to_end",
        scenario="Full end-to-end happy path — greeting → intent → bridge → verification routing",
    )

    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification_agent"),
            (
                lambda: assert_any_agent_message_contains(record, "first name"),
                "bridge_message_asks_first_name",
            ),
            (lambda: assert_not_escalated(record), "no_escalation"),
            (lambda: _assert_greeting_received(record), "greeting_delivered_on_turn_0"),
            (lambda: assert_turn_count(record, min_turns=2, max_turns=5), "turn_count_reasonable"),
            (
                lambda: _assert_active_agent_was_intake(record),
                "active_agent_was_intake_agent",
            ),
        ],
    )


# ===========================================================================
# Latency helpers
# ===========================================================================


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
    p50 = _compute_latency_percentile(record, 50)
    assert p50 <= threshold_sec, f"p50 latency {p50:.3f}s exceeds threshold {threshold_sec:.3f}s"


def assert_p95_under(record: ConversationRecord, threshold_sec: float) -> None:
    p95 = _compute_latency_percentile(record, 95)
    assert p95 <= threshold_sec, f"p95 latency {p95:.3f}s exceeds threshold {threshold_sec:.3f}s"


def _print_latency_summary(record: ConversationRecord) -> None:
    """Print a per-turn and aggregate latency table to stdout."""
    durations = [t.duration_sec for t in record.turns if t.duration_sec > 0]
    if not durations:
        return
    p50 = _compute_latency_percentile(record, 50)
    p95 = _compute_latency_percentile(record, 95)
    avg = statistics.mean(durations)
    print(f"\n  Latency — turns={len(durations)}  avg={avg:.3f}s  p50={p50:.3f}s  p95={p95:.3f}s")
    for t in record.turns:
        if t.duration_sec > 0:
            print(f"    turn {t.turn_number:>2} ({t.user_input[:40]!r:<42}) {t.duration_sec:.3f}s")


# ===========================================================================
# GROUP G — Latency Benchmarks
# Per-scenario p50/p95 thresholds. Intentionally generous (10s p95) so they
# catch regressions rather than enforce SLA. Tighten as baseline is established.
# ===========================================================================

_LATENCY_P50_SEC = 2.0  # per-turn median threshold
_LATENCY_P95_SEC = 3.0  # per-turn tail threshold


@pytest.mark.live
async def test_latency_happy_path(run_intake_conversation, assert_and_record):
    """
    p50/p95 per-turn latency for the happy-path flow (greeting → intent → bridge).
    Validates that the common case does not regress beyond generous thresholds.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to find a doctor in my network"],
        test_name="test_latency_happy_path",
        scenario="Latency: happy-path provider_services intent classification",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_unclear_then_clarifies(run_intake_conversation, assert_and_record):
    """
    p50/p95 for a two-turn flow: vague greeting then explicit claim intent.
    Covers the multi-turn path where the LLM runs twice (once per user turn).
    """
    record = await run_intake_conversation(
        user_inputs=["Hi", "I need to check on a claim"],
        test_name="test_latency_unclear_then_clarifies",
        scenario="Latency: unclear first turn then claim_services clarification",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_transfer_request(run_intake_conversation, assert_and_record):
    """
    p50/p95 for a guard-trigger path (TRANSFER_REQUEST → escalation_agent).
    Guard paths are cheaper (static message, no LLM generation) so p95 should
    comfortably fit within the threshold.
    """
    record = await run_intake_conversation(
        user_inputs=["I want to speak to a real person please"],
        test_name="test_latency_transfer_request",
        scenario="Latency: TRANSFER_REQUEST guard path through escalation_agent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_out_of_scope(run_intake_conversation, assert_and_record):
    """
    p50/p95 for the out-of-scope routing path (direct to END, keyword routing,
    no LLM response generation). Should be the fastest non-trivial path.
    """
    record = await run_intake_conversation(
        user_inputs=["I need to pay my bill"],
        test_name="test_latency_out_of_scope",
        scenario="Latency: out-of-scope billing intent routed to END",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_call_ended(record), "call_ended_cleanly"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_max_clarification_escalation(run_intake_conversation, assert_and_record):
    """
    p50/p95 for the longest clarification path (3 unclear turns → escalation).
    This is the worst-case LLM call count in the intake flow; p95 captures
    any turn that incurs extra generation LLM cost.
    """
    record = await run_intake_conversation(
        user_inputs=["I don't know", "Not sure what I need", "I'm confused"],
        test_name="test_latency_max_clarification_escalation",
        scenario="Latency: three unclear turns exhausting MAX_CLARIFICATION_ATTEMPTS",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_escalated(record), "escalation_triggered"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_offtopic_then_valid(run_intake_conversation, assert_and_record):
    """p50/p95 for the two-turn offtopic-recovery path."""
    record = await run_intake_conversation(
        user_inputs=[
            "What's the weather like today?",
            "I need to find a doctor in my network",
        ],
        test_name="test_latency_offtopic_then_valid",
        scenario="Latency: one off-topic turn then valid provider intent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


@pytest.mark.live
async def test_latency_two_offtopic_then_valid(run_intake_conversation, assert_and_record):
    """p50/p95 for the three-turn two-offtopic-then-valid path."""
    record = await run_intake_conversation(
        user_inputs=[
            "Hi",
            "who am i talking to?",
            "okay lets focus on claims. I want to follow up on a claim I submitted",
        ],
        test_name="test_latency_two_offtopic_then_valid",
        scenario="Latency: two off-topic turns then valid claim intent",
    )
    assert_and_record(
        record,
        [
            (lambda: assert_intent(record, "claim_services"), "intent==claim_services"),
            (lambda: assert_p50_under(record, _LATENCY_P50_SEC), f"p50<{_LATENCY_P50_SEC}s"),
            (lambda: assert_p95_under(record, _LATENCY_P95_SEC), f"p95<{_LATENCY_P95_SEC}s"),
        ],
    )


# ===========================================================================
# Private assertion helpers (not part of the public API)
# ===========================================================================


def _assert_no_intent(record: ConversationRecord) -> None:
    intent = record.final_state.get("call_intent", "")
    assert not intent, f"Expected call_intent to be empty/unset, got {intent!r}"


def _assert_still_in_intake(record: ConversationRecord) -> None:
    next_node = record.final_state.get("next_node", "")
    assert next_node == "intake_agent", (
        f"Expected next_node='intake_agent' (agent asking for clarification), got {next_node!r}"
    )


def _assert_caller_type_handled(record: ConversationRecord) -> None:
    handled = record.final_state.get("caller_type_handled", False)
    assert handled is True, f"Expected caller_type_handled=True, got {handled!r}"


def _assert_not_caller_type(record: ConversationRecord, bad_type: str) -> None:
    actual = record.final_state.get("caller_type", "")
    assert actual != bad_type, f"Expected caller_type != {bad_type!r}, but got {actual!r}"


def _assert_intent_is_set(record: ConversationRecord) -> None:
    intent = record.final_state.get("call_intent", "")
    assert intent and intent not in ("", "unclear"), (
        f"Expected call_intent to be set to a real intent, got {intent!r}"
    )


def _assert_greeting_received(record: ConversationRecord) -> None:
    if not record.turns:
        raise AssertionError("No turns recorded")
    # Turn 0 is the system start — the agent message is the greeting
    turn_0 = record.turns[0]
    msg = (turn_0.agent_message or "").lower()
    assert "sagility" in msg or "quality assurance" in msg or "thank you for calling" in msg, (
        f"Expected regulatory greeting on turn 0, got: {turn_0.agent_message!r}"
    )


def _assert_active_agent_was_intake(record: ConversationRecord) -> None:
    # At least one turn should show active_agent == intake_agent
    was_intake = any(t.active_agent == "intake_agent" for t in record.turns)
    assert was_intake, "Expected intake_agent to be active_agent in at least one turn"
