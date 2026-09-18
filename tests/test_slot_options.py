"""Tests for the closed-set slot options registry and the gap fill it feeds.

The registry is the single source of truth for "what counts as an answer to
this slot". Three things have to hold for that to be true:

  1. every option's value is one the slot's own validator accepts — otherwise
     the prompt offers the model a value that cannot be stored;
  2. the utterances from the production transcripts read to the right option;
  3. the gap fill in request_detection fills only, and only in the window it
     is allowed to.
"""

from __future__ import annotations

import pytest

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import EventType, GuardType, WorkerResult
from agent.slots import validators
from agent.slots.options import (
    SLOT_OPTIONS,
    get_option_set,
    is_closed_set,
    match_option,
    option_values,
    render_open_options,
    render_options,
)

# Slot → the validator that guards it in production. upload_method has none;
# the agent branches on the literal value, which is why it is listed with the
# branch values it is read against instead.
_VALIDATOR_FOR = {
    "delivery_method": validators.validate_delivery_method,
    "notification_method": validators.validate_notification_method,
    "provider_type": validators.validate_provider_type,
    "relationship": validators.validate_relationship,
}


# ── The registry agrees with the rest of the system ──────────────────────────


@pytest.mark.parametrize("slot", sorted(_VALIDATOR_FOR))
def test_every_option_passes_the_slots_validator(slot):
    """A value the prompt offers must be a value the system can store.

    This is the check that catches drift: notification_method's normalizer can
    return "both" while its validator accepts only sms and email, and offering
    "both" would put the caller through a choice that fails on save.
    """
    validate = _VALIDATOR_FOR[slot]
    for value in option_values(slot):
        assert validate(value).valid, f"{slot}={value!r} is offered but the validator rejects it"


def test_upload_method_values_match_the_agents_branches():
    from agent.agents.records_coordination import agent as records_agent  # noqa: F401

    assert option_values("upload_method") == (
        "personal_guide",
        "doctor_direct",
        "member_upload",
        "decline",
    )


def test_registry_is_keyed_by_its_own_slot_names():
    for key, option_set in SLOT_OPTIONS.items():
        assert key == option_set.slot


def test_lookups_are_quiet_on_unknown_and_empty_slots():
    assert get_option_set("first_name") is None
    assert get_option_set("") is None
    assert get_option_set(None) is None
    assert not is_closed_set("dob")
    assert option_values("dob") == ()


# ── Reading an option out of the caller's words ──────────────────────────────


@pytest.mark.parametrize(
    "utterance,expected",
    [
        # The transcript-1 utterance. This is the whole reason the module exists.
        ("Can I ask my doctor to send them over?", "doctor_direct"),
        ("My doctor will send it", "doctor_direct"),
        ("the provider can send it", "doctor_direct"),
        ("I'll have my doctor's office send them", "doctor_direct"),
        # personal_guide has to beat doctor_direct: both name the provider.
        ("Can you contact my doctor's office for me?", "personal_guide"),
        ("could someone there chase it up on my behalf", "personal_guide"),
        # doctor_direct has to beat member_upload: "I" is the subject, the
        # doctor is the sender.
        ("I'll upload them myself", "member_upload"),
        ("send me the link", "member_upload"),
        ("I'll do it online", "member_upload"),
        # Never guessed — inferring it escalates the call.
        ("no thanks", ""),
        ("I don't want to proceed", ""),
        ("", ""),
    ],
)
def test_match_upload_method(utterance, expected):
    assert match_option("upload_method", utterance) == expected


@pytest.mark.parametrize(
    "slot,utterance,expected",
    [
        ("delivery_method", "And you send it to my fax, please?", "fax"),
        ("delivery_method", "email it to me", "email"),
        ("delivery_method", "both would be great", "both"),
        ("delivery_method", "whatever is easiest", ""),
        # An ASR mishearing of "fax" the normalizer already knows about, and
        # which the registry inherits rather than re-listing.
        ("delivery_method", "send it by facts", "fax"),
        ("provider_type", "I need a heart doctor", "Cardiologist"),
        ("provider_type", "my regular doctor", "Primary Care Physician"),
        ("provider_type", "someone for my knee", ""),
        ("notification_method", "text me", "sms"),
        ("notification_method", "by email please", "email"),
        ("caller_role", "Calling for myself.", "plan_holder"),
        ("caller_role", "it's for my daughter", "dependent"),
        ("relationship", "I'm the policy holder", "plan_holder"),
    ],
)
def test_match_via_the_slots_normalizer(slot, utterance, expected):
    assert match_option(slot, utterance) == expected


def test_match_is_empty_for_slots_that_are_not_closed_sets():
    assert match_option("dob", "April twelfth nineteen eighty eight") == ""
    assert match_option(None, "fax") == ""


def test_notification_method_never_yields_both():
    """normalize_notification_method maps "both" → "both"; the validator does
    not accept it, so the matcher must discard it rather than pass it on."""
    assert match_option("notification_method", "both, please") == ""


# ── What the extraction model is shown ───────────────────────────────────────


def test_render_names_every_value_and_refuses_to_force_a_fit():
    block = render_options("upload_method")
    for value in option_values("upload_method"):
        assert value in block
    assert "never force a fit" in block


def test_render_is_empty_for_an_open_slot():
    assert render_options("dob") == ""
    assert render_options("") == ""


def test_render_open_options_skips_the_slot_being_asked():
    block = render_open_options(["upload_method", "delivery_method"], exclude="delivery_method")
    assert "upload_method" in block
    assert "delivery_method" not in block


def test_render_open_options_is_empty_when_nothing_open_is_a_closed_set():
    assert render_open_options(["dob", "member_id"]) == ""
    assert render_open_options(None) == ""


def test_accepted_answers_reach_the_extraction_prompt():
    messages = build_worker_input(
        "SYSTEM",
        awaiting_slot="upload_method",
        last_agent_message="Are you able to provide those records?",
        last_user_message="Can I ask my doctor to send them over?",
    )
    user_content = messages[1]["content"]
    assert "Accepted answers for upload_method:" in user_content
    assert "doctor_direct" in user_content


def test_a_still_open_closed_set_reaches_the_extraction_prompt():
    """Transcript 1's shape: the answer arrives one turn after the question.

    upload_consent is what is being asked; upload_method was asked before it
    and is still open, so its values have to be visible or the caller's answer
    has nowhere to land.
    """
    messages = build_worker_input(
        "SYSTEM",
        awaiting_slot="upload_consent",
        last_agent_message="Would you like me to send that link over?",
        last_user_message="Can I ask my doctor to send them over?",
        pending_slots=["upload_method"],
    )
    user_content = messages[1]["content"]
    assert "Also answerable" in user_content
    assert "doctor_direct" in user_content


def test_an_open_slot_adds_nothing_to_the_prompt():
    messages = build_worker_input(
        "SYSTEM",
        awaiting_slot="dob",
        last_agent_message="And your date of birth?",
        last_user_message="April twelfth nineteen eighty eight.",
    )
    assert "Accepted answers" not in messages[1]["content"]


# ── The gap fill: fills only, and only in its window ─────────────────────────


def test_fills_a_closed_set_the_model_missed():
    result = reconcile_worker_result(
        WorkerResult(event_type=EventType.AMBIGUOUS),
        "Can I ask my doctor to send them over?",
        awaiting_slot="upload_method",
    )
    assert result.extracted == {"upload_method": "doctor_direct"}
    # AMBIGUOUS carrying a value is a combination no downstream branch expects.
    assert result.event_type == EventType.ANSWERED


def test_never_contradicts_a_value_the_model_returned():
    result = reconcile_worker_result(
        WorkerResult(extracted={"upload_method": "member_upload"}),
        "Can I ask my doctor to send them over?",
        awaiting_slot="upload_method",
    )
    assert result.extracted == {"upload_method": "member_upload"}


def test_never_contradicts_a_correction_the_model_returned():
    result = reconcile_worker_result(
        WorkerResult(
            corrections={"delivery_method": "email"},
            event_type=EventType.CORRECTED,
        ),
        "actually send it to my fax",
        awaiting_slot="delivery_method",
    )
    assert not (result.extracted or {}).get("delivery_method")


def test_leaves_answered_with_followup_alone():
    """The caller answered AND asked. Flattening the event drops the question."""
    result = reconcile_worker_result(
        WorkerResult(
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            followup_query="Can I change my ZIP code?",
        ),
        "My doctor can send them. But can I change my ZIP code?",
        awaiting_slot="upload_method",
    )
    assert result.extracted == {"upload_method": "doctor_direct"}
    assert result.event_type == EventType.ANSWERED_WITH_FOLLOWUP


def test_a_guard_outranks_the_option_reading():
    result = reconcile_worker_result(
        WorkerResult(guard=GuardType.TRANSFER_REQUEST, guard_confidence=0.9),
        "Forget it, have my doctor's office call a real person instead",
        awaiting_slot="upload_method",
    )
    assert not (result.extracted or {})


def test_cannot_provide_outranks_the_option_reading():
    result = reconcile_worker_result(
        WorkerResult(cannot_provide=True),
        "I can't get hold of my doctor to send anything",
        awaiting_slot="upload_method",
    )
    assert not (result.extracted or {})


def test_a_wait_is_not_an_answer():
    result = reconcile_worker_result(
        WorkerResult(event_type=EventType.WAIT),
        "Hold on, let me call my doctor's office and ask them to send it",
        awaiting_slot="upload_method",
    )
    assert not (result.extracted or {})
    assert result.event_type == EventType.WAIT


def test_only_the_slot_being_asked_is_filled():
    """delivery_method is a closed set, but it is not what is being asked.

    Reading a value out of a turn about something else is how a caller ends up
    confirming a choice they never made.
    """
    result = reconcile_worker_result(
        WorkerResult(event_type=EventType.AMBIGUOUS),
        "send it to my fax",
        awaiting_slot="upload_consent",
    )
    assert not (result.extracted or {})


def test_no_awaiting_slot_is_a_no_op():
    """Call sites that do not pass awaiting_slot keep their old behaviour."""
    result = reconcile_worker_result(WorkerResult(event_type=EventType.AMBIGUOUS), "send it to my fax")
    assert not (result.extracted or {})


def test_a_miss_changes_nothing():
    result = reconcile_worker_result(
        WorkerResult(event_type=EventType.AMBIGUOUS),
        "sorry, what was the question?",
        awaiting_slot="upload_method",
    )
    assert not (result.extracted or {})
    assert result.event_type == EventType.AMBIGUOUS
