"""The turn said no and yes about the same thing.

    AI    I can also generate a secure upload link and send it to your email.
          Would you like me to do that?
    User  no — actually, I've changed my mind, I don't want the link. could you
          just reach out to my doctor's office for me instead?
    AI    I understand you'd prefer we contact your doctor's office directly.
          A representative would need to make that change. I can also have one
          of our Personal Guides contact your doctor's office on your behalf.
          Would you like us to proceed with that?

One turn, built by two layers that cannot see each other. The caller's side
question went to the generation LLM, which found no answer for it and declined;
that sentence was carried in pending_side_answer; and the agent then spoke the
Personal Guide offer — which is that same outreach, offered.

The decline should never have been written. "Coming up:" is built from two
lists: the stages after this agent, which have spoken labels, and the slots this
agent has left, which went in as their own field names with the underscores
swapped for spaces. The answer to the caller's question was the first item on
that line — as "personal guide consent", which is not a thing that happens to a
caller, it is the name of a field. So the model read it as a step the call could
not reach and declined.

Both halves of the line are spoken now. Behind that, a carried decline is
dropped when the turn it rides in front of offers the same thing — whatever the
generation layer produced, a caller is never told no and yes about one thing.
"""

from __future__ import annotations

import pytest

from agent.agents.records_coordination.constants import (
    MSG_PERSONAL_GUIDE_OFFER,
    MSG_PERSONAL_GUIDE_OFFER_ALSO,
)
from agent.core.call_stages import SLOT_STAGE_LABELS, spoken_slot_stage
from agent.core.slot_manager import SlotManagerMixin

DECLINE = (
    "I understand you'd prefer we contact your doctor's office directly. "
    "A representative would need to make that change."
)
GUIDE_OFFER = (
    "I can have one of our Personal Guides contact your doctor's office on your behalf. "
    "Would you like us to proceed with that?"
)


def _join(answer: str, message: str) -> str:
    return SlotManagerMixin.join_side_answer(answer, message)


# ── the reported turn ────────────────────────────────────────────────────────


def test_the_turn_does_not_say_no_before_saying_yes():
    assert _join(DECLINE, GUIDE_OFFER) == GUIDE_OFFER


@pytest.mark.parametrize("offer", MSG_PERSONAL_GUIDE_OFFER + MSG_PERSONAL_GUIDE_OFFER_ALSO)
def test_whichever_line_the_offer_pool_draws(offer):
    assert "representative" not in _join(DECLINE, offer)


@pytest.mark.parametrize(
    "decline",
    [
        "A representative would need to make that change to the doctor's office contact.",
        "Contacting your doctor's office isn't something this line handles.",
        "I'm not able to do that — reaching your doctor's office would need a representative.",
    ],
)
def test_every_shape_the_decline_takes(decline):
    assert _join(decline, GUIDE_OFFER) == GUIDE_OFFER


def test_the_offer_still_speaks_when_the_decline_is_dropped():
    """Only the carried sentence is ever dropped. The handler's own message —
    the question the caller has to answer — always survives."""
    out = _join(DECLINE, GUIDE_OFFER)
    assert out.endswith("?")
    assert "Personal Guides" in out


# ── a decline this turn does not answer is still owed to the caller ──────────


def test_an_unrelated_decline_is_not_swallowed():
    """ "A representative would need to change your mailing address", spoken in
    front of a fax offer, answers a question the offer does not."""
    unrelated = "A representative would need to make that change to your mailing address."
    fax_offer = "I can send the provider list to your fax at 6175554199. Would you like me to do that?"
    assert _join(unrelated, fax_offer) == f"{unrelated} {fax_offer}"


@pytest.mark.parametrize(
    "answer",
    [
        "Your ZIP is on file as 90210.",
        "Got it on your email — you'll choose fax or email in just a moment.",
        "Prescriptions are handled by our pharmacy benefits team.",
        "Yes, we'll send a notification to the number on file.",
    ],
)
def test_an_ordinary_side_answer_rides_along_as_before(answer):
    assert _join(answer, GUIDE_OFFER) == f"{answer} {GUIDE_OFFER}"


def test_a_decline_with_nothing_to_ride_in_front_of_still_speaks():
    assert _join(DECLINE, "") == DECLINE


def test_an_offer_with_no_decline_in_front_is_untouched():
    assert _join("", GUIDE_OFFER) == GUIDE_OFFER


# ── the line the model reads ─────────────────────────────────────────────────


def test_the_step_the_caller_asked_about_is_named_in_words():
    spoken = spoken_slot_stage("personal_guide_consent")
    assert "personal guide consent" != spoken
    for word in ("Personal Guide", "doctor's office", "records"):
        assert word in spoken


@pytest.mark.parametrize("slot,label", sorted(SLOT_STAGE_LABELS.items()))
def test_a_slot_label_reads_as_a_step_not_a_field(slot, label):
    assert "_" not in label
    assert label == label.strip()
    assert label != slot.replace("_", " ")


def test_a_slot_with_no_label_still_reads_as_itself():
    """ "provider type" and "zip code" are already things a caller would
    recognise — only the steps that are not need translating."""
    assert spoken_slot_stage("provider_type") == "provider type"
    assert spoken_slot_stage("zip_code") == "zip code"


@pytest.mark.parametrize(
    "order,module",
    [
        ("RECORDS_SLOT_ORDER", "agent.agents.records_coordination.constants"),
        ("NOTIFICATION_SLOT_ORDER", "agent.agents.notification_setup.constants"),
        ("DELIVERY_SLOT_ORDER", "agent.agents.delivery_management.constants"),
    ],
)
def test_every_step_of_an_offer_pipeline_is_spoken(order, module):
    """These pipelines are made of offers and channel choices — none of their
    slots read as a step under their own field name."""
    import importlib

    slots = getattr(importlib.import_module(module), order)
    missing = [s for s in slots if s not in SLOT_STAGE_LABELS]
    assert not missing, f"{order} slots with no spoken label: {missing}"


# ── "also" belongs to the turn that has something to add to ──────────────────


def test_the_declined_link_is_not_offered_a_guide_as_well():
    """The caller has just refused the upload link. An offer made "also" reads
    as though we were not listening."""
    for line in MSG_PERSONAL_GUIDE_OFFER:
        assert " also " not in line


def test_the_sent_link_is():
    """The link went out and this is the other thing we can do on top of it."""
    for line in MSG_PERSONAL_GUIDE_OFFER_ALSO:
        assert "also" in line


def test_each_path_draws_from_the_pool_written_for_it():
    import inspect

    from agent.agents.records_coordination.agent import RecordsCoordinationAgent as R

    declined = inspect.getsource(R._handle_guide_consent_ask)
    assert "MSG_PERSONAL_GUIDE_OFFER)" in declined
    sent = inspect.getsource(R._send_link_and_proceed)
    assert "MSG_PERSONAL_GUIDE_OFFER_ALSO)" in sent
