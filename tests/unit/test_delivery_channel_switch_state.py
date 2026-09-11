"""Channel switch (fax → email) must survive into the Care Coach dispatch.

Transcript bug: the caller confirmed fax, then changed their mind mid-readback
("can you send it to an email instead?"), gave the email, and received the
provider list by email — but the Care Coach details that followed went out to
the abandoned fax:

    ai   I'll send it to 2155553299 — is that the right fax number?
    human Oh, I'm sorry. Can you send it to an email instead?
    ...
    ai   I'll send it to the same fax 2155553299 you provided.   ← wrong

Root cause is the slot record, not the message template. slot_attempts is
shared state: delivery_management rewrote state["delivery_method"] to "email"
but left the delivery_method slot record confirmed as "fax", and
SignalsMixin.ask_member re-persists confirmed slot values on every interrupt —
so the very next agent's question (benefits' Care Coach offer) wrote "fax"
back over "email" and care_wellness resolved the fax contact.

Covers both halves of the fix:
  1. ask_member never resurrects a stale slot record over a diverged live value.
  2. the switch keeps the delivery_method slot record in step with state.
"""

from __future__ import annotations

import pytest

from agent.agents.benefits.agent import BenefitsAgent
from agent.agents.care_wellness.handlers import _resolve_delivery_contact
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import WorkerResult

FAX = "2155553299"
EMAIL = "james.wilson@gmail.com"


def _state(**overrides) -> dict:
    state = {
        "messages": [],
        "app_run_id": "test-run",
        "slot_attempts": {},
        "member_id": "M310188",
        "fax": FAX,
        "email": EMAIL,
        "delivery_method": "",
    }
    state.update(overrides)
    return state


def _merge(state: dict, result: dict) -> dict:
    """Apply an agent's state-update dict the way LangGraph would."""
    return {**state, **{k: v for k, v in result.items() if k != "messages"}}


# ── 1. ask_member must not resurrect a stale slot record ─────────────────────


def test_stale_slot_record_does_not_overwrite_a_diverged_live_value():
    """benefits asks its own question — it must not rewrite delivery_method."""
    state = _state(
        delivery_method="email",
        slot_attempts={"delivery_method": {"attempt_count": 0, "confirmed": True, "last_value": "fax"}},
    )
    result = BenefitsAgent.from_state(state).ask_member(state, "Care Coach details?")

    assert result.get("delivery_method", "email") == "email"
    assert _resolve_delivery_contact(_merge(state, result)) == ("email", EMAIL)


def test_a_slot_confirmed_this_turn_still_reaches_state():
    """The Phase 2 mid-pipeline persist is unchanged for live confirmations."""
    state = _state(delivery_method="fax")
    agent = DeliveryManagementAgent.from_state(state)
    agent.slot_ok("delivery_method", "email")

    result = agent.ask_member(state, "Is that the right email address?")

    assert result["delivery_method"] == "email"


def test_a_slot_record_matching_state_is_still_persisted():
    state = _state(delivery_method="fax")
    state["slot_attempts"] = {"delivery_method": {"attempt_count": 0, "confirmed": True, "last_value": "fax"}}
    result = BenefitsAgent.from_state(state).ask_member(state, "Care Coach details?")

    assert result["delivery_method"] == "fax"


# ── 2. the switch keeps the slot record in step with state ───────────────────


@pytest.mark.parametrize(
    "awaiting, old_method",
    [("fax_confirmed", "fax"), ("email_confirmed", "email")],
)
def test_channel_switch_rewrites_the_delivery_method_slot_record(awaiting, old_method):
    new_method = "email" if old_method == "fax" else "fax"
    state = _state(
        delivery_method=old_method,
        awaiting_slot=awaiting,
        slot_attempts={"delivery_method": {"attempt_count": 0, "confirmed": True, "last_value": old_method}},
    )
    agent = DeliveryManagementAgent.from_state(state)

    switch = agent._maybe_switch_method(
        state,
        WorkerResult(extracted={"delivery_method": new_method}),
        awaiting,
        old_method,
        FAX,
        EMAIL,
    )

    assert switch is not None
    assert switch["delivery_method"] == new_method
    assert switch["slot_attempts"]["delivery_method"]["last_value"] == new_method


def test_switched_method_survives_the_next_agents_question():
    """End-to-end on the transcript: fax → email, then the Care Coach offer."""
    state = _state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        slot_attempts={"delivery_method": {"attempt_count": 0, "confirmed": True, "last_value": "fax"}},
    )
    agent = DeliveryManagementAgent.from_state(state)

    switch = agent._maybe_switch_method(
        state,
        WorkerResult(extracted={"delivery_method": "email"}),
        "fax_confirmed",
        "fax",
        FAX,
        EMAIL,
    )
    state = _merge(state, switch)
    assert state["delivery_method"] == "email"

    # benefits asks the Care Coach offer — the interrupt used to write "fax" back
    offer = BenefitsAgent.from_state(state).ask_member(state, "Care Coach details?")
    state = _merge(state, offer)

    assert state["delivery_method"] == "email"
    assert _resolve_delivery_contact(state) == ("email", EMAIL)


def test_unconfirmed_replacement_contact_is_dropped_on_a_switch():
    """A fax the caller gave but never confirmed must not leak into state."""
    state = _state(
        delivery_method="fax",
        awaiting_slot="fax",
        slot_attempts={"fax": {"attempt_count": 0, "confirmed": True, "last_value": "9995551234"}},
    )
    agent = DeliveryManagementAgent.from_state(state)

    switch = agent._maybe_switch_method(
        state,
        WorkerResult(extracted={"delivery_method": "email"}),
        "fax",
        "fax",
        FAX,
        EMAIL,
    )
    state = _merge(state, switch)

    assert state["fax"] == FAX
