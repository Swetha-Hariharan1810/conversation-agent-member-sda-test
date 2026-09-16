"""
call_stages.py — what the call still has to cover, in spoken form.

"Coming up:" lets the generation LLM answer a side question about a step the
call has not reached yet ("will I get a text about this?") in the sentence the
caller is already getting, instead of promising to return to it. Until now that
line carried only the remaining slots of the CURRENT agent's pipeline, so the
question was answerable while the current agent had slots left and declined
otherwise — at the tail of every pipeline, and throughout verification, which
is where most side questions are actually asked.

This module widens it to the rest of the call: the stages still ahead of the
agent that is running, with the conditional ones filtered the way the router
filters them.

Order and conditions mirror orchestration/fast_path.get_fast_path_route — the
two must be kept in step. This module is the spoken view of that routing, not a
second source of truth for it: nothing routes from here.

Keep this module dependency-free (core ↔ agents import safety).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Stage order per intent. follow_up and closure are deliberately absent: "is
# there anything else?" is not a step a caller asks about in advance, and
# naming it would invite the model to promise the end of the call.
_PROVIDER_FLOW: tuple[str, ...] = (
    "provider_search_agent",
    "delivery_management_agent",
    "benefits_agent",
    "care_wellness_agent",
)
_CLAIM_FLOW: tuple[str, ...] = (
    "claim_adjustment_agent",
    "records_coordination_agent",
    "notification_setup_agent",
)

_FLOWS: dict[str, tuple[str, ...]] = {
    "provider_services": _PROVIDER_FLOW,
    "claim_services": _CLAIM_FLOW,
}

# Agents that run before the domain flow begins — the whole flow is ahead.
_PRE_FLOW_AGENTS = ("intake_agent", "verification_agent")

# Spoken labels. These reach the generation LLM as "Coming up:" entries, so each
# has to read as a step the caller would recognise being told about — not a node
# name, and not a field to hand over.
STAGE_LABELS: dict[str, str] = {
    "provider_search_agent": "finding in-network providers near you",
    "delivery_management_agent": "choosing whether the provider list comes by fax or email",
    "benefits_agent": "going over the benefits for office visits",
    "care_wellness_agent": "an offer of a free health and wellness coach",
    "claim_adjustment_agent": "looking up the claim",
    "records_coordination_agent": "arranging the medical records for the claim",
    "notification_setup_agent": "choosing SMS or email for claim status updates",
}


# The same rule, applied to the other half of the line. "Coming up:" is built
# from two lists: the stages after this agent, labelled above, and the slots
# this agent has left — and those went in as their own field names, with the
# underscores swapped for spaces:
#
#     Coming up: personal guide consent, choosing SMS or email for claim status updates
#
# The caller had just asked us to contact their doctor's office. The answer was
# the first item on that line, and the model did not recognise it as one,
# because "personal guide consent" is not a thing that happens to a caller —
# it is the name of a field. So it took the question as one the call could not
# reach, declined it ("a representative would need to make that change"), and
# the system appended the offer it had been about to make anyway:
#
#     AI  I understand you'd prefer we contact your doctor's office directly.
#         A representative would need to make that change. I can also have one
#         of our Personal Guides contact your doctor's office on your behalf.
#         Would you like us to proceed with that?
#
# A slot with no entry here still goes in as its own name, which is right for
# the ones that read as themselves ("provider type", "zip code"). Only the
# steps a caller would not recognise under their field name need translating.
SLOT_STAGE_LABELS: dict[str, str] = {
    # ── records coordination ─────────────────────────────────────────────────
    "upload_method": "choosing how the medical records reach us",
    "upload_consent": "an offer of a secure upload link by email",
    "personal_guide_consent": (
        "an offer to have a Personal Guide contact your doctor's office for the records"
    ),
    # ── delivery management ──────────────────────────────────────────────────
    "delivery_method": "choosing whether the provider list comes by fax or email",
    "fax_confirmed": "confirming the fax number the list goes to",
    "fax": "the fax number the list goes to",
    "email_confirmed": "confirming the email address",
    "email": "the email address to use",
    "benefits_response": "an offer to go over the benefits for office visits",
    # ── notification setup ───────────────────────────────────────────────────
    "notification_method": "choosing SMS or email for claim status updates",
    "phone_confirmed": "confirming the phone number for updates",
    "phone": "the phone number for updates",
    "timeline_question": "an offer of updates as the claim moves along",
    "n2_notification_method": "choosing how those timeline updates reach you",
}


def spoken_slot_stage(slot: str) -> str:
    """A pipeline slot as a step the caller would recognise being told about."""
    return SLOT_STAGE_LABELS.get(slot, (slot or "").replace("_", " "))


def _still_ahead(agent: str, state: Mapping[str, Any]) -> bool:
    """Is this stage still going to happen?

    The conditional stages mirror the guards the router applies. A stage this
    call will skip must never be promised: the caller is told about a step that
    never arrives, which is the failure parking used to produce.
    """
    if agent == "delivery_management_agent":
        return not state.get("provider_list_sent")
    if agent == "benefits_agent":
        return not state.get("benefits_explained")
    if agent == "care_wellness_agent":
        return not state.get("care_coach_offered")
    if agent == "records_coordination_agent":
        return bool(state.get("records_required")) and not state.get("records_branch_taken")
    if agent == "notification_setup_agent":
        channel = str(state.get("notification_channel") or "").strip()
        return not channel or channel == "not_set"
    return True


def remaining_call_stages(*, intent: str, current_agent: str, state: Mapping[str, Any]) -> list[str]:
    """Spoken labels for the stages still ahead of ``current_agent``.

    ``intent`` is passed rather than read from state because intake knows the
    intent a turn before state does — and intake's first turn is exactly when
    the whole flow is still ahead.

    Returns [] for an agent outside the flow (follow_up, closure, escalation)
    and for an unknown intent: with nothing to promise, a question the payload
    cannot answer is declined where it is asked.
    """
    flow = _FLOWS.get((intent or "").strip())
    if not flow:
        return []
    if current_agent in flow:
        rest = flow[flow.index(current_agent) + 1 :]
    elif current_agent in _PRE_FLOW_AGENTS:
        rest = flow
    else:
        return []
    return [STAGE_LABELS[agent] for agent in rest if _still_ahead(agent, state)]
