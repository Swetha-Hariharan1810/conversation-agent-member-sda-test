"""
slot_ownership.py — ownership registries: which agent owns each slot or
capability and how a caller's cross-agent request may be honored.

SLOT_OWNERSHIP (Phase 4, fixes Bug C) covers slot VALUE updates.
updatable modes:
  in_flow        — the owning agent's own pipeline collects/updates the slot;
                   when another agent is active, the update is parked as a
                   kind="action" item for follow_up (Phase 3 behavior).
  route_to_owner — the update is honored NOW: the current agent hands off to
                   the owner (pending_cross_agent_request carries the way
                   back). Never park or say "later" for these.
  human_only     — only a representative can change it; decline honestly.

invalidates: state keys stale the moment the slot changes; a routing agent
clears them before the hand-off so the owner recomputes them (and dispatch
preconditions can block on them). Keys are literal State keys.

CAPABILITY_REGISTRY (Phase 6) covers the two further request kinds real
calls contain beyond value updates:
  redo   — re-perform a completed action with a changed parameter
           ("send that list to my email instead of fax")
  replay — re-state information already given this call
           ("can you repeat my benefits again")
Keyed by (kind, topic); topics are canonicalized via resolve_capability so
extraction-level targets like "delivery_method" reach ("redo", "delivery").
Unknown topics resolve to None — callers degrade to the Phase-3
park-as-question path, never a hard decline.

Consumers: core.slot_manager (resolve_update_target / routing / parking),
agents.delivery_management + provider_search + benefits (round-trips and
re-entry contracts), orchestration.fast_path (return hop), agents.follow_up
(parked actions + live redo/replay), agents.verification.handlers
(correction scoping).

Keep this module dependency-free (core ↔ agents import safety).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Updatable = Literal["in_flow", "route_to_owner", "human_only"]


@dataclass(frozen=True)
class SlotOwnership:
    agent: str
    updatable: Updatable
    invalidates: tuple[str, ...] = ()


SLOT_OWNERSHIP: dict[str, SlotOwnership] = {
    # ZIP drives the provider search — an update must re-run the search, so
    # the current provider list and the ZIP it was built from are invalidated.
    "zip_code": SlotOwnership(
        agent="provider_search_agent",
        updatable="route_to_owner",
        invalidates=("provider_list_sent", "zip_code_used"),
    ),
    # Delivery contact details — delivery_management's own confirmation flow
    # updates them inline.
    "fax": SlotOwnership(agent="delivery_management_agent", updatable="in_flow"),
    "email": SlotOwnership(agent="delivery_management_agent", updatable="in_flow"),
    # Identity slots — verification's existing correction/detour machinery
    # updates them in flow; elsewhere they park for a re-verify.
    "first_name": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "last_name": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "member_id": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "dob": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "relationship": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    # Claim notification preferences — collected by notification_setup within
    # the claims flow.
    "notification_method": SlotOwnership(agent="notification_setup_agent", updatable="in_flow"),
    "n2_notification_method": SlotOwnership(agent="notification_setup_agent", updatable="in_flow"),
    # Human-only: system flags and SF-verified phone.
    "phone_number": SlotOwnership(agent="", updatable="human_only"),
    "member_status_verify": SlotOwnership(agent="", updatable="human_only"),
    "call_intent": SlotOwnership(agent="", updatable="human_only"),
}


def get_ownership(slot: str) -> SlotOwnership | None:
    """Registry entry for ``slot``, or None when unknown (treat as human-only)."""
    return SLOT_OWNERSHIP.get((slot or "").strip().lower())


# ── Capability registry (Phase 6: redo / replay cross-agent requests) ────────


@dataclass(frozen=True)
class Capability:
    agent: str
    description: str = ""


CAPABILITY_REGISTRY: dict[tuple[str, str], Capability] = {
    # "actually send that list to my email instead of fax" after dispatch —
    # not a slot update: re-dispatch the provider list with a new
    # method/destination.
    ("redo", "delivery"): Capability(
        agent="delivery_management_agent",
        description="re-dispatch the provider list with a new method/destination",
    ),
    # "can you repeat my benefits again" — re-explain the plan benefits
    # (fetch_benefits is idempotent from Salesforce).
    ("replay", "benefits"): Capability(
        agent="benefits_agent",
        description="re-explain the plan benefits",
    ),
    # "what did you send me / where did it go" — re-state what was sent,
    # where, and the delivery window (answerable from state:
    # delivery_method, fax/email, zip_code_used, delivery_timestamp).
    ("replay", "provider_list"): Capability(
        agent="delivery_management_agent",
        description="re-state what was sent, where, and the delivery window",
    ),
    # "actually notify me by email instead" after notification setup finished —
    # re-collect the claim notification method keeping the claim context
    # (never re-runs the timeline question).
    ("redo", "notification"): Capability(
        agent="notification_setup_agent",
        description="re-collect the claim notification method keeping the claim context",
    ),
    # "what's happening with my claim again?" — re-state the adjustment
    # status from state (idempotent read: claim_status, last_update_date,
    # reference_number).
    ("replay", "claim_status"): Capability(
        agent="claim_adjustment_agent",
        description="re-state the adjustment status from state",
    ),
}

# Extraction-level targets → canonical capability topics. update_target for a
# redo arrives as "delivery_method" (see CROSS-CALL REQUESTS in the extraction
# headers); replay targets arrive as loose topic words.
_CAPABILITY_TOPIC_ALIASES: dict[str, str] = {
    "delivery": "delivery",
    "delivery_method": "delivery",
    "provider_list": "provider_list",
    "provider list": "provider_list",
    "providers": "provider_list",
    "list": "provider_list",
    "benefits": "benefits",
    "benefit": "benefits",
    "my_benefits": "benefits",
    # Claims-path topics (Phase 7). notification_method canonicalizes to the
    # notification capability topic for redo/replay routing; slot-ownership
    # lookups (get_ownership) are unaffected — they key on the slot name.
    "notification": "notification",
    "notifications": "notification",
    "notification method": "notification",
    "notification preference": "notification",
    "claim": "claim_status",
    "claim_status": "claim_status",
    "claim status": "claim_status",
    "my claim": "claim_status",
}


def capability_topic(target: str) -> str:
    """Canonical capability topic for an extraction-level target, or ""."""
    key = (target or "").strip().lower().replace("-", "_")
    return _CAPABILITY_TOPIC_ALIASES.get(key) or _CAPABILITY_TOPIC_ALIASES.get(key.replace("_", " "), "")


# Redo-topic equivalences: re-sending the provider list IS a delivery redo —
# "send that list again" and "send it by email instead" both re-dispatch the
# list, so a redo whose topic canonicalizes to provider_list resolves to the
# ("redo", "delivery") capability. Replay is NOT equivalent: replaying the
# provider_list re-states what was sent; replaying delivery makes no sense.
_REDO_TOPIC_EQUIVALENTS: dict[str, str] = {"provider_list": "delivery"}


def canonical_capability_topic(kind: str, target: str) -> str:
    """Registry topic a (kind, target) request actually resolves to, applying
    the redo equivalences above — "" when the request is not routable.

    Callers recording a hop (pending_cross_agent_request) must use THIS topic,
    not capability_topic(target): the owner's re-entry gates key off the
    canonical registry topic (e.g. delivery's redo_active)."""
    kind = (kind or "").strip().lower()
    topic = capability_topic(target)
    if not kind or not topic:
        return ""
    if (kind, topic) in CAPABILITY_REGISTRY:
        return topic
    alias = _REDO_TOPIC_EQUIVALENTS.get(topic, "") if kind == "redo" else ""
    if alias and (kind, alias) in CAPABILITY_REGISTRY:
        return alias
    return ""


def resolve_capability(kind: str, target: str) -> Capability | None:
    """Capability entry for a (kind, target) request, or None when unknown.

    None means the topic is not routable — callers park it as a question
    (Phase 3), never hard-decline.
    """
    topic = canonical_capability_topic(kind, target)
    if not topic:
        return None
    return CAPABILITY_REGISTRY.get(((kind or "").strip().lower(), topic))


def invalidated_state_updates(slot: str) -> dict:
    """State updates that clear everything an update to ``slot`` invalidates."""
    own = get_ownership(slot)
    if not own:
        return {}
    return {key: (False if key.endswith(("_sent", "_updated")) else "") for key in own.invalidates}


# ── follow_up compatibility layer (Phase 3 parked-action routing) ────────────
# follow_up routes parked kind="action" items by intake intent. Map each
# owning agent to the intent whose flow re-runs it; verification is special
# (re-verify with the CURRENT intent) and human-only stays human.
OWNER_HUMAN = "human"
OWNER_VERIFICATION = "verification"

_AGENT_TO_INTAKE_INTENT: dict[str, str] = {
    "provider_search_agent": "provider_services",
    "delivery_management_agent": "provider_services",
    "notification_setup_agent": "claim_services",
}


def slot_update_owner(slot: str) -> str:
    """Owner label for follow_up's parked-action routing.

    Returns OWNER_VERIFICATION, an intake intent, or OWNER_HUMAN. Unknown and
    human-only slots are OWNER_HUMAN — never route an update nowhere.
    """
    own = get_ownership(slot)
    if not own or own.updatable == "human_only":
        return OWNER_HUMAN
    if own.agent == "verification_agent":
        return OWNER_VERIFICATION
    return _AGENT_TO_INTAKE_INTENT.get(own.agent, OWNER_HUMAN)
