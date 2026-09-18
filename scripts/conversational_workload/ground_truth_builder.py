"""
Ground-truth builder — decides what the simulated user should say in reply
to a live agent message, combining the transcript cursor (positional ground
truth from the static reference transcripts) with the slot-level fallback.

FIXES (stability — these were sources of intermittent eval failures):

  1. CURSOR-FIRST RESTART HANDLING. The old builder reset the cursor to 0
     whenever the agent asked for first_name with visits > 0, on the theory
     that a first-name re-ask means verification restarted. But the
     pcp_correction_first_name reference transcript ALREADY CONTAINS the
     restart inline — its cursor position naturally reaches the second
     first-name ask. Resetting to 0 there desynced the cursor for the rest
     of the conversation. The builder now tries the cursor AT ITS CURRENT
     POSITION FIRST and only treats a first-name re-ask as a verification
     restart when the cursor MISSES — i.e. when the re-ask is genuinely not
     part of the scripted flow.

  2. On a genuine restart the identity slot counters (first_name, last_name,
     member_id, dob) are cleared so the slot fallback re-serves the same
     identity answers instead of escalating its variants.

  3. The slot fallback import is defensive: the canonical implementation in
     slot_ground_truth.py is used when available; a minimal internal table
     keeps the builder importable in isolation (e.g. offline tests).

The cursor position lives in the runner's turn_counters dict under the key
(CURSOR_KEY, scenario_tag) — the same storage the slot counters use, keyed
(scenario_tag, slot), so no runner change is needed.
"""

from __future__ import annotations

from typing import Any, Callable

from . import intent_classifier, transcript_cursor

CURSOR_KEY = "__cursor__"
IDENTITY_SLOTS = ("first_name", "last_name", "member_id", "dob")

# ---------------------------------------------------------------------------
# Slot-level fallback
# ---------------------------------------------------------------------------

_slot_fallback_fn: Callable[..., str] | None = None
try:  # canonical implementation, unchanged
    from . import slot_ground_truth as _sgt

    for _name in ("slot_ground_truth", "get_slot_ground_truth", "build", "get"):
        _candidate = getattr(_sgt, _name, None)
        if callable(_candidate):
            _slot_fallback_fn = _candidate
            break
except Exception:  # pragma: no cover - module absent in isolated checkouts
    _sgt = None


def _entity_get(entities: Any, field: str, default: str = "") -> str:
    if entities is None:
        return default
    if isinstance(entities, dict):
        return str(entities.get(field, default))
    return str(getattr(entities, field, default))


def _internal_slot_fallback(slot: str, entities: Any) -> str:
    """Minimal slot → reply table used only when slot_ground_truth is absent."""
    table = {
        "first_name": _entity_get(entities, "first_name", "emily"),
        "last_name": _entity_get(entities, "last_name", "carter"),
        "member_id": _entity_get(entities, "member_id_spoken", "") or _entity_get(entities, "member_id", ""),
        "dob": _entity_get(entities, "dob_spoken", "") or _entity_get(entities, "dob", ""),
        "subscriber_confirm": "I'm calling for myself",
        "intent_selection": _entity_get(
            entities, "intent_utterance", "I need to find a primary care physician in my area."
        ),
        "provider_type": "Primary Care Physician",
        "zip_confirm": "yes that's correct",
        "zip_update": _entity_get(entities, "zip_spoken", "") or _entity_get(entities, "zip_code", ""),
        "fax_or_email": "send it to my fax",
        "fax_confirm": "yes that's correct",
        "fax_update": _entity_get(entities, "fax_spoken", "") or _entity_get(entities, "fax", ""),
        "benefits_offer": "yes please",
        "coach_offer": "yes that sounds interesting",
        "clarification": "sorry — yes that's correct",
        "correction_ack": "thank you",
        "reference_number": _entity_get(entities, "reference_number", ""),
        "records_method": "Can I ask my doctor to send it over?",
        "upload_link_offer": "Yes, please",
        "email_confirm": "Yes, that's correct",
        "personal_guide_consent": "Perfect. Please do that",
        "notification_method": "You can send me the updates to my phone",
        "phone_confirm": "Yes, that's correct",
        "n2_notification_method": "email them to me",
        "guide_scheduled": "Okay, how long will it take to finalize the request?",
        "timeline_question": "Okay",
        "follow_up": "No, that's all. Thanks!",
        "closing": "",
    }
    return table.get(slot, "Okay.")


def _curated_override(slot: str, ai_lower: str) -> str | None:
    """Slot-fallback corrections for asks where the canonical slot table is
    context-blind. Applied only on a cursor MISS, before delegating to
    slot_ground_truth.

    timeline_question covers two opposite contexts:
      * the OFFER ("I can walk you through the expected timeline" / "let me
        share the expected timeline") — the scripted user ASKS the question;
      * the ANSWER ("...5 to 10 business days...") — a bare acknowledgement
        is right (and that message usually carries the N2 channel ask, which
        classifies n2_notification_method anyway).
    The slot table returns "Okay" for both, which scored 0.25 on live runs
    whenever the cursor missed the rephrased offer.
    """
    if slot == "timeline_question" and "business days" not in ai_lower:
        return "Okay, how long will it take to finalize the request?"
    return None


def _slot_fallback(slot: str, entities: Any) -> str:
    if _slot_fallback_fn is not None:
        try:
            return _slot_fallback_fn(slot, entities)
        except TypeError:
            try:
                return _slot_fallback_fn(slot)
            except Exception:
                pass
        except Exception:
            pass
    return _internal_slot_fallback(slot, entities)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _flow_for(scenario_tag: str) -> str:
    return "claim" if scenario_tag.startswith("claim") else "pcp"


def build_ground_truth(
    scenario_tag: str,
    ai_message: str,
    turn_counters: dict,
    entities: Any = None,
) -> str:
    """Return the simulated user's reply to ``ai_message``.

    Order of resolution:
      1. transcript cursor at its current position (positional ground truth);
      2. on a cursor MISS where the agent re-asks first_name after it was
         already visited → verification restart: reset the cursor and the
         identity slot counters, then retry the cursor from the top;
      3. slot-level fallback keyed by the deterministic intent classifier.
    """
    flow = _flow_for(scenario_tag)
    cursor_slot = (CURSOR_KEY, scenario_tag)
    cursor = int(turn_counters.get(cursor_slot, 0))

    reply, new_cursor = transcript_cursor.get_ground_truth(scenario_tag, ai_message, cursor)

    slot = intent_classifier.classify_ai_slot(ai_message, flow=flow)

    if reply is None and slot == "first_name" and turn_counters.get((scenario_tag, "first_name"), 0) > 0:
        # Genuine verification restart: the re-ask was NOT part of the
        # scripted flow (cursor missed). Restart positional tracking and
        # clear identity counters so the same identity answers are re-served.
        cursor = 0
        for identity_slot in IDENTITY_SLOTS:
            turn_counters[(scenario_tag, identity_slot)] = 0
        reply, new_cursor = transcript_cursor.get_ground_truth(scenario_tag, ai_message, cursor)

    if reply is not None:
        turn_counters[cursor_slot] = new_cursor
        return reply

    # Cursor miss — keep the cursor where it is (do NOT corrupt it) and
    # answer from the slot table, with curated overrides for context-blind
    # entries applied first.
    turn_counters[cursor_slot] = cursor
    override = _curated_override(slot, ai_message.lower())
    if override is not None:
        return override
    return _slot_fallback(slot, entities)


# Backwards-compatible alias — the runner historically imported get_ground_truth.
get_ground_truth = build_ground_truth


def build_dynamic_ground_truth(*args: Any, **kwargs: Any) -> str:
    """Entry point imported by runner.py.

    The runner calls this with ``ai_msg`` positional and the rest as
    keywords (``scenario_tag=..., turn_counters=...``); other call sites have
    historically used different orders. Rather than pin one signature, the
    arguments are resolved by KIND:

      * a string that is a registered scenario tag  -> scenario_tag
      * any other string                            -> ai_message
      * a dict                                      -> turn_counters
      * anything else                               -> entities

    Recognised keywords: scenario_tag, ai_message/ai_msg/agent_message,
    turn_counters, entities/entity. Unknown keywords (e.g. ``flow``) are
    ignored — the flow is derived from the scenario tag.
    """
    scenario_tag = kwargs.pop("scenario_tag", None)
    ai_message = (
        kwargs.pop("ai_message", None) or kwargs.pop("ai_msg", None) or kwargs.pop("agent_message", None)
    )
    turn_counters = kwargs.pop("turn_counters", None)
    entities = kwargs.pop("entities", None)
    if entities is None:
        entities = kwargs.pop("entity", None)

    for arg in args:
        if isinstance(arg, str):
            if scenario_tag is None and arg in transcript_cursor._SCENARIO_FILE_MAP:
                scenario_tag = arg
            elif ai_message is None:
                ai_message = arg
            elif scenario_tag is None:
                scenario_tag = arg
        elif isinstance(arg, dict):
            if turn_counters is None:
                turn_counters = arg
            elif entities is None:
                entities = arg
        elif entities is None:
            entities = arg

    if scenario_tag is None or ai_message is None:
        raise TypeError(
            "build_dynamic_ground_truth: could not resolve scenario_tag and "
            f"ai_message from args={args!r} kwargs={kwargs!r}"
        )
    if turn_counters is None:
        turn_counters = {}
    return build_ground_truth(scenario_tag, ai_message, turn_counters, entities)
