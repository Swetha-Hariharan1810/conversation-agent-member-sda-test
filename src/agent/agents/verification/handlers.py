"""
handlers.py — Verification workflow handlers. Updated to use pick().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.agents.verification.constants import (
    IDENTITY_SLOT_ORDER,
    LOG_PARTIAL_REASK,
    MAX_LOOKUP_ATTEMPTS,
    MSG_REASK_DOB,
    MSG_REASK_FIRST_NAME,
    MSG_REASK_GENERIC,
    MSG_REASK_LAST_NAME,
)
from agent.conversation.context import ConversationContext

if TYPE_CHECKING:
    from agent.llm.schema import WorkerResult
from agent.responses.builder import build_initial_prompt
from agent.responses.message_builders import (
    build_offtopic_redirect,
    build_phone_confirmation_prompt,
    build_relationship_confirmation_prompt,
)
from agent.slots.normalizers import (
    normalize_dob,
    normalize_member_id,
    normalize_name,
    normalize_yes_no,
)
from agent.slots.validators import (
    validate_dob,
    validate_member_id,
    validate_name,
)
from agent.utils import _last_user_msg, pick

# Escalation messages — delivered at the moment of verification failure handoff
MSG_ESCALATE = [
    (
        "I'm sorry, I wasn't able to verify your account with the details provided. "
        "Let me connect you with a representative who can help verify your identity directly."
    ),
    (
        "Unfortunately I wasn't able to confirm your account after a couple of attempts. "
        "A representative will be able to look into this with you and get things sorted out."
    ),
    (
        "I wasn't able to match the details you provided to an account in our system. "
        "Let me get a representative on the line who can assist you further."
    ),
]
MSG_RESTART = [
    (
        "I wasn't able to find an account with those details — "
        "let's try once more. "
        "Could I start with your first name?"
    ),
    (
        "I wasn't able to match those details to an account. "
        "Let's give it one more try — could I get your first name again?"
    ),
    (
        "Those details didn't quite match what we have on file. "
        "Let's try again — could I start with your first name please?"
    ),
]
MSG_OFFTOPIC_PREFIX = [
    "I'm currently verifying your identity. ",
    "Let me finish verifying your account first. ",
]

# ── Phone not confirmed — static end message ──────────────────────────────────
MSG_PHONE_NOT_CONFIRMED = (
    "I'm sorry, I'm unable to verify your account without confirming "
    "the phone number on file. I'm transferring you to a live representative."
    "Thank you for calling Sagility Health."
)

# Truly human-only fields: system flags and the SF-verified phone number.
# Callers cannot change these values — they must be referred to a human agent.
# Any value the LLM puts in corrections{} for a locked slot is silently dropped
# AND does not trigger a correction acknowledgement.
#
# Phase 4: zip_code / fax / email are no longer globally locked — they are
# owned by provider_search / delivery_management (core.slot_ownership).
# During verification they are still not correctable here (no normalizer in
# _NORMALIZERS, so apply_corrections never applies them), but instead of
# being silently dropped they are parked as kind="action" items by
# slot_manager's CORRECTED path so the owning flow honors the request later.
CALLER_LOCKED_SLOTS: frozenset[str] = frozenset(
    {
        "phone_number",  # from SF record — disputes go to a human
        "member_status_verify",  # system flag, never caller-stated
        "call_intent",  # classified by intake agent
        # Add domain-agent locked slots here as they are built:
        # "coverage_tier", "plan_type", "benefit_level", etc.
    }
)

_NORMALIZERS = {
    "first_name": normalize_name,
    "last_name": normalize_name,
    "member_id": normalize_member_id,
    "dob": normalize_dob,
}
_VALIDATORS = {
    "first_name": validate_name,
    "last_name": validate_name,
    "member_id": validate_member_id,
    "dob": validate_dob,
}


def _reask_message(mismatched: list[str]) -> str:
    """Pick the targeted re-ask prompt for the mismatch set.

    Single-field mismatches use the disclosing, field-named pools (Phase 0
    decision). Any multi-field mismatch falls back to the non-disclosing
    generic pool so we don't read every wrong detail back to the caller.
    """
    if mismatched == ["first_name"]:
        return pick(MSG_REASK_FIRST_NAME)
    if mismatched == ["last_name"]:
        return pick(MSG_REASK_LAST_NAME)
    if mismatched == ["dob"]:
        return pick(MSG_REASK_DOB)
    return pick(MSG_REASK_GENERIC)


def _full_restart(agent, state) -> dict:
    """Wipe all four identity fields and re-ask from the top (MSG_RESTART).

    Used when the Member ID isn't found (Phase 0: re-ask everything) or when no
    usable field-match info is available.
    """
    # Reset attempt counters so the next round starts with a full budget.
    for _slot in ("first_name", "last_name", "member_id", "dob"):
        agent.get_slot(_slot).reset()
    restart = agent.ask_member(state, pick(MSG_RESTART))
    restart.update({"first_name": "", "last_name": "", "member_id": "", "dob": ""})
    restart["name_confirmed"] = False
    restart["name_confirm_attempts"] = 0
    # A restart abandons any in-flight update detour and wait streak.
    # parked_followups deliberately survives — those get answered in follow_up.
    restart["correction_return_to"] = ""
    restart["wait_count"] = 0

    ctx = ConversationContext.from_state(state)
    ctx.caller_first_name = ""
    ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s not in ("first_name", "last_name")]
    restart["conversation_context"] = ctx.to_dict()

    restart["verification_restart_index"] = len(state.get("messages") or [])
    return restart


def _partial_reask(agent, state, mismatched: list[str]) -> dict:
    """Clear only the mismatched identity slots and re-ask just those fields.

    Preserves the Member ID and every matched field (slot value, attempt count,
    and confirmation). Name confirmation is only reset when a name field is in
    the mismatch set. ``verification_restart_index`` is refreshed so the
    extractor re-reads recent turns for the corrected field(s).
    """
    name_mismatch = any(f in mismatched for f in ("first_name", "last_name"))

    # Reset attempt counters ONLY for the mismatched slots — matched fields keep
    # their state. Done before ask_member so the cleared slots are not persisted
    # back as confirmed values.
    for _slot in mismatched:
        agent.get_slot(_slot).reset()

    result = agent.ask_member(state, _reask_message(mismatched))

    # Clear ONLY the mismatched slot values; matched fields (incl. member_id)
    # were persisted by ask_member and stay intact.
    for _slot in mismatched:
        result[_slot] = ""

    # Name confirmation only reset if a name field mismatched (Phase 0 decision).
    if name_mismatch:
        result["name_confirmed"] = False
        result["name_confirm_attempts"] = 0

    # Drop only the re-asked slots from confirmed_slots; clear the cached caller
    # name only when first_name itself is being re-asked.
    ctx = ConversationContext.from_state(state)
    ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s not in mismatched]
    if "first_name" in mismatched:
        ctx.caller_first_name = ""
    result["conversation_context"] = ctx.to_dict()

    # Point awaiting_slot at the FIRST mismatched field (identity order). run()
    # recomputes awaiting_slot as `state.get("awaiting_slot") or <first empty>`,
    # so a stale truthy pointer (e.g. "member_id" left over from a multi-slot
    # utterance, or "dob" after a last-name mismatch) would otherwise mislabel the
    # extraction context on the re-ask turn. The identity pipeline collects in
    # IDENTITY_SLOT_ORDER and stops at the first empty slot, which — now that the
    # matched fields stay populated — is exactly this first mismatched slot.
    result["awaiting_slot"] = next(s for s in IDENTITY_SLOT_ORDER if s in mismatched)

    # A targeted re-ask abandons any in-flight update detour and wait streak;
    # parked_followups deliberately survives (answered later in follow_up).
    result["correction_return_to"] = ""
    result["wait_count"] = 0

    result["verification_restart_index"] = len(state.get("messages") or [])

    import logging

    logging.getLogger(__name__).info(LOG_PARTIAL_REASK, extra={"mismatched": mismatched})
    return result


async def lookup_and_verify(agent, state, collected):
    import asyncio as _asyncio

    # Lazy import — avoids circular import risk at module level.
    from agent.storage.queries.benefits import get_member_benefits

    # Lazy import: agents/verification → storage.tools → storage.db → storage.client;
    # importing at module level would create a agent → storage → agent cycle via
    # lookup_member usage in handlers.py being imported by agent.py at startup.
    from agent.storage.tools import lookup_member

    member_id = collected.get("member_id", "")

    # Run member lookup and benefits fetch concurrently.
    # Benefits failure must never block or fail the verification step —
    # benefits_agent has its own fallback fetch path.
    try:
        result, benefits_record = await _asyncio.gather(
            lookup_member.ainvoke(collected),
            get_member_benefits(member_id),
            return_exceptions=False,
        )
    except Exception:
        # If gather raises (e.g. benefits call throws), fall back to
        # lookup-only so verification is never blocked by a benefits error.
        import logging

        logging.getLogger(__name__).warning(
            "lookup_and_verify: benefits prefetch raised — retrying lookup alone"
        )
        result = await lookup_member.ainvoke(collected)
        benefits_record = None

    verified = result is not None and result.get("verified") is True

    if not verified:
        # Global attempt cap (unchanged): escalate to a human after
        # MAX_LOOKUP_ATTEMPTS failed lookups, counted across all fields.
        if escalation := agent.guard_loop_limit(
            state,
            "lookup_fail",
            MAX_LOOKUP_ATTEMPTS,
            escalate_message=pick(MSG_ESCALATE),
            escalate_reason="Verification failed after max lookup attempts",
        ):
            return None, escalation

        # Phase 2 lookup attaches member_id_found + field_matches on failure.
        # Older/exception failure shapes (just {"verified": False}) fall through
        # to the full-restart branch below.
        field_matches = (result or {}).get("field_matches") or {}
        mismatched = [f for f in IDENTITY_SLOT_ORDER if field_matches.get(f) is False]
        member_id_found = bool(result and result.get("member_id_found"))

        # No record for this Member ID (or no usable field-match info) → full
        # restart per Phase 0: wipe all four identity fields and re-ask with
        # MSG_RESTART.
        if not member_id_found or not mismatched:
            return None, _full_restart(agent, state)

        # Member ID found but some identity fields differ → targeted re-ask:
        # clear only the mismatched slots; keep Member ID and every matched field.
        return None, _partial_reask(agent, state, mismatched)

    # Merge prefetched benefits into the result dict so _signal_verified()
    # can pass them through context_updates into state, making them available
    # to benefits_agent without a second Salesforce call.
    if benefits_record:
        result["individual_deductible"] = str(benefits_record.get("individual_deductible") or "")
        result["family_deductible"] = str(benefits_record.get("family_deductible") or "")
        result["coinsurance_percent"] = str(benefits_record.get("coinsurance_percent") or "")
        result["individual_oop_max"] = str(benefits_record.get("individual_oop_max") or "")
        result["family_oop_max"] = str(benefits_record.get("family_oop_max") or "")
        import logging

        logging.getLogger(__name__).info(
            "lookup_and_verify: benefits prefetched and merged into verification result"
        )
    else:
        import logging

        logging.getLogger(__name__).warning(
            "lookup_and_verify: benefits prefetch returned None — benefits_agent will fetch on demand"
        )

    return result, None


async def collect_post_lookup(
    agent,
    state,
    messages,
    collected,
    call_intent,
    member_record,
    decision: "WorkerResult | None",
    claims_pipeline,
    provider_pipeline,
):
    post_collected: dict = {}

    if call_intent == "claim_services":
        phone = (member_record or {}).get("phone_number") or state.get("phone_number") or ""
        prompt = (
            build_phone_confirmation_prompt(phone)
            if phone
            else "Thank you. Could you confirm your phone number on file?"
        )
        claims_pipeline.configs["phone_confirmed"].prompt = prompt
        pipeline = claims_pipeline
        # Deterministic fallback: same fix as notification_setup_agent. The LLM
        # often returns empty for phone_confirmed when the user gives a plain
        # "yes correct" — map the raw utterance directly to avoid a retry.
        # Guard: only inject canonical "yes"/"no" — normalize_yes_no returns the
        # cleaned input as-is for non-yes/no strings (e.g. a DOB answer on the
        # turn verification completes), which would fail validate_yes_no and
        # produce a spurious RETRY message.
        if decision is not None:
            pc = (decision.extracted or {}).get("phone_confirmed", "")
            if not pc:
                fallback = normalize_yes_no(_last_user_msg(messages))
                if fallback in ("yes", "no"):
                    decision.extracted = {**(decision.extracted or {}), "phone_confirmed": fallback}
    else:
        # relationship_str = (member_record or {}).get("relationship") or ""
        relationship_str = "planholder or dependent"
        prompt = build_relationship_confirmation_prompt(relationship_str)
        provider_pipeline.configs["relationship"].prompt = prompt
        pipeline = provider_pipeline

    interrupt = await pipeline.collect(state, messages, post_collected, decision=decision)
    if interrupt:
        # find member_status_verify=True and skip the Salesforce lookup entirely.
        # Without this, every retry enters the `if not state.get("member_status_verify")`
        # branch and fires another SF HTTP call.
        # setdefault: an update detour on an identity slot (Phase 4) puts
        # member_status_verify=False in the interrupt to force re-verification —
        # that must not be stomped back to True here.
        interrupt.setdefault("member_status_verify", True)
        if member_record:
            for field in ("phone_number", "zip_code", "fax", "email", "relationship"):
                if val := member_record.get(field):
                    interrupt[field] = val
        return interrupt

    collected.update(post_collected)

    # ── Phone not confirmed — end call immediately with static message ────────
    # When the caller says "no" to the phone confirmation (claims flow), we
    # cannot verify their identity. Route directly to END with a static message
    # rather than completing verification or escalating to a live agent.
    if call_intent == "claim_services":
        phone_answer = post_collected.get("phone_confirmed", "")
        # normalize_yes_no maps "no" → "no"; the slot value may also be stored
        # as the boolean False when collected["phone_confirmed"] == "no".
        phone_declined = phone_answer == "no" or phone_answer is False or str(phone_answer).lower() == "no"
        if phone_declined:
            import logging

            logging.getLogger(__name__).info(
                "collect_post_lookup: phone_confirmed=no — ending call with static message"
            )
            result = agent.ask_member(state, MSG_PHONE_NOT_CONFIRMED)
            result["next_node"] = "END"
            result["is_interrupt"] = False
            result["phone_update_requested"] = True
            return result

    if "phone_confirmed" in post_collected:
        collected["phone_confirmed"] = True
        collected["phone_update_requested"] = post_collected["phone_confirmed"] == "no"

    return None


def redirect_off_topic(agent, state, collected, identity_pipeline):
    next_slot = next((s for s in IDENTITY_SLOT_ORDER if not collected.get(s)), None)
    if not next_slot:
        awaiting = state.get("awaiting_slot", "")
        if awaiting == "relationship":
            # Count this as a failed attempt — off-topic is still a non-answer
            agent.slot_fail("relationship")
            if agent.get_slot("relationship").is_exhausted():
                from agent.responses.static import build_slot_exhausted_message

                return agent.signal_escalate(
                    state,
                    build_slot_exhausted_message("relationship"),
                    "relationship exhausted",
                    initiator="Agent",
                )
            relationship_str = "planholder or dependent"
            next_prompt = build_relationship_confirmation_prompt(relationship_str)
        elif awaiting in ("phone_confirmed", "phone_confirmation"):
            agent.slot_fail("phone_confirmed")
            if agent.get_slot("phone_confirmed").is_exhausted():
                from agent.responses.static import build_slot_exhausted_message

                return agent.signal_escalate(
                    state,
                    build_slot_exhausted_message("phone_confirmed"),
                    "phone_confirmed exhausted",
                    initiator="Agent",
                )
            phone = state.get("phone_number", "")
            next_prompt = (
                build_phone_confirmation_prompt(phone)
                if phone
                else "Could you confirm your phone number on file?"
            )
        else:
            message = pick(MSG_OFFTOPIC_PREFIX).strip()
            result = agent.ask_member(state, message)
            result.update({k: v for k, v in collected.items() if v})
            return result
    else:
        cfg = identity_pipeline.configs[next_slot]
        if cfg.slot_type:
            next_prompt = build_initial_prompt(cfg.slot_type)
        else:
            next_prompt = cfg.prompt(collected) if callable(cfg.prompt) else cfg.prompt

    prefix = pick(MSG_OFFTOPIC_PREFIX)
    result = agent.ask_member(state, build_offtopic_redirect(next_prompt, prefix=prefix))
    result.update({k: v for k, v in collected.items() if v})
    return result


def apply_corrections(agent, collected, state, decision: "WorkerResult | None"):
    """
    Apply slot corrections from the LLM extraction result.

    Returns the list of slot names that were successfully corrected.
    Locked slots (CALLER_LOCKED_SLOTS) are silently dropped — they do NOT
    appear in the returned list, so no correction acknowledgement fires.
    """
    corrections = (decision.corrections or {}) if decision else {}
    corrected: list[str] = []

    for slot_name, raw in corrections.items():
        # Silently drop any correction targeting a locked slot.
        # This prevents ghost acknowledgements ("Got it — updated zip code")
        # when the system never actually changed the value.
        if slot_name in CALLER_LOCKED_SLOTS:
            continue
        if not raw:
            continue
        norm = _NORMALIZERS.get(slot_name)
        val = _VALIDATORS.get(slot_name)
        if norm and val:
            normalized = norm(str(raw))
            if normalized and val(normalized).valid:
                collected[slot_name] = normalized
                agent.slot_ok(slot_name, normalized)
                corrected.append(slot_name)

    # only cascade-clear if value not provided in same utterance
    extracted_this_turn = (decision.extracted or {}) if decision else {}
    if corrections.get("first_name") and not extracted_this_turn.get("last_name"):
        collected["last_name"] = ""
    if corrections.get("member_id") and not extracted_this_turn.get("dob"):
        collected["dob"] = ""

    # Re-verification consequence (Phase 4): correcting an identity slot after a
    # successful Salesforce match invalidates the verification. Clear the flag
    # in-place so run() re-enters lookup_and_verify this same turn; run() also
    # persists the cleared flag onto any interrupt it returns.
    if state.get("member_status_verify") and any(f in IDENTITY_SLOT_ORDER for f in corrected):
        state["member_status_verify"] = False

    return corrected
