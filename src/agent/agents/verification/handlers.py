"""
handlers.py — Verification workflow handlers. Updated to use pick().
"""

from __future__ import annotations

import re
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
from agent.core.metadata_events import ON_FILE_FIELDS, mark_on_file
from agent.logger import get_logger

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

# ── Phone not confirmed — escalation pre-message ───────────────────────────────
# Spoken by the escalation agent, which appends the reference number and the
# sign-off — so this carries no goodbye of its own. It used to end the call
# where it stood, while telling the caller they were being transferred; the
# transfer is real now and the wording no longer promises one twice.
MSG_PHONE_NOT_CONFIRMED = (
    "I'm sorry, I'm unable to verify your account without confirming "
    "the phone number on file. Let me connect you with a live representative."
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
    # parked_followups deliberately survives — those are update actions, routed
    # to their owner in follow_up.
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
    # parked_followups deliberately survives (update actions, routed to their
    # owner later in follow_up).
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


# ── Reading a confirmation of the number on file ─────────────────────────────
#     AI      Thank you. Is your phone number 512-555-6101?
#     Caller  yep, that's the right number
#     AI      Sorry, I didn't catch that — is 512-555-6101 still the best
#             number to reach you?
#
# The caller confirmed and the turn re-asked. This slot lost its last backstop
# when the raw-utterance normalize_yes_no fallback was removed, on the ground
# that the model has the context and the contract by this turn. It mostly does.
# When it does not, nothing catches it, and what the caller hears is the
# question they just answered.
#
# Both directions are read, and only when the model placed nothing at all.
#
# An affirmation is a closed set — people have a handful of ways to agree and
# they are not inventing more (see core.confirmation) — so it reads without a
# list that has to finish. A refusal is not a closed set, and this reads only
# the explicit ones, never "anything that is not a yes": a garbled turn or a
# hold has taken no position, and re-asking is the right answer to those.
#
# What made the decline unreadable before was where it went. It ended the call
# where it stood, so a wrong read cost the caller the call. It escalates now,
# which is where the re-ask was taking them anyway once the attempts ran out —
# so the cost of reading one wrongly is reaching a representative sooner, and
# the cost of not reading one is asking a caller the question they answered.
#
# The mistake that stays guarded is the mixed turn: normalize_yes_no("no,
# that's right") is "no", and that caller confirmed. A turn that both affirms
# and refuses is read as neither.
_AFFIRMS_RE = re.compile(
    r"^\W*(?:"
    r"yes|yeah|yep|yup|ya|uh\s*huh|mm\s*hmm|sure|absolutely|definitely|certainly"
    r"|correct|right|true|affirmative|of\s+course"
    r"|that'?s\s+(?:the\s+)?(?:right|correct|it|one|mine|my\s+(?:number|cell|phone|mobile))"
    r"|it\s+is|still\s+(?:right|correct|good|current|the\s+same)"
    r"|sounds?\s+(?:right|good)|looks?\s+(?:right|good)|all\s+good|perfect"
    r")\b",
    re.IGNORECASE,
)

# A refusal of the number, or a statement that it is not the current one. Read
# now because a decline no longer ends the call where it stands — it escalates,
# so the cost of reaching this wrongly is a caller handed to a representative,
# which is where a re-ask of a question they already answered was taking them
# anyway, three attempts later.
#
# Explicit refusals only. NOT "anything that is not a yes": a garbled turn, a
# side question or a hold have all said nothing about the number, and they
# still re-ask.
_DECLINES_RE = re.compile(
    r"^\W*(?:no|nope|nah)\b"
    r"|\b(?:not\s+(?:my|mine|right|correct|it|that|the\s+right|anymore|any\s+more)"
    r"|wrong|incorrect|different|another|new\s+(?:number|phone|cell)"
    r"|old|older|outdated|stale|previous|former"
    r"|changed|change[sd]?\s+it|switch(?:ed)?|replaced|moved|disconnected)\b"
    r"|\b(?:is|was|are|were|does|do|did|ca|wo|has|have|had|could|would|should)n'?t\b",
    re.IGNORECASE,
)

# An affirmation anywhere in the turn, not just at the start — the guard for
# "no, that's right", which opens on a refusal and then confirms. Turns that do
# both say nothing this can act on, and re-ask.
#
# The phrases require their words adjacent, so "that's not right" is not one of
# them: "that's right" and "that's not right" differ by one word and mean
# opposite things.
_AFFIRMS_ANYWHERE_RE = re.compile(
    r"\b(?:"
    r"that'?s\s+(?:the\s+)?(?:right|correct|it|one|mine|my\s+(?:number|cell|phone|mobile))"
    r"|it\s+is|still\s+(?:right|correct|good|current|the\s+same)"
    r"|sounds?\s+(?:right|good)|looks?\s+(?:right|good)|all\s+good"
    r")\b",
    re.IGNORECASE,
)


def screen_phone_confirmation(utterance: str) -> str:
    """``"yes"``, ``"no"``, or "" — the caller's position on the number read back.

    "" for a turn that takes no position, which is most turns: a question, a
    hold, an answer to something else. Both directions are read only when the
    turn is unmixed — a caller who affirms and refuses in one breath gets the
    question again rather than a guess at which half won.
    """
    text = (utterance or "").strip()
    if not text or text.endswith("?"):
        return ""

    declines = bool(_DECLINES_RE.search(text))
    affirms = bool(_AFFIRMS_RE.match(text) or _AFFIRMS_ANYWHERE_RE.search(text))

    if declines and affirms:
        return ""
    if declines:
        return "no"
    if affirms and _AFFIRMS_RE.match(text):
        return "yes"
    return ""


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

        # A turn the model cannot place re-asks, and deliberately does not fall
        # either way: unlike a fax or email read-back there is no new value to
        # collect — the phone on file is human_only — so "anything that is not
        # a yes is a decline" buys nothing here, and a decline read into a
        # caller who did say yes ends their call on a refusal they never made.
        #
        # What DOES get read is the two ways a confirmation the model made
        # never reaches the slot. The field is `phone_confirmed` in
        # verification_claims.md and `phone_confirmation` in llm.py's own
        # docstring, and redirect_off_topic below already treats both names as
        # this slot — the pipeline read only the first, so a confirmation filed
        # under the second was dropped. And an affirmation in the caller's
        # words, when the model placed nothing at all: only "yes", never "no",
        # and only when nothing in the turn contradicts it — see
        # screen_phone_affirmation.
        if decision is not None:
            extracted = dict(decision.extracted or {})
            answer = extracted.get("phone_confirmed", "") or extracted.get("phone_confirmation", "")
            # A turn with nothing usable in it, or one asking for time: "I'm
            # not sure" and "hold on" are not positions on the number, whatever
            # words carry them, and the screen is for the turn the model placed
            # nowhere — not for one it placed as unsure. Same ordering, and the
            # same reason, as core.confirmation.is_not_an_answer.
            placed_no_position = decision is not None and (
                decision.no_usable_value or decision.asked_for_time
            )
            if (
                not answer
                and not placed_no_position
                and (screened := screen_phone_confirmation(_last_user_msg(messages)))
            ):
                get_logger(__name__).info(
                    "collect_post_lookup: phone position read from the caller's words",
                    extra={"verdict": screened},
                )
                answer = screened
            if answer and not extracted.get("phone_confirmed"):
                decision.extracted = {**extracted, "phone_confirmed": answer}
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
            # The record's contact fields are carried so the agents downstream
            # have them without a second lookup — on file, not captured, so
            # they are marked and stay unreported until the call asks (see
            # core/metadata_events). relationship is not carried at all: the
            # record's value is the account's list of allowed relationships,
            # not this caller's, and the field holds the caller's own answer.
            hydrated = [f for f in ON_FILE_FIELDS if member_record.get(f)]
            for field in hydrated:
                interrupt[field] = member_record[field]
            interrupt["fields_on_file"] = mark_on_file(
                state.get("fields_on_file"), hydrated, emitted=state.get("emitted_fields")
            )
        return interrupt

    collected.update(post_collected)

    # ── Phone not confirmed — escalate to a representative ───────────────────
    # The caller says the number on file is not theirs. Identity cannot be
    # verified without it and the field is human_only, so there is nothing this
    # call can do with the answer — it goes to someone who can.
    #
    # This used to route straight to END while saying "I'm transferring you to
    # a live representative", so the caller was told about a transfer that
    # never happened: no AgentCallTransfer event, no reference number, and the
    # call reported itself finished. signal_escalate raises the event, hands
    # the pre-message to escalation_agent for the reference number, and makes
    # the sentence true.
    if call_intent == "claim_services":
        phone_answer = post_collected.get("phone_confirmed", "")
        # normalize_yes_no maps "no" → "no"; the slot value may also be stored
        # as the boolean False when collected["phone_confirmed"] == "no".
        phone_declined = phone_answer == "no" or phone_answer is False or str(phone_answer).lower() == "no"
        if phone_declined:
            get_logger(__name__).info(
                "collect_post_lookup: phone_confirmed=no — escalating to a representative"
            )
            result = agent.signal_escalate(
                state,
                MSG_PHONE_NOT_CONFIRMED,
                "phone_not_confirmed",
                initiator="Agent",
            )
            result["phone_update_requested"] = True
            return result

    if "phone_confirmed" in post_collected:
        collected["phone_confirmed"] = True
        collected["phone_update_requested"] = post_collected["phone_confirmed"] == "no"
        # The caller heard the number on file read back and said it was theirs.
        # That is this call capturing the phone number, not merely carrying it,
        # so it is reported now — a decline never reaches here (the branch
        # above escalates), and the claims pipeline is the only one that
        # asks, so a flow that never puts the number to the caller still
        # reports nothing.
        phone_on_file = (member_record or {}).get("phone_number") or state.get("phone_number") or ""
        if phone_on_file:
            agent.field_captured("phone_number", phone_on_file)

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


# ── Named-part name corrections, read from the words ─────────────────────────
#
#     AI    Thank you. Just to confirm — is your name Emily Watson. That's
#           spelled E-M-I-L-Y-W-A-T-S-O-N, correct?
#     User  Actually, my surname is Carter, not Watson.
#     AI    Sure, what is the correct name?
#
# The caller had just given the correct name. They said which half of it was
# wrong and what it should be, and were asked to say it again.
#
# name_confirmation.md teaches the correction shape as "no, it's Jhon Doe" — a
# replacement offered whole, with no part of the name named. This caller named
# the part ("surname") and contrasted it with the value that had been read back
# ("not Watson"), and neither phrasing is in the contract. The trailing "not
# Watson" is the problem: it is the shape of every rejection listed under
# OUTCOME 3 ("that's not right", "no that's not me"), so the turn comes back a
# bare no — or, worse, as last_name="Watson", which reads the wrong name back
# as though the caller had asked for it.
#
# Which part of their name a caller is correcting is a fact about their words
# whenever they name it, so it is read from the words, the way a named
# specialty and a named appeal already are in intake.

_NAME_WORD = r"[A-Za-z][A-Za-z'\-]*"

# Words that can stand where a name would and are not one. Without these the
# capture runs past the name into the rest of the sentence: "my last name is
# wrong" yields "Wrong", and "my surname is Carter, not Watson" yields "Carter
# Not Watson" — a correction assembled out of the sentence complaining about it.
_NOT_A_NAME = (
    r"(?:not|and|or|but|instead|rather|it|its|that|this|these|those|the|a|an|is|was|"
    r"are|were|my|your|his|her|their|our|please|thanks|thank|actually|really|correct|"
    r"right|wrong|incorrect|different|spelled|spelt|sorry|no|yes|yeah|yep|nope|"
    r"name|names|surname|first|last|family|maiden|given)"
)
_NAME_TOKEN = rf"\b(?!{_NOT_A_NAME}\b){_NAME_WORD}"
# Up to three words: "Carter", "Van Der Berg". The lookbehind keeps the run from
# starting inside a word or after an apostrophe — without it "it's Carter" reads
# as the name "T's Carter", the tail of a contraction the stop list already
# rejected whole.
_NAME_RUN = rf"((?<![\w']){_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})"

_PART_LABELS: dict[str, str] = {
    "last_name": r"(?:sur\s?name|last\s+name|family\s+name|second\s+name|maiden\s+name)",
    "first_name": r"(?:first\s+name|given\s+name|fore\s?name|christian\s+name)",
}
# Carries its own leading space so "my surname's Carter" reads the same as
# "my surname is Carter".
_IS = r"(?:\s*'s|\s+(?:is|was|should\s+be|would\s+be|needs?\s+to\s+be|must\s+be|goes\s+by))"
_FILLER = r"(?:\s+(?:actually|really|spelled|spelt|just))?"

# A bare "X, not Y" names no part, but which part it corrects is still knowable
# when Y is the name the call is currently holding: X replaces it.
_CONTRASTIVE_MAX_WORDS = 6

_NAMED_PART_PATTERNS: dict[str, re.Pattern[str]] = {
    slot: re.compile(
        rf"\b(?:(?:my|the|her|his|their|our)\s+)?{labels}{_IS}{_FILLER}\s+{_NAME_RUN}",
        re.IGNORECASE,
    )
    for slot, labels in _PART_LABELS.items()
}
_CONTRASTIVE_PATTERN: re.Pattern[str] = re.compile(
    rf"{_NAME_RUN}\s*,?\s+not\s+({_NAME_WORD})\b", re.IGNORECASE
)


def recover_name_correction(
    utterance: str,
    current_first: str = "",
    current_last: str = "",
) -> dict[str, str]:
    """The name parts the caller's words name, as ``{slot: value}``.

    Empty when the words name nothing — the ordinary case, where the LLM's
    reading stands unchanged. Only ever returns a part the caller identified:
    either by naming it ("my surname is Carter") or by contrasting the
    replacement with the value the call is currently holding ("Carter, not
    Watson").
    """
    text = (utterance or "").replace("\u2019", "'")
    found: dict[str, str] = {}

    for slot, pattern in _NAMED_PART_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        candidate = normalize_name(match.group(1))
        if candidate and validate_name(candidate).valid:
            found[slot] = candidate

    # The contrastive reading only applies to a short turn. "X, not Y" carries a
    # correction when it is most of what the caller said; inside a longer
    # sentence it is as likely to be about someone else ("I spoke to Sarah, not
    # Watson, about the claim"), and that one is the model's to read, not a
    # regex's.
    if len(text.split()) <= _CONTRASTIVE_MAX_WORDS:
        for match in _CONTRASTIVE_PATTERN.finditer(text):
            replacement = normalize_name(match.group(1))
            replaced = normalize_name(match.group(2)).lower()
            if not replacement or not validate_name(replacement).valid or not replaced:
                continue
            if replaced == (current_last or "").strip().lower():
                found.setdefault("last_name", replacement)
            elif replaced == (current_first or "").strip().lower():
                found.setdefault("first_name", replacement)

    return found
