"""
agent.py — FollowUpAgent

Two escalation rules, both owned entirely by Python:

  UPDATE_REQUEST
    Signal escalate immediately on every occurrence.
    Fixed message + reference number. No counting, no threshold.

  MSG_CANNOT_ANSWER
    Count consecutive cannot-answer turns (resets when any real answer
    is given, or on done/unsure/update_request).
    On the 3rd consecutive cannot-answer, escalate using that same
    cannot-answer message as the prefix.
"""

from __future__ import annotations

import re

from agent.agents.follow_up.constants import (
    AGENT_NAME,
    APPEAL_GRIEVANCE_KEYWORDS,
    BARE_AFFIRMATIONS,
    CLOSURE_KEYWORDS,
    FLOW_COMPLETE_FLAGS,
    INTAKE_INTENTS,
    INTAKE_RESCREEN_INTENTS,
    LOG_ANSWERED,
    LOG_CANNOT_ANSWER,
    LOG_CLOSURE,
    LOG_ENTERED,
    LOG_NEW_INTENT,
    MAX_CANNOT_ANSWER_BEFORE_ESCALATE,
    MAX_FOLLOW_UP_TURNS,
    MSG_CANNOT_ANSWER,
    MSG_CONTINUATION,
    MSG_FOLLOW_UP_ASK,
    MSG_NUDGE,
    MSG_UPDATE_REQUEST_ESCALATE,
)
from agent.agents.follow_up.llm import extract_follow_up_decision
from agent.core.agent import BaseAgent
from agent.core.constants import WAIT_PATTERNS as _WAIT_PATTERNS_RAW
from agent.core.slot_ownership import (
    OWNER_HUMAN,
    OWNER_VERIFICATION,
    canonical_capability_topic,
    slot_update_owner,
)
from agent.llm.config import get_follow_up_llm
from agent.llm.schema import FollowUpIntent
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.responses.static import MSG_WAIT_ACK
from agent.state import State, normalize_parked_followups, reset_for_new_intent
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    pick,
)

logger = get_logger(__name__)

# Wait detection for follow-up: no slot-value continuation guard because
# follow-up is conversational — any wait phrase should be honoured.
# Also covers "give me two/a few/some minutes" that the shared util misses.
_FOLLOW_UP_WAIT_RE: list = [re.compile(p, re.IGNORECASE) for p in _WAIT_PATTERNS_RAW] + [
    re.compile(
        r"\b(give me|gimme)\s+\w+\s+(sec|second|seconds|minute|minutes|min|moment|moments)\b",
        re.IGNORECASE,
    )
]


def _detect_wait_follow_up(text: str | None) -> bool:
    """True when the caller is asking for a moment — no continuation guard."""
    if not text:
        return False
    lowered = text.lower().strip()
    return any(pat.search(lowered) for pat in _FOLLOW_UP_WAIT_RE)


_FORBIDDEN_ANSWER_PHRASES = (
    "i can help you find",
    "i can help you with",
    "i can help with",
)

# Whole-word match over the appeal/grievance keywords. Word boundaries keep
# "appeal" from matching inside unrelated words and let each surface form
# (appeal/appeals/appealing/…) match exactly the keyword listed.
_APPEAL_GRIEVANCE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(APPEAL_GRIEVANCE_KEYWORDS)) + r")\b",
    re.IGNORECASE,
)


def _is_appeal_or_grievance(text: str) -> bool:
    """True when the member's utterance mentions an appeal or grievance.

    Appeals and grievances are out_of_scope topics, but the follow-up classifier
    has no tag for them and its new_intent branch only fires on a cross-intent
    switch — so mid-call they arrive as a plain `question`. This keyword gate
    detects them directly so follow_up can reroute back through intake, which
    classifies them out_of_scope and routes the caller to the appeals/grievance
    team. Keyword-based by design: it must not depend on the LLM's follow_up tag.
    """
    if not text:
        return False
    return bool(_APPEAL_GRIEVANCE_RE.search(text))


def _last_user_is_question(last_user: str) -> bool:
    """
    Return True only when the last user message is a genuine question or
    statement that follow_up_agent should classify and answer via the LLM,
    rather than overriding it with the canned opening prompt.

    The opener should be suppressed when:
      - The message is non-empty AND
      - It is not a bare affirmation ("yes", "ok", …) AND
      - It is not a closure signal ("no", "bye", "that's all", …)

    Bare affirmations and closure signals should still receive the opener
    (or be handled by the bare-affirmation fast path / LLM DONE detection
    on the *next* turn after the opener is sent).
    """
    if not last_user:
        return False
    lowered = last_user.lower().strip()
    # Bare affirmations: let the opener fire, then the member can respond.
    if lowered in BARE_AFFIRMATIONS:
        return False
    # Closure signals (e.g. "no", "bye"): the opener must NOT fire because
    # it would echo their "no" back as a new question. Instead fall through
    # to the LLM which will classify it as DONE and close the call.
    if lowered in CLOSURE_KEYWORDS:
        return True  # treat as substantive so LLM classifies DONE
    return True


def _prior_flow_complete(state: State, intent: str) -> bool:
    """True when the flow for `intent` has already finished this call.

    Lets a same-intent request (e.g. a second, distinct claim) qualify as a
    fresh intake while a same-intent clarification — where the flow is still
    open — does not.
    """
    flag = FLOW_COMPLETE_FLAGS.get(intent)
    return bool(state.get(flag)) if flag else False


def is_new_intake_intent(detected_intent: str | None, state: State) -> bool:
    """True only when `detected_intent` is a fresh, routable intake intent that
    warrants restarting the call from verification.

    Returns False for:
      * empty / missing detected_intent — covers "no"/goodbye/end-call, survey
        responses, and same-intent clarifications, which the classifier returns
        without a detected_intent (as done/unsure/question, not new_intent);
      * any intent outside the routable intake set (INTAKE_INTENTS);
      * the same intent just served, unless that prior flow already completed
        (otherwise it is a same-intent clarification, not a new request).
    """
    if not detected_intent:
        return False
    intent = detected_intent.strip().lower()
    if intent not in INTAKE_INTENTS:
        return False
    served = (state.get("call_intent") or "").strip().lower()
    if intent != served:
        return True
    return _prior_flow_complete(state, intent)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        # ── Parked follow-ups (Phase 6, structured in Phase 3) ──────────────
        # Items parked earlier in the call (FOLLOWUP_PARK during slot
        # collection / intake). kind="action" items are update requests —
        # routed via the slot ownership registry below, never
        # blanket-escalated. kind="question" items are NOT answered here:
        # answering stale parked questions in follow_up produced misleading
        # responses, so they are dropped (logged) and the list is cleared on
        # every outcome — the member can re-ask if it still matters.
        parked_items = normalize_parked_followups(state.get("parked_followups"))
        parked_questions = [p["query"] for p in parked_items if p["kind"] == "question"]
        parked_actions = [p for p in parked_items if p["kind"] == "action"]

        if parked_actions:
            return self._route_parked_action(state, parked_actions[0])

        if parked_questions:
            logger.warning(
                "follow_up_agent: dropping parked questions unanswered — "
                "parked-question answering is disabled in follow_up",
                extra={"dropped_parked": parked_questions},
            )

        # Every return path below carries the cleared parked list so dropped
        # questions never resurface on a later turn (reroutes clear it via
        # reset_for_new_intent already).
        def _consume_parked(result: dict) -> dict:
            if parked_items and "parked_followups" not in result:
                result["parked_followups"] = []
            return result

        # ── PHASE 0: First entry — ask opening question ───────────────────────
        if not state.get("follow_up_turn_count"):
            if not state.get("claim_timeline_notification_channel"):
                # Normal entry: opening question not yet delivered.
                #
                # Two cases where we must NOT send the canned opener and should
                # instead fall through to LLM classification:
                #
                # 1. The member already asked a substantive question before
                #    follow_up_agent ran for the first time — e.g. care_wellness
                #    asked "Is there anything else?" and the member replied with
                #    "What is my deductible?" or "What is my OOP max?".
                #    Suppressing the opener lets the LLM answer the question.
                #
                # 2. The member said a closure/DONE word (e.g. "no") in reply
                #    to care_wellness's closing question.  Sending the opener
                #    would discard their "no" and loop them back into a
                #    redundant "Aside from this…?" prompt.  Instead fall through
                #    so the LLM classifies it as DONE and closes the call.
                #
                # Only send the opener when last_user is truly empty (no prior
                # human turn) or is a bare affirmation that carries no intent.
                if _last_user_is_question(last_user):
                    # Fall through to LLM classification.
                    # Seed counters so the rest of the function runs correctly.
                    state = {
                        **state,
                        "follow_up_turn_count": 1,
                        "follow_up_cannot_answer_count": 0,
                    }
                else:
                    result = self.ask_member(state, pick(MSG_FOLLOW_UP_ASK))
                    result["follow_up_turn_count"] = 1
                    result["follow_up_cannot_answer_count"] = 0
                    return _consume_parked(result)

            else:
                # Handoff from notification_setup already delivered the combined
                # confirm + "Aside from this..." message. The last human message
                # in history (e.g. "email") is the N2 preference answer — it must
                # NOT be treated as the member's follow-up response.
                # Guard: if the last conversation message is an AI message, the
                # member has not yet replied to "Aside from this..." — yield
                # silently with is_interrupt=True (no new AI turn) and wait.
                last_msg = messages[-1] if messages else {}
                last_role = (
                    last_msg.get("role", "") if isinstance(last_msg, dict) else getattr(last_msg, "role", "")
                )
                if last_role == "assistant":
                    return _consume_parked(
                        {
                            "next_node": self.AGENT_NAME,
                            "is_interrupt": True,
                            "active_agent": self.AGENT_NAME,
                            "slot_attempts": self.slots_dict(),
                            "metadata_events": [],
                            "app_run_id": state.get("app_run_id", ""),
                            "follow_up_turn_count": 1,
                            "follow_up_cannot_answer_count": 0,
                        }
                    )
                # Last message is human — member already responded to the
                # "Aside from this..." question; fall through to LLM classification.

        turn_count = (state.get("follow_up_turn_count") or 0) + 1
        consecutive_cannot_answer = state.get("follow_up_cannot_answer_count") or 0

        # ── Hard turn cap ────────────────────────────────────────────────────
        if turn_count > MAX_FOLLOW_UP_TURNS:
            logger.info("follow_up_agent: max turns reached — routing to closure")
            return _consume_parked(
                self.signal_complete(
                    state,
                    message="",
                    resolved_intents=["follow_up"],
                    closure_requested=True,
                )
            )

        # ── FAST PATH: bare affirmations — zero LLM calls ───────────────────
        if last_user and last_user.lower().strip() in BARE_AFFIRMATIONS:
            logger.info("follow_up_agent: bare affirmation — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = consecutive_cannot_answer
            return _consume_parked(result)

        # ── FAST PATH: wait request — caller needs a moment ──────────────────
        if _detect_wait_follow_up(last_user):
            logger.info("follow_up_agent: wait request detected — acknowledging")
            result = self.ask_member(state, pick(MSG_WAIT_ACK))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = consecutive_cannot_answer
            return _consume_parked(result)

        # ── LLM call: classify intent + generate answer ──────────────────────
        call_intent = state.get("call_intent", "")
        if call_intent == "claim_services":
            prompt_file = "extraction/follow_up_claims.md"
        else:
            prompt_file = "extraction/follow_up.md"

        extraction_result = await extract_follow_up_decision(
            get_follow_up_llm(),
            build_extraction_prompt_extraction(prompt_file),
            last_agent_message=last_agent,
            last_user_message=last_user,
            pending_slots=None,  # post-flow Q&A: no slots left to collect
            recent_messages=messages[-6:],
            state=state,
        )

        # ── Conversation guards ──────────────────────────────────────────────
        if interrupt := await self.run_conversation_guards(
            state, user_text=last_user, result=extraction_result
        ):
            return _consume_parked(interrupt)

        follow_up_intent = extraction_result.follow_up_intent if extraction_result else FollowUpIntent.UNSURE
        answer = (extraction_result.answer or "").strip() if extraction_result else ""

        # ── WAIT — LLM detected caller is asking for time ────────────────────
        if follow_up_intent == FollowUpIntent.WAIT:
            logger.info("follow_up_agent: LLM WAIT intent — acknowledging")
            result = self.ask_member(state, pick(MSG_WAIT_ACK))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = consecutive_cannot_answer
            return _consume_parked(result)

        # ── APPEAL / GRIEVANCE — keyword gate ────────────────────────────────
        # Appeals/grievances are out_of_scope, but the follow-up classifier has no
        # tag for them and new_intent only fires on cross-intent switches, so they
        # surface here as a plain `question`. Detect them by keyword (not LLM tag)
        # and reroute back through intake, whose out_of_scope screening hands the
        # caller to the appeals/grievance team.
        if _is_appeal_or_grievance(last_user):
            logger.info("follow_up_agent: appeal/grievance keyword detected — rerouting through intake")
            # Pass "" so reset_for_new_intent sets call_intent="" and intake runs its
            # LLM classifier (→ out_of_scope) rather than hitting the same-member
            # early-trigger that fires when call_intent is a known domain intent.
            return self._reroute_through_intake(state, "")

        # ── DONE ─────────────────────────────────────────────────────────────
        if follow_up_intent == FollowUpIntent.DONE:
            logger.info(LOG_CLOSURE)
            result = self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )
            return _consume_parked(result)

        # ── Live cross-call requests (Phase 6): redo / replay route to owner ─
        # Known capabilities are honored by re-running the owning agent —
        # never escalated. Applies whether the classifier tagged the turn
        # update_request or question-without-answer (the request_kind field
        # is authoritative). Unknown topics fall through to the QUESTION
        # machinery below — an answer attempt or the cannot-answer path,
        # never a hard decline.
        kind_raw = getattr(extraction_result, "request_kind", None) if extraction_result else None
        request_kind = str(getattr(kind_raw, "value", kind_raw) or "").strip().lower()
        request_target = (
            (getattr(extraction_result, "request_target", None) or "").strip() if extraction_result else ""
        )
        if (
            request_kind in ("redo", "replay")
            and request_target
            and (follow_up_intent == FollowUpIntent.UPDATE_REQUEST or not answer)
        ):
            if hop := self.route_capability_request(
                state, kind=request_kind, target=request_target, return_awaiting=""
            ):
                logger.info(
                    "follow_up_agent: live %s request routed to owner",
                    request_kind,
                    extra={"target": request_target},
                )
                return _consume_parked(hop)
            if follow_up_intent == FollowUpIntent.UPDATE_REQUEST:
                logger.info(
                    "follow_up_agent: unknown %s topic — degrading to question path",
                    request_kind,
                    extra={"target": request_target},
                )
                follow_up_intent = FollowUpIntent.QUESTION

        # ── UPDATE_REQUEST — route via slot ownership; escalate human-only ───
        if follow_up_intent == FollowUpIntent.UPDATE_REQUEST:
            if request_target and slot_update_owner(request_target) != OWNER_HUMAN:
                logger.info(
                    "follow_up_agent: update_request routed via slot ownership",
                    extra={"target": request_target},
                )
                return self._route_parked_action(
                    state, {"query": last_user, "kind": "action", "target": request_target}
                )
            logger.info("follow_up_agent: update_request — escalating (human-only or unknown target)")
            return _consume_parked(
                self.signal_escalate(
                    state,
                    MSG_UPDATE_REQUEST_ESCALATE,
                    reason="update_request_in_follow_up",
                    initiator="Agent",
                )
            )

        # ── NEW_INTENT — member asked about a different service ──────────────────
        if follow_up_intent == FollowUpIntent.NEW_INTENT:
            detected = (extraction_result.detected_intent or "").strip()
            if is_new_intake_intent(detected, state):
                # Guard: anaphoric back-references ("what about that claim history?",
                # "can you go over my X again?") and explicit replay classifications
                # indicate the caller is referring to something mentioned this call,
                # not starting a fresh service. Degrade to question/cannot-answer.
                _is_anaphoric = bool(
                    re.search(
                        r"\bwhat\s+about\s+that\b"
                        r"|\bgo\s+over\b.*\bagain\b"
                        r"|\bagain\b.*\bhistory\b"
                        r"|\bthat\s+\w+\s+history\b",
                        last_user or "",
                        re.IGNORECASE,
                    )
                )
                if _is_anaphoric or request_kind == "replay":
                    logger.info(
                        "follow_up_agent: new_intent suppressed — "
                        "anaphoric or replay reference detected; falling through to question path",
                        extra={
                            "detected_intent": detected,
                            "request_kind": request_kind,
                            "utterance": last_user,
                        },
                    )
                    # Fall through to QUESTION / cannot-answer path
                elif detected.lower() in INTAKE_RESCREEN_INTENTS:
                    return self._reroute_through_intake(state, detected)
                else:
                    # Fresh intake intent → full reset + re-verify before the new flow.
                    return self._reroute_through_verification(state, detected)
            # Not a fresh intake intent (missing/unrecognised detected_intent, or a
            # same-intent clarification). Fall through to the QUESTION/cannot-answer
            # handling below so we stay in follow_up rather than restarting the call.
            logger.info(
                "follow_up_agent: new_intent did not qualify as fresh intake — staying in follow_up",
                extra={"detected_intent": detected, "call_intent": state.get("call_intent", "")},
            )

        # ── UNSURE ────────────────────────────────────────────────────────────
        if follow_up_intent == FollowUpIntent.UNSURE:
            logger.info("follow_up_agent: unsure — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = 0  # reset streak
            return _consume_parked(result)

        # ── QUESTION ─────────────────────────────────────────────────────────
        # Strip forbidden redirect phrases — only block if the answer STARTS with them
        if answer:
            answer_lower = answer.lower().strip()
            if any(answer_lower.startswith(p) for p in _FORBIDDEN_ANSWER_PHRASES):
                logger.info(LOG_CANNOT_ANSWER)
                answer = ""

        if not answer:
            # Cannot answer this question
            new_cannot_answer_count = consecutive_cannot_answer + 1
            logger.info(LOG_CANNOT_ANSWER + " (consecutive=%d)", new_cannot_answer_count)
            cannot_answer_msg = pick(MSG_CANNOT_ANSWER)

            # ── Escalate on Nth consecutive cannot-answer ────────────────────
            if new_cannot_answer_count >= MAX_CANNOT_ANSWER_BEFORE_ESCALATE:
                logger.info(
                    "follow_up_agent: %d consecutive cannot-answer turns — escalating",
                    new_cannot_answer_count,
                )
                return _consume_parked(
                    self.signal_escalate(
                        state,
                        cannot_answer_msg,
                        reason="repeated_cannot_answer_in_follow_up",
                        initiator="Agent",
                    )
                )

            # Still under threshold — reply and count
            message = f"{cannot_answer_msg} {pick(MSG_CONTINUATION)}"
            result = self.ask_member(state, message)
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = new_cannot_answer_count
            return _consume_parked(result)

        # Real answer — reset cannot-answer streak.
        # Append a continuation prompt if the answer doesn't already end with one.
        logger.info(LOG_ANSWERED)
        _continuation = pick(MSG_CONTINUATION)
        _answer_stripped = answer.lower().rstrip("? ")
        if not any(_answer_stripped.endswith(phrase.lower().rstrip("? ")) for phrase in MSG_CONTINUATION):
            answer = f"{answer} {_continuation}"
        result = self.ask_member(state, answer)
        result["follow_up_turn_count"] = turn_count
        result["follow_up_last_question"] = last_user
        result["follow_up_cannot_answer_count"] = 0
        return _consume_parked(result)

    def _route_parked_action(self, state: State, action: dict) -> dict:
        """Route a parked kind="action" update request via the capability and
        slot ownership registries (core.slot_ownership) instead of
        blanket-escalating.

        - Capability-first (Phase 6): a target the capability registry maps
          to a redo (e.g. delivery_method after the list was dispatched)
          hands off directly to the owning agent — a lightweight hop, not a
          full flow reset.
        - OWNER_VERIFICATION slots (identity): re-run verification for the
          current intent — re-verification re-collects the slot.
        - Intent-owned slots: re-run the owning flow, respecting the intake
          re-screen split exactly like a NEW_INTENT hand-off.
        - OWNER_HUMAN slots (or verification-owned with no intent to resume):
          escalate with the update-request message — the only case that still
          reaches MSG_UPDATE_REQUEST_ESCALATE for a parked item.

        Reroutes clear parked_followups via reset_for_new_intent, matching the
        pre-existing behavior of every mid-call reroute.
        """
        target = (action.get("target") or "").strip()
        # Capability-first gate, per topic: a redo only makes sense when the
        # thing to re-do exists (Phase 7 extends delivery's provider_list_sent
        # gate to the notification capability).
        topic = canonical_capability_topic("redo", target)
        redo_data_exists = {
            "delivery": bool(state.get("provider_list_sent")),
            "notification": bool(state.get("notification_channel"))
            and state.get("notification_channel") != "not_set",
        }.get(topic, False)
        if redo_data_exists:
            if hop := self.route_capability_request(state, kind="redo", target=target, return_awaiting=""):
                logger.info(
                    "follow_up_agent: parked action routed via capability registry",
                    extra={"target": target},
                )
                hop["parked_followups"] = [
                    p
                    for p in normalize_parked_followups(state.get("parked_followups"))
                    if not (p.get("kind") == "action" and p.get("target") == target)
                ]
                return hop
        owner = slot_update_owner(target)
        logger.info(
            "follow_up_agent: parked update action — routing via slot ownership",
            extra={"target": target, "owner": owner, "query": action.get("query", "")},
        )
        if owner == OWNER_VERIFICATION:
            intent = (state.get("call_intent") or "").strip()
            if intent:
                return self._reroute_through_verification(state, intent)
            owner = OWNER_HUMAN  # no flow to resume after re-verifying
        if owner != OWNER_HUMAN:
            # Owner is an intake intent — re-run the owning flow.
            if owner in INTAKE_RESCREEN_INTENTS:
                return self._reroute_through_intake(state, owner)
            return self._reroute_through_verification(state, owner)
        result = self.signal_escalate(
            state,
            MSG_UPDATE_REQUEST_ESCALATE,
            reason=f"parked_update_{target or 'unknown'}_human_only",
            initiator="Agent",
        )
        result["parked_followups"] = []
        return result

    def _reroute_through_verification(self, state: State, detected_intent: str) -> dict:
        """
        A fresh intake intent was detected mid-follow-up. Fully reset the
        conversation (identity + verification flags + every domain field) and
        route to the verification node so the caller re-verifies before the new
        flow begins.

        Routing uses the same mechanism every other hand-off uses: set
        ``next_node`` and let ``conditional_routing`` (the follow_up node's
        conditional edge) dispatch — no Command(goto=...) needed. ``is_interrupt``
        is False so verification runs in the same super-step and owns its own
        first prompt.

        ``reset_for_new_intent`` stages the new intent in BOTH ``call_intent``
        (which verification reads to choose its claims/provider pipeline, and
        which the post-verification fast-path reads to pick the domain agent) and
        ``pending_intent`` (a durable marker that this is a mid-call switch).
        """
        logger.info(
            LOG_NEW_INTENT,
            extra={"detected_intent": detected_intent, "previous_intent": state.get("call_intent", "")},
        )
        updates = reset_for_new_intent(state, detected_intent)
        updates["next_node"] = AgentNode.VERIFICATION.value
        updates["is_interrupt"] = False
        updates["active_agent"] = self.AGENT_NAME
        updates["metadata_events"] = []
        updates["app_run_id"] = state.get("app_run_id", "")
        return updates

    def _reroute_through_intake(self, state: State, detected_intent: str) -> dict:
        """
        A fresh intake intent that must be re-screened was detected mid-follow-up.
        Like ``_reroute_through_verification`` this fully resets the conversation,
        but routes to the *intake* node instead of verification so intake re-applies
        its front-door screening (e.g. the unsupported-provider-type gate) before
        identity is re-collected.

        Same-member shortcut eligibility:
          The shortcut (save verified context → let intake skip its LLM and ask
          "same member or different?") only applies when the pivot intent is
          GUARANTEED to reach a domain agent — i.e. the LLM would be redundant.
          Two cases qualify:
            1. provider_services with a KNOWN SUPPORTED type (normalize_provider_type
               returns a non-empty string).  Unsupported types must still run the LLM
               so intake can fire the unsupported-type gate at the front door.
            2. claim_services (genuine new-intent pivot; the follow-up LLM already
               verified the classification).
          Anything else — unsupported provider, appeal/grievance ("")  — clears
          call_intent and lets intake re-classify via LLM as usual.

        Fast-path bypass for provider_services + known supported type + unverified
        member: skip intake entirely and go straight to verification with
        provider_type pre-set.
        """
        from agent.slots.normalizers import normalize_provider_type

        last_user = _last_user_msg(list(state.get("messages") or []))

        resolved_pt = ""
        if detected_intent == "provider_services":
            resolved_pt = normalize_provider_type(last_user)
            if resolved_pt and not state.get("member_status_verify"):
                # Known supported type, no verified context — skip intake.
                logger.info(
                    LOG_NEW_INTENT,
                    extra={
                        "detected_intent": detected_intent,
                        "previous_intent": state.get("call_intent", ""),
                        "route": "intake_rescreen_bypassed",
                        "provider_type": resolved_pt,
                    },
                )
                updates = reset_for_new_intent(state, detected_intent)
                updates["provider_type"] = resolved_pt
                updates["next_node"] = AgentNode.VERIFICATION.value
                updates["is_interrupt"] = False
                updates["active_agent"] = self.AGENT_NAME
                updates["metadata_events"] = []
                updates["app_run_id"] = state.get("app_run_id", "")
                return updates

        # Determine whether to save member context and skip intake's LLM
        # (routing straight to the same-member disambiguation question instead).
        #
        # Rules for provider_services:
        #   - SUPPORTED or UNSPECIFIED type  → save context, same-member check
        #     (unspecified types like "a doctor" are fine — provider_search will
        #     ask for the type after the member confirms same/different)
        #   - KNOWN UNSUPPORTED type         → do NOT save context; let intake's
        #     LLM classify provider_type_unsupported and escalate at the front door
        #     before any identity is re-collected.
        # Rules for claim_services: always eligible.
        # Anything else ("", appeal/grievance): always run intake LLM.
        skip_llm_eligible = False
        if state.get("member_status_verify"):
            if detected_intent == "claim_services":
                skip_llm_eligible = True
            elif detected_intent == "provider_services":
                # Import locally to avoid module-level circular dependency.
                from agent.agents.intake.handlers import _extract_provider_type_from_utterance

                is_unsupported = _extract_provider_type_from_utterance(last_user) != "this provider type"
                skip_llm_eligible = not is_unsupported

        saved_member_context: dict | None = None
        if skip_llm_eligible:
            saved_member_context = {
                "first_name": state.get("first_name"),
                "last_name": state.get("last_name"),
                "member_id": state.get("member_id"),
                "dob": state.get("dob"),
                "relationship": state.get("relationship"),
                "caller_role": state.get("caller_role"),
                "zip_code": state.get("zip_code"),
                "phone_number": state.get("phone_number"),
                "fax": state.get("fax"),
                "email": state.get("email"),
                "conversation_context": state.get("conversation_context"),
                "name_confirmed": state.get("name_confirmed", False),
                "phone_confirmed": state.get("phone_confirmed", False),
            }

        logger.info(
            LOG_NEW_INTENT,
            extra={
                "detected_intent": detected_intent,
                "previous_intent": state.get("call_intent", ""),
                "route": "intake_rescreen",
                "saved_member_context": saved_member_context is not None,
            },
        )
        updates = reset_for_new_intent(state, detected_intent)
        if saved_member_context is not None:
            # Intent already classified — keep call_intent so intake skips its LLM
            # and goes directly to the same-member disambiguation question.
            pass
        else:
            updates["call_intent"] = ""  # force intake to re-classify + re-screen
            updates["pending_intent"] = ""  # intake → verification routes by call_intent
        updates["reverify_bridge_pending"] = False  # intake delivers its own bridge
        updates["next_node"] = AgentNode.INTAKE.value
        updates["is_interrupt"] = False
        updates["active_agent"] = self.AGENT_NAME
        updates["metadata_events"] = []
        updates["app_run_id"] = state.get("app_run_id", "")
        if saved_member_context is not None:
            updates["saved_member_context"] = saved_member_context
        return updates


async def follow_up_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await FollowUpAgent.from_state(state).execute(state)
