"""
agent.py — BenefitsAgent: fetches plan benefits from Salesforce,
explains them to the member, and makes the Care Coach proactive offer.
"""

from __future__ import annotations

import random

from agent.agents.benefits.constants import (
    AGENT_NAME,
    BENEFITS_EXPLANATION_TEMPLATE,
    BENEFITS_NOEXPLANATION_TEMPLATES,
    CARE_COACH_OFFER_TEMPLATES,
    LOG_BENEFITS_EXPLAINED,
    LOG_BENEFITS_FETCHED,
    LOG_ENTERED,
)
from agent.agents.benefits.handlers import fetch_benefits
from agent.agents.benefits.llm import extract_benefits_decision
from agent.core.agent import BaseAgent
from agent.core.request_detection import reconcile_worker_result
from agent.core.slot_ownership import capability_topic
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.slots.normalizers import normalize_yes_no
from agent.state import State, normalize_parked_followups
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt_core, pick

logger = get_logger(__name__)

_CARE_COACH_SLOT = "care_coach_response"


def _clean_amount(val) -> str:
    """
    Convert a Salesforce Decimal/int/str amount to a clean integer string.
    "600.0" → "600",  "600.00" → "600",  "600" → "600",  None → "0"
    Only strips trailing zeros when a decimal point is present — avoids
    the str.rstrip(".0") pitfall which treats chars individually.
    """
    s = str(val or "").strip()
    if not s:
        return "0"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


class BenefitsAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        current_awaiting = state.get("awaiting_slot", "")

        # ── CROSS-AGENT RE-ENTRY (Phase 6): replay of the benefits summary ───
        # Checked BEFORE the care_coach_offered early exit — the whole point
        # of the re-entry contract is serving requests after completion.
        replay_request = self.consume_cross_agent_request(state, kinds=("replay",), targets=("benefits",))
        if replay_request:
            return await self._replay_benefits(state, request=replay_request)

        # ── RESUME after a routed request completed (Phase 6 round-trip) ─────
        # The orchestrator sent us back after the owner (delivery re-dispatch)
        # finished. Acknowledge and re-ask the still-unanswered Care Coach
        # offer — no extraction on this hop (the last user utterance was
        # consumed by the owner). The benefits offer itself is NOT repeated.
        if state.get("slot_update_resume") and current_awaiting == _CARE_COACH_SLOT:
            method = (state.get("delivery_method") or "").strip()
            ack = (
                f"All set — I've sent that same list to your {method} as well. "
                if method
                else "All set — that's been taken care of. "
            )
            result = self.ask_member(state, ack + random.choice(CARE_COACH_OFFER_TEMPLATES))
            result["awaiting_slot"] = _CARE_COACH_SLOT
            result["slot_update_resume"] = False
            result["benefits_explained"] = state.get("benefits_explained", False)
            return result

        # ── EARLY EXIT: only gate on care_coach_offered; benefits_explained may be False on NO path ─
        if state.get("care_coach_offered"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, bool(state.get("proactive_offer_available"))),
                proactive_offer_available=bool(state.get("proactive_offer_available")),
            )

        # ── PHASE B: Member responded to the Care Coach offer ────────────────
        if current_awaiting == _CARE_COACH_SLOT:
            return await self._handle_care_coach_response(state, messages, last_user, last_agent)

        # ── PHASE A: Branch on proactive_offer_available ─────────────────────
        benefits_wanted = bool(state.get("proactive_offer_available"))

        if benefits_wanted:
            # YES path — fetch SF, explain benefits, append care coach offer
            benefits, interrupt = await fetch_benefits(self, state)
            if interrupt:
                return interrupt

            logger.info(LOG_BENEFITS_FETCHED)

            indv_ded = _clean_amount(benefits.get("individual_deductible"))
            fam_ded = _clean_amount(benefits.get("family_deductible"))
            coins_pct = _clean_amount(benefits.get("coinsurance_percent"))
            indv_oop = _clean_amount(benefits.get("individual_oop_max"))
            fam_oop = _clean_amount(benefits.get("family_oop_max"))

            explanation = BENEFITS_EXPLANATION_TEMPLATE.format(
                individual_deductible=indv_ded,
                family_deductible=fam_ded,
                coinsurance_percent=coins_pct,
                individual_oop_max=indv_oop,
                family_oop_max=fam_oop,
            )
            care_coach_offer = random.choice(CARE_COACH_OFFER_TEMPLATES)
            full_message = f"{explanation}\n\n{care_coach_offer}"

            logger.info(LOG_BENEFITS_EXPLAINED)

            result = self.ask_member(state, full_message)
            result["awaiting_slot"] = _CARE_COACH_SLOT
            result["benefits_explained"] = True
            result["individual_deductible"] = indv_ded
            result["family_deductible"] = fam_ded
            result["coinsurance_percent"] = coins_pct
            result["individual_oop_max"] = indv_oop
            result["family_oop_max"] = fam_oop
            return result

        else:
            # NO path — skip SF fetch, go straight to Care Coach offer
            message = pick(BENEFITS_NOEXPLANATION_TEMPLATES)
            result = self.ask_member(state, message)
            result["awaiting_slot"] = _CARE_COACH_SLOT
            result["benefits_explained"] = False
            return result

    # -------------------------------------------------------------------------

    async def _handle_care_coach_response(
        self, state: State, messages: list, last_user: str, last_agent: str
    ) -> dict:
        """Extract and act on the member's yes/no to the Care Coach offer."""
        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(_CARE_COACH_SLOT, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_benefits_decision(
            get_extraction_llm(),
            build_extraction_prompt_core("extraction/benefits.md"),
            awaiting_slot=_CARE_COACH_SLOT,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            pending_slots=[_CARE_COACH_SLOT],
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        # ── DETERMINISTIC RECONCILE (Phase 1) ────────────────────────────────
        # llm.py already reconciles on success, but extraction fallbacks (and
        # monkeypatched results) bypass it — re-running here is idempotent and
        # guarantees "send that list to my email instead of fax" always yields
        # kind redo / target delivery before the yes/no extraction below.
        result = reconcile_worker_result(result, last_user)

        # ── Check care_coach_response BEFORE routing ─────────────────────────
        # A direct yes/no answer to the offer must not be suppressed by an
        # incidental replay/redo classification the LLM picks up from the
        # conversation context (e.g. "no thank you" after a parked replay
        # is still the answer to the care coach, not a replay request).
        extracted = (result.extracted or {}) if result else {}
        raw_response = extracted.get("care_coach_response", "")
        normalized = normalize_yes_no(raw_response) if raw_response else ""

        # ── CROSS-CALL REQUESTS (Phase 6): redo / replay voiced mid-offer ────
        kind_raw = getattr(result, "request_kind", None) if result else None
        request_kind = str(getattr(kind_raw, "value", kind_raw) or "").strip().lower()
        request_target = ((getattr(result, "update_target", None) or "").strip()) if result else ""
        if not normalized and request_kind in ("redo", "replay") and request_target:
            # Replay of our own material stays in-flow — re-explain and
            # re-ask the offer, zero routing.
            if request_kind == "replay" and capability_topic(request_target) == "benefits":
                return await self._replay_benefits(state, request=None)
            # Another agent owns it (e.g. redo → delivery re-dispatch):
            # hand off now; the way back is the Care Coach offer.
            if hop := self.route_capability_request(
                state, kind=request_kind, target=request_target, return_awaiting=_CARE_COACH_SLOT
            ):
                return hop
            # Unknown topic — park as a question for follow_up (Phase 3
            # degrade path), acknowledge, and re-ask the offer. Never a
            # hard decline. Any delivery-phrased redo resolves via the
            # registry (redo/provider_list → delivery) and routes above, so
            # landing here with one means the tables have a gap — warn with
            # the raw target for observability.
            parked = normalize_parked_followups(state.get("parked_followups"))
            parked.append(
                {"query": last_user or f"{request_kind} {request_target}", "kind": "question", "target": ""}
            )
            logger.warning(
                "benefits_agent: unknown %s topic parked as question",
                request_kind,
                extra={"kind": request_kind, "target": request_target, "utterance": last_user},
            )
            retry = self.ask_member(
                state,
                "I'll come back to that in just a moment. " + random.choice(CARE_COACH_OFFER_TEMPLATES),
            )
            retry["awaiting_slot"] = _CARE_COACH_SLOT
            retry["parked_followups"] = parked
            retry["benefits_explained"] = state.get("benefits_explained", False)
            return retry

        # ── UPDATE of delivery contact (email/fax) while list already sent ───
        # "please change my email address?" during the Care Coach offer means
        # the caller wants the list re-dispatched to a different/confirmed
        # contact. Route to delivery_management as a redo so it handles the
        # contact update + re-send. Only when request_kind is "update" (not
        # redo — that is handled above) and the list has already been sent.
        if (
            not normalized
            and request_kind == "update"
            and request_target in ("email", "fax")
            and state.get("provider_list_sent")
        ):
            if hop := self.route_capability_request(
                state, kind="redo", target="delivery", return_awaiting=_CARE_COACH_SLOT
            ):
                logger.info(
                    "benefits_agent: delivery contact update mid-offer — routing as redo",
                    extra={"target": request_target},
                )
                return hop

        if normalized == "yes":
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, True),
                proactive_offer_available=True,
            )

        if normalized == "no":
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, False),
                proactive_offer_available=False,
            )

        # No clear yes/no — retry or exhaust gracefully
        self.slot_fail(_CARE_COACH_SLOT)
        if self.get_slot(_CARE_COACH_SLOT).is_exhausted():
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, False),
                proactive_offer_available=False,
            )

        retry_msg = random.choice(CARE_COACH_OFFER_TEMPLATES)
        retry_result = self.ask_member(state, retry_msg)
        retry_result["awaiting_slot"] = _CARE_COACH_SLOT
        retry_result["benefits_explained"] = True
        return retry_result

    async def _replay_benefits(self, state: State, request: dict | None) -> dict:
        """Replay capability (Phase 6): re-explain the plan benefits.

        request set   — routed re-entry after our flow completed: re-run the
                        Phase A fetch+explain (fetch_benefits is idempotent
                        from Salesforce), skip re-offering the Care Coach,
                        clear the request key, and hand back to
                        return_to_agent (or follow_up when none).
        request None  — in-flow replay voiced during the Care Coach offer:
                        re-explain and re-ask the offer, no routing.
        """
        amounts = {
            "individual_deductible": (state.get("individual_deductible") or "").strip(),
            "family_deductible": (state.get("family_deductible") or "").strip(),
            "coinsurance_percent": (state.get("coinsurance_percent") or "").strip(),
            "individual_oop_max": (state.get("individual_oop_max") or "").strip(),
            "family_oop_max": (state.get("family_oop_max") or "").strip(),
        }
        if not all(amounts.values()):
            # Benefits were never fetched this call (NO path) — fetch now.
            benefits, interrupt = await fetch_benefits(self, state)
            if interrupt:
                return interrupt
            amounts = {
                "individual_deductible": _clean_amount(benefits.get("individual_deductible")),
                "family_deductible": _clean_amount(benefits.get("family_deductible")),
                "coinsurance_percent": _clean_amount(benefits.get("coinsurance_percent")),
                "individual_oop_max": _clean_amount(benefits.get("individual_oop_max")),
                "family_oop_max": _clean_amount(benefits.get("family_oop_max")),
            }
        explanation = BENEFITS_EXPLANATION_TEMPLATE.format(**amounts)
        logger.info(
            "benefits_agent: benefits replay",
            extra={"routed": bool(request), "return_to": (request or {}).get("return_to_agent", "")},
        )

        if request is None:
            # In-flow: the Care Coach offer is still unanswered — re-ask it.
            message = f"Of course — here are your benefits again.\n\n{explanation}\n\n" + random.choice(
                CARE_COACH_OFFER_TEMPLATES
            )
            result = self.ask_member(state, message)
            result["awaiting_slot"] = _CARE_COACH_SLOT
        else:
            # Routed re-entry: no Care Coach re-offer — hand straight back.
            message = f"Of course — here are your benefits again.\n\n{explanation}"
            result = self.ask_member(state, message)
            result["next_node"] = request.get("return_to_agent") or "follow_up_agent"
            result["awaiting_slot"] = request.get("return_awaiting", "")
            result["pending_cross_agent_request"] = {}
            result["pending_slot_update"] = {}  # legacy key
        result["benefits_explained"] = True
        result.update(amounts)
        return result

    @staticmethod
    def _completion_context(state: State, care_coach_accepted: bool) -> dict:
        return {
            "benefits_explained": state.get("benefits_explained", False),
            "care_coach_offered": True,
            "care_coach_offer_made": True,
            "proactive_offer_available": care_coach_accepted,
            "individual_deductible": state.get("individual_deductible", ""),
            "family_deductible": state.get("family_deductible", ""),
            "coinsurance_percent": state.get("coinsurance_percent", ""),
            "individual_oop_max": state.get("individual_oop_max", ""),
            "family_oop_max": state.get("family_oop_max", ""),
        }


async def benefits_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await BenefitsAgent.from_state(state).execute(state)
