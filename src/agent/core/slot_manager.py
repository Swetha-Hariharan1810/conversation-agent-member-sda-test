"""
slot_manager.py — SlotManagerMixin: slot state + per-turn slot collection.

Two concerns kept together because _collect_slot directly uses slot state:
  1. Slot CRUD (get_slot, slot_ok, slot_fail, slots_dict, guard_loop_limit)
  2. _collect_slot — the reusable per-turn collector used by every agent

_collect_slot returns:
  (value, None)      → slot confirmed, caller proceeds
  (None, interrupt)  → ask/retry dict, caller returns immediately
  (None, escalation) → retries exhausted, caller returns immediately

See git history for migration notes from v4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from agent.conversation.context import ConversationContext
from agent.core.constants import MAX_WAIT_TURNS
from agent.core.models import SlotAttempt
from agent.responses.builder import (
    build_initial_prompt,
    build_transition_prompt,
)
from agent.responses.static import MSG_WAIT_ACK, MSG_WAIT_NUDGE, build_slot_exhausted_message
from agent.slots.types import SlotType
from agent.state import State, normalize_parked_followups
from agent.utils import _last_user_msg, detect_cannot_provide, detect_wait_request, pick

# ── Empathetic "cannot provide" escalation message ────────────────────────────
# Slot-aware: {slot_label} is filled at runtime from the SlotType label.
# Keeps exactly the same warm tone as the rest of the codebase.
_CANNOT_PROVIDE_MSG = "No problem — let me connect you with a representative "


def _mk_session_ctx(
    *,
    extracted_val: str | None = None,
    followup_query: str | None = None,
    confirmed_values: dict | None = None,
) -> dict:
    """Build a lightweight session-context dict for _generate_slot_retry_response."""
    ctx: dict = {}
    if extracted_val is not None:
        ctx["extracted_val"] = extracted_val
    if followup_query:
        ctx["followup_query"] = followup_query
    if confirmed_values is not None:
        ctx["confirmed_values"] = confirmed_values
    return ctx


# WorkerResult.followup_disposition → generation-LLM guard label.
# Missing/none defaults to FOLLOWUP_RESPOND — the generation LLM self-triages
# (answers from Confirmed: if it can, gracefully declines if it cannot).
_DISPOSITION_GUARDS: dict[str, str] = {
    "answer": "FOLLOWUP_RESPOND",
    "park": "FOLLOWUP_PARK",
    # Legacy aliases — kept so old cached extraction results still route correctly
    "answer_now": "FOLLOWUP_RESPOND",
    "decline": "FOLLOWUP_RESPOND",
}

# Max detours per update target before escalating (counter: f"update_{target}").
_MAX_UPDATE_DETOURS = 2


# _InternalSlotConfig is the low-level config used by _collect_slot.
# The public SlotConfig used by SlotPipeline lives in slots/pipeline.py.
# pipeline.py imports this class for constructing _collect_slot calls.
@dataclass
class _InternalSlotConfig:
    """Low-level slot configuration consumed by _collect_slot."""

    slot_name: str
    prompt: str
    normalizer: Callable
    validator: Callable
    slot_type: Optional["SlotType"] = None


class SlotManagerMixin:
    """Mixin that adds slot management and collection to BaseAgent."""

    # -------------------------------------------------------------------------
    # Slot CRUD
    # -------------------------------------------------------------------------

    def get_slot(self, name: str) -> SlotAttempt:
        if name not in self._slots:
            self._slots[name] = SlotAttempt(name)
        return self._slots[name]

    def slot_ok(self, name: str, value: Any) -> None:
        """Mark slot confirmed. Also queues a CallAgentField event."""
        self.get_slot(name).record_attempt(value, success=True)
        self._newly_confirmed.add(name)

    def slot_fail(self, name: str, value: Any = None, is_asr: bool = False) -> None:
        """Record a failed slot attempt."""
        self.get_slot(name).record_attempt(value, success=False, is_asr=is_asr)

    def slot_exhausted(self, name: str) -> bool:
        return self.get_slot(name).is_exhausted()

    def slots_dict(self) -> dict:
        """Serialize slot state for LangGraph persistence."""
        return {
            k: {"attempt_count": v.attempt_count, "confirmed": v.confirmed, "last_value": v.last_value}
            for k, v in self._slots.items()
        }

    @staticmethod
    def _restore_slot(slot_name: str, data) -> SlotAttempt:
        slot = SlotAttempt(slot_name)
        slot.attempt_count = data.get("attempt_count", 0)
        slot.confirmed = data.get("confirmed", False)
        slot.last_value = data.get("last_value")
        return slot

    # -------------------------------------------------------------------------
    # ask_member_with_context — carries context in every interrupt
    # -------------------------------------------------------------------------

    def ask_member_with_context(
        self,
        state: State,
        message: str,
        context: ConversationContext,
    ) -> dict:
        """
        Like ask_member() but also persists the updated ConversationContext
        into state so it's available on the next turn.
        """
        result = self.ask_member(state, message)
        result["conversation_context"] = context.to_dict()
        return result

    # -------------------------------------------------------------------------
    # Loop guard
    # -------------------------------------------------------------------------

    def guard_loop_limit(
        self,
        state: State,
        counter_name: str,
        max_attempts: int,
        escalate_message: str,
        escalate_reason: str,
        *,
        initiator: str = "Agent",
    ) -> Optional[dict]:
        """
        Increment a named counter and return an escalation dict if the limit is hit.
        Returns None while still under the limit.
        """
        self.slot_fail(counter_name)
        count = self.get_slot(counter_name).attempt_count
        if count >= max_attempts:
            return self.signal_escalate(state, escalate_message, escalate_reason, initiator=initiator)
        return None

    # -------------------------------------------------------------------------
    # LLM 2 retry response helper
    # -------------------------------------------------------------------------

    async def _generate_slot_retry_response(
        self,
        state: State,
        slot_name: str,
        ctx: ConversationContext,
        messages: list[dict],
        *,
        guard: str = "RETRY",
        session_context: dict | None = None,
        extracted_this_turn: str | None = None,
        next_slot_label: str | None = None,
        will_append_ask: bool = False,
        fallback_slot_label: str | None = None,
    ) -> str:
        # Lazy import: core.slot_manager → llm.response_generator → llm.config → core (via schema);
        # importing at module level would create a core → llm → core cycle.
        from agent.llm.response_generator import generate_recovery_message, sanitize_generated

        slot_state = self.get_slot(slot_name)
        slot_label_override: str | None = None
        if slot_name == "relationship" and state.get("relationship"):
            slot_label_override = "relationship — whether they are the subscriber or dependent"
        elif slot_name in ("phone_confirmed", "phone_confirmation") and state.get("phone_number"):
            digits = "".join(c for c in state["phone_number"] if c.isdigit())
            formatted = (
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else state["phone_number"]
            )
            slot_label_override = (
                f"phone confirmation — whether {formatted} is still the number on file (yes or no)"
            )
        sc = session_context or {}
        text = await generate_recovery_message(
            slot_name=slot_name,
            attempt=slot_state.attempt_count,
            guard=guard,
            last_messages=messages[-4:],
            slot_label_override=slot_label_override,
            caller_name=ctx.caller_first_name,
            # FOLLOWUP_ANSWER passes real (masked) values so the LLM can answer
            # from Confirmed; all other guards pass actual state values so LLM 2
            # sees meaningful context — mask_confirmed handles sensitive slots
            # (member_id, dob) by rendering them as "on file".
            confirmed_slots=sc.get("confirmed_values")
            or {k: (str(state.get(k) or "").strip() or "confirmed") for k in (ctx.confirmed_slots or [])},
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted_this_turn
            if extracted_this_turn is not None
            else sc.get("extracted_val"),
            followup_query=sc.get("followup_query"),
        )
        # Single-ask invariant: strip re-asks of confirmed slots always; when a
        # static ask is appended after this text, also strip any competing
        # question so the appended ask is the only one. slot_name itself is
        # exempt — it is the slot being collected (RETRY/CLARIFY re-ask it) or
        # the value just captured (FOLLOWUP acks mention it).
        return sanitize_generated(
            text,
            guard=guard,
            next_slot_label=next_slot_label,
            confirmed_labels=tuple(s for s in (ctx.confirmed_slots or []) if s != slot_name),
            will_append_ask=will_append_ask,
            fallback_slot_label=fallback_slot_label or slot_name.replace("_", " "),
        )

    async def _generate_correction_ack(
        self,
        state: State,
        corrected_fields: list[str],
        awaiting_slot: str,
        ctx: "ConversationContext",
        messages: list,
        *,
        decision=None,
    ) -> str:
        from agent.agents.verification.handlers import _NORMALIZERS
        from agent.llm.response_generator import generate_recovery_message, sanitize_generated

        corrected_label = corrected_fields[0].replace("_", " ") if corrected_fields else "that"

        corrected_value = ""
        if corrected_fields:
            raw = getattr(decision, "corrections", {}) if decision else {}
            norm = _NORMALIZERS.get(corrected_fields[0])
            corrected_value = (
                norm(str(raw.get(corrected_fields[0], ""))) if norm else str(raw.get(corrected_fields[0], ""))
            )

        # explicit correction ack for name slots
        if corrected_fields and corrected_fields[0] in ("first_name", "last_name") and corrected_value:
            # Names: read back explicitly so caller hears their name confirmed
            slot_label_override = (
                f"caller corrected their {corrected_label} to '{corrected_value}' — "
                f"acknowledge by explicitly saying the corrected {corrected_label} "
                f"is '{corrected_value}', "
                f"then ask for their {awaiting_slot.replace('_', ' ')}"
            )
        elif corrected_value:
            # Sensitive slots (member_id, dob etc.): acknowledge WITHOUT reading
            # the value back out loud — just confirm the update and re-ask
            slot_label_override = (
                f"caller corrected their {corrected_label} — "
                f"acknowledge the correction without repeating the value, "
                f"then ask for their {awaiting_slot.replace('_', ' ')}"
            )
        else:
            # No new value provided: re-ask naturally for awaiting_slot
            slot_label_override = (
                f"caller corrected {corrected_label}" + f" — now re-ask for {awaiting_slot.replace('_', ' ')}"
            )

        text = await generate_recovery_message(
            slot_name=awaiting_slot,
            attempt=0,
            guard="CORRECTION",
            last_messages=messages[-6:],
            slot_label_override=slot_label_override,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages[-6:]),
        )
        # The ack's own re-ask targets awaiting_slot (not confirmed) and it is
        # explicitly told to read corrected fields back — exempt those; any
        # OTHER confirmed slot re-asked here is a double-ask and is stripped.
        return sanitize_generated(
            text,
            guard="CORRECTION",
            confirmed_labels=tuple(
                s for s in (ctx.confirmed_slots or []) if s not in corrected_fields and s != awaiting_slot
            ),
            will_append_ask=False,
            fallback_slot_label=awaiting_slot.replace("_", " "),
        )

    # -------------------------------------------------------------------------
    # Update / correction detours (Phase 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_slot_tools(slot_name: str, slot_configs: Optional[dict]) -> tuple:
        """Resolve (normalizer, validator) for a slot.

        The current pipeline's configs win, so any agent using SlotPipeline can
        validate corrections against its own slot definitions; the verification
        maps are the fallback for identity slots corrected mid-flow elsewhere.
        Returns (None, None) when the slot is unknown to both.
        """
        cfg = (slot_configs or {}).get(slot_name)
        if cfg is not None and getattr(cfg, "normalizer", None) and getattr(cfg, "validator", None):
            return cfg.normalizer, cfg.validator
        from agent.agents.verification.handlers import _NORMALIZERS, _VALIDATORS

        return _NORMALIZERS.get(slot_name), _VALIDATORS.get(slot_name)

    def resolve_update_target(
        self,
        target: str,
        ctx: ConversationContext,
        state: State,
        slot_configs: Optional[dict],
    ) -> str:
        """Three-way routing decision for a caller-requested slot update.

        "allow"   — the current agent's pipeline collects the slot and it has
                    a value to correct: take the existing detour path.
        "route"   — the ownership registry says route_to_owner and another
                    agent owns it: hand off NOW via _route_slot_update.
        "decline" — human-only / unknown slots, or nothing to update here.
                    Call sites may still park "decline" targets whose registry
                    entry is in_flow under another agent (Phase 3 behavior).
        """
        from agent.agents.verification.handlers import CALLER_LOCKED_SLOTS
        from agent.core.slot_ownership import get_ownership

        target = (target or "").strip()
        if not target or target in CALLER_LOCKED_SLOTS:
            return "decline"
        own = get_ownership(target)
        if own and own.updatable == "human_only":
            return "decline"
        collectable = slot_configs is None or target in slot_configs
        has_value = target in (ctx.confirmed_slots or []) or bool(str(state.get(target) or "").strip())
        if collectable and has_value:
            return "allow"
        if own and own.updatable == "route_to_owner" and own.agent and own.agent != self.AGENT_NAME:
            return "route"
        return "decline"

    def _route_slot_update(
        self,
        state: State,
        target: str,
        ctx: ConversationContext,
        *,
        return_awaiting: str,
        ack_prefix: str = "",
    ) -> dict:
        """Hand off to the slot's owning agent to honor an update NOW (Bug C).

        Asks the caller for the new value, routes the next turn to the owner
        (next_node), records the way back in pending_cross_agent_request
        (kind="update"), and clears every state key the registry says the
        update invalidates — so e.g. provider_search cannot early-exit on a
        stale zip_code_used and the list is rebuilt from the new ZIP. Never
        says "later".
        """
        from agent.core.slot_ownership import get_ownership, invalidated_state_updates
        from agent.llm.response_generator import _SLOT_LABELS

        own = get_ownership(target)
        label = target.replace("_", " ")
        ask_label = _SLOT_LABELS.get(target, label).split("—")[0].strip()
        msg = f"{ack_prefix}Sure — let me update your {label} first. Could you give me your {ask_label}?"

        self.logger.info(
            "%s: routing slot update to owner",
            self.AGENT_NAME,
            extra={"target": target, "owner": own.agent if own else "", "return_awaiting": return_awaiting},
        )
        interrupt = self.ask_member_with_context(state, msg, ctx)
        interrupt["next_node"] = own.agent if own else self.AGENT_NAME
        interrupt["awaiting_slot"] = target
        interrupt["pending_cross_agent_request"] = {
            "kind": "update",
            "target": target,
            "return_to_agent": self.AGENT_NAME,
            "return_awaiting": return_awaiting,
        }
        interrupt.update(invalidated_state_updates(target))
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        return interrupt

    def _route_foreign_update(
        self,
        state: State,
        target: str,
        *,
        return_awaiting: str,
    ) -> Optional[dict]:
        """Route a foreign slot update to its owner (Phase 7, claims-path parity).

        "route" targets (zip_code) hand off per the registry. Identity slots —
        in_flow under verification — are honored by routing to verification
        with the slot cleared for re-collection and the completed verification
        invalidated (a changed identity value always re-verifies; a changed
        name also re-triggers the readback). Returns None when the target is
        not routable from here: in-flow branches or a retry handle it.
        """
        from agent.agents.verification.constants import IDENTITY_SLOT_ORDER
        from agent.core.slot_ownership import get_ownership

        target = (target or "").strip()
        if not target:
            return None
        ctx = ConversationContext.from_state(state)
        if self.resolve_update_target(target, ctx, state, {}) == "route":
            return self._route_slot_update(state, target, ctx, return_awaiting=return_awaiting)
        own = get_ownership(target)
        if (
            target in IDENTITY_SLOT_ORDER
            and own is not None
            and own.updatable == "in_flow"
            and own.agent == "verification_agent"
            and own.agent != self.AGENT_NAME
        ):
            route = self._route_slot_update(state, target, ctx, return_awaiting=return_awaiting)
            route[target] = ""  # verification must re-collect, not skip
            route["member_status_verify"] = False
            if target in ("first_name", "last_name"):
                route["name_confirmed"] = False  # new name → readback re-runs
            return route
        return None

    def _reroute_detected_update(self, state: State, *, return_awaiting: str) -> Optional[dict]:
        """Never-verbatim-repeat guard for hand-rolled confirmation branches.

        Before a terminal retry re-asks the same question, check whether the
        turn is actually a foreign update request the registry can route.
        Returns None for genuinely unclassifiable turns — those may burn a
        slot_fail retry.
        """
        from agent.core.request_detection import detect_request

        detected = detect_request(_last_user_msg(list(state.get("messages") or [])))
        if detected is None or detected.kind != "update" or not detected.target:
            return None
        return self._route_foreign_update(state, detected.target, return_awaiting=return_awaiting)

    def route_capability_request(
        self,
        state: State,
        *,
        kind: str,
        target: str,
        return_awaiting: str,
    ) -> Optional[dict]:
        """Hand a live redo/replay request to its owning agent (Phase 6).

        Returns the hand-off state updates, or None when the request should
        NOT be routed:
          - kind is not redo/replay, or target maps to no known capability
            (callers park unknown topics as questions — never hard-decline);
          - the owner IS this agent (mid-flow interruption of the current
            agent stays in-flow — the agent's own branches handle it, no
            orchestrator hop, no pending_cross_agent_request).

        The hop runs the owner in the same super-step (is_interrupt=False via
        conditional_routing) so the owner processes this same caller turn and
        speaks the acknowledgement — mirroring follow_up's reroute mechanics.
        pending_cross_agent_request records the way back; the owner signals
        COMPLETE with it still set (or clears it when handing back directly).
        """
        from agent.core.slot_ownership import canonical_capability_topic, resolve_capability

        kind = (kind or "").strip().lower()
        if kind not in ("redo", "replay"):
            return None
        cap = resolve_capability(kind, target)
        if cap is None or cap.agent == self.AGENT_NAME:
            return None
        # Record the CANONICAL registry topic (redo/provider_list → delivery):
        # the owner's re-entry gates key off it (e.g. delivery's redo_active).
        topic = canonical_capability_topic(kind, target)
        self.logger.info(
            "%s: routing %s request to owner",
            self.AGENT_NAME,
            kind,
            extra={"target": topic, "owner": cap.agent, "return_awaiting": return_awaiting},
        )
        return {
            "next_node": cap.agent,
            "is_interrupt": False,
            "active_agent": self.AGENT_NAME,
            "awaiting_slot": "",  # the owner recomputes its own entry point
            "pending_cross_agent_request": {
                "kind": kind,
                "target": topic,
                "return_to_agent": self.AGENT_NAME,
                "return_awaiting": return_awaiting,
            },
            "slot_attempts": self.slots_dict(),
            "metadata_events": [],
            "app_run_id": state.get("app_run_id", ""),
            "wait_count": 0,  # non-WAIT turn resets the wait streak
        }

    def _ignored_request_guard(self, state: State, target: str) -> Optional[dict]:
        """Escalate honestly on the SECOND identical declined request.

        Counter f"ignored_request_{target}" via guard_loop_limit (max 2): the
        first decline explains and re-asks; a repeat means the caller is being
        ignored — never re-ask the same thing verbatim a third time.
        """
        from agent.responses.static import MSG_REPEATED_REQUEST_ESCALATE

        return self.guard_loop_limit(
            state,
            f"ignored_request_{target}",
            2,
            escalate_message=pick(MSG_REPEATED_REQUEST_ESCALATE),
            escalate_reason=f"repeated_ignored_request_{target}",
        )

    @staticmethod
    def _park_action_item(parked: list[dict], target: str, query: str = "") -> list[dict]:
        """Append a kind="action" item for ``target`` (deduped) to ``parked``."""
        if not any(p.get("kind") == "action" and p.get("target") == target for p in parked):
            parked.append(
                {
                    "query": query or f"update {target.replace('_', ' ')}",
                    "kind": "action",
                    "target": target,
                }
            )
        return parked

    def _match_promised_item(self, state: State, followup_query: str) -> str:
        """Promise text when ``followup_query`` asks about a parked/pending item.

        Meta-questions like "when will you update my zip?" must be answered
        concretely from the promise, never declined or treated as a slot
        answer. Fuzzy match: the pending update target's words, or ≥2 content
        words shared with a parked item's query.
        """
        from agent.state import normalize_cross_agent_request

        query_words = set(re.findall(r"[a-z]+", (followup_query or "").lower()))
        if not query_words:
            return ""
        pending = normalize_cross_agent_request(state)
        pending_label = (pending.get("target") or "").replace("_", " ")
        if pending_label and set(pending_label.split()) <= query_words:
            return f"your {pending_label} update is already in progress"
        for item in normalize_parked_followups(state.get("parked_followups")):
            item_words = set(re.findall(r"[a-z]+", item.get("query", "").lower())) | set(
                (item.get("target") or "").replace("_", " ").split()
            )
            meaningful = {w for w in (item_words & query_words) if len(w) > 3}
            if len(meaningful) >= 2 or (
                item.get("target") and set(item["target"].replace("_", " ").split()) <= query_words
            ):
                if item.get("kind") == "action":
                    return f"the {item['target'].replace('_', ' ')} update is queued and will be handled"
                return "that question is queued and will be answered"
        return ""

    def _confirmed_slot_values(
        self,
        ctx: ConversationContext,
        state: State,
        collected: Optional[dict] = None,
    ) -> dict:
        """Real confirmed slot values for the FOLLOWUP_ANSWER Confirmed: line.

        member_id / dob are never passed raw — masked as "on file".
        """
        from agent.llm.redaction import MASKED_SLOTS

        values: dict = {}
        for s in ctx.confirmed_slots:
            if s in MASKED_SLOTS:
                values[s] = "on file"
                continue
            v = (collected or {}).get(s) or state.get(s) or ""
            values[s] = str(v) if v else "confirmed"
        return values

    def _next_slot_ask(self, next_slot: str, slot_configs: Optional[dict], ctx: ConversationContext) -> str:
        """Static ask for the next pending slot, appended after the Gemini text.

        Typed slots use the transition template; untyped slots fall back to the
        config prompt (callable prompts get an empty collected dict — they only
        read already-collected values, all irrelevant to a next-slot ask).
        """
        cfg = (slot_configs or {}).get(next_slot)
        slot_type = getattr(cfg, "slot_type", None) if cfg is not None else None
        if slot_type:
            return build_transition_prompt(slot_type, ctx)
        prompt = getattr(cfg, "prompt", "") if cfg is not None else ""
        if callable(prompt):
            try:
                prompt = prompt({})
            except Exception:
                prompt = ""
        return prompt or f"Could you provide your {next_slot.replace('_', ' ')}?"

    async def _open_update_detour(  # noqa: C901
        self,
        state: State,
        target: str,
        return_to: str,
        ctx: ConversationContext,
        messages: list,
        *,
        flavor: str,
        collected: Optional[dict] = None,
        extracted_this_turn: str | None = None,
        followup_query: str = "",
    ) -> dict:
        """Open a detour to re-collect ``target`` before returning to ``return_to``.

        flavor:
          "answer"  — Case B: awaiting slot was just confirmed; FOLLOWUP_ANSWER
                      message acknowledges the answer AND asks for the new value.
          "bare"    — Case C2: bare update request; CORRECTION message asks for
                      the new value.
          "invalid" — Case C1: corrected value failed validation; CORRECTION
                      message says the new value didn't look right.

        Returns an interrupt dict (or an escalation dict when the per-target
        detour budget is exhausted).
        """
        from agent.llm.response_generator import generate_recovery_message

        # Loop guard: at most _MAX_UPDATE_DETOURS detours per target per call.
        if escalation := self.guard_loop_limit(
            state,
            f"update_{target}",
            _MAX_UPDATE_DETOURS,
            escalate_message=build_slot_exhausted_message(target),
            escalate_reason=f"update_{target} exhausted",
        ):
            return escalation

        target_label = target.replace("_", " ")
        if flavor == "answer":
            guard = "FOLLOWUP_ANSWER"
            override = (
                f"caller answered the current question AND asked to update their {target_label} "
                f"without giving a new value — acknowledge the captured answer, "
                f"then ask for the new {target_label}"
            )
        elif flavor == "invalid":
            guard = "CORRECTION"
            override = (
                f"caller gave a new {target_label} but it didn't look right — "
                f"acknowledge without repeating the value and ask for the correct {target_label}"
            )
        else:  # "bare"
            guard = "CORRECTION"
            override = (
                f"caller wants to update their {target_label} — "
                f"acknowledge and ask for the new {target_label}"
            )

        msg = await generate_recovery_message(
            slot_name=target,
            attempt=0,
            guard=guard,
            last_messages=messages[-6:],
            slot_label_override=override,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted_this_turn,
            followup_query=followup_query or None,
            # Case B (FOLLOWUP_ANSWER flavor): the sentence must end by asking
            # for the new value — followup_answer.md keys off this line.
            ask_for_new_value=(flavor == "answer"),
        )

        # Reset the target's slot record BEFORE ask_member: a stale confirmed
        # last_value would otherwise be re-persisted into later interrupts and
        # silently resurrect the value we're about to clear.
        self.get_slot(target).reset()
        ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s != target]
        if target == "first_name":
            ctx.caller_first_name = ""

        interrupt = self.ask_member_with_context(state, msg, ctx)
        interrupt["awaiting_slot"] = target
        interrupt["correction_return_to"] = return_to
        interrupt[target] = ""  # clear the state key so the pipeline re-collects
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        if collected is not None:
            collected[target] = ""

        # Re-verification consequence: updating an identity slot invalidates the
        # completed Salesforce verification — clear the flag so lookup_and_verify
        # re-runs once the new value is collected (the detour completion path).
        from agent.agents.verification.constants import IDENTITY_SLOT_ORDER

        if target in IDENTITY_SLOT_ORDER and state.get("member_status_verify"):
            interrupt["member_status_verify"] = False

        return interrupt

    async def _handle_answered_followup(  # noqa: C901
        self,
        state: State,
        config: _InternalSlotConfig,
        messages: list,
        normalized: str,
        ctx: ConversationContext,
        *,
        decision: Any,
        pending_slots: list[str] | None,
        slot_configs: Optional[dict],
        collected: Optional[dict],
    ) -> Tuple[Optional[str], Optional[dict]]:
        """ANSWERED_WITH_FOLLOWUP: confirm the slot, route the follow-up disposition.

        Handles update Cases A (answer + valid corrections) and B (answer +
        value-less update_target) inline; plain follow-ups route to the
        FOLLOWUP_ANSWER / FOLLOWUP_PARK / FOLLOWUP_DECLINE guard and get the
        next pending slot's static ask appended (Option A: Gemini never asks
        for a slot — Python appends the ask).
        """
        from agent.agents.verification.constants import IDENTITY_SLOT_ORDER
        from agent.agents.verification.handlers import CALLER_LOCKED_SLOTS

        slot_name = config.slot_name
        extracted = getattr(decision, "extracted", None) or {}
        corrections = {k: v for k, v in (getattr(decision, "corrections", None) or {}).items() if v}
        update_target = (getattr(decision, "update_target", None) or "").strip()
        followup_query = (getattr(decision, "followup_query", None) or "").strip()
        disposition = getattr(decision, "followup_disposition", None)
        disposition_value = str(getattr(disposition, "value", disposition) or "none")

        # Regex fallback (request_detection): the LLM sometimes answers the
        # slot but drops a plainly-phrased update request from update_target.
        # Backfill BEFORE disposition routing so the detour/route invariant
        # below wins over a park/decline the LLM chose for the same words.
        if not update_target:
            from agent.core.request_detection import detect_request

            detected = detect_request(followup_query) or detect_request(_last_user_msg(messages))
            # Meta-questions about an already-parked/pending item stay on the
            # promise-answer path below — never re-open a route for them.
            # A target equal to the slot just answered is likewise skipped: a
            # stale followup_query echo ("change my last name") from the turn
            # that OPENED the detour must not re-request the update the
            # current answer just fulfilled.
            if (
                detected
                and detected.kind == "update"
                and detected.target
                and detected.target != slot_name
                and not self._match_promised_item(state, followup_query or _last_user_msg(messages))
            ):
                update_target = detected.target
                self.logger.info(
                    "_handle_answered_followup: regex fallback set update_target",
                    extra={
                        "source": "regex_fallback",
                        "matched": detected.matched,
                        "target": detected.target,
                    },
                )

        # An update_target equal to the slot just answered is fulfilled by
        # that answer (detour completion) — consume it so it can neither
        # re-open a detour for the same slot (Case B would trip the
        # _MAX_UPDATE_DETOURS loop guard and escalate a successful capture)
        # nor park as a kind="action" item. The fully-consumed shape (no
        # genuine side question) never reaches this handler; a followup_query
        # still present here is a real side question and routes through the
        # disposition messaging as usual.
        if update_target == slot_name:
            self.logger.info(
                "_handle_answered_followup: update_target fulfilled by this answer — consumed",
                extra={"target": update_target},
            )
            update_target = ""

        # ── Case A: apply validated corrections + cascade clears BEFORE the
        # awaiting slot is confirmed, so the confirm order matches apply_corrections.
        from agent.core.slot_ownership import get_ownership

        applied: list[str] = []
        cascade_cleared: list[str] = []
        invalid_target = ""
        foreign_parked: list[str] = []
        for target, raw in corrections.items():
            if target in CALLER_LOCKED_SLOTS or target == slot_name:
                continue
            norm_fn, val_fn = self._resolve_slot_tools(target, slot_configs)
            if not (norm_fn and val_fn):
                # Foreign slot — this pipeline cannot apply it. Park it as a
                # kind="action" item when the registry names an owner (Phase 4:
                # never silently drop a caller's correction); truly ownerless /
                # human-only slots still drop with no ghost acknowledgement.
                own = get_ownership(target)
                if own and own.updatable != "human_only":
                    foreign_parked.append(target)
                continue
            corrected_val = norm_fn(str(raw))
            if corrected_val and val_fn(corrected_val).valid:
                self.slot_ok(target, corrected_val)
                if collected is not None:
                    collected[target] = corrected_val
                applied.append(target)
            elif not invalid_target:
                # New value failed validation — never applied. After the awaiting
                # slot confirms, a detour re-collects this slot (Phase 7).
                invalid_target = target

        # Existing cascade clears (mirrors handlers.apply_corrections): a new
        # first_name invalidates last_name, a new member_id invalidates dob —
        # unless the dependent value arrived in the same utterance.
        if "first_name" in applied and slot_name != "last_name" and not extracted.get("last_name"):
            cascade_cleared.append("last_name")
        if "member_id" in applied and slot_name != "dob" and not extracted.get("dob"):
            cascade_cleared.append("dob")
        for cleared in cascade_cleared:
            self.get_slot(cleared).reset()
            ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s != cleared]
            if collected is not None:
                collected[cleared] = ""

        clear_verify = bool(state.get("member_status_verify")) and any(
            t in IDENTITY_SLOT_ORDER for t in applied
        )

        # ── Confirm the awaiting slot ────────────────────────────────────────
        self.slot_ok(slot_name, normalized)
        ctx.record_slot_success(slot_name)
        if slot_name == "first_name":
            ctx.update_caller_name(normalized)
        self._pending_ambiguous_resets.add(slot_name)
        if collected is not None:
            collected[slot_name] = normalized

        # Option-A next ask target: first pending slot after this one. pending
        # was computed before this slot confirmed, so it still contains slot_name;
        # cascade-cleared slots are re-collected on later turns by the pipeline.
        pending = list(pending_slots or [])
        if slot_name in pending:
            remaining = pending[pending.index(slot_name) + 1 :]
        else:
            remaining = [s for s in pending if s != slot_name]
        next_slot = next((s for s in remaining if s not in applied), "")

        # ── Case B: answer + value-less update request ───────────────────────
        # allow → detour (the detour ask REPLACES the normal next-slot ask);
        # route → hand off to the owning agent NOW (pending_cross_agent_request);
        # otherwise park in_flow-elsewhere targets, decline human-only ones.
        # INVARIANT: when update_target is set and resolution is "allow" or
        # "route", the LLM's followup_disposition is IGNORED — the detour /
        # route path below returns unconditionally, so a park/decline the LLM
        # chose for the same turn can never shadow an honorable update.
        # (update_target == slot_name was consumed above, so a detour here is
        # always for a DIFFERENT slot than the one just answered.)
        if update_target and update_target not in applied:
            resolution = self.resolve_update_target(update_target, ctx, state, slot_configs)
            if resolution == "allow":
                detour = await self._open_update_detour(
                    state,
                    update_target,
                    next_slot,
                    ctx,
                    messages,
                    flavor="answer",
                    collected=collected,
                    extracted_this_turn=normalized,
                    followup_query=followup_query,
                )
                for cleared in cascade_cleared:
                    detour.setdefault(cleared, "")
                if clear_verify:
                    detour.setdefault("member_status_verify", False)
                return normalized, detour
            if resolution == "route":
                route = self._route_slot_update(
                    state,
                    update_target,
                    ctx,
                    return_awaiting=next_slot or state.get("awaiting_slot") or "",
                    ack_prefix="Got that. ",
                )
                for cleared in cascade_cleared:
                    route.setdefault(cleared, "")
                if clear_verify:
                    route.setdefault("member_status_verify", False)
                return normalized, route
            # "decline": park when the registry says another flow owns it
            # in_flow — never blanket-decline the caller's own request.
            own = get_ownership(update_target)
            if own and own.updatable == "in_flow" and own.agent != self.AGENT_NAME:
                disposition_value = "park"
                followup_query = followup_query or f"update {update_target.replace('_', ' ')}"
            else:
                # Decline is only legitimate for human_only or unknown
                # ownership — any other registry entry reaching this branch
                # means resolve_update_target and the registry disagree.
                if own is not None and own.updatable != "human_only":
                    self.logger.warning(
                        "_handle_answered_followup: declining update for a slot the "
                        "registry says is updatable — resolution/registry mismatch",
                        extra={
                            "target": update_target,
                            "updatable": own.updatable,
                            "owner": own.agent,
                            "agent": self.AGENT_NAME,
                        },
                    )
                # Human-only decline: escalate honestly on the second
                # identical ignored request instead of re-asking verbatim.
                if escalation := self._ignored_request_guard(state, update_target):
                    return normalized, escalation
                disposition_value = "decline"

        # ── Answer + INVALID corrected value: the awaiting slot IS confirmed,
        # but the correction was never applied — open a detour to re-collect
        # the corrected slot instead of silently dropping the bad value.
        # (Case B above wins when both an update_target and an invalid
        # correction arrive in the same turn.)
        if invalid_target and self.resolve_update_target(invalid_target, ctx, state, slot_configs) == "allow":
            detour = await self._open_update_detour(
                state,
                invalid_target,
                next_slot,
                ctx,
                messages,
                flavor="invalid",
                collected=collected,
                extracted_this_turn=normalized,
                followup_query=followup_query,
            )
            for cleared in cascade_cleared:
                detour.setdefault(cleared, "")
            if clear_verify:
                detour.setdefault("member_status_verify", False)
            return normalized, detour

        # ── Disposition routing (answer_now / park / decline / none) ────────
        # Pure correction (corrections applied, no side question, no
        # disposition): a decline would wrongly tell the caller "I can't help
        # with that" about their own correction — acknowledge it instead.
        if applied and not followup_query and disposition_value in ("none", ""):
            guard = "CORRECTION_ACK"
        else:
            guard = _DISPOSITION_GUARDS.get(disposition_value, "FOLLOWUP_RESPOND")

        # Meta-question about a parked/promised item ("when will you update my
        # zip?"): answer concretely from the promise — never decline or re-park
        # something already promised.
        promise = self._match_promised_item(state, followup_query) if followup_query else ""
        if promise and guard == "FOLLOWUP_PARK":
            guard = "FOLLOWUP_RESPOND"

        # Always pass real confirmed values — the generation LLM uses them to
        # decide whether it can answer; FOLLOWUP_ANSWER (detour) uses same path.
        confirmed_values = (
            self._confirmed_slot_values(ctx, state, collected)
            if guard in ("FOLLOWUP_RESPOND", "FOLLOWUP_ANSWER")
            else None
        )
        if promise and confirmed_values is not None:
            step = f"right after we confirm the {next_slot.replace('_', ' ')}" if next_slot else "next"
            confirmed_values["promised next step"] = f"{promise} {step}"
        session_context = _mk_session_ctx(
            extracted_val=normalized,
            followup_query=followup_query,
            confirmed_values=confirmed_values,
        )
        msg = await self._generate_slot_retry_response(
            state,
            slot_name,
            ctx,
            messages,
            guard=guard,
            session_context=session_context,
            extracted_this_turn=normalized,
            # The static ask below must be the ONLY ask in the combined turn.
            next_slot_label=next_slot or None,
            will_append_ask=bool(next_slot),
            fallback_slot_label=applied[0].replace("_", " ") if guard == "CORRECTION_ACK" else None,
        )

        if next_slot:
            msg = msg.rstrip() + " " + self._next_slot_ask(next_slot, slot_configs, ctx)

        interrupt = self.ask_member_with_context(state, msg, ctx)
        interrupt["awaiting_slot"] = next_slot  # "" when the pipeline finished
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        for cleared in cascade_cleared:
            interrupt[cleared] = ""
        if clear_verify:
            interrupt["member_status_verify"] = False
        if guard == "FOLLOWUP_PARK" and followup_query:
            # Structured parked item: an unhonored update_target parks as an
            # actionable item follow_up routes via the ownership registry;
            # plain side questions park as kind="question".
            parked = normalize_parked_followups(state.get("parked_followups"))
            is_action = bool(update_target and update_target not in applied)
            parked.append(
                {
                    "query": followup_query,
                    "kind": "action" if is_action else "question",
                    "target": update_target if is_action else "",
                }
            )
            interrupt["parked_followups"] = parked
        # Foreign corrections this pipeline could not apply (Case A) park as
        # actions so the owning flow honors them — never silently dropped.
        if foreign_parked:
            parked = normalize_parked_followups(
                interrupt.get("parked_followups") or state.get("parked_followups")
            )
            for foreign in foreign_parked:
                self._park_action_item(parked, foreign)
            interrupt["parked_followups"] = parked
        # Return normalized so slot_ok's persistence carries the value forward;
        # with no next_slot this matches the pre-Phase-4 return exactly, so the
        # agent's post-pipeline logic proceeds next turn.
        return normalized, interrupt

    # -------------------------------------------------------------------------
    # Per-turn slot collector — with contextual response generation
    # -------------------------------------------------------------------------

    async def _collect_slot(  # noqa: C901
        self,
        state: State,
        config: _InternalSlotConfig,
        messages: list,
        pre_extracted: str = "",
        *,
        context: Optional[ConversationContext] = None,
        is_transition: bool = False,
        decision: Optional[Any] = None,
        pending_slots: Optional[list] = None,
        slot_configs: Optional[dict] = None,
        collected: Optional[dict] = None,
    ) -> Tuple[Optional[str], Optional[dict]]:
        slot_name = config.slot_name
        normalizer = config.normalizer
        validator = config.validator
        slot_type = config.slot_type

        ctx = context or ConversationContext()
        ctx.increment_turn(self.AGENT_NAME)

        # ------------------------------------------------------------------
        # Build a human-readable slot label for use in the cannot-provide
        # escalation message (e.g. "member ID", "date of birth").
        # ------------------------------------------------------------------
        if slot_type:
            slot_label = slot_type.value.replace("_", " ")
        else:
            slot_label = slot_name.replace("_", " ")

        # ------------------------------------------------------------------
        # Response builder: use contextual builder when slot_type provided,
        # fall back to static strings for backward compatibility.
        # ------------------------------------------------------------------
        def _ask_first() -> str:
            if slot_type:
                return (
                    build_transition_prompt(slot_type, ctx)
                    if is_transition
                    else build_initial_prompt(slot_type)
                )
            return config.prompt

        # ------------------------------------------------------------------
        # 1. Already valid in state — nothing to do
        # ------------------------------------------------------------------
        existing = (state.get(slot_name) or "").strip()
        if existing:
            r = validator(existing)
            if r.valid if hasattr(r, "valid") else bool(r):
                return existing, None

        slot = self.get_slot(slot_name)

        # ------------------------------------------------------------------
        # 2. LLM pre-extracted a candidate value
        # ------------------------------------------------------------------
        if pre_extracted:
            normalized = normalizer(pre_extracted)
            if normalized:
                r = validator(normalized)
                valid = r.valid if hasattr(r, "valid") else bool(r)
                if valid:
                    # Check whether caller also said something that needs addressing.
                    # Import here to avoid circular imports at module level.
                    from agent.llm.schema import EventType

                    event_type = getattr(decision, "event_type", None)

                    # A valid answer accompanied by a value-less update request
                    # must reach the followup handler even when the LLM
                    # flattened the event to ANSWERED/CORRECTED — otherwise the
                    # caller's request is silently dropped on the clean-confirm
                    # path. Only bare "update" shapes route here: corrections
                    # with values (Case A/C1) and redo/replay keep their paths.
                    update_hint = (getattr(decision, "update_target", None) or "").strip()
                    kind_raw = getattr(decision, "request_kind", None)
                    kind_hint = str(getattr(kind_raw, "value", kind_raw) or "").strip().lower()
                    corrections_hint = {
                        k: v for k, v in (getattr(decision, "corrections", None) or {}).items() if v
                    }
                    answered_with_request = bool(
                        update_hint
                        and update_hint != slot_name
                        and not corrections_hint
                        and kind_hint in ("", "none", "update")
                    )

                    # Detour-completion turn: the LLM re-emits the update
                    # request it can still see in recent context ("change my
                    # last name") alongside the answer that fulfills it. When
                    # that request is the event's ONLY content — its target is
                    # the slot just answered, there are no other corrections,
                    # and the followup_query (if any) is that same request —
                    # the followup is already satisfied: take the clean
                    # confirm path below. Entering the handler would generate
                    # decline/park messaging about a change that was just
                    # made ("a representative will need to make that change").
                    self_fulfilled_update = False
                    if event_type == EventType.ANSWERED_WITH_FOLLOWUP and not corrections_hint:
                        from agent.core.request_detection import detect_request

                        followup_hint = (getattr(decision, "followup_query", None) or "").strip()
                        fq_detected = detect_request(followup_hint) if followup_hint else None
                        fq_is_self_update = bool(
                            fq_detected and fq_detected.kind == "update" and fq_detected.target == slot_name
                        )
                        self_fulfilled_update = (
                            (update_hint == slot_name or (not update_hint and fq_is_self_update))
                            and (not followup_hint or fq_is_self_update)
                            and kind_hint in ("", "none", "update")
                        )
                        if self_fulfilled_update:
                            self.logger.info(
                                "_collect_slot: followup update request fulfilled by this "
                                "answer — downgrading to clean ANSWERED",
                                extra={"slot": slot_name},
                            )

                    if (
                        event_type == EventType.ANSWERED_WITH_FOLLOWUP or answered_with_request
                    ) and not self_fulfilled_update:
                        # Slot will be confirmed inside the handler — corrections
                        # (Case A) must apply BEFORE slot_ok of the awaiting slot.
                        # Attempt counter is never incremented on this path.
                        return await self._handle_answered_followup(
                            state,
                            config,
                            messages,
                            normalized,
                            ctx,
                            decision=decision,
                            pending_slots=pending_slots,
                            slot_configs=slot_configs,
                            collected=collected,
                        )

                    # Clean answered path — confirm and move on immediately
                    self.slot_ok(slot_name, normalized)
                    ctx.record_slot_success(slot_name)
                    if slot_name == "first_name":
                        ctx.update_caller_name(normalized)
                    self._pending_ambiguous_resets.add(slot_name)
                    return normalized, None

            # ── CHANGE 1: cannot-provide check on rejected extraction ─────
            # The LLM extracted something but it failed normalisation /
            # validation (e.g. member_id without M-prefix, partial number).
            # Before burning a retry attempt, check whether the raw text is
            # actually "I don't have it" in disguise.
            last_user = _last_user_msg(messages)
            if detect_cannot_provide(last_user):
                self.logger.info(
                    "_collect_slot: cannot-provide detected (rejected-extraction path) — "
                    "escalating immediately",
                    extra={"slot": slot_name},
                )
                return None, self.signal_escalate(
                    state,
                    _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                    f"{slot_name}_cannot_provide",
                    initiator="Agent",
                )
            # ── END CHANGE 1 ──────────────────────────────────────────────

            # Extracted but normaliser/validator rejected it — count as failure
            self.slot_fail(slot_name, pre_extracted)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state,
                    build_slot_exhausted_message(slot_name),
                    f"{slot_name} exhausted",
                    initiator="Agent",
                )
            msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
            interrupt = self.ask_member_with_context(state, msg, ctx)
            interrupt["awaiting_slot"] = slot_name
            interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
            return None, interrupt

        # ------------------------------------------------------------------
        # 2b. No extraction — classify the turn before counting a failure
        # ------------------------------------------------------------------
        if state.get("awaiting_slot") == slot_name:
            from agent.llm.schema import EventType

            event_type = getattr(decision, "event_type", None)
            event_value = event_type.value if event_type is not None else EventType.ANSWERED.value

            # ── WAIT: caller asked for time — not a failure, not ambiguous ─
            # Detection is deliberately post-extraction only: a pre-GPT regex
            # short-circuit in agent.run() would swallow utterances like
            # "hold on... it's M451982" where a valid value follows the wait
            # phrase — extraction must see the turn first so the value wins
            # (a turn with a valid value never reaches this block).
            # The regex fallback also rescues waits the LLM mislabels as
            # ambiguous. cannot-provide outranks wait: "I don't have it"
            # falls through to the escalation checks below.
            last_user = _last_user_msg(messages)
            if (
                event_value == EventType.WAIT.value or detect_wait_request(last_user)
            ) and not detect_cannot_provide(last_user):
                wait_count = int(state.get("wait_count") or 0) + 1
                if wait_count < MAX_WAIT_TURNS:
                    msg = pick(MSG_WAIT_ACK)
                else:
                    msg = pick(MSG_WAIT_NUDGE).format(slot_label=slot_label)
                # Static response only: no slot_fail, no ambiguous_counts,
                # no generation-LLM call — waiting is not a failed attempt.
                interrupt = self.ask_member_with_context(state, msg, ctx)
                interrupt["awaiting_slot"] = slot_name
                interrupt["wait_count"] = wait_count
                return None, interrupt

            # ── CORRECTED: caller fixed a confirmed slot, not answering us ─
            if event_value == EventType.CORRECTED.value:
                corrections = {k: v for k, v in (getattr(decision, "corrections", None) or {}).items() if v}
                update_target = (getattr(decision, "update_target", None) or "").strip()

                # Regex fallback (request_detection): a CORRECTED turn the LLM
                # left targetless still carries the target in plain words
                # ("I need to change my email") — recover it and take the C2
                # path instead of downgrading the caller's request to ANSWERED.
                if not corrections and not update_target:
                    from agent.core.request_detection import detect_request

                    detected = detect_request(last_user)
                    if detected and detected.kind == "update" and detected.target:
                        update_target = detected.target
                        self.logger.info(
                            "_collect_slot: regex fallback set update_target on bare CORRECTED turn",
                            extra={
                                "source": "regex_fallback",
                                "matched": detected.matched,
                                "target": detected.target,
                            },
                        )

                if not corrections and not update_target:
                    # Downgrade only when BOTH corrections and update_target are
                    # empty — a bare update request (C2) is handled below.
                    self.logger.warning(
                        "_collect_slot: CORRECTED event with empty corrections{} and no "
                        "update_target — treating as ANSWERED. Slot: %s. "
                        "This is a prompt-following failure.",
                        slot_name,
                    )
                    event_value = EventType.ANSWERED.value
                elif not corrections and update_target:
                    # ── C2: bare update request ("I need to change my email") ─
                    # allow → detour: target becomes awaiting, current awaiting
                    # slot is preserved in correction_return_to to resume.
                    resolution = self.resolve_update_target(update_target, ctx, state, slot_configs)
                    if resolution == "allow":
                        detour = await self._open_update_detour(
                            state,
                            update_target,
                            slot_name,
                            ctx,
                            messages,
                            flavor="bare",
                            collected=collected,
                        )
                        return None, detour
                    if resolution == "route":
                        # Hand off to the owning agent NOW — never say "later".
                        return None, self._route_slot_update(
                            state, update_target, ctx, return_awaiting=slot_name
                        )
                    # Park when another flow owns the slot in_flow; only
                    # human-only targets decline — and a second identical
                    # ignored request escalates instead of re-asking verbatim.
                    from agent.core.slot_ownership import get_ownership

                    own = get_ownership(update_target)
                    if own and own.updatable == "in_flow" and own.agent != self.AGENT_NAME:
                        msg = await self._generate_slot_retry_response(
                            state,
                            slot_name,
                            ctx,
                            messages,
                            guard="FOLLOWUP_PARK",
                            session_context=_mk_session_ctx(
                                followup_query=f"update {update_target.replace('_', ' ')}",
                            ),
                        )
                        interrupt = self.ask_member_with_context(state, msg, ctx)
                        interrupt["awaiting_slot"] = slot_name
                        interrupt["wait_count"] = 0
                        interrupt["parked_followups"] = self._park_action_item(
                            normalize_parked_followups(state.get("parked_followups")), update_target
                        )
                        return None, interrupt
                    if escalation := self._ignored_request_guard(state, update_target):
                        return None, escalation
                    msg = await self._generate_slot_retry_response(
                        state,
                        slot_name,
                        ctx,
                        messages,
                        guard="FOLLOWUP_RESPOND",
                        session_context=_mk_session_ctx(
                            followup_query=f"update {update_target.replace('_', ' ')}",
                        ),
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    return None, interrupt
                else:
                    # ── C1: corrections with values — validate before acking ──
                    # An invalid corrected value must NOT be applied; open a
                    # detour to re-collect that slot instead of acknowledging it.
                    # Foreign corrections (no tools in this pipeline) must never
                    # be acknowledged as applied ("ghost ack") — route or park
                    # them via the ownership registry instead.
                    from agent.core.slot_ownership import get_ownership

                    invalid_target = ""
                    appliable_fields: list[str] = []
                    foreign_fields: list[str] = []
                    for target, raw in corrections.items():
                        norm_fn, val_fn = self._resolve_slot_tools(target, slot_configs)
                        if not (norm_fn and val_fn):
                            foreign_fields.append(target)
                            continue
                        appliable_fields.append(target)
                        corrected_val = norm_fn(str(raw))
                        if not (corrected_val and val_fn(corrected_val).valid):
                            invalid_target = target
                            break
                    if (
                        invalid_target
                        and self.resolve_update_target(invalid_target, ctx, state, slot_configs) == "allow"
                    ):
                        detour = await self._open_update_detour(
                            state,
                            invalid_target,
                            slot_name,
                            ctx,
                            messages,
                            flavor="invalid",
                            collected=collected,
                        )
                        return None, detour

                    # Foreign-only corrections: honor via route (route_to_owner)
                    # or park (in_flow elsewhere); human-only declines honestly.
                    if foreign_fields and not appliable_fields:
                        target = foreign_fields[0]
                        resolution = self.resolve_update_target(target, ctx, state, slot_configs)
                        if resolution == "route":
                            return None, self._route_slot_update(
                                state, target, ctx, return_awaiting=slot_name
                            )
                        own = get_ownership(target)
                        if own and own.updatable != "human_only":
                            msg = await self._generate_slot_retry_response(
                                state,
                                slot_name,
                                ctx,
                                messages,
                                guard="FOLLOWUP_PARK",
                                session_context=_mk_session_ctx(
                                    followup_query=f"update {target.replace('_', ' ')}",
                                ),
                            )
                            interrupt = self.ask_member_with_context(state, msg, ctx)
                            interrupt["awaiting_slot"] = slot_name
                            interrupt["wait_count"] = 0
                            interrupt["parked_followups"] = self._park_action_item(
                                normalize_parked_followups(state.get("parked_followups")), target
                            )
                            return None, interrupt
                        if escalation := self._ignored_request_guard(state, target):
                            return None, escalation
                        msg = await self._generate_slot_retry_response(
                            state,
                            slot_name,
                            ctx,
                            messages,
                            guard="FOLLOWUP_RESPOND",
                            session_context=_mk_session_ctx(
                                followup_query=f"update {target.replace('_', ' ')}",
                            ),
                        )
                        interrupt = self.ask_member_with_context(state, msg, ctx)
                        interrupt["awaiting_slot"] = slot_name
                        interrupt["wait_count"] = 0
                        return None, interrupt

                    # Existing CORRECTED path: valid corrections were already
                    # applied by the agent (apply_corrections) — acknowledge
                    # ONLY the fields this pipeline applied and re-ask the
                    # awaiting slot; foreign extras park as actions.
                    msg = await self._generate_correction_ack(
                        state, appliable_fields, slot_name, ctx, messages, decision=decision
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                    ambiguous_counts[slot_name] = 0
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    if foreign_fields:
                        parked = normalize_parked_followups(state.get("parked_followups"))
                        for foreign in foreign_fields:
                            own = get_ownership(foreign)
                            if own and own.updatable != "human_only":
                                self._park_action_item(parked, foreign)
                        interrupt["parked_followups"] = parked
                    return None, interrupt

            # ── AMBIGUOUS: caller signalled correction intent with no value ─
            if event_value == EventType.AMBIGUOUS.value:
                ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                ambiguous_counts[slot_name] = ambiguous_counts.get(slot_name, 0) + 1

                # ── cannot-provide check in AMBIGUOUS branch ──────────────
                # Utterances like "sorry I dont have it my member ID",
                # "I lost the letter", "I never received a card" are frequently
                last_user = _last_user_msg(messages)
                if detect_cannot_provide(last_user):
                    self.logger.info(
                        "_collect_slot: cannot-provide detected (ambiguous path) — escalating immediately",
                        extra={"slot": slot_name},
                    )
                    return None, self.signal_escalate(
                        state,
                        _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                        f"{slot_name}_cannot_provide",
                        initiator="Agent",
                    )
                # ── END cannot-provide check ──────────────────────────────

                # Two consecutive AMBIGUOUS turns → treat as a genuine non-answer.
                # (The count was just incremented, so the first ambiguous turn
                # arrives here as 1 and must fall through to the CLARIFY block —
                # no attempt cost; the second, as 2, burns an attempt.)
                if ambiguous_counts[slot_name] >= 2:
                    self.slot_fail(slot_name, None, is_asr=True)
                    if slot.is_exhausted():
                        return None, self.signal_escalate(
                            state,
                            build_slot_exhausted_message(slot_name),
                            f"{slot_name} exhausted",
                            initiator="Agent",
                        )
                    msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    return None, interrupt

                # First AMBIGUOUS turn — ask for clarification without counting a failure
                corrections = getattr(decision, "corrections", None) or {}
                if corrections:
                    corrected_fields = list(corrections.keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                else:
                    # If the caller asked a clarifying question (followup_query set
                    # on an ambiguous turn), forward it so the generation LLM can
                    # answer the question before re-asking — not just re-ask blindly.
                    clarify_fq = (getattr(decision, "followup_query", None) or "").strip()
                    # Route through _generate_slot_retry_response so sanitize_generated
                    # strips any confirmed-slot re-asks the generation LLM produces.
                    msg = await self._generate_slot_retry_response(
                        state,
                        slot_name,
                        ctx,
                        messages,
                        guard="CLARIFY",
                        session_context=_mk_session_ctx(followup_query=clarify_fq) if clarify_fq else None,
                    )
                interrupt = self.ask_member_with_context(state, msg, ctx)
                interrupt["awaiting_slot"] = slot_name
                interrupt["ambiguous_counts"] = ambiguous_counts
                interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                return None, interrupt

            # ── ANSWERED (default): genuine non-answer — check cannot-provide
            #    BEFORE counting a failure and retrying ─────────────────────

            # ── CHANGE 2: cannot-provide check on genuine non-answer ──────
            last_user = _last_user_msg(messages)
            if detect_cannot_provide(last_user):
                self.logger.info(
                    "_collect_slot: cannot-provide detected (no-extraction path) — escalating immediately",
                    extra={"slot": slot_name},
                )
                return None, self.signal_escalate(
                    state,
                    _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                    f"{slot_name}_cannot_provide",
                    initiator="Agent",
                )
            # ── END CHANGE 2 ──────────────────────────────────────────────

            ambiguous_counts = dict(state.get("ambiguous_counts") or {})
            ambiguous_counts[slot_name] = 0
            self.slot_fail(slot_name, None, is_asr=True)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state,
                    build_slot_exhausted_message(slot_name),
                    f"{slot_name} exhausted",
                    initiator="Agent",
                )
            msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
            interrupt = self.ask_member_with_context(state, msg, ctx)
            interrupt["awaiting_slot"] = slot_name
            interrupt["ambiguous_counts"] = ambiguous_counts
            interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
            return None, interrupt

        # ------------------------------------------------------------------
        # 3. First ask
        # ------------------------------------------------------------------
        interrupt = self.ask_member_with_context(state, _ask_first(), ctx)
        interrupt["awaiting_slot"] = slot_name
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        return None, interrupt
