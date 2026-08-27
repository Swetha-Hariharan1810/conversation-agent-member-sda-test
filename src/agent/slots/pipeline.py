"""
slot_pipeline.py — Reusable ordered slot collection engine.

SlotPipeline.collect() drives the slot collection loop across turns:
  - Loads ConversationContext from state for name personalization.
  - Tracks prev_slot_just_confirmed to enable transition prompts
    ("Thank you, Emily. And your date of birth?").
  - Passes context and is_transition into every _collect_slot call.
  - Persists updated context into every interrupt dict.

Two SlotConfig classes exist in this codebase:
  - SlotConfig (this file)       — public declarative config used by agents
  - _InternalSlotConfig (core/slot_manager.py) — low-level config for _collect_slot
SlotPipeline.collect() translates SlotConfig → _InternalSlotConfig on each turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from agent.llm.schema import WorkerResult

from agent.conversation.context import (
    ConversationContext,
)
from agent.core.slot_manager import _InternalSlotConfig
from agent.logger import get_logger
from agent.slots.types import SlotType, SlotValues
from agent.state import State

logger = get_logger(__name__)

_ESCALATE_MSG = (
    "I'm sorry, I wasn't able to verify your details after several attempts. "
    "Let me connect you with a representative. Please hold."
)


# ---------------------------------------------------------------------------
# Slot configuration
# ---------------------------------------------------------------------------


@dataclass
class SlotConfig:
    """
    Declarative configuration for a single slot.

    slot_type: when provided, response_builder generates context-aware prompts
    automatically; prompt is unused for typed slots.

    prompt can be a plain string or a callable that receives the
    already-collected slots dict — used for untyped slots (slot_type=None).
    """

    name: str
    normalizer: Callable
    validator: Callable
    prompt: str | Callable[[dict], str]
    slot_type: Optional[SlotType] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SlotPipeline:
    """
    Collects an ordered list of slots, one conversational turn at a time.

    Design contract:
      - Returns None when every slot is collected and valid.
      - Returns a dict (interrupt or escalation) the moment the conversation
        needs to pause, so the caller can return it immediately to LangGraph.
      - Mutates `collected` in-place so the caller always has the latest values.
      - Delegates all state I/O to the owning agent.
    """

    def __init__(self, agent: Any, slot_configs: list[SlotConfig]) -> None:
        self._agent = agent
        self.configs = {cfg.name: cfg for cfg in slot_configs}
        self.order = [cfg.name for cfg in slot_configs]

    async def collect(
        self,
        state: State,
        messages: list,
        collected: SlotValues,
        *,
        decision: "WorkerResult | None" = None,
        escalate_message: str = _ESCALATE_MSG,
    ) -> Optional[dict]:
        """
        Drive the slot collection loop.

        Args:
            state:            LangGraph state dict.
            messages:         Conversation history (full, not just last message).
            collected:        Mutable dict of already-collected values.
            decision:         Optional LLM-extracted decision object.
            escalate_message: Message when retries are exhausted.

        Returns:
            None  — all slots collected, caller can proceed.
            dict  — interrupt or escalation, caller must return this.
        """
        # Load accumulated conversation context from state
        ctx = ConversationContext.from_state(state)

        # Tell context how many slots we're collecting so "almost there" cues work
        # Only set if not already set (don't override on re-entry mid-pipeline)
        if not ctx.total_slots_in_pipeline:
            ctx.total_slots_in_pipeline = len(self.order)

        # Track whether the previous slot was just confirmed THIS turn
        # so we can use transition templates ("Thank you. And your X?")
        prev_slot_just_confirmed = False

        # Slots still to be collected, computed ONCE per call (Phase 4).
        # _collect_slot's ANSWERED_WITH_FOLLOWUP branch hands this to the
        # generation LLM and uses it to pick the next slot for the appended
        # static ask / detour return pointer.
        pending = [
            s for s in self.order if not (collected.get(s) and self.configs[s].validator(collected[s]).valid)
        ]

        for slot_name in self.order:
            config = self.configs[slot_name]

            # If a correction detour is active for a *different* slot, skip this slot
            # and let the pipeline run to completion on the correction target slot first.
            correction_return_to = state.get("correction_return_to") or ""
            if correction_return_to and slot_name != correction_return_to:
                # Skip slots that aren't the correction target or the original awaiting slot
                if collected.get(slot_name) and config.validator(collected[slot_name]).valid:
                    prev_slot_just_confirmed = True
                    continue
                # If this is a slot that is pending but not the correction target,
                # continue processing normally — the loop will reach correction_return_to naturally.

            # Already collected and valid — skip, note it was confirmed
            if collected.get(slot_name) and config.validator(collected[slot_name]).valid:
                if slot_name not in ctx.confirmed_slots:
                    ctx.confirmed_slots.append(slot_name)
                    if slot_name == "first_name":
                        ctx.update_caller_name(collected[slot_name])
                prev_slot_just_confirmed = True
                continue

            pre_extracted = decision.extracted.get(slot_name, "") if decision and decision.extracted else ""

            # Build static fallback prompt (used when slot_type not set)
            prompt = config.prompt(collected) if callable(config.prompt) else config.prompt

            sm_config = _InternalSlotConfig(
                slot_name=slot_name,
                prompt=prompt,
                normalizer=config.normalizer,
                validator=config.validator,
                slot_type=config.slot_type,
            )

            value, interrupt = await self._agent._collect_slot(
                state,
                sm_config,
                messages=messages,
                pre_extracted=pre_extracted,
                context=ctx,
                is_transition=prev_slot_just_confirmed,
                decision=decision,
                pending_slots=pending,
                slot_configs=self.configs,
                collected=collected,
            )

            if interrupt:
                # Carry already-collected values forward
                interrupt.update({k: v for k, v in collected.items() if v})
                # Ensure updated context is always in the interrupt
                if "conversation_context" not in interrupt:
                    interrupt["conversation_context"] = ctx.to_dict()
                # Propagate correction_return_to clearing — but never stomp a
                # pointer the interrupt just set (an update detour opened this
                # turn writes its own correction_return_to).
                if state.get("correction_return_to") == "" and "correction_return_to" not in interrupt:
                    interrupt["correction_return_to"] = ""
                return interrupt

            collected[slot_name] = value
            if slot_name not in ctx.confirmed_slots:
                ctx.confirmed_slots.append(slot_name)
            if slot_name == "first_name":
                ctx.update_caller_name(value)
            prev_slot_just_confirmed = True

            # Clear correction detour pointer when the correction target is collected
            if slot_name == correction_return_to:
                state = {**state, "correction_return_to": ""}

        return None  # all slots collected
