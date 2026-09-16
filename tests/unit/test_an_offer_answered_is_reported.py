"""The two offers a provider call makes are decisions, so the call reports them.

A completed provider call reported its intent, its provider type, the caller,
the ZIP, the delivery method and the address the list went to — and nothing
about the two questions it asked afterwards:

    ai    …would you like me to go over your plan benefits?
    ai    …shall I send you the details of our Care Coach program?

Neither answer is a slot the pipelines collect, so neither passed through
slot_ok, and neither reaches state as a value: they set flags
(benefits_offer_made, proactive_offer_available, care_coach_offered,
care_coach_details_sent) that route the rest of the call. A flag that is False
is not a capture — _format_value drops it — so a caller who declined both was
indistinguishable from a caller who was never asked.

The yes/no is what the caller decided, so that is what gets reported, under the
slot each offer is asked as: benefits_response and care_coach_response.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agents.benefits import agent as benefits_module
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import WorkerResult


def _fields(update: dict) -> dict:
    return {
        e["data"]["field"]: e["data"]["value"]
        for e in update.get("metadata_events") or []
        if e["eventType"] == "CallAgentField"
    }


# ── the benefits offer, answered in delivery_management ──────────────────────


async def _benefits_answer(answer: str) -> dict:
    state = {
        "app_run_id": "test-run",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "delivery_method": "email",
        "awaiting_slot": "benefits_response",
        "messages": [],
    }
    agent = DeliveryManagementAgent.from_state(state)
    result = SimpleNamespace(extracted={"benefits_response": answer})
    return agent.stamp_metadata_events(state, await agent._handle_benefits_response(state, result))


@pytest.mark.parametrize("answer,reported", [("yes", "yes"), ("no", "no"), ("nope", "no")])
async def test_the_benefits_offer_is_reported_either_way(answer, reported):
    assert _fields(await _benefits_answer(answer))["benefits_response"] == reported


async def test_an_unanswered_benefits_offer_reports_nothing():
    """A turn that is not an answer is a retry, not a decision."""
    assert "benefits_response" not in _fields(await _benefits_answer("hold on a second"))


# ── the Care Coach offer, answered in benefits ───────────────────────────────


async def _care_coach_answer(answer: str) -> dict:
    async def _extract(*_args, **_kwargs):
        return WorkerResult(extracted={"care_coach_response": answer})

    async def _generate(**_kwargs):
        return "Generated."

    state = {
        "app_run_id": "test-run",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "benefits_explained": True,
        "awaiting_slot": "care_coach_response",
        "messages": [
            {"role": "assistant", "content": "Shall I send you the details of our Care Coach program?"},
            {"role": "user", "content": answer},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(benefits_module, "extract_benefits_decision", _extract))
        stack.enter_context(patch.object(benefits_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        return await benefits_module.BenefitsAgent.from_state(state).execute(state)


@pytest.mark.parametrize("answer,reported", [("yes", "yes"), ("no", "no")])
async def test_the_care_coach_offer_is_reported_either_way(answer, reported):
    assert _fields(await _care_coach_answer(answer))["care_coach_response"] == reported


async def test_an_unanswered_care_coach_offer_reports_nothing():
    assert "care_coach_response" not in _fields(await _care_coach_answer(""))
