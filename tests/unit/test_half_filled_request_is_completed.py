"""A request the extractor half-reported is still a request.

    AI      …Would you like us to send you details about our Care Coach Guides?
    Caller  please change my email address?
    AI      One more thing — you're eligible for a complimentary health and
            wellness coach. Would you like me to send you information on how to
            get started?

The request vanished and the offer was re-asked.

Not a detection failure: detect_request reads that utterance as update/email
without difficulty. And not a missing branch either — benefits_agent has one
written for this exact case, commented "please change my email address?" during
the Care Coach offer, which routes to delivery_management as a redo so the
contact update and the re-send are handled together.

The branch tests request_kind == "update" AND update_target in ("email", "fax"),
and the extractor returned update_target "email" with request_kind left "none".
reconcile_worker_result's gap-filler only wrote the pair when BOTH were empty:

    if not llm_target and not llm_kind:
        result.update_target = detected.target
        result.request_kind = ...

so a half-filled result stayed half-filled, and every downstream branch needing
the pair behaved as though nothing had been requested. The fields are now filled
one at a time — which is what "fills gaps" always meant.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from agent.agents.benefits import agent as benefits_module
from agent.core.request_detection import reconcile_worker_result
from agent.llm.schema import RequestKind, WorkerResult

SAID = "please change my email address?"


def _kind(result) -> str:
    raw = getattr(result, "request_kind", None)
    return str(getattr(raw, "value", raw) or "")


# ── the gap-filler ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label, kwargs",
    [
        ("neither field", {}),
        ("target only — the reported shape", {"update_target": "email"}),
        ("kind only", {"request_kind": RequestKind.UPDATE}),
        ("both already set", {"update_target": "email", "request_kind": RequestKind.UPDATE}),
    ],
)
def test_both_fields_end_up_set(label, kwargs):
    result = reconcile_worker_result(WorkerResult(**kwargs), SAID)

    assert result.update_target == "email", label
    assert _kind(result) == "update", label


def test_a_target_the_llm_named_is_never_overwritten():
    """Filling a gap is not second-guessing a detection. The regex reads this
    utterance as update/email; the LLM named delivery_method, and that stands —
    only the empty side is written."""
    result = reconcile_worker_result(WorkerResult(update_target="delivery_method"), SAID)

    assert result.update_target == "delivery_method"
    assert _kind(result) == "update"


def test_a_turn_with_no_request_is_untouched():
    result = reconcile_worker_result(WorkerResult(), "yes please")

    assert not (result.update_target or "")
    assert _kind(result) in ("none", "")


# ── end to end, through the real agent ───────────────────────────────────────


async def _turn(extracted: dict | None = None, **extraction_fields) -> dict:
    async def _extract(*_a, **_k):
        return WorkerResult(extracted=extracted or {}, **extraction_fields)

    async def _generate(**_kw):
        return "GENERATED"

    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "call_intent": "provider_services",
        "benefits_explained": True,
        "provider_list_sent": True,
        "delivery_method": "email",
        "email": "emily.new@example.com",
        "provider_type": "Primary Care Physician",
        "zip_code": "16783",
        "zip_code_used": "16783",
        "first_name": "Emily",
        "last_name": "New",
        "member_id": "M451982",
        "awaiting_slot": "care_coach_response",
        "messages": [
            {"role": "assistant", "content": "Would you like us to send you details about our Care Coach?"},
            {"role": "user", "content": SAID},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(benefits_module, "extract_benefits_decision", _extract))
        stack.enter_context(patch.object(benefits_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        return await benefits_module.BenefitsAgent.from_state(state).execute(state)


@pytest.mark.parametrize(
    "label, kwargs",
    [
        ("neither field", {}),
        ("target only — the reported shape", {"update_target": "email"}),
        ("kind only", {"request_kind": RequestKind.UPDATE}),
        ("both already set", {"update_target": "email", "request_kind": RequestKind.UPDATE}),
    ],
)
async def test_the_email_change_reaches_delivery_whatever_the_extractor_gave(label, kwargs):
    result = await _turn(**kwargs)

    assert result["next_node"] == "delivery_management_agent", label
    request = result.get("pending_cross_agent_request") or {}
    assert request.get("kind") == "redo"
    assert request.get("target") == "delivery"
    assert request.get("return_awaiting") == "care_coach_response", "the offer is still owed an answer"


async def test_the_care_coach_answer_still_works():
    """The route must not take the turn away from a caller who simply said yes."""
    result = await _turn(extracted={"care_coach_response": "yes"})

    assert result["next_node"] != "delivery_management_agent"


async def test_the_request_is_not_invented_from_the_offer_itself():
    """ "yes please" names no slot — nothing should route."""

    async def _extract(*_a, **_k):
        return WorkerResult(extracted={"care_coach_response": "yes"})

    async def _generate(**_kw):
        return "GENERATED"

    state = {
        "slot_attempts": {},
        "app_run_id": "test-run",
        "member_status_verify": True,
        "call_intent": "provider_services",
        "benefits_explained": True,
        "provider_list_sent": True,
        "awaiting_slot": "care_coach_response",
        "messages": [
            {"role": "assistant", "content": "Would you like details about our Care Coach?"},
            {"role": "user", "content": "yes please"},
        ],
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(benefits_module, "extract_benefits_decision", _extract))
        stack.enter_context(patch.object(benefits_module, "get_extraction_llm", lambda: object()))
        stack.enter_context(patch("agent.llm.response_generator.generate_recovery_message", _generate))
        result = await benefits_module.BenefitsAgent.from_state(state).execute(state)

    assert not (result.get("pending_cross_agent_request") or {})
