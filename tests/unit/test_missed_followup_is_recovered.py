"""The same request gets the same answer, whichever slot the caller is on.

    AI      …I can also provide your benefits information for Pediatrician
            visits — would that be helpful?
    Caller  No. But I lost my credit ID card. Can you help me with the new one?
    →       {"event_type": "answered", "followup_query": null}

    AI      …Do you want us to send the details of our Care Coach Guides?
    Caller  That sounds interesting, but I lost my ID card. Can you help me to
            get a new one?
    →       {"event_type": "answered_with_followup",
             "followup_query": "can you help me to get a new one"}

One request, two turns apart, classified both ways. It is not model variance —
those two slots run different prompt stacks:

    benefits_response    delivery_management  header_extraction.md +
                                              delivery_management.md  (~4,100 words,
                                              follow-up rules far from FIELDS)
    care_coach_response  benefits agent       header_core.md + benefits.md
                                              (~1,300 words, request block
                                              directly under FIELDS)

Whether a caller's question is heard depended on which prompt file the slot
they happened to be on lived in. Nineteen prompt files cannot be kept in step
by hand, and a missed question is invisible downstream: BaseAgent.execute's
safety net only fires when followup_query is set, so a dropped one is
indistinguishable from a caller who asked nothing — on that turn and every
repeat of it.

So the shape is read from the caller's words. recover_side_question fires only
on the clean "answer, then ask" split, because a false positive here puts words
in the caller's mouth.
"""

from __future__ import annotations

import pytest

from agent.core.followup_grounding import recover_side_question
from agent.core.request_detection import reconcile_worker_result
from agent.llm.schema import EventType, FollowupDisposition, RequestKind, WorkerResult

ID_CARD_AT_BENEFITS = "No. But I lost my credit ID card. Can you help me with the new one?"
ID_CARD_AT_CARE_COACH = "That sounds interesting, but I lost my ID card. Can you help me to get a new one?"


# ── recover_side_question ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance, expected",
    [
        (ID_CARD_AT_BENEFITS, "I lost my credit ID card. Can you help me with the new one?"),
        (ID_CARD_AT_CARE_COACH, "I lost my ID card. Can you help me to get a new one?"),
        ("Yes, but can you also send it to my email?", "can you also send it to my email?"),
        ("Fax. By the way, how do I get a replacement card?", "how do I get a replacement card?"),
        ("90210. One more thing, what about my copay?", "what about my copay?"),
    ],
)
def test_an_answer_then_an_ask_is_recovered(utterance, expected):
    assert recover_side_question(utterance) == expected


@pytest.mark.parametrize(
    "utterance",
    [
        # A courtesy question about the thing just answered is part of the
        # answer, not a second topic.
        "Fax please. Can you do that for me today?",
        "Email. Is that okay?",
        "Yes. Does that work?",
        "November 5th 1992. Did you get that?",
        "SMS. Sounds good?",
        # One segment: the caller answered, or asked instead of answering.
        # Either way there is no answer-plus-question split to make.
        "Fax, please send it to 231-555-3211.",
        "Do you have pediatricians?",
        "M451982",
        "No.",
        # A cue in the first segment — the whole utterance reads as one ask,
        # so the extractor's own classification stands.
        "Can you repeat that? It's M451982.",
        # An answer in two parts is not an answer and a question. Bare "and"
        # is deliberately not a separator.
        "M451982 and my dob is November 5th 1992.",
        # A correction, which has its own path.
        "90210. Actually my email is a at b dot com.",
        "It's 90210. Thanks.",
        "",
    ],
)
def test_everything_else_is_left_to_the_extractor(utterance):
    assert recover_side_question(utterance) == ""


def test_a_transcript_is_not_a_question():
    """Past a sentence or two it stops being a question and starts being a
    transcript, and it is quoted verbatim into the Followup: line."""
    rambling = "Yes. But " + " ".join(["something about my plan"] * 8) + " can you help me?"
    assert recover_side_question(rambling) == ""


# ── reconcile_worker_result: both turns now agree ────────────────────────────


def _answered(slot: str, value: str) -> WorkerResult:
    """What the extractor returned on the benefits_response turn."""
    return WorkerResult(
        event_type=EventType.ANSWERED,
        followup_disposition=FollowupDisposition.NONE,
        followup_query=None,
        extracted={slot: value},
    )


@pytest.mark.parametrize(
    "slot, value, utterance",
    [
        ("benefits_response", "no", ID_CARD_AT_BENEFITS),
        ("care_coach_response", "yes", ID_CARD_AT_CARE_COACH),
    ],
)
def test_both_slots_now_hear_the_same_request(slot, value, utterance):
    result = reconcile_worker_result(_answered(slot, value), utterance)

    assert result.event_type == EventType.ANSWERED_WITH_FOLLOWUP
    assert "ID card" in (result.followup_query or "")
    assert result.followup_disposition == FollowupDisposition.ANSWER
    assert result.extracted == {slot: value}, "the answer itself is untouched"


def test_a_reported_question_is_never_overwritten():
    """The LLM stays primary — recovery fills a gap, it never second-guesses a
    question the model did report."""
    reported = WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.ANSWER,
        followup_query="can you help me to get a new one",
        extracted={"care_coach_response": "yes"},
    )
    result = reconcile_worker_result(reported, ID_CARD_AT_CARE_COACH)

    assert result.followup_query == "can you help me to get a new one"


def test_a_turn_with_no_answer_is_left_alone():
    """Recovery needs both halves. With no extracted value there is no answer
    to have been given, and the ambiguous/question-only paths own that turn."""
    result = reconcile_worker_result(WorkerResult(event_type=EventType.AMBIGUOUS), ID_CARD_AT_BENEFITS)

    assert result.followup_query is None
    assert result.event_type == EventType.AMBIGUOUS


def test_the_veto_and_the_recovery_do_not_fight():
    """A phantom question on a turn that does contain a real one is replaced by
    the real one, not merely cleared."""
    phantom = WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.ANSWER,
        followup_query="help with claim status",
        extracted={"benefits_response": "no"},
    )
    result = reconcile_worker_result(phantom, ID_CARD_AT_BENEFITS)

    assert result.followup_query == "I lost my credit ID card. Can you help me with the new one?"
    assert result.event_type == EventType.ANSWERED_WITH_FOLLOWUP


# ── schema keys the model writes into extracted{} ────────────────────────────


def test_schema_keys_are_dropped_from_extracted():
    """Seen in production on the care_coach turn:

        "extracted": {"care_coach_response": "yes",
                      "update_target": "ID card", "request_kind": "update"}

    extracted{} is slot name -> caller value, and every consumer reads it that
    way: note_side_question joins its values into the "Extracted this turn:"
    line the generation LLM reads back, which became "yes, ID card, update".
    """
    polluted = WorkerResult(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_query="can you help me to get a new one",
        extracted={"care_coach_response": "yes", "update_target": "ID card", "request_kind": "update"},
        update_target="ID card",
        request_kind=RequestKind.UPDATE,
    )
    result = reconcile_worker_result(polluted, ID_CARD_AT_CARE_COACH)

    assert result.extracted == {"care_coach_response": "yes"}
    assert result.update_target == "ID card", "the real field is untouched"
    assert result.request_kind == RequestKind.UPDATE


def test_corrections_are_cleaned_the_same_way():
    polluted = WorkerResult(
        event_type=EventType.CORRECTED,
        corrections={"email": "a@b.com", "request_kind": "update"},
    )
    result = reconcile_worker_result(polluted, "actually my email is a at b dot com")

    assert result.corrections == {"email": "a@b.com"}


# ── topic overlap: the veto asks "did they ask?", not "did they ask THIS?" ───


@pytest.mark.parametrize(
    "query, utterance, shares",
    [
        # The phantom: a topic the AI raised, against a turn about something else.
        ("help with claim status", ID_CARD_AT_BENEFITS, False),
        # A legitimate paraphrase — one shared topic word is enough.
        ("can you help me to get a new one", ID_CARD_AT_CARE_COACH, True),
        ("when will the list arrive", "It's 90210 — when will I get the list?", True),
        ("repeat my zip code", "sure, can you repeat my zip code first?", True),
        # "help" and "need" appear on both sides of almost every service call,
        # so an overlap on them alone is not a shared topic.
        ("I need help", "Customer. I need my deductible.", False),
        # Nothing to disagree with — the extractor is left alone.
        ("when will the list arrive", "Yes.", True),
    ],
)
def test_topic_overlap(query, utterance, shares):
    from agent.core.followup_grounding import quotes_the_caller

    assert quotes_the_caller(query, utterance) is shares


# ── end to end: the two reported turns, through their real agents ────────────

ANSWER = "ID card replacements are handled on a different line."


@pytest.fixture
def generation(monkeypatch):
    calls: list[dict] = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return ANSWER

    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", _generate)
    return calls


async def test_the_benefits_turn_now_hears_the_question(monkeypatch, generation):
    """The reported failure: the extractor returned event "answered" and
    followup_query null, so the ID-card request was dropped on the floor."""
    from agent.agents.delivery_management import agent as dm

    async def _extract(*_a, **_k):
        return _answered("benefits_response", "no")

    monkeypatch.setattr(dm, "extract_delivery_management_decision", _extract)
    monkeypatch.setattr(dm, "get_extraction_llm", lambda: object())

    state = {
        "messages": [
            {"role": "assistant", "content": "…would that be helpful?"},
            {"role": "user", "content": ID_CARD_AT_BENEFITS},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "benefits_response",
        "fax": "2315553211",
        "provider_type": "Pediatrician",
        "zip_code": "16783",
        "zip_code_used": "16783",
        "call_intent": "provider_services",
        "provider_list_sent": True,
        "delivery_method": "fax",
    }
    result = await dm.DeliveryManagementAgent.from_state(state).execute(state)

    assert len(generation) == 1, "the question must reach the generation LLM exactly once"
    assert generation[0]["guard"] == "FOLLOWUP_RESPOND"
    assert "ID card" in generation[0]["followup_query"]
    # This turn hands off to benefits, so it says nothing of its own; the answer
    # rides along and ask_member puts it in front of the next agent's opener.
    assert result["pending_side_answer"] == ANSWER


async def test_the_care_coach_turn_is_unchanged(monkeypatch, generation):
    """The turn that already worked must keep working, and the schema keys the
    model wrote into extracted{} must not reach the generation payload."""
    from agent.agents.benefits import agent as ba

    async def _extract(*_a, **_k):
        return WorkerResult(
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            followup_disposition=FollowupDisposition.NONE,
            followup_query="can you help me to get a new one",
            extracted={"care_coach_response": "yes", "update_target": "ID card", "request_kind": "update"},
            update_target="ID card",
        )

    monkeypatch.setattr(ba, "extract_benefits_decision", _extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    state = {
        "messages": [
            {"role": "assistant", "content": "Do you want us to send the details of our Care Coach Guides?"},
            {"role": "user", "content": ID_CARD_AT_CARE_COACH},
        ],
        "slot_attempts": {},
        "app_run_id": "test-run",
        "awaiting_slot": "care_coach_response",
        "call_intent": "provider_services",
        "benefits_explained": True,
    }
    result = await ba.BenefitsAgent.from_state(state).execute(state)

    assert len(generation) == 1
    assert generation[0]["guard"] == "FOLLOWUP_RESPOND"
    assert generation[0]["followup_query"] == "can you help me to get a new one"
    assert "update" not in (generation[0].get("extracted_value") or ""), (
        'the "Extracted this turn:" line read "yes, ID card, update" before the schema keys were stripped'
    )
    assert result["pending_side_answer"] == ANSWER


def test_the_reconcile_runs_before_the_guards_in_every_agent():
    """note_side_question is called from run_conversation_guards, and reconcile
    is what recovers a dropped question and clears an invented one. An agent
    that reconciles afterwards leaves both invisible to the safety net on the
    one path this second reconcile exists for — where llm.py's did not run.

    Checked per method, on the agent modules' own source: any method that does
    both must do them in that order.
    """
    import pathlib
    import re

    import agent as agent_pkg

    root = pathlib.Path(agent_pkg.__file__).parent / "agents"
    agents = sorted(root.glob("*/agent.py"))
    assert agents, "no agent modules found"

    checked = 0
    for path in agents:
        source = path.read_text()
        # Split on def boundaries so each method is compared against itself.
        for body in re.split(r"\n    (?=async def |def )", source):
            reconcile = body.find("reconcile_worker_result(")
            guards = body.find("run_conversation_guards(")
            if reconcile == -1 or guards == -1:
                continue
            checked += 1
            assert reconcile < guards, (
                f"{path.parent.name} reconciles after running its guards:\n{body[: body.find(chr(10))]}"
            )
    assert checked >= 5, f"expected several agents to do both; found {checked}"
