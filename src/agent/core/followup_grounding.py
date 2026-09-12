"""
followup_grounding.py — is the side question the extractor reported actually
present in what the caller just said?

`WorkerResult.followup_query` is the single most expensive field on a turn. A
non-empty value routes the turn away from the deterministic confirm/re-ask path
and into FOLLOWUP_RESPOND, which generates prose whose whole contract is to
answer the "Followup:" line. When the line is real that is exactly right. When
the extractor invented it, the generator has nothing to answer and pads — and
because the padding is written against the same conversation the last AI turn
was written against, what it pads with is usually a restatement of the sentence
the caller just heard:

    AI      Thanks — and could I get your last name?
    Caller  Customer.
    AI      Got your last name as Customer, and I can certainly help you with
            your claim status today. Thanks — now, could I get your date of
            birth?

Nothing in the caller's turn asked anything. `followup_query` came back as
"help with claim status" — a topic the AI itself had raised two turns earlier
— and the turn paid for a generation call to answer a question nobody asked.

The same phantom reaches BaseAgent.execute's side-question safety net, which
generates a SECOND sentence for the turn and puts it in front of the first, so
one hallucinated field costs two generation calls and produces two sentences
that say overlapping things.

Three extraction headers already tell the model, in capitals, that
followup_query "MUST be derived from the caller's current utterance" and to
"NEVER synthesize a followup_query from topics the AI raised in prior turns".
Repeating an instruction three times is evidence it is not being followed.
Nothing in Python was checking.

This module is that check: a side question needs a trace of a question or a
request in the caller's own words. The test is deliberately generous — a
dropped real question is worse than a kept marginal one — so it accepts any
interrogative, any second-person modal, any ask-the-agent verb, and the plain
"I need / I want / help me" shapes a caller uses instead of a question mark.
What it rejects is the turn with no ask in it at all: a bare value, a yes, a
no, a name.

Keep this module dependency-free (core <-> agents import safety).
"""

from __future__ import annotations

import re

# Each entry is (name, pattern). The name is what gets logged when a reported
# side question is kept or dropped, so production can be read for which cue
# family is carrying the decisions.
_REQUEST_CUES: tuple[tuple[str, re.Pattern], ...] = (
    # A question mark is the cheapest evidence there is. On a voice call the
    # ASR supplies it inconsistently, which is why it is one cue among many
    # rather than the only one.
    ("question_mark", re.compile(r"\?")),
    # Interrogatives, anywhere — "and what about my copay", "so how long does
    # that take". Not anchored to the start: a side question almost never
    # starts the utterance, because the slot answer does.
    (
        "interrogative",
        re.compile(r"\b(?:what|what'?s|why|when|where|which|who|whose|whom|how|how'?s)\b", re.IGNORECASE),
    ),
    # Second-person modals — the shape of a request put to the agent.
    (
        "modal_request",
        re.compile(
            r"\b(?:can|could|would|will|should|may|shall|do|does|did|is|are|was|were|have|has)\s+"
            r"(?:you|i|we|it|that|there|they|my)\b",
            re.IGNORECASE,
        ),
    ),
    # Ask-the-agent verbs that need no modal — "say that again", "repeat that",
    # "tell me about the deductible", "explain the coinsurance".
    (
        "ask_verb",
        re.compile(
            r"\b(?:repeat|explain|clarify|remind\s+me|tell\s+me|read\s+(?:it\s+|that\s+)?back|"
            r"go\s+over|say\s+that|come\s+again)\b",
            re.IGNORECASE,
        ),
    ),
    # "again" on its own is a repeat request in spoken English — "sorry, your
    # name again", "the zip again".
    ("again", re.compile(r"\bagain\b", re.IGNORECASE)),
    # Requests stated rather than asked. A caller mid-verification says "I lost
    # my ID card, I need a new one" far more often than they form a question.
    #
    # "please" only counts with something after it ("please check my claim
    # status"). Trailing, it is politeness attached to the answer — "SMS
    # please", "fax please", "Customer, please" — and treating that as a
    # question is the very over-trigger this module exists to stop.
    (
        "stated_request",
        re.compile(
            r"\b(?:i\s+need|i\s+want|i'?d\s+like|i\s+have\s+to|i'?ve\s+got\s+to|"
            r"help\s+me|helping\s+me|let\s+me\s+know)\b|\bplease\s+\S",
            re.IGNORECASE,
        ),
    ),
    # Change / redo shapes. These usually arrive as update_target rather than
    # followup_query, but when they arrive as both the question is real.
    (
        "change_request",
        re.compile(
            r"\b(?:change|update|correct|fix|resend|re-?send|instead|different|"
            r"wrong|incorrect|no\s+longer)\b",
            re.IGNORECASE,
        ),
    ),
)

# Below this, an utterance has no room for a side question alongside whatever
# else it is doing. "what?" and "sorry?" are requests to repeat, and the canned
# re-ask answers them better than generated prose does — see
# carries_freeform_content.
_MIN_FREEFORM_WORDS = 3


def request_cue(text: str | None) -> str:
    """The name of the first request/question cue in ``text``, or "".

    Truthy means the caller asked or requested something in this utterance.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    for name, pattern in _REQUEST_CUES:
        if pattern.search(stripped):
            return name
    return ""


def is_grounded_followup(query: str | None, utterance: str | None) -> bool:
    """Is a reported ``followup_query`` supported by the caller's own words?

    False means the extractor produced a side question the caller did not ask
    — drop it rather than generate an answer to it.

    An empty ``query`` is trivially not a follow-up. An empty ``utterance``
    means the call site has no transcript to check against, in which case the
    extractor is trusted: this module only ever vetoes on evidence, never on
    the absence of it.
    """
    if not (query or "").strip():
        return False
    if not (utterance or "").strip():
        return True
    return bool(request_cue(utterance))


def carries_freeform_content(text: str | None) -> bool:
    """Does this utterance contain something a canned re-ask cannot address?

    Used as the safety net under ``WorkerResult.needs_freeform_response``: the
    flag comes from a model that can be wrong about its own output, and it was
    wrong on "Please check my claim status today" — a clear request that got
    "Sorry, I didn't catch that" twice running.

    The net used to be a word count (four words or more => generate). Word
    count is the wrong proxy: on a voice call nearly every non-answer clears
    four words ("um, I'm not really sure about that"), so the deterministic
    re-ask path almost never ran and the generation LLM got a free hand on
    precisely the turns where — as the fast path's own comment puts it — there
    is nothing for it to add and a great deal for it to get wrong.

    What actually distinguishes the claim-status turn from a mumble is not its
    length but that it asks for something. So: a request cue, and enough words
    for the cue to be doing real work. "what?" stays on the canned path, where
    "Sorry, I didn't catch that — could you say your first name again?" is both
    true and exactly what the caller wanted.
    """
    stripped = (text or "").strip()
    if len(stripped.split()) < _MIN_FREEFORM_WORDS:
        return False
    return bool(request_cue(stripped))
