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


# ── Recovering a follow-up the extractor dropped ─────────────────────────────
# The veto above handles a question the model invented. This handles the other
# half, which turned out to be the same bug seen from the other side:
#
#     AI      …I can also provide your benefits information for Pediatrician
#             visits — would that be helpful?
#     Caller  No. But I lost my credit ID card. Can you help me with the new one?
#     → {"event_type": "answered", "followup_query": null}
#
#     AI      …Do you want us to send the details of our Care Coach Guides?
#     Caller  That sounds interesting, but I lost my ID card. Can you help me to
#             get a new one?
#     → {"event_type": "answered_with_followup",
#        "followup_query": "can you help me to get a new one"}
#
# The same request, two turns apart, classified both ways. It is not model
# variance: those two slots run different prompt stacks. benefits_response is
# collected by delivery_management against header_extraction.md +
# delivery_management.md — 4,100 words, with the follow-up rules a long way
# from the field definitions. care_coach_response is collected by the benefits
# agent against header_core.md + benefits.md — 1,300 words, with a request
# block sitting directly under FIELDS. Whether a caller's question is heard
# depends on which prompt file the current slot happens to live in.
#
# Nineteen prompt files cannot be kept in step by hand, and a missed question
# is invisible downstream: BaseAgent.execute's safety net only fires when
# followup_query is set, so a dropped one is indistinguishable from a caller
# who asked nothing — on that turn and on every repeat of it.
#
# So the shape is read from the caller's words instead. This is deliberately
# far stricter than request_cue: it fires only on the clean "answer, then ask"
# shape, because a false positive here puts words in the caller's mouth.

# Where the answer stops and the ask starts. Sentence boundaries, plus the
# contrastive markers that carry a second thought inside one sentence. Bare
# "and" is NOT a separator: "M451982 and my dob is November 5th" is one answer
# in two parts, not an answer and a question.
_SIDE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|[,;]?\s+\b(?:but|however|although|though|by\s+the\s+way|one\s+more\s+thing|"
    r"quick\s+question|and\s+also)\b[,]?\s+",
    re.IGNORECASE,
)

# A strict ask. Every alternative here is unambiguously directed at the agent —
# no bare interrogatives ("what a day"), no "is my" (which matches an ordinary
# correction, "actually my email is …").
_SIDE_REQUEST_RE = re.compile(
    r"\?"
    r"|\b(?:can|could|would|will)\s+(?:you|i|we)\b"
    r"|\bhow\s+(?:do|can)\s+i\b"
    r"|\bwhat\s+(?:do|does|about)\s+\w+\b"
    r"|\bhelp\s+me\b"
    r"|\bi\s+(?:need|want|lost|misplaced)\b"
    r"|\bis\s+there\s+(?:a|any)\b",
    re.IGNORECASE,
)

# A courtesy question is not a side question. "Fax please. Can you do that for
# me today?" asks about the very thing just answered — the whole utterance is
# one answer, and treating the tail as a second topic makes the turn generate
# prose to answer a question that was already answered by saying yes.
#
# These are a closed set in this domain, so they are listed rather than guessed
# at. They are only consulted when the recovery is a SINGLE segment: a
# multi-segment recovery carries a statement that introduces a topic ("I lost
# my credit ID card") in front of its question, which a courtesy question never
# does.
_COURTESY_QUESTION_RE = re.compile(
    r"^(?:so\s+|and\s+|then\s+)?(?:"
    r"(?:can|could|will|would)\s+you\s+(?:do|manage|handle)\s+(?:that|this|it|so)\b"
    r"|(?:can|could|will|would)\s+you\s+help\s+me\s+with\s+(?:that|this|it)\b"
    r"|(?:is|was|would)\s+that\s+(?:be\s+)?"
    r"(?:ok|okay|alright|all\s+right|fine|possible|right|correct|what\s+you\s+\w+)\b"
    r"|(?:does|will|would)\s+that\s+work\b"
    r"|(?:did|do)\s+you\s+(?:get|hear|catch|have)\s+(?:that|me|it|all\s+that)\b"
    r"|sounds?\s+good\b"
    r"|is\s+that\s+(?:everything|all)\b"
    r")",
    re.IGNORECASE,
)

# A recovered question is quoted back to the generation LLM as the "Followup:"
# line. Past this many words it stops being a question and starts being a
# transcript, so it is left to the extractor.
_MAX_RECOVERED_WORDS = 30

_LEADING_CONJUNCTION_RE = re.compile(
    r"^(?:but|however|although|though|and|by\s+the\s+way|one\s+more\s+thing|quick\s+question)\b[,\s]*",
    re.IGNORECASE,
)


# Words that carry no topic. A reported question and the caller's utterance
# overlapping only on these says nothing about whether they are about the same
# thing.
# The second block is service-generic: on a member-services call "help",
# "need" and "call" appear in almost every turn on both sides, so an overlap on
# them is not evidence of a shared topic. "claim", "card", "email", "fax",
# "benefits" and the rest of the real subject matter stay.
_STOPWORDS = frozenset(
    """
    a an and are can could did do does for from get got has have how i if in is it
    its me my not of on one or please that the their them then there they this to
    was were what when where which who will with would you your about also but

    help helping need needs want wants like would thanks thank sorry sure okay
    call calling today now just know tell told say said give given make made take
    use used back again thing things something anything everything yes yeah nope
    """.split()
)


def _content_words(text: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 2}


def quotes_the_caller(query: str | None, utterance: str | None) -> bool:
    """Is the reported question about the same thing the caller talked about?

    A weaker test than it sounds: the headers ask for a "short paraphrase", so
    wording legitimately differs, and one shared topic word is enough. What it
    catches is the reported question with NOTHING in common with the turn —
    "help with claim status" against "No. But I lost my credit ID card. Can you
    help me with the new one?" — which is the signature of a question lifted
    from the AI's own earlier turns.

    An utterance with no topic words of its own returns True: there is nothing
    to disagree with, so the extractor is left alone.
    """
    said = _content_words(utterance)
    if not said:
        return True
    asked = _content_words(query)
    if not asked:
        return False
    return bool(asked & said)


def recover_side_question(utterance: str | None) -> str:
    """The side question in ``utterance`` that the extractor did not report.

    Returns "" unless the utterance has the clean two-part shape: an answer
    that asks nothing, followed by something that plainly does. Both halves are
    required. A single segment is the caller answering ("Fax, please send it to
    231-555-3211") or asking instead of answering ("Do you have
    pediatricians?") — neither is an answer-plus-question, and neither is this
    function's business.

    A request cue in the FIRST segment also returns "": the whole utterance
    reads as one ask, there is no clean split to make, and the extractor's own
    classification of such a turn is left alone. So does a lone courtesy
    question — "Fax please. Can you do that for me today?" asks about the thing
    just answered, and is part of the answer, not a second topic.

    The recovered text is the caller's own words from the first asking segment
    onward — not a paraphrase. "I lost my credit ID card" is kept in front of
    "can you help me with the new one", because without it the question does
    not say what it is about.
    """
    text = (utterance or "").strip()
    if not text:
        return ""
    segments = [s.strip(" ,;") for s in _SIDE_SPLIT_RE.split(text) if s and s.strip(" ,;")]
    if len(segments) < 2:
        return ""
    if _SIDE_REQUEST_RE.search(segments[0]):
        return ""
    for index, segment in enumerate(segments[1:], start=1):
        if not _SIDE_REQUEST_RE.search(segment):
            continue
        tail = segments[index:]
        recovered = _LEADING_CONJUNCTION_RE.sub("", " ".join(tail)).strip()
        if not recovered or len(recovered.split()) > _MAX_RECOVERED_WORDS:
            return ""
        if len(tail) == 1 and _COURTESY_QUESTION_RE.match(recovered):
            return ""
        return recovered
    return ""
