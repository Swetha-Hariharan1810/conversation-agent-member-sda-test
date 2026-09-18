"""
confirmation.py — the one rule for reading an answer to a read-back.

A read-back asks the caller to confirm a value already on file: "your ZIP code
is 58797, correct?", "I'll send it to 2155553299 — is that the right fax?".
Three answers matter, and the asymmetry between them is the whole point:

  - an affirmation. A closed set. "yes", "yeah", "correct", "that's right" —
    people have a handful of ways to agree and they are not inventing more.
  - a replacement value. Not a vocabulary at all: a ZIP, a fax, an email is a
    SHAPE, recognised by its form no matter how it is introduced.
  - a decline. Endless. "I moved", "that's my old one", "we relocated last
    spring", "my daughter handles my mail now", "that hasn't been right since
    the divorce". No list of these is ever finished, and every call that
    arrives with a phrasing the list lacks is a bug report.

So agents recognise the first two and treat everything else as a decline.
Nothing has to know how a decline was worded to act on one, because asking for
the current value is what a decline needs and that needs no phrase matched.

What that leaves to identify is the narrow set this module names: turns that
are not an answer to the question at all. Those re-ask the read-back, and
reading a decline into them would be wrong. It is decided from the extraction
model's classification of the turn, never from the words in it.

Risk runs one way here. Mistaking a decline for a non-answer re-asks a question
the caller already answered — the failure this rule exists to stop. Mistaking a
non-answer for a decline asks for a value that was going to be asked for
anyway. Neither confirms a stale contact, which is the outcome that would
actually cost something: a provider list faxed to a number the caller has just
told us is wrong.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from agent.utils import detect_wait_request

# Words that say "not this value" on a turn that reads one back. Not a list of
# DECLINE PHRASINGS — that list is the one that never finishes, and the reason
# _STALE_CONTACT_RE was deleted from delivery_management. This is the far
# smaller set of words that name the ACT of changing a value, and it is
# consulted only when the extractor reported no position at all (see below),
# so a turn the model classified is never overruled by vocabulary.
_CHANGE_INTENT_RE = re.compile(
    r"\b(?:new|newer|change|changed|changing|update|updated|updating|"
    r"different|another|old|older|outdated|obsolete|wrong|incorrect|"
    r"switch|switched|replace|replaced|stale|moved|moving|relocated)\b"
    r"|\bno\s+longer\b"
    r"|\bnot\s+(?:right|correct|valid|current)\b",
    re.IGNORECASE,
)


def is_not_an_answer(result: Any, last_user: str, *, owned_slots: Sequence[str] = ()) -> bool:
    """Is this turn about something other than the value we just read back?

    Each case below says "the caller took no position on the value on file":

      - nothing came back (the extraction call threw and returned an empty
        result is NOT this case — that is a caller whose words the model could
        not place, which on a confirmation question is a decline); here it
        means no result object at all, or the caller said nothing we can see
      - no usable value: "I'm not sure", "I think so?" — they do not know
      - asked for time: "hold on a second" — they are not answering yet
      - a side question rides the turn — it gets answered or parked
      - they want to change a DIFFERENT slot — that routes to its owner

    owned_slots: the slot names this read-back is about (e.g. ("fax",
    "fax_confirmed")). An update aimed at one of these is the caller declining
    the value, not a request to route elsewhere — and so is a VALUE extracted
    for one of them, which answers the read-back outright and beats every case
    below it.
    """
    if result is None or not (last_user or "").strip():
        return True

    if result.no_usable_value or result.asked_for_time:
        return True

    # The same case, read from the caller's words rather than from the label.
    # The WAIT above fires only when the extractor tagged the turn, and on the
    # reported transcripts it did not — it returned a plain ANSWERED with
    # nothing extracted, which falls through to the bottom of this function as
    # False and is read as a decline of the value on file:
    #
    #     AI      Just to be sure — your fax number is 231-555-3211, correct?
    #     Caller  hold on, let me dig out the letter... one second
    #     AI      No problem — what is the correct fax number?
    #
    # The caller asked for a second and lost the number we already had. A wait
    # is not a position on the value, whoever labelled the turn.
    if detect_wait_request(last_user):
        return True

    # A position on the value beats anything else riding the turn.
    #
    #     AI      The fax number we have on file is 4155553211. Is this correct?
    #     Caller  Yeah. That's kind of an old fax number. I'll give you a new
    #             number if you can do that?
    #     AI      No worries at all, I can update that for you. I'll send it to
    #             4155553211 — is that the right fax number?
    #
    # The caller declined the number and offered a replacement in one breath.
    # The extractor heard the decline — fax_confirmed "no" — and the tail was
    # read as a side question, which used to be enough on its own to call the
    # turn a non-answer. So the number the caller had just rejected was read
    # back to them again, and the decline branch that would have asked for the
    # new one was never reached.
    #
    # owned_slots was already the answer to this and only guarded the
    # change-target test at the bottom, so the same intent put as a QUESTION
    # rather than an update walked past it. A value extracted for one of these
    # slots is the caller answering the read-back; the side question rides
    # along in pending_side_answer and is answered in front of whatever is
    # asked next.
    #
    # AMBIGUOUS and WAIT stay ahead of this deliberately: "I'm not sure" and
    # "hold on" are not positions, whatever else the extractor filled in.
    owned = {s.strip().lower() for s in owned_slots}
    extracted = getattr(result, "extracted", None) or {}
    if owned and any(
        str(value or "").strip() for key, value in extracted.items() if str(key).strip().lower() in owned
    ):
        return False

    # The same position, when the extractor reported no position at all.
    #
    #     AI      The fax number we have on file is 4155553211. Is this correct?
    #     Caller  Yeah. That's kind of an old fax number. I'll give you a new
    #             number if you can do that.
    #     →       extracted {}, no change target,
    #             followup_query "I'll give you a new number if you can do that"
    #
    # The caller declined and offered a replacement, and the extractor reported
    # the whole turn as a side question — no fax_confirmed, no change target.
    # Both of the outs the prompt gives it were unused, so the checks below saw
    # a bare question and re-read the number the caller had just called old.
    #
    # This runs only after the two above have found nothing: a value for an
    # owned slot returns before it, and AMBIGUOUS/WAIT return before that. So
    # the words are consulted only when the model's classification gives the
    # branch nothing to act on — the same exception, and the same reasoning, as
    # detect_wait_request above.
    #
    # Reaching it wrongly costs a question that was coming anyway ("what fax
    # number should we use?"). Not reaching it costs the caller being read back
    # a value they rejected, and a provider list sent to it. The module
    # docstring's asymmetry decides which way to lean.
    # A request the extractor DID place decides on its own, and it decides
    # both ways: aimed at an owned slot it is the caller declining the value
    # (so an answer), aimed at anything else it routes to that owner (so not
    # one). This moved above the follow-up checks with the same reasoning as
    # the value test above — "can you use a different fax?" is a position on
    # the fax, however the extractor labelled the sentence carrying it.
    target = result.change_target.lower()
    if target:
        return target not in owned

    # Last, the caller's own words — and only here, where the extractor has
    # placed nothing at all: no value, no target, no usable label.
    #
    #     AI      The fax number we have on file is 4155553211. Is this correct?
    #     Caller  Yeah. That's kind of an old fax number. I'll give you a new
    #             number if you can do that.
    #     →       extracted {}, no change target,
    #             followup_query "I'll give you a new number if you can do that"
    #
    # The caller declined and offered a replacement, and the extractor reported
    # the whole turn as a side question. Both outs the prompt gives it went
    # unused, so the checks below saw a bare question and re-read the number
    # the caller had just called old.
    #
    # Reaching this wrongly costs a question that was coming anyway ("what fax
    # number should we use?"). Not reaching it reads a rejected value back and
    # can send a provider list to it. The module docstring's asymmetry decides
    # which way to lean.
    if owned and _CHANGE_INTENT_RE.search(last_user or ""):
        return False

    return bool(str(getattr(result, "followup_query", "") or "").strip())


def confirms_value(verdict: str, last_user: str, *, owned_slots: Sequence[str] = ()) -> bool:
    """Is this normalized "yes" actually a confirmation of the value read back?

    A caller who opens with the affirmative and then asks to change the value
    has not confirmed it:

        AI      Just to confirm — your ZIP code is 16783?
        Caller  yeah. Actually, you know what? I want to update the ZIP code
                because I moved to a new address. So can I do that now?
        →       zip_confirmed "yes"

    The leading affirmative acknowledges the question; the rest answers it. Six
    branches read a bare "yes" and act on it immediately — dispatching a
    provider list, writing a contact to Salesforce — BEFORE is_not_an_answer
    is ever consulted, so nothing downstream of them can undo it. The caller's
    update request is dropped without a word, and the list goes to the value
    they were in the middle of replacing.

    The extraction prompts are where this belongs and where it is now stated
    for every slot (extraction/_confirmation_contract.md). This is the backstop
    for when the model reads the first word and stops, which is what the
    reported calls show: the cost of missing it is silent and lands on the
    caller, and a prompt rule cannot be regression-tested.

    Same narrow vocabulary as is_not_an_answer, and the same asymmetry behind
    it: a "yes" wrongly rejected here costs the question the branch was about
    to ask anyway, because the caller falls through to the decline path that
    asks for the current value. A "yes" wrongly accepted sends the list
    somewhere the caller has just told us not to.
    """
    if verdict != "yes":
        return False
    return not (owned_slots and _CHANGE_INTENT_RE.search(last_user or ""))


def _fold_spoken(text: str) -> str:
    """Fold the spoken punctuation callers use for addresses into its symbols.

    "james dot one at example dot com" → "james.one@example.com". Only used to
    ask whether a value appears in what the caller said, so folding a stray
    "at" inside an ordinary sentence costs nothing.
    """
    folded = (text or "").lower()
    folded = re.sub(r"\s*\bdot\b\s*", ".", folded)
    folded = re.sub(r"\s*\bat\b\s*", "@", folded)
    return folded


def caller_spoke(value: str, last_user: str, normalizer: Callable[[str], str]) -> bool:
    """Did the caller actually say this value, or did the extractor supply it?

    The extraction model is handed the value on file on a ``Confirmed:`` context
    line (llm.extractor.build_worker_input). When it returns that same value it
    may be reporting what the caller said — or parroting its own context. The
    two are indistinguishable in ``extracted{}`` and mean opposite things, so
    this asks the only source that can tell them apart: the utterance.

    Digit values are recovered from a whole sentence by their own normalizers,
    spoken digits and all, and come back empty when the caller spoke none:

        normalize_fax_number("no, send it to six one seven ...") → "6175554199"
        normalize_fax_number("no")                               → ""

    Addresses are not — the model is what turns "dot"/"at" into punctuation —
    so those fold the spoken form and look for the value in it.
    """
    value = (value or "").strip()
    utterance = (last_user or "").strip()
    if not value or not utterance:
        return False
    if value.isdigit():
        return value in normalizer(utterance)
    return value.lower() in _fold_spoken(utterance)


def is_read_back_echo(
    new_value: Any,
    read_back: str,
    normalizer: Callable[[str], str],
    *,
    last_user: str | None = None,
) -> bool:
    """Is the value the extractor returned one the caller never gave us?

    The extraction contract says a replacement contact and a yes/no on the
    read-back are mutually exclusive. The model breaks it both ways, so four
    confirmation branches compensated — and three of them compensated in the
    wrong direction, clearing the replacement whenever a "no" arrived with it:

        AI      The email address we have on file is james.wilson@gmail.com.
                Is this correct or has it been changed?
        Caller  no, use james.one@example.com
        AI      No problem — what is the correct email address?
        Caller  no, actually use james.two@example.com
        AI      No problem — what is the correct email address?

    The caller handed over the new address twice and was asked for it twice
    more. records_coordination is the clearest case: the line that threw the
    value away sat three lines above a branch commented "Inline replacement:
    member declined AND provided new email in same utterance", which it made
    unreachable.

    A replacement is easy: it is a value different from the one we read back.
    Comparison settles it, against what the read-back actually put to the caller
    (the pending value when there is one, otherwise the value on file), since a
    second read-back reads the pending value back.

    Equality is the case that needed ``last_user``. Both of these reach this
    function as a "no" carrying the value on file, and they are opposite turns:

        AI      The fax number we have on file is 6175554199. Is that correct?
        Caller  no, send it to six one seven five five five four one nine nine
        →       the caller named the destination. Not an echo — take it.

        AI      The fax number we have on file is 6175554199. Is that correct?
        Caller  no
        →       the model filled the field from its Confirmed: context line.
                An echo — clear it, and let the decline path ask.

    Treating the first as an echo asks the caller for a value they just spoke.
    Treating the second as a replacement dispatches to the contact they just
    declined — the failure delivery_management's fax branch carries a comment
    about. Only the utterance separates them, so when it is given, it decides;
    when it is not, equality alone means echo, as before.
    """
    text = str(new_value or "").strip()
    if not text:
        return True  # nothing to keep
    normalized = normalizer(text)
    if normalized != normalizer((read_back or "").strip()):
        return False  # a different value — the caller's replacement
    if last_user is None:
        return True
    return not caller_spoke(normalized, last_user, normalizer)


# Extraction field and validation for each delivery channel a caller can name.
# "sms" and "phone" are the same contact under two names — the channel is how
# it will be used, the field is where the extraction puts it.
_CHANNEL_CONTACT: dict[str, str] = {
    "fax": "fax",
    "email": "email",
    "sms": "phone",
    "phone": "phone",
}


def carried_contact(result: Any, channel: str) -> str:
    """A valid contact for ``channel`` the caller gave in this same utterance.

    "Send it by fax, use 415-555-3211" names the channel and the number in one
    breath; "text me at 415-555-3211" and "email it to jim at example dot com"
    do the same. The contact is the caller's answer just as much as the channel
    is, and an agent that takes only the channel then reads the value on file
    back invites a "yes" to a destination the caller never gave — which is how
    a provider list, or a claim notification, reaches the wrong number carrying
    the caller's own apparent agreement to it.

    Returns "" when nothing usable was given: a half-heard number that does not
    validate is not a value the caller can be held to, and the contact on file
    is confirmed instead, as before.
    """
    from agent.slots.normalizers import normalize_email, normalize_fax_number, normalize_phone_number
    from agent.slots.validators import validate_email, validate_fax_number, validate_phone_number

    field = _CHANNEL_CONTACT.get((channel or "").strip().lower())
    if not field:
        return ""

    extracted = (getattr(result, "extracted", None) or {}) if result else {}
    raw = str(extracted.get(field) or "")
    if not raw:
        return ""

    normalize, validate = {
        "fax": (normalize_fax_number, validate_fax_number),
        "email": (normalize_email, validate_email),
        "phone": (normalize_phone_number, validate_phone_number),
    }[field]

    candidate = normalize(raw)
    return candidate if candidate and validate(candidate).valid else ""
