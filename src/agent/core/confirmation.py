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

from typing import Any, Sequence

from agent.llm.schema import EventType, FollowupDisposition
from agent.utils import detect_wait_request


def is_not_an_answer(result: Any, last_user: str, *, owned_slots: Sequence[str] = ()) -> bool:
    """Is this turn about something other than the value we just read back?

    Each case below says "the caller took no position on the value on file":

      - nothing came back (the extraction call threw and returned an empty
        result is NOT this case — that is a caller whose words the model could
        not place, which on a confirmation question is a decline); here it
        means no result object at all, or the caller said nothing we can see
      - AMBIGUOUS: "I'm not sure", "I think so?" — they do not know
      - WAIT: "hold on a second" — they are not answering yet
      - a side question rides the turn — it gets answered or parked
      - they want to change a DIFFERENT slot — that routes to its owner

    owned_slots: the slot names this read-back is about (e.g. ("fax",
    "fax_confirmed")). An update aimed at one of these is the caller declining
    the value, not a request to route elsewhere.
    """
    if result is None or not (last_user or "").strip():
        return True

    if getattr(result, "event_type", None) in (EventType.AMBIGUOUS, EventType.WAIT):
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

    if getattr(result, "followup_disposition", None) in (
        FollowupDisposition.ANSWER,
        FollowupDisposition.PARK,
    ):
        return True

    if str(getattr(result, "followup_query", "") or "").strip():
        return True

    target = str(getattr(result, "update_target", "") or "").strip().lower()
    return bool(target) and target not in {s.strip().lower() for s in owned_slots}


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
