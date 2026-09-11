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

    if getattr(result, "followup_disposition", None) in (
        FollowupDisposition.ANSWER,
        FollowupDisposition.PARK,
    ):
        return True

    if str(getattr(result, "followup_query", "") or "").strip():
        return True

    target = str(getattr(result, "update_target", "") or "").strip().lower()
    return bool(target) and target not in {s.strip().lower() for s in owned_slots}
