<!--
_confirmation_contract.md — the read-back rules, in ONE place.

This is a partial, not an agent prompt. Every extraction prompt gets it, and
gets the same copy: build_extraction_prompt, build_extraction_prompt_core and
build_extraction_prompt_extraction all compose it between their header and the
agent file. The leading underscore marks it as a shared fragment.

It exists for the same reason _followup_contract.md does. The rules below were
written once, inside delivery_management.md, after a caller was read their own
stale fax number back. Six other prompts collect a confirmation slot —
provider_search, notification_setup, records_coordination, name_confirmation,
verification_claims, header_extraction — and none of them had the rule, so
whether "yeah, but actually…" was heard depended on which value happened to be
under confirmation:

    AI      The fax number we have on file is 4155553211. Is this correct?
    Caller  Yeah. That's kind of an old fax number. I'll give you a new
            number if you can do that.
    →       fax_confirmed "no"                      (rule present)

    AI      Just to confirm — your ZIP code is 16783?
    Caller  yeah. Actually, you know what? I want to update the ZIP code
            because I moved to a new address. So can I do that now?
    →       zip_confirmed "yes"                     (rule absent)

The second sent the caller to the delivery question with the ZIP they were in
the middle of replacing still on file.

Change the rules here and every agent changes with them. Do not restate any of
this in a header or an agent file — a second copy is how the drift started.
-->

## THE READ-BACK CONTRACT — the same on every confirmation slot

A **read-back** asks the caller to confirm a value already on file: "your ZIP
code is 58797, correct?", "I'll send it to 2155553211 — is that the right
fax?". The confirmation field (`zip_confirmed`, `fax_confirmed`,
`email_confirmed`, `phone_confirmed`, …) carries their answer.

Three answers matter, and they are not symmetrical.

- **An affirmation** is a closed set: "yes", "yeah", "yep", "correct",
  "that's right". People have a handful of ways to agree and are not
  inventing more.
- **A replacement value** is a SHAPE — a ZIP, a fax, an email — recognised by
  its form however it is introduced.
- **A decline** is endless: "I moved", "that's my old one", "we relocated last
  spring", "that hasn't been right since the divorce". No list of these is
  ever finished.

So: recognise the first two, and treat **everything else the caller SAID as a
decline** → `"no"`. You do not need to match a phrasing to decline, and asking
for the current value is always safer than reading the same one back.

`"unusable"` has exactly two homes, both named at the end of this contract: the
caller who does not know, and the turn that carried no speech at all. Outside
those two, an intelligible turn that is not an affirmation is a decline.

### The leading affirmative does not decide the turn

A caller may open with "yeah", "yes", "sure" or "right" and then immediately
qualify it. **The qualifying content decides, not the opening word.** The
leading affirmative acknowledges the question; the rest answers it.

| Caller just said | confirmation field |
|------------------|--------------------|
| "Yeah. That's kind of an old fax number. I'll give you a new number." | "no" |
| "yeah. Actually, you know what? I want to update the ZIP code because I moved." | "no" |
| "Yes, but that number has changed." | "no" |
| "Right, although I'd want to update that." | "no" |
| "Yeah, that's right." | "yes" |
| "Yep, that's the one." | "yes" |

Read the WHOLE utterance before setting the field. A turn that contains any
statement that the value is wrong, outdated, or should change is a decline,
whatever word it opens with.

### A proposal to change the value IS the answer

Declines arrive as statements, as questions, and as offers. All three are
declines, and all three set the field to `"no"`:

- statement — "that's my old email", "it needs to be updated"
- question — "can you use a different fax?", "is it possible to send it to
  another number?", "so can I do that now?"
- offer — "I'll give you a new number if you can do that", "let me give you
  the current one", "I can give you a better one"

**Never report one of these only as `followup_query` with the confirmation
field left empty.** A proposal to change the value being read back is the
answer to the read-back, not a side question riding alongside it. Reported as
a side question it reads as a caller who took no position, and the value they
just rejected gets read back to them a second time — or, worse, treated as
confirmed.

### The first case that is NEITHER: the caller does not know

The caller genuinely does not know: "maybe", "I'm not sure", "I think so?",
"not sure if that's still active" → `turn_intent` "unusable", leave the
confirmation field empty.

Keep this narrow. It is the difference between a caller who CANNOT answer and
one who is answering no. "I moved recently" is a DECLINE — they know the value
is wrong. "I'm not sure if that's still right" is "unusable" — they do not know.

### The second case that is NEITHER: nothing was heard

A decline is something the caller SAID. An utterance that is not words at all
— "csaxv", a stray syllable, a fragment of line noise — said nothing, and
"everything else is a decline" does not reach it, because there is no position
in it to read.

    AI      Just to confirm — your ZIP code is 12139?
    Caller  csaxv
    →       turn_intent "unusable", zip_confirmed EMPTY

Read as a decline instead, a garbled second of audio becomes "the caller says
this value is wrong", and they are asked to replace something they never
mentioned — while the retry budget, which exists for exactly this, never
engages, because the turn was reported as an answer rather than an unusable
one.

Still narrow, and narrow in a different direction from the case above. That one
turns on whether the caller KNOWS; this one turns on whether they SPOKE. So
anything INTELLIGIBLE that is not an affirmation remains a DECLINE, whatever it
says and however little it engages with the question — "I moved", "that's my
old one", "ugh, don't get me started", "my daughter handles all this" are all
declines. Only genuine non-speech is "unusable" here.
