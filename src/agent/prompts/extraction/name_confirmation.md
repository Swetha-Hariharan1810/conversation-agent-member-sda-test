ROLE: Extract the member's response to a name readback / confirmation.

The agent just read back the member's full name, spelled letter by letter, and asked
"is that correct?". There are exactly three valid outcomes — extract accordingly.

FIELDS
  name_confirmed  "yes" | "no"
    "yes" — member confirmed the name is correct:
      "yes", "yep", "correct", "that's right", "yes that's me",
      "yes that's correct", "yes that's my name" → yes

    "no" — member rejected the name (bare no, no replacement given):
      "no", "nope", "that's wrong", "that's not right",
      "no that's not me", "incorrect" → no
      Only extract "no" here when the member gives NO new name in the same utterance.

  first_name  Title Case string
    The corrected first name, when the member declines AND gives the right name inline:
      "no it's Jhon", "no, my name is Jhon Doe", "actually it's Jhon" → Jhon
    Only extract when the member is actively providing a replacement first name.

  last_name  Title Case string
    The corrected last name, when the member declines AND gives the right name inline.
    Often given together with first_name in the same utterance.
    "surname", "family name", "second name" and "maiden name" all mean last_name.
    "given name" and "forename" mean first_name.

CRITICAL EXTRACTION RULE — three outcomes, mutually exclusive:

  OUTCOME 1 — confirmed (yes):
    name_confirmed = "yes", first_name omitted, last_name omitted

  OUTCOME 2 — declined with inline correction:
    name_confirmed omitted, first_name = "<corrected>", last_name = "<corrected>"
    (or just first_name if only first name was given)
    This fires when the member says the wrong name AND provides the correct one
    in the same utterance. Do NOT also set name_confirmed = "no".

  OUTCOME 3 — bare no (no correction given):
    name_confirmed = "no", first_name omitted, last_name omitted
    Only use when the member clearly rejected the name but gave no replacement.

NAMING THE PART, AND CONTRASTING IT — the two shapes that are still OUTCOME 2:

  The member may say WHICH HALF of the name is wrong rather than repeating the
  whole name. Extract that half; leave the other one alone.
    "my surname is Carter"        → last_name="Carter"
    "my first name is Emma"       → first_name="Emma"

  The member may name the correction and the value it replaces in one breath,
  as "<correct>, not <wrong>". The trailing "not <wrong>" is part of the
  correction — it names what is being replaced. It is NEVER a bare rejection,
  even though it reads like the OUTCOME 3 phrases above, and the name after
  "not" is NEVER the correction.
    "my surname is Carter, not Watson"   → last_name="Carter"  (NOT "Watson",
                                            and NOT name_confirmed="no")
    "it's Carter, not Watson"            → last_name="Carter"
    "Emma, not Emily"                    → first_name="Emma"

  A member who says only what is wrong, with no replacement, is still OUTCOME 3:
    "my surname is not Watson"           → name_confirmed="no"
    "my last name is wrong"              → name_confirmed="no"

Examples:
  "yes that's correct"                     → name_confirmed="yes"
  "yes"                                    → name_confirmed="yes"
  "yep, that's me"                         → name_confirmed="yes"
  "no"                                     → name_confirmed="no"
  "nope that's wrong"                      → name_confirmed="no"
  "no it's Jhon"                           → first_name="Jhon"
  "no, it's Jhon Doe"                   → first_name="Jhon", last_name="Doe"
  "actually my name is Jhon Doe"        → first_name="Jhon", last_name="Doe"
  "no that's not right, it's Jhon Doe"  → first_name="Jhon", last_name="Doe"
  "actually, my surname is Carter, not Watson" → last_name="Carter"
  "yes j h o n d o e"             → name_confirmed="yes"

SPELLED-OUT CORRECTIONS — letter-by-letter responses:
  When the caller spells out a name letter by letter in direct response to the
  readback, this is ALWAYS an inline correction (OUTCOME 2) — even without an
  explicit "no" or "wrong" prefix. The caller is providing the correct spelling.
  Reconstruct the spelled letters into the word they form.

  Spelling that DIFFERS from what was read back → OUTCOME 2 (inline correction):
    "R e e d"  (when readback showed "R-E-A-D") → last_name="Reed"
    "r e e d"                                    → last_name="Reed"
    "S m i t h"                                  → last_name="Smith"
    "J o h n s o n"                              → last_name="Johnson"
    "e m i l y"  (when readback showed different first name) → first_name="Emily"

  Spelling that MATCHES what was read back exactly → OUTCOME 1 (confirmed):
    "y e s" or confirmed form of exactly the same letters → name_confirmed="yes"

  NEVER treat a spelled correction as AMBIGUOUS when the letters clearly form
  a plausible name.

NAME PLAUSIBILITY — same rule as verification:
  Accept any plausible human name (any culture, hyphenated, apostrophe).
  Reject only clear non-names (numbers, gibberish).

CONFIDENCE NOTES
  Only extract when the member's intent is unambiguous.
  Hesitation or spelling filler words ("um", "let me think") → event_type "ambiguous".
  If the member only gives a first name (no last name), extract first_name only.
