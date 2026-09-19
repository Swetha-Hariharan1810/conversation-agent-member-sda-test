<!--
_followup_contract.md — the side-question rules, in ONE place.

This is a partial, not an agent prompt. Every extraction prompt gets it, and
gets the same copy: build_extraction_prompt, build_extraction_prompt_core and
build_extraction_prompt_extraction all compose it between their header and the
agent file. The leading underscore marks it as a shared fragment.

It exists because these rules used to live inside each of the three headers,
written three different ways, and a caller's question was heard or not
depending on which header the slot they were on happened to use:

    AI      …would that be helpful?          (benefits_response,
    Caller  No. But I lost my credit ID       header_extraction.md)
            card. Can you help me with
            the new one?
    →       followup_query null

    AI      …send the details of our Care     (care_coach_response,
            Coach Guides?                      header_core.md)
    Caller  That sounds interesting, but I
            lost my ID card. Can you help
            me to get a new one?
    →       followup_query "can you help me to get a new one"

The same request, two turns apart, heard one time in two. header_core.md did not
describe a side question at all; header.md and header_extraction.md each
described it, and followup_disposition, in their own words.

Nineteen agent prompts and three headers cannot be kept in step by hand, so
they are no longer asked to be. Change the rules here and every agent changes
with them. Do not restate any of this in a header or an agent file — a second
copy is how the drift started.

WHY THIS FILE CARRIES MORE THAN IT USED TO
------------------------------------------
Python used to sit behind this contract and second-guess it, in four places:

  - request_detection._reconcile_followup_query   (the grounding veto)
  - request_detection._recover_missed_followup    (the regex recovery)
  - slot_manager.note_side_question               (the veto again, per agent)
  - response_generator.needs_freeform_response    (a regex cue on the utterance)

All four are gone. followup_query is the extraction model's call and nothing
downstream repairs it, so the judgements those layers made in regex now have to
be made here, in the prompt, on the model's one read of the turn:

  - "is there a question in the caller's words at all"  → THE TEST, step 2
  - "is the reported question lifted from the AI's own  → YOU MAY ONLY QUOTE
     earlier turns"
  - "did the caller answer and THEN ask"                → THE TEST, step 1
  - "is the question about the answer just given"       → THE TEST, step 3

Each bullet below that looks oddly specific is a production transcript. Keep
them specific; the failures they describe are the ones that recur.
-->

## THE FOLLOW-UP CONTRACT — the same on every slot, in every agent

A **side question** is something the caller asked or requested THIS TURN,
alongside (or instead of) answering the question they were asked.

`followup_query` carries it, and it is the ONLY thing that decides whether the
turn is handed to a second model to answer. Nothing downstream re-checks this
field. A question you invent is answered aloud to a caller who never asked it;
a question you drop is never heard again, on this turn or any repeat of it —
nothing later in the call comes back to it. Run the test below every turn.

### The test — three steps, in order

**1. Split what the caller just said.** Cut it at sentence ends, and at the
words that carry a second thought: "but", "however", "though", "by the way",
"one more thing", "quick question", "and also". Plain "and" does NOT split a
thought — "M451982 and my DOB is November 5th" is one answer in two parts.

**2. Find a part that ASKS.** A part asks when it puts a question to you ("can
you…", "could you…", "what about…", "how do I…", "is there…", "when will…"),
or states a request flat ("I need…", "I want…", "I lost my ID card", "help me
with…"). A voice caller often asks without a question mark, so do not rely on
one. If NO part asks, `followup_query` is null — stop here.

**3. Check the asking part against the answering part.** Report it only when it
brings up something the caller's answer did not:

- **New subject → report it.** Quote from the first asking part through to the
  end of the utterance, so the question says what it is about: "I lost my ID
  card, can you help me get a new one" — not "can you help me with the new
  one", which names nothing.
- **About the very thing just answered → null.** "Fax please. Can you do that
  for me today?" asks about the fax they just chose; saying yes to the fax is
  already the whole answer. Same for "does that work?", "is that okay?", "can
  you manage that?" — courtesy attached to an answer, not a second topic. The
  test is the subject, not the wording: if every word of the question is about
  the value just captured or the slot it filled, there is no side question,
  whatever verb the caller reached for.
- **The whole utterance asks and nothing answers → `turn_intent` "unusable"**,
  nothing in `extracted{}`, the question in `followup_query`.

### You may only quote — never supply

**Point at the words in the "Caller just said:" line that are the question.**
If you cannot quote them, there is no follow-up: leave `followup_query` null.

Three ways this field gets filled with something the caller never said:

- **From their own answer.** "Monique" answers the question; it does not also
  ask one. Neither does "yes", "90210", or "November 5th, 1992".
- **From your own history.** If the AI said "I can help with your claim status"
  two turns ago, "help with claim status" is not something the caller asked —
  it is something you read in the transcript above. A `followup_query` that
  shares no subject word with the "Caller just said:" line came from there.
  Delete it.
- **From a decline.** When the AI has just read a value back for confirmation,
  anything proposing a change to that value — "can you use a different one?",
  "I'll give you a new number if you can do that" — is the ANSWER to the
  read-back, not a question riding alongside it. Answer the confirmation field
  and leave `followup_query` null. Reported as a side question it reads as a
  caller who took no position, and the value they just rejected is read back to
  them again.

| Caller just said                        | turn_intent | followup_query |
|-----------------------------------------|-------------|----------------|
| "Customer."                             | answered    | null |
| "M451982."                              | answered    | null |
| "Yes, that's right."                    | answered    | null |
| "November 5th, 1992."                   | answered    | null |
| "Smith — sorry, bad line."              | answered    | null |
| "Fax please. Can you do that for me today?" | answered | null — the question is about the answer just given |
| "Yeah, that's my old fax. I'll give you a new number if you can do that." | answered | null — offering a replacement IS the answer to the read-back |
| "90210, and when will I get the list?"  | answered    | "when will the list arrive" |
| "No. But I lost my ID card. Can you help me with a new one?" | answered | "I lost my ID card, can you help me get a new one" |
| "Two. I know fax is easier, but I might have to send it to a few people. So can you just fax it to me?" | answered | null — the question is the delivery choice just made |
| "What do you need exactly?"             | unusable    | "what do you need exactly" |
| "Please check my claim status today."   | unusable    | "check my claim status" |

### What counts as a side question

Any of these, when they appear in the caller's own words this turn and step 3
above clears them:

- a repeat request — "can you say that again", "sorry what was that"
- a question this call can answer from what is already known — "what email do
  you have for me?"
- a question about a step still ahead — "will I get a text when it's sent?"
- a question or request this call does not handle at all — "I lost my ID card,
  can you help me get a new one?", "do you sell car insurance?"
- format uncertainty about their own answer — "I think it's…", "not sure if
  that's right"

The last two still belong in `followup_query`. Whether the system can act on a
question is never your decision — it answers from what it knows, or declines
gracefully, and it can do neither if you did not report the question.

"Did you get that?" is the one shape that goes both ways, and step 3 settles
it: asked after the caller gave a value, it is courtesy about that value and
`followup_query` is null; asked when they gave no value, it is a repeat request
and belongs in the field.

### It rides alongside `turn_intent`, it does not replace it

`followup_query` is its own field, so a question never costs you the
classification of the turn:

- the caller answered AND asked → `turn_intent` "answered", the value in
  `extracted{}`, the question in `followup_query`
- the caller asked INSTEAD of answering → `turn_intent` "unusable", nothing in
  `extracted{}`, the question in `followup_query`. The system answers it before
  re-asking the slot.

### Quoting it

`followup_query` is the caller's question condensed — their words, not a
paraphrase of the topic. "I lost my ID card, can you help me get a new one"
beats "ID card". Keep enough of it that the question says what it is about: a
bare "can you help me with the new one" does not. Keep it under thirty words;
past that you are transcribing the turn, not quoting the question.

A question about the timing or status of something the agent just PROMISED
("when?", "when will you update my zip?", "did you change it yet?") is NOT a
slot answer and must never be extracted as one. It is a side question like any
other; `followup_query` = "timing of <promised item>".
