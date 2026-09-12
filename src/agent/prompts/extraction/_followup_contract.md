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
    →       event_type "answered", followup_query null

    AI      …send the details of our Care     (care_coach_response,
            Coach Guides?                      header_core.md)
    Caller  That sounds interesting, but I
            lost my ID card. Can you help
            me to get a new one?
    →       event_type "answered_with_followup",
            followup_query "can you help me to get a new one"

The same request, two turns apart, classified both ways. header_core.md did not
describe answered_with_followup at all; header.md and header_extraction.md each
described it, and followup_disposition, in their own words.

Nineteen agent prompts and three headers cannot be kept in step by hand, so
they are no longer asked to be. Change the rules here and every agent changes
with them. Do not restate any of this in a header or an agent file — a second
copy is how the drift started.
-->

## THE FOLLOW-UP CONTRACT — the same on every slot, in every agent

A **side question** is something the caller asked or requested THIS TURN,
alongside (or instead of) answering the question they were asked.

`followup_query` carries it. A non-null value routes the turn to a second model
whose only job is to answer that line, so the field has to be right in both
directions: a question invented costs the caller a sentence answering something
they never asked, and a question missed is never heard again — nothing later in
the call comes back to it.

### The test — apply it before setting `answered_with_followup`

**Point at the words in the "Caller just said:" line that are the question.**
If you cannot quote them, there is no follow-up: use `answered`, and leave
`followup_query` null.

Two mistakes to avoid, both seen in production:

- Do NOT turn the caller's own answer into a follow-up. "Monique" answers the
  question; it does not also ask one.
- Do NOT turn a topic the AI raised into a follow-up. If the AI said "I can help
  with your claim status" two turns ago, "help with claim status" is not
  something the caller asked — it is something you read in your own history.

| Caller just said                        | event_type | followup_query |
|-----------------------------------------|------------|----------------|
| "Customer."                             | answered   | null           |
| "M451982."                              | answered   | null           |
| "Yes, that's right."                    | answered   | null           |
| "November 5th, 1992."                   | answered   | null           |
| "Smith — sorry, bad line."              | answered   | null           |
| "Fax please. Can you do that for me today?" | answered | null — the question is about the answer just given |
| "90210, and when will I get the list?"  | answered_with_followup | "when will the list arrive" |
| "No. But I lost my ID card. Can you help me with a new one?" | answered_with_followup | "I lost my ID card, can you help me get a new one" |

### What counts as a side question

Any of these, when they appear in the caller's own words this turn:

- a repeat request — "can you say that again", "sorry what was that"
- a confirmation request — "did you get that", "is that right"
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

### The two event types that carry one

- **`answered_with_followup`** — the caller answered the awaiting slot AND asked
  something. `extracted{}` must contain the slot value. Never use this event
  type when the utterance contains only the slot value.
- **`ambiguous`** — the caller asked INSTEAD of answering, so there is no value
  to extract ("what do you need exactly?", "what format?", "why do you need
  that?"). Set `followup_query` here too: the system answers the question
  before re-asking the slot.

### Quoting it

`followup_query` is the caller's question condensed — their words, not a
paraphrase of the topic. "I lost my ID card, can you help me get a new one"
beats "ID card". Keep enough of it that the question says what it is about: a
bare "can you help me with the new one" does not.

A question about the timing or status of something the agent just PROMISED
("when?", "when will you update my zip?", "did you change it yet?") is NOT a
slot answer and must never be extracted as one. It is a side question like any
other; `followup_query` = "timing of <promised item>".

### `followup_disposition`

Leave it `"none"`. The system decides what happens to a side question — every
value you could set means the same thing to it. Reporting the question
accurately is the whole job.
